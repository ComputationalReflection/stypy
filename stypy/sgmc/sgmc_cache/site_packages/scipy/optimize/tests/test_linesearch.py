
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Tests for line search routines
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: from numpy.testing import assert_, assert_equal, \
7:      assert_array_almost_equal, assert_array_almost_equal_nulp, assert_warns
8: from scipy._lib._numpy_compat import suppress_warnings
9: import scipy.optimize.linesearch as ls
10: from scipy.optimize.linesearch import LineSearchWarning
11: import numpy as np
12: 
13: 
14: def assert_wolfe(s, phi, derphi, c1=1e-4, c2=0.9, err_msg=""):
15:     '''
16:     Check that strong Wolfe conditions apply
17:     '''
18:     phi1 = phi(s)
19:     phi0 = phi(0)
20:     derphi0 = derphi(0)
21:     derphi1 = derphi(s)
22:     msg = "s = %s; phi(0) = %s; phi(s) = %s; phi'(0) = %s; phi'(s) = %s; %s" % (
23:         s, phi0, phi1, derphi0, derphi1, err_msg)
24: 
25:     assert_(phi1 <= phi0 + c1*s*derphi0, "Wolfe 1 failed: " + msg)
26:     assert_(abs(derphi1) <= abs(c2*derphi0), "Wolfe 2 failed: " + msg)
27: 
28: 
29: def assert_armijo(s, phi, c1=1e-4, err_msg=""):
30:     '''
31:     Check that Armijo condition applies
32:     '''
33:     phi1 = phi(s)
34:     phi0 = phi(0)
35:     msg = "s = %s; phi(0) = %s; phi(s) = %s; %s" % (s, phi0, phi1, err_msg)
36:     assert_(phi1 <= (1 - c1*s)*phi0, msg)
37: 
38: 
39: def assert_line_wolfe(x, p, s, f, fprime, **kw):
40:     assert_wolfe(s, phi=lambda sp: f(x + p*sp),
41:                  derphi=lambda sp: np.dot(fprime(x + p*sp), p), **kw)
42: 
43: 
44: def assert_line_armijo(x, p, s, f, **kw):
45:     assert_armijo(s, phi=lambda sp: f(x + p*sp), **kw)
46: 
47: 
48: def assert_fp_equal(x, y, err_msg="", nulp=50):
49:     '''Assert two arrays are equal, up to some floating-point rounding error'''
50:     try:
51:         assert_array_almost_equal_nulp(x, y, nulp)
52:     except AssertionError as e:
53:         raise AssertionError("%s\n%s" % (e, err_msg))
54: 
55: 
56: class TestLineSearch(object):
57:     # -- scalar functions; must have dphi(0.) < 0
58:     def _scalar_func_1(self, s):
59:         self.fcount += 1
60:         p = -s - s**3 + s**4
61:         dp = -1 - 3*s**2 + 4*s**3
62:         return p, dp
63: 
64:     def _scalar_func_2(self, s):
65:         self.fcount += 1
66:         p = np.exp(-4*s) + s**2
67:         dp = -4*np.exp(-4*s) + 2*s
68:         return p, dp
69: 
70:     def _scalar_func_3(self, s):
71:         self.fcount += 1
72:         p = -np.sin(10*s)
73:         dp = -10*np.cos(10*s)
74:         return p, dp
75: 
76:     # -- n-d functions
77: 
78:     def _line_func_1(self, x):
79:         self.fcount += 1
80:         f = np.dot(x, x)
81:         df = 2*x
82:         return f, df
83: 
84:     def _line_func_2(self, x):
85:         self.fcount += 1
86:         f = np.dot(x, np.dot(self.A, x)) + 1
87:         df = np.dot(self.A + self.A.T, x)
88:         return f, df
89: 
90:     # --
91: 
92:     def setup_method(self):
93:         self.scalar_funcs = []
94:         self.line_funcs = []
95:         self.N = 20
96:         self.fcount = 0
97: 
98:         def bind_index(func, idx):
99:             # Remember Python's closure semantics!
100:             return lambda *a, **kw: func(*a, **kw)[idx]
101: 
102:         for name in sorted(dir(self)):
103:             if name.startswith('_scalar_func_'):
104:                 value = getattr(self, name)
105:                 self.scalar_funcs.append(
106:                     (name, bind_index(value, 0), bind_index(value, 1)))
107:             elif name.startswith('_line_func_'):
108:                 value = getattr(self, name)
109:                 self.line_funcs.append(
110:                     (name, bind_index(value, 0), bind_index(value, 1)))
111: 
112:         np.random.seed(1234)
113:         self.A = np.random.randn(self.N, self.N)
114: 
115:     def scalar_iter(self):
116:         for name, phi, derphi in self.scalar_funcs:
117:             for old_phi0 in np.random.randn(3):
118:                 yield name, phi, derphi, old_phi0
119: 
120:     def line_iter(self):
121:         for name, f, fprime in self.line_funcs:
122:             k = 0
123:             while k < 9:
124:                 x = np.random.randn(self.N)
125:                 p = np.random.randn(self.N)
126:                 if np.dot(p, fprime(x)) >= 0:
127:                     # always pick a descent direction
128:                     continue
129:                 k += 1
130:                 old_fv = float(np.random.randn())
131:                 yield name, f, fprime, x, p, old_fv
132: 
133:     # -- Generic scalar searches
134: 
135:     def test_scalar_search_wolfe1(self):
136:         c = 0
137:         for name, phi, derphi, old_phi0 in self.scalar_iter():
138:             c += 1
139:             s, phi1, phi0 = ls.scalar_search_wolfe1(phi, derphi, phi(0),
140:                                                     old_phi0, derphi(0))
141:             assert_fp_equal(phi0, phi(0), name)
142:             assert_fp_equal(phi1, phi(s), name)
143:             assert_wolfe(s, phi, derphi, err_msg=name)
144: 
145:         assert_(c > 3)  # check that the iterator really works...
146: 
147:     def test_scalar_search_wolfe2(self):
148:         for name, phi, derphi, old_phi0 in self.scalar_iter():
149:             s, phi1, phi0, derphi1 = ls.scalar_search_wolfe2(
150:                 phi, derphi, phi(0), old_phi0, derphi(0))
151:             assert_fp_equal(phi0, phi(0), name)
152:             assert_fp_equal(phi1, phi(s), name)
153:             if derphi1 is not None:
154:                 assert_fp_equal(derphi1, derphi(s), name)
155:             assert_wolfe(s, phi, derphi, err_msg="%s %g" % (name, old_phi0))
156: 
157:     def test_scalar_search_armijo(self):
158:         for name, phi, derphi, old_phi0 in self.scalar_iter():
159:             s, phi1 = ls.scalar_search_armijo(phi, phi(0), derphi(0))
160:             assert_fp_equal(phi1, phi(s), name)
161:             assert_armijo(s, phi, err_msg="%s %g" % (name, old_phi0))
162: 
163:     # -- Generic line searches
164: 
165:     def test_line_search_wolfe1(self):
166:         c = 0
167:         smax = 100
168:         for name, f, fprime, x, p, old_f in self.line_iter():
169:             f0 = f(x)
170:             g0 = fprime(x)
171:             self.fcount = 0
172:             s, fc, gc, fv, ofv, gv = ls.line_search_wolfe1(f, fprime, x, p,
173:                                                            g0, f0, old_f,
174:                                                            amax=smax)
175:             assert_equal(self.fcount, fc+gc)
176:             assert_fp_equal(ofv, f(x))
177:             if s is None:
178:                 continue
179:             assert_fp_equal(fv, f(x + s*p))
180:             assert_array_almost_equal(gv, fprime(x + s*p), decimal=14)
181:             if s < smax:
182:                 c += 1
183:                 assert_line_wolfe(x, p, s, f, fprime, err_msg=name)
184: 
185:         assert_(c > 3)  # check that the iterator really works...
186: 
187:     def test_line_search_wolfe2(self):
188:         c = 0
189:         smax = 512
190:         for name, f, fprime, x, p, old_f in self.line_iter():
191:             f0 = f(x)
192:             g0 = fprime(x)
193:             self.fcount = 0
194:             with suppress_warnings() as sup:
195:                 sup.filter(LineSearchWarning,
196:                            "The line search algorithm could not find a solution")
197:                 sup.filter(LineSearchWarning,
198:                            "The line search algorithm did not converge")
199:                 s, fc, gc, fv, ofv, gv = ls.line_search_wolfe2(f, fprime, x, p,
200:                                                                g0, f0, old_f,
201:                                                                amax=smax)
202:             assert_equal(self.fcount, fc+gc)
203:             assert_fp_equal(ofv, f(x))
204:             assert_fp_equal(fv, f(x + s*p))
205:             if gv is not None:
206:                 assert_array_almost_equal(gv, fprime(x + s*p), decimal=14)
207:             if s < smax:
208:                 c += 1
209:                 assert_line_wolfe(x, p, s, f, fprime, err_msg=name)
210:         assert_(c > 3)  # check that the iterator really works...
211: 
212:     def test_line_search_wolfe2_bounds(self):
213:         # See gh-7475
214: 
215:         # For this f and p, starting at a point on axis 0, the strong Wolfe
216:         # condition 2 is met if and only if the step length s satisfies
217:         # |x + s| <= c2 * |x|
218:         f = lambda x: np.dot(x, x)
219:         fp = lambda x: 2 * x
220:         p = np.array([1, 0])
221: 
222:         # Smallest s satisfying strong Wolfe conditions for these arguments is 30
223:         x = -60 * p
224:         c2 = 0.5
225: 
226:         s, _, _, _, _, _ = ls.line_search_wolfe2(f, fp, x, p, amax=30, c2=c2)
227:         assert_line_wolfe(x, p, s, f, fp)
228: 
229:         s, _, _, _, _, _ = assert_warns(LineSearchWarning,
230:                                         ls.line_search_wolfe2, f, fp, x, p,
231:                                         amax=29, c2=c2)
232:         assert_(s is None)
233: 
234:         # s=30 will only be tried on the 6th iteration, so this won't converge
235:         assert_warns(LineSearchWarning, ls.line_search_wolfe2, f, fp, x, p,
236:                      c2=c2, maxiter=5)
237: 
238:     def test_line_search_armijo(self):
239:         c = 0
240:         for name, f, fprime, x, p, old_f in self.line_iter():
241:             f0 = f(x)
242:             g0 = fprime(x)
243:             self.fcount = 0
244:             s, fc, fv = ls.line_search_armijo(f, x, p, g0, f0)
245:             c += 1
246:             assert_equal(self.fcount, fc)
247:             assert_fp_equal(fv, f(x + s*p))
248:             assert_line_armijo(x, p, s, f, err_msg=name)
249:         assert_(c >= 9)
250: 
251:     # -- More specific tests
252: 
253:     def test_armijo_terminate_1(self):
254:         # Armijo should evaluate the function only once if the trial step
255:         # is already suitable
256:         count = [0]
257: 
258:         def phi(s):
259:             count[0] += 1
260:             return -s + 0.01*s**2
261:         s, phi1 = ls.scalar_search_armijo(phi, phi(0), -1, alpha0=1)
262:         assert_equal(s, 1)
263:         assert_equal(count[0], 2)
264:         assert_armijo(s, phi)
265: 
266:     def test_wolfe_terminate(self):
267:         # wolfe1 and wolfe2 should also evaluate the function only a few
268:         # times if the trial step is already suitable
269: 
270:         def phi(s):
271:             count[0] += 1
272:             return -s + 0.05*s**2
273: 
274:         def derphi(s):
275:             count[0] += 1
276:             return -1 + 0.05*2*s
277: 
278:         for func in [ls.scalar_search_wolfe1, ls.scalar_search_wolfe2]:
279:             count = [0]
280:             r = func(phi, derphi, phi(0), None, derphi(0))
281:             assert_(r[0] is not None, (r, func))
282:             assert_(count[0] <= 2 + 2, (count, func))
283:             assert_wolfe(r[0], phi, derphi, err_msg=str(func))
284: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_209121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nTests for line search routines\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_, assert_equal, assert_array_almost_equal, assert_array_almost_equal_nulp, assert_warns' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_209122 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_209122) is not StypyTypeError):

    if (import_209122 != 'pyd_module'):
        __import__(import_209122)
        sys_modules_209123 = sys.modules[import_209122]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_209123.module_type_store, module_type_store, ['assert_', 'assert_equal', 'assert_array_almost_equal', 'assert_array_almost_equal_nulp', 'assert_warns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_209123, sys_modules_209123.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal, assert_array_almost_equal, assert_array_almost_equal_nulp, assert_warns

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal', 'assert_array_almost_equal', 'assert_array_almost_equal_nulp', 'assert_warns'], [assert_, assert_equal, assert_array_almost_equal, assert_array_almost_equal_nulp, assert_warns])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_209122)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_209124 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat')

if (type(import_209124) is not StypyTypeError):

    if (import_209124 != 'pyd_module'):
        __import__(import_209124)
        sys_modules_209125 = sys.modules[import_209124]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', sys_modules_209125.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_209125, sys_modules_209125.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', import_209124)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import scipy.optimize.linesearch' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_209126 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize.linesearch')

if (type(import_209126) is not StypyTypeError):

    if (import_209126 != 'pyd_module'):
        __import__(import_209126)
        sys_modules_209127 = sys.modules[import_209126]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'ls', sys_modules_209127.module_type_store, module_type_store)
    else:
        import scipy.optimize.linesearch as ls

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'ls', scipy.optimize.linesearch, module_type_store)

else:
    # Assigning a type to the variable 'scipy.optimize.linesearch' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize.linesearch', import_209126)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.optimize.linesearch import LineSearchWarning' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_209128 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize.linesearch')

if (type(import_209128) is not StypyTypeError):

    if (import_209128 != 'pyd_module'):
        __import__(import_209128)
        sys_modules_209129 = sys.modules[import_209128]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize.linesearch', sys_modules_209129.module_type_store, module_type_store, ['LineSearchWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_209129, sys_modules_209129.module_type_store, module_type_store)
    else:
        from scipy.optimize.linesearch import LineSearchWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize.linesearch', None, module_type_store, ['LineSearchWarning'], [LineSearchWarning])

else:
    # Assigning a type to the variable 'scipy.optimize.linesearch' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.optimize.linesearch', import_209128)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_209130 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_209130) is not StypyTypeError):

    if (import_209130 != 'pyd_module'):
        __import__(import_209130)
        sys_modules_209131 = sys.modules[import_209130]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_209131.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_209130)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def assert_wolfe(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_209132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'float')
    float_209133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 45), 'float')
    str_209134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 58), 'str', '')
    defaults = [float_209132, float_209133, str_209134]
    # Create a new context for function 'assert_wolfe'
    module_type_store = module_type_store.open_function_context('assert_wolfe', 14, 0, False)
    
    # Passed parameters checking function
    assert_wolfe.stypy_localization = localization
    assert_wolfe.stypy_type_of_self = None
    assert_wolfe.stypy_type_store = module_type_store
    assert_wolfe.stypy_function_name = 'assert_wolfe'
    assert_wolfe.stypy_param_names_list = ['s', 'phi', 'derphi', 'c1', 'c2', 'err_msg']
    assert_wolfe.stypy_varargs_param_name = None
    assert_wolfe.stypy_kwargs_param_name = None
    assert_wolfe.stypy_call_defaults = defaults
    assert_wolfe.stypy_call_varargs = varargs
    assert_wolfe.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_wolfe', ['s', 'phi', 'derphi', 'c1', 'c2', 'err_msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_wolfe', localization, ['s', 'phi', 'derphi', 'c1', 'c2', 'err_msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_wolfe(...)' code ##################

    str_209135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Check that strong Wolfe conditions apply\n    ')
    
    # Assigning a Call to a Name (line 18):
    
    # Assigning a Call to a Name (line 18):
    
    # Call to phi(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 's' (line 18)
    s_209137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 's', False)
    # Processing the call keyword arguments (line 18)
    kwargs_209138 = {}
    # Getting the type of 'phi' (line 18)
    phi_209136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'phi', False)
    # Calling phi(args, kwargs) (line 18)
    phi_call_result_209139 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), phi_209136, *[s_209137], **kwargs_209138)
    
    # Assigning a type to the variable 'phi1' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'phi1', phi_call_result_209139)
    
    # Assigning a Call to a Name (line 19):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to phi(...): (line 19)
    # Processing the call arguments (line 19)
    int_209141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_209142 = {}
    # Getting the type of 'phi' (line 19)
    phi_209140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'phi', False)
    # Calling phi(args, kwargs) (line 19)
    phi_call_result_209143 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), phi_209140, *[int_209141], **kwargs_209142)
    
    # Assigning a type to the variable 'phi0' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'phi0', phi_call_result_209143)
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to derphi(...): (line 20)
    # Processing the call arguments (line 20)
    int_209145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_209146 = {}
    # Getting the type of 'derphi' (line 20)
    derphi_209144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'derphi', False)
    # Calling derphi(args, kwargs) (line 20)
    derphi_call_result_209147 = invoke(stypy.reporting.localization.Localization(__file__, 20, 14), derphi_209144, *[int_209145], **kwargs_209146)
    
    # Assigning a type to the variable 'derphi0' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'derphi0', derphi_call_result_209147)
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to derphi(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 's' (line 21)
    s_209149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 's', False)
    # Processing the call keyword arguments (line 21)
    kwargs_209150 = {}
    # Getting the type of 'derphi' (line 21)
    derphi_209148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 14), 'derphi', False)
    # Calling derphi(args, kwargs) (line 21)
    derphi_call_result_209151 = invoke(stypy.reporting.localization.Localization(__file__, 21, 14), derphi_209148, *[s_209149], **kwargs_209150)
    
    # Assigning a type to the variable 'derphi1' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'derphi1', derphi_call_result_209151)
    
    # Assigning a BinOp to a Name (line 22):
    
    # Assigning a BinOp to a Name (line 22):
    str_209152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'str', "s = %s; phi(0) = %s; phi(s) = %s; phi'(0) = %s; phi'(s) = %s; %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_209153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    # Getting the type of 's' (line 23)
    s_209154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), tuple_209153, s_209154)
    # Adding element type (line 23)
    # Getting the type of 'phi0' (line 23)
    phi0_209155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'phi0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), tuple_209153, phi0_209155)
    # Adding element type (line 23)
    # Getting the type of 'phi1' (line 23)
    phi1_209156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'phi1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), tuple_209153, phi1_209156)
    # Adding element type (line 23)
    # Getting the type of 'derphi0' (line 23)
    derphi0_209157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'derphi0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), tuple_209153, derphi0_209157)
    # Adding element type (line 23)
    # Getting the type of 'derphi1' (line 23)
    derphi1_209158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'derphi1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), tuple_209153, derphi1_209158)
    # Adding element type (line 23)
    # Getting the type of 'err_msg' (line 23)
    err_msg_209159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 41), 'err_msg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), tuple_209153, err_msg_209159)
    
    # Applying the binary operator '%' (line 22)
    result_mod_209160 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 10), '%', str_209152, tuple_209153)
    
    # Assigning a type to the variable 'msg' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'msg', result_mod_209160)
    
    # Call to assert_(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Getting the type of 'phi1' (line 25)
    phi1_209162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'phi1', False)
    # Getting the type of 'phi0' (line 25)
    phi0_209163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'phi0', False)
    # Getting the type of 'c1' (line 25)
    c1_209164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'c1', False)
    # Getting the type of 's' (line 25)
    s_209165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 's', False)
    # Applying the binary operator '*' (line 25)
    result_mul_209166 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 27), '*', c1_209164, s_209165)
    
    # Getting the type of 'derphi0' (line 25)
    derphi0_209167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'derphi0', False)
    # Applying the binary operator '*' (line 25)
    result_mul_209168 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 31), '*', result_mul_209166, derphi0_209167)
    
    # Applying the binary operator '+' (line 25)
    result_add_209169 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 20), '+', phi0_209163, result_mul_209168)
    
    # Applying the binary operator '<=' (line 25)
    result_le_209170 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 12), '<=', phi1_209162, result_add_209169)
    
    str_209171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'str', 'Wolfe 1 failed: ')
    # Getting the type of 'msg' (line 25)
    msg_209172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 62), 'msg', False)
    # Applying the binary operator '+' (line 25)
    result_add_209173 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 41), '+', str_209171, msg_209172)
    
    # Processing the call keyword arguments (line 25)
    kwargs_209174 = {}
    # Getting the type of 'assert_' (line 25)
    assert__209161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 25)
    assert__call_result_209175 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), assert__209161, *[result_le_209170, result_add_209173], **kwargs_209174)
    
    
    # Call to assert_(...): (line 26)
    # Processing the call arguments (line 26)
    
    
    # Call to abs(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'derphi1' (line 26)
    derphi1_209178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'derphi1', False)
    # Processing the call keyword arguments (line 26)
    kwargs_209179 = {}
    # Getting the type of 'abs' (line 26)
    abs_209177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'abs', False)
    # Calling abs(args, kwargs) (line 26)
    abs_call_result_209180 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), abs_209177, *[derphi1_209178], **kwargs_209179)
    
    
    # Call to abs(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'c2' (line 26)
    c2_209182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 32), 'c2', False)
    # Getting the type of 'derphi0' (line 26)
    derphi0_209183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'derphi0', False)
    # Applying the binary operator '*' (line 26)
    result_mul_209184 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 32), '*', c2_209182, derphi0_209183)
    
    # Processing the call keyword arguments (line 26)
    kwargs_209185 = {}
    # Getting the type of 'abs' (line 26)
    abs_209181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'abs', False)
    # Calling abs(args, kwargs) (line 26)
    abs_call_result_209186 = invoke(stypy.reporting.localization.Localization(__file__, 26, 28), abs_209181, *[result_mul_209184], **kwargs_209185)
    
    # Applying the binary operator '<=' (line 26)
    result_le_209187 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 12), '<=', abs_call_result_209180, abs_call_result_209186)
    
    str_209188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 45), 'str', 'Wolfe 2 failed: ')
    # Getting the type of 'msg' (line 26)
    msg_209189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 66), 'msg', False)
    # Applying the binary operator '+' (line 26)
    result_add_209190 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 45), '+', str_209188, msg_209189)
    
    # Processing the call keyword arguments (line 26)
    kwargs_209191 = {}
    # Getting the type of 'assert_' (line 26)
    assert__209176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 26)
    assert__call_result_209192 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), assert__209176, *[result_le_209187, result_add_209190], **kwargs_209191)
    
    
    # ################# End of 'assert_wolfe(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_wolfe' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_209193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_209193)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_wolfe'
    return stypy_return_type_209193

# Assigning a type to the variable 'assert_wolfe' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'assert_wolfe', assert_wolfe)

@norecursion
def assert_armijo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_209194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'float')
    str_209195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 43), 'str', '')
    defaults = [float_209194, str_209195]
    # Create a new context for function 'assert_armijo'
    module_type_store = module_type_store.open_function_context('assert_armijo', 29, 0, False)
    
    # Passed parameters checking function
    assert_armijo.stypy_localization = localization
    assert_armijo.stypy_type_of_self = None
    assert_armijo.stypy_type_store = module_type_store
    assert_armijo.stypy_function_name = 'assert_armijo'
    assert_armijo.stypy_param_names_list = ['s', 'phi', 'c1', 'err_msg']
    assert_armijo.stypy_varargs_param_name = None
    assert_armijo.stypy_kwargs_param_name = None
    assert_armijo.stypy_call_defaults = defaults
    assert_armijo.stypy_call_varargs = varargs
    assert_armijo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_armijo', ['s', 'phi', 'c1', 'err_msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_armijo', localization, ['s', 'phi', 'c1', 'err_msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_armijo(...)' code ##################

    str_209196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n    Check that Armijo condition applies\n    ')
    
    # Assigning a Call to a Name (line 33):
    
    # Assigning a Call to a Name (line 33):
    
    # Call to phi(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 's' (line 33)
    s_209198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 's', False)
    # Processing the call keyword arguments (line 33)
    kwargs_209199 = {}
    # Getting the type of 'phi' (line 33)
    phi_209197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'phi', False)
    # Calling phi(args, kwargs) (line 33)
    phi_call_result_209200 = invoke(stypy.reporting.localization.Localization(__file__, 33, 11), phi_209197, *[s_209198], **kwargs_209199)
    
    # Assigning a type to the variable 'phi1' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'phi1', phi_call_result_209200)
    
    # Assigning a Call to a Name (line 34):
    
    # Assigning a Call to a Name (line 34):
    
    # Call to phi(...): (line 34)
    # Processing the call arguments (line 34)
    int_209202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'int')
    # Processing the call keyword arguments (line 34)
    kwargs_209203 = {}
    # Getting the type of 'phi' (line 34)
    phi_209201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'phi', False)
    # Calling phi(args, kwargs) (line 34)
    phi_call_result_209204 = invoke(stypy.reporting.localization.Localization(__file__, 34, 11), phi_209201, *[int_209202], **kwargs_209203)
    
    # Assigning a type to the variable 'phi0' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'phi0', phi_call_result_209204)
    
    # Assigning a BinOp to a Name (line 35):
    
    # Assigning a BinOp to a Name (line 35):
    str_209205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 10), 'str', 's = %s; phi(0) = %s; phi(s) = %s; %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_209206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 's' (line 35)
    s_209207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 52), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 52), tuple_209206, s_209207)
    # Adding element type (line 35)
    # Getting the type of 'phi0' (line 35)
    phi0_209208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 55), 'phi0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 52), tuple_209206, phi0_209208)
    # Adding element type (line 35)
    # Getting the type of 'phi1' (line 35)
    phi1_209209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 61), 'phi1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 52), tuple_209206, phi1_209209)
    # Adding element type (line 35)
    # Getting the type of 'err_msg' (line 35)
    err_msg_209210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 67), 'err_msg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 52), tuple_209206, err_msg_209210)
    
    # Applying the binary operator '%' (line 35)
    result_mod_209211 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 10), '%', str_209205, tuple_209206)
    
    # Assigning a type to the variable 'msg' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'msg', result_mod_209211)
    
    # Call to assert_(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Getting the type of 'phi1' (line 36)
    phi1_209213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'phi1', False)
    int_209214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'int')
    # Getting the type of 'c1' (line 36)
    c1_209215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'c1', False)
    # Getting the type of 's' (line 36)
    s_209216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 's', False)
    # Applying the binary operator '*' (line 36)
    result_mul_209217 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 25), '*', c1_209215, s_209216)
    
    # Applying the binary operator '-' (line 36)
    result_sub_209218 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 21), '-', int_209214, result_mul_209217)
    
    # Getting the type of 'phi0' (line 36)
    phi0_209219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'phi0', False)
    # Applying the binary operator '*' (line 36)
    result_mul_209220 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 20), '*', result_sub_209218, phi0_209219)
    
    # Applying the binary operator '<=' (line 36)
    result_le_209221 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), '<=', phi1_209213, result_mul_209220)
    
    # Getting the type of 'msg' (line 36)
    msg_209222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'msg', False)
    # Processing the call keyword arguments (line 36)
    kwargs_209223 = {}
    # Getting the type of 'assert_' (line 36)
    assert__209212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 36)
    assert__call_result_209224 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), assert__209212, *[result_le_209221, msg_209222], **kwargs_209223)
    
    
    # ################# End of 'assert_armijo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_armijo' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_209225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_209225)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_armijo'
    return stypy_return_type_209225

# Assigning a type to the variable 'assert_armijo' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'assert_armijo', assert_armijo)

@norecursion
def assert_line_wolfe(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_line_wolfe'
    module_type_store = module_type_store.open_function_context('assert_line_wolfe', 39, 0, False)
    
    # Passed parameters checking function
    assert_line_wolfe.stypy_localization = localization
    assert_line_wolfe.stypy_type_of_self = None
    assert_line_wolfe.stypy_type_store = module_type_store
    assert_line_wolfe.stypy_function_name = 'assert_line_wolfe'
    assert_line_wolfe.stypy_param_names_list = ['x', 'p', 's', 'f', 'fprime']
    assert_line_wolfe.stypy_varargs_param_name = None
    assert_line_wolfe.stypy_kwargs_param_name = 'kw'
    assert_line_wolfe.stypy_call_defaults = defaults
    assert_line_wolfe.stypy_call_varargs = varargs
    assert_line_wolfe.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_line_wolfe', ['x', 'p', 's', 'f', 'fprime'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_line_wolfe', localization, ['x', 'p', 's', 'f', 'fprime'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_line_wolfe(...)' code ##################

    
    # Call to assert_wolfe(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 's' (line 40)
    s_209227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 's', False)
    # Processing the call keyword arguments (line 40)

    @norecursion
    def _stypy_temp_lambda_65(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_65'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_65', 40, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_65.stypy_localization = localization
        _stypy_temp_lambda_65.stypy_type_of_self = None
        _stypy_temp_lambda_65.stypy_type_store = module_type_store
        _stypy_temp_lambda_65.stypy_function_name = '_stypy_temp_lambda_65'
        _stypy_temp_lambda_65.stypy_param_names_list = ['sp']
        _stypy_temp_lambda_65.stypy_varargs_param_name = None
        _stypy_temp_lambda_65.stypy_kwargs_param_name = None
        _stypy_temp_lambda_65.stypy_call_defaults = defaults
        _stypy_temp_lambda_65.stypy_call_varargs = varargs
        _stypy_temp_lambda_65.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_65', ['sp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_65', ['sp'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to f(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'x' (line 40)
        x_209229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 37), 'x', False)
        # Getting the type of 'p' (line 40)
        p_209230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 'p', False)
        # Getting the type of 'sp' (line 40)
        sp_209231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'sp', False)
        # Applying the binary operator '*' (line 40)
        result_mul_209232 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 41), '*', p_209230, sp_209231)
        
        # Applying the binary operator '+' (line 40)
        result_add_209233 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 37), '+', x_209229, result_mul_209232)
        
        # Processing the call keyword arguments (line 40)
        kwargs_209234 = {}
        # Getting the type of 'f' (line 40)
        f_209228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'f', False)
        # Calling f(args, kwargs) (line 40)
        f_call_result_209235 = invoke(stypy.reporting.localization.Localization(__file__, 40, 35), f_209228, *[result_add_209233], **kwargs_209234)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'stypy_return_type', f_call_result_209235)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_65' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_209236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_65'
        return stypy_return_type_209236

    # Assigning a type to the variable '_stypy_temp_lambda_65' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), '_stypy_temp_lambda_65', _stypy_temp_lambda_65)
    # Getting the type of '_stypy_temp_lambda_65' (line 40)
    _stypy_temp_lambda_65_209237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), '_stypy_temp_lambda_65')
    keyword_209238 = _stypy_temp_lambda_65_209237

    @norecursion
    def _stypy_temp_lambda_66(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_66'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_66', 41, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_66.stypy_localization = localization
        _stypy_temp_lambda_66.stypy_type_of_self = None
        _stypy_temp_lambda_66.stypy_type_store = module_type_store
        _stypy_temp_lambda_66.stypy_function_name = '_stypy_temp_lambda_66'
        _stypy_temp_lambda_66.stypy_param_names_list = ['sp']
        _stypy_temp_lambda_66.stypy_varargs_param_name = None
        _stypy_temp_lambda_66.stypy_kwargs_param_name = None
        _stypy_temp_lambda_66.stypy_call_defaults = defaults
        _stypy_temp_lambda_66.stypy_call_varargs = varargs
        _stypy_temp_lambda_66.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_66', ['sp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_66', ['sp'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to dot(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to fprime(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'x' (line 41)
        x_209242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 49), 'x', False)
        # Getting the type of 'p' (line 41)
        p_209243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 53), 'p', False)
        # Getting the type of 'sp' (line 41)
        sp_209244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 55), 'sp', False)
        # Applying the binary operator '*' (line 41)
        result_mul_209245 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 53), '*', p_209243, sp_209244)
        
        # Applying the binary operator '+' (line 41)
        result_add_209246 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 49), '+', x_209242, result_mul_209245)
        
        # Processing the call keyword arguments (line 41)
        kwargs_209247 = {}
        # Getting the type of 'fprime' (line 41)
        fprime_209241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 42), 'fprime', False)
        # Calling fprime(args, kwargs) (line 41)
        fprime_call_result_209248 = invoke(stypy.reporting.localization.Localization(__file__, 41, 42), fprime_209241, *[result_add_209246], **kwargs_209247)
        
        # Getting the type of 'p' (line 41)
        p_209249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 60), 'p', False)
        # Processing the call keyword arguments (line 41)
        kwargs_209250 = {}
        # Getting the type of 'np' (line 41)
        np_209239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'np', False)
        # Obtaining the member 'dot' of a type (line 41)
        dot_209240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 35), np_209239, 'dot')
        # Calling dot(args, kwargs) (line 41)
        dot_call_result_209251 = invoke(stypy.reporting.localization.Localization(__file__, 41, 35), dot_209240, *[fprime_call_result_209248, p_209249], **kwargs_209250)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'stypy_return_type', dot_call_result_209251)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_66' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_209252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_66'
        return stypy_return_type_209252

    # Assigning a type to the variable '_stypy_temp_lambda_66' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), '_stypy_temp_lambda_66', _stypy_temp_lambda_66)
    # Getting the type of '_stypy_temp_lambda_66' (line 41)
    _stypy_temp_lambda_66_209253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), '_stypy_temp_lambda_66')
    keyword_209254 = _stypy_temp_lambda_66_209253
    # Getting the type of 'kw' (line 41)
    kw_209255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 66), 'kw', False)
    kwargs_209256 = {'kw_209255': kw_209255, 'phi': keyword_209238, 'derphi': keyword_209254}
    # Getting the type of 'assert_wolfe' (line 40)
    assert_wolfe_209226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'assert_wolfe', False)
    # Calling assert_wolfe(args, kwargs) (line 40)
    assert_wolfe_call_result_209257 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), assert_wolfe_209226, *[s_209227], **kwargs_209256)
    
    
    # ################# End of 'assert_line_wolfe(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_line_wolfe' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_209258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_209258)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_line_wolfe'
    return stypy_return_type_209258

# Assigning a type to the variable 'assert_line_wolfe' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'assert_line_wolfe', assert_line_wolfe)

@norecursion
def assert_line_armijo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_line_armijo'
    module_type_store = module_type_store.open_function_context('assert_line_armijo', 44, 0, False)
    
    # Passed parameters checking function
    assert_line_armijo.stypy_localization = localization
    assert_line_armijo.stypy_type_of_self = None
    assert_line_armijo.stypy_type_store = module_type_store
    assert_line_armijo.stypy_function_name = 'assert_line_armijo'
    assert_line_armijo.stypy_param_names_list = ['x', 'p', 's', 'f']
    assert_line_armijo.stypy_varargs_param_name = None
    assert_line_armijo.stypy_kwargs_param_name = 'kw'
    assert_line_armijo.stypy_call_defaults = defaults
    assert_line_armijo.stypy_call_varargs = varargs
    assert_line_armijo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_line_armijo', ['x', 'p', 's', 'f'], None, 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_line_armijo', localization, ['x', 'p', 's', 'f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_line_armijo(...)' code ##################

    
    # Call to assert_armijo(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 's' (line 45)
    s_209260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 's', False)
    # Processing the call keyword arguments (line 45)

    @norecursion
    def _stypy_temp_lambda_67(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_67'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_67', 45, 25, True)
        # Passed parameters checking function
        _stypy_temp_lambda_67.stypy_localization = localization
        _stypy_temp_lambda_67.stypy_type_of_self = None
        _stypy_temp_lambda_67.stypy_type_store = module_type_store
        _stypy_temp_lambda_67.stypy_function_name = '_stypy_temp_lambda_67'
        _stypy_temp_lambda_67.stypy_param_names_list = ['sp']
        _stypy_temp_lambda_67.stypy_varargs_param_name = None
        _stypy_temp_lambda_67.stypy_kwargs_param_name = None
        _stypy_temp_lambda_67.stypy_call_defaults = defaults
        _stypy_temp_lambda_67.stypy_call_varargs = varargs
        _stypy_temp_lambda_67.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_67', ['sp'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_67', ['sp'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to f(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'x' (line 45)
        x_209262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 38), 'x', False)
        # Getting the type of 'p' (line 45)
        p_209263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 42), 'p', False)
        # Getting the type of 'sp' (line 45)
        sp_209264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'sp', False)
        # Applying the binary operator '*' (line 45)
        result_mul_209265 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 42), '*', p_209263, sp_209264)
        
        # Applying the binary operator '+' (line 45)
        result_add_209266 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 38), '+', x_209262, result_mul_209265)
        
        # Processing the call keyword arguments (line 45)
        kwargs_209267 = {}
        # Getting the type of 'f' (line 45)
        f_209261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'f', False)
        # Calling f(args, kwargs) (line 45)
        f_call_result_209268 = invoke(stypy.reporting.localization.Localization(__file__, 45, 36), f_209261, *[result_add_209266], **kwargs_209267)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'stypy_return_type', f_call_result_209268)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_67' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_209269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_67'
        return stypy_return_type_209269

    # Assigning a type to the variable '_stypy_temp_lambda_67' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), '_stypy_temp_lambda_67', _stypy_temp_lambda_67)
    # Getting the type of '_stypy_temp_lambda_67' (line 45)
    _stypy_temp_lambda_67_209270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), '_stypy_temp_lambda_67')
    keyword_209271 = _stypy_temp_lambda_67_209270
    # Getting the type of 'kw' (line 45)
    kw_209272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 51), 'kw', False)
    kwargs_209273 = {'phi': keyword_209271, 'kw_209272': kw_209272}
    # Getting the type of 'assert_armijo' (line 45)
    assert_armijo_209259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'assert_armijo', False)
    # Calling assert_armijo(args, kwargs) (line 45)
    assert_armijo_call_result_209274 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), assert_armijo_209259, *[s_209260], **kwargs_209273)
    
    
    # ################# End of 'assert_line_armijo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_line_armijo' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_209275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_209275)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_line_armijo'
    return stypy_return_type_209275

# Assigning a type to the variable 'assert_line_armijo' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'assert_line_armijo', assert_line_armijo)

@norecursion
def assert_fp_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_209276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'str', '')
    int_209277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 43), 'int')
    defaults = [str_209276, int_209277]
    # Create a new context for function 'assert_fp_equal'
    module_type_store = module_type_store.open_function_context('assert_fp_equal', 48, 0, False)
    
    # Passed parameters checking function
    assert_fp_equal.stypy_localization = localization
    assert_fp_equal.stypy_type_of_self = None
    assert_fp_equal.stypy_type_store = module_type_store
    assert_fp_equal.stypy_function_name = 'assert_fp_equal'
    assert_fp_equal.stypy_param_names_list = ['x', 'y', 'err_msg', 'nulp']
    assert_fp_equal.stypy_varargs_param_name = None
    assert_fp_equal.stypy_kwargs_param_name = None
    assert_fp_equal.stypy_call_defaults = defaults
    assert_fp_equal.stypy_call_varargs = varargs
    assert_fp_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_fp_equal', ['x', 'y', 'err_msg', 'nulp'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_fp_equal', localization, ['x', 'y', 'err_msg', 'nulp'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_fp_equal(...)' code ##################

    str_209278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'str', 'Assert two arrays are equal, up to some floating-point rounding error')
    
    
    # SSA begins for try-except statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to assert_array_almost_equal_nulp(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'x' (line 51)
    x_209280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 39), 'x', False)
    # Getting the type of 'y' (line 51)
    y_209281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'y', False)
    # Getting the type of 'nulp' (line 51)
    nulp_209282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 45), 'nulp', False)
    # Processing the call keyword arguments (line 51)
    kwargs_209283 = {}
    # Getting the type of 'assert_array_almost_equal_nulp' (line 51)
    assert_array_almost_equal_nulp_209279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'assert_array_almost_equal_nulp', False)
    # Calling assert_array_almost_equal_nulp(args, kwargs) (line 51)
    assert_array_almost_equal_nulp_call_result_209284 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert_array_almost_equal_nulp_209279, *[x_209280, y_209281, nulp_209282], **kwargs_209283)
    
    # SSA branch for the except part of a try statement (line 50)
    # SSA branch for the except 'AssertionError' branch of a try statement (line 50)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'AssertionError' (line 52)
    AssertionError_209285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'AssertionError')
    # Assigning a type to the variable 'e' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'e', AssertionError_209285)
    
    # Call to AssertionError(...): (line 53)
    # Processing the call arguments (line 53)
    str_209287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'str', '%s\n%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_209288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    # Getting the type of 'e' (line 53)
    e_209289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'e', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 41), tuple_209288, e_209289)
    # Adding element type (line 53)
    # Getting the type of 'err_msg' (line 53)
    err_msg_209290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 44), 'err_msg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 41), tuple_209288, err_msg_209290)
    
    # Applying the binary operator '%' (line 53)
    result_mod_209291 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 29), '%', str_209287, tuple_209288)
    
    # Processing the call keyword arguments (line 53)
    kwargs_209292 = {}
    # Getting the type of 'AssertionError' (line 53)
    AssertionError_209286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 53)
    AssertionError_call_result_209293 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), AssertionError_209286, *[result_mod_209291], **kwargs_209292)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 53, 8), AssertionError_call_result_209293, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'assert_fp_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_fp_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_209294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_209294)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_fp_equal'
    return stypy_return_type_209294

# Assigning a type to the variable 'assert_fp_equal' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'assert_fp_equal', assert_fp_equal)
# Declaration of the 'TestLineSearch' class

class TestLineSearch(object, ):

    @norecursion
    def _scalar_func_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_scalar_func_1'
        module_type_store = module_type_store.open_function_context('_scalar_func_1', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_function_name', 'TestLineSearch._scalar_func_1')
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_param_names_list', ['s'])
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch._scalar_func_1.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch._scalar_func_1', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_scalar_func_1', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_scalar_func_1(...)' code ##################

        
        # Getting the type of 'self' (line 59)
        self_209295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Obtaining the member 'fcount' of a type (line 59)
        fcount_209296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_209295, 'fcount')
        int_209297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'int')
        # Applying the binary operator '+=' (line 59)
        result_iadd_209298 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 8), '+=', fcount_209296, int_209297)
        # Getting the type of 'self' (line 59)
        self_209299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'fcount' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_209299, 'fcount', result_iadd_209298)
        
        
        # Assigning a BinOp to a Name (line 60):
        
        # Assigning a BinOp to a Name (line 60):
        
        # Getting the type of 's' (line 60)
        s_209300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 's')
        # Applying the 'usub' unary operator (line 60)
        result___neg___209301 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), 'usub', s_209300)
        
        # Getting the type of 's' (line 60)
        s_209302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 's')
        int_209303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'int')
        # Applying the binary operator '**' (line 60)
        result_pow_209304 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 17), '**', s_209302, int_209303)
        
        # Applying the binary operator '-' (line 60)
        result_sub_209305 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), '-', result___neg___209301, result_pow_209304)
        
        # Getting the type of 's' (line 60)
        s_209306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 's')
        int_209307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 27), 'int')
        # Applying the binary operator '**' (line 60)
        result_pow_209308 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 24), '**', s_209306, int_209307)
        
        # Applying the binary operator '+' (line 60)
        result_add_209309 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 22), '+', result_sub_209305, result_pow_209308)
        
        # Assigning a type to the variable 'p' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'p', result_add_209309)
        
        # Assigning a BinOp to a Name (line 61):
        
        # Assigning a BinOp to a Name (line 61):
        int_209310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 13), 'int')
        int_209311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 18), 'int')
        # Getting the type of 's' (line 61)
        s_209312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 's')
        int_209313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'int')
        # Applying the binary operator '**' (line 61)
        result_pow_209314 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 20), '**', s_209312, int_209313)
        
        # Applying the binary operator '*' (line 61)
        result_mul_209315 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 18), '*', int_209311, result_pow_209314)
        
        # Applying the binary operator '-' (line 61)
        result_sub_209316 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 13), '-', int_209310, result_mul_209315)
        
        int_209317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'int')
        # Getting the type of 's' (line 61)
        s_209318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 's')
        int_209319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 32), 'int')
        # Applying the binary operator '**' (line 61)
        result_pow_209320 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 29), '**', s_209318, int_209319)
        
        # Applying the binary operator '*' (line 61)
        result_mul_209321 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 27), '*', int_209317, result_pow_209320)
        
        # Applying the binary operator '+' (line 61)
        result_add_209322 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 25), '+', result_sub_209316, result_mul_209321)
        
        # Assigning a type to the variable 'dp' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'dp', result_add_209322)
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_209323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        # Getting the type of 'p' (line 62)
        p_209324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), tuple_209323, p_209324)
        # Adding element type (line 62)
        # Getting the type of 'dp' (line 62)
        dp_209325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'dp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), tuple_209323, dp_209325)
        
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', tuple_209323)
        
        # ################# End of '_scalar_func_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_scalar_func_1' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_209326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_scalar_func_1'
        return stypy_return_type_209326


    @norecursion
    def _scalar_func_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_scalar_func_2'
        module_type_store = module_type_store.open_function_context('_scalar_func_2', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_function_name', 'TestLineSearch._scalar_func_2')
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_param_names_list', ['s'])
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch._scalar_func_2.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch._scalar_func_2', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_scalar_func_2', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_scalar_func_2(...)' code ##################

        
        # Getting the type of 'self' (line 65)
        self_209327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Obtaining the member 'fcount' of a type (line 65)
        fcount_209328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_209327, 'fcount')
        int_209329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'int')
        # Applying the binary operator '+=' (line 65)
        result_iadd_209330 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 8), '+=', fcount_209328, int_209329)
        # Getting the type of 'self' (line 65)
        self_209331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'fcount' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_209331, 'fcount', result_iadd_209330)
        
        
        # Assigning a BinOp to a Name (line 66):
        
        # Assigning a BinOp to a Name (line 66):
        
        # Call to exp(...): (line 66)
        # Processing the call arguments (line 66)
        int_209334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'int')
        # Getting the type of 's' (line 66)
        s_209335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 's', False)
        # Applying the binary operator '*' (line 66)
        result_mul_209336 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 19), '*', int_209334, s_209335)
        
        # Processing the call keyword arguments (line 66)
        kwargs_209337 = {}
        # Getting the type of 'np' (line 66)
        np_209332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'np', False)
        # Obtaining the member 'exp' of a type (line 66)
        exp_209333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), np_209332, 'exp')
        # Calling exp(args, kwargs) (line 66)
        exp_call_result_209338 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), exp_209333, *[result_mul_209336], **kwargs_209337)
        
        # Getting the type of 's' (line 66)
        s_209339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 27), 's')
        int_209340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'int')
        # Applying the binary operator '**' (line 66)
        result_pow_209341 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 27), '**', s_209339, int_209340)
        
        # Applying the binary operator '+' (line 66)
        result_add_209342 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 12), '+', exp_call_result_209338, result_pow_209341)
        
        # Assigning a type to the variable 'p' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'p', result_add_209342)
        
        # Assigning a BinOp to a Name (line 67):
        
        # Assigning a BinOp to a Name (line 67):
        int_209343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 13), 'int')
        
        # Call to exp(...): (line 67)
        # Processing the call arguments (line 67)
        int_209346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'int')
        # Getting the type of 's' (line 67)
        s_209347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 's', False)
        # Applying the binary operator '*' (line 67)
        result_mul_209348 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 23), '*', int_209346, s_209347)
        
        # Processing the call keyword arguments (line 67)
        kwargs_209349 = {}
        # Getting the type of 'np' (line 67)
        np_209344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'np', False)
        # Obtaining the member 'exp' of a type (line 67)
        exp_209345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 16), np_209344, 'exp')
        # Calling exp(args, kwargs) (line 67)
        exp_call_result_209350 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), exp_209345, *[result_mul_209348], **kwargs_209349)
        
        # Applying the binary operator '*' (line 67)
        result_mul_209351 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 13), '*', int_209343, exp_call_result_209350)
        
        int_209352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'int')
        # Getting the type of 's' (line 67)
        s_209353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 's')
        # Applying the binary operator '*' (line 67)
        result_mul_209354 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 31), '*', int_209352, s_209353)
        
        # Applying the binary operator '+' (line 67)
        result_add_209355 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 13), '+', result_mul_209351, result_mul_209354)
        
        # Assigning a type to the variable 'dp' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'dp', result_add_209355)
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_209356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        # Getting the type of 'p' (line 68)
        p_209357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 15), tuple_209356, p_209357)
        # Adding element type (line 68)
        # Getting the type of 'dp' (line 68)
        dp_209358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'dp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 15), tuple_209356, dp_209358)
        
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type', tuple_209356)
        
        # ################# End of '_scalar_func_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_scalar_func_2' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_209359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209359)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_scalar_func_2'
        return stypy_return_type_209359


    @norecursion
    def _scalar_func_3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_scalar_func_3'
        module_type_store = module_type_store.open_function_context('_scalar_func_3', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_function_name', 'TestLineSearch._scalar_func_3')
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_param_names_list', ['s'])
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch._scalar_func_3.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch._scalar_func_3', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_scalar_func_3', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_scalar_func_3(...)' code ##################

        
        # Getting the type of 'self' (line 71)
        self_209360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Obtaining the member 'fcount' of a type (line 71)
        fcount_209361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_209360, 'fcount')
        int_209362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'int')
        # Applying the binary operator '+=' (line 71)
        result_iadd_209363 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 8), '+=', fcount_209361, int_209362)
        # Getting the type of 'self' (line 71)
        self_209364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member 'fcount' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_209364, 'fcount', result_iadd_209363)
        
        
        # Assigning a UnaryOp to a Name (line 72):
        
        # Assigning a UnaryOp to a Name (line 72):
        
        
        # Call to sin(...): (line 72)
        # Processing the call arguments (line 72)
        int_209367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'int')
        # Getting the type of 's' (line 72)
        s_209368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 's', False)
        # Applying the binary operator '*' (line 72)
        result_mul_209369 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 20), '*', int_209367, s_209368)
        
        # Processing the call keyword arguments (line 72)
        kwargs_209370 = {}
        # Getting the type of 'np' (line 72)
        np_209365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'np', False)
        # Obtaining the member 'sin' of a type (line 72)
        sin_209366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), np_209365, 'sin')
        # Calling sin(args, kwargs) (line 72)
        sin_call_result_209371 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), sin_209366, *[result_mul_209369], **kwargs_209370)
        
        # Applying the 'usub' unary operator (line 72)
        result___neg___209372 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 12), 'usub', sin_call_result_209371)
        
        # Assigning a type to the variable 'p' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'p', result___neg___209372)
        
        # Assigning a BinOp to a Name (line 73):
        
        # Assigning a BinOp to a Name (line 73):
        int_209373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'int')
        
        # Call to cos(...): (line 73)
        # Processing the call arguments (line 73)
        int_209376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'int')
        # Getting the type of 's' (line 73)
        s_209377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 's', False)
        # Applying the binary operator '*' (line 73)
        result_mul_209378 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 24), '*', int_209376, s_209377)
        
        # Processing the call keyword arguments (line 73)
        kwargs_209379 = {}
        # Getting the type of 'np' (line 73)
        np_209374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'np', False)
        # Obtaining the member 'cos' of a type (line 73)
        cos_209375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 17), np_209374, 'cos')
        # Calling cos(args, kwargs) (line 73)
        cos_call_result_209380 = invoke(stypy.reporting.localization.Localization(__file__, 73, 17), cos_209375, *[result_mul_209378], **kwargs_209379)
        
        # Applying the binary operator '*' (line 73)
        result_mul_209381 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 13), '*', int_209373, cos_call_result_209380)
        
        # Assigning a type to the variable 'dp' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'dp', result_mul_209381)
        
        # Obtaining an instance of the builtin type 'tuple' (line 74)
        tuple_209382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 74)
        # Adding element type (line 74)
        # Getting the type of 'p' (line 74)
        p_209383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 15), tuple_209382, p_209383)
        # Adding element type (line 74)
        # Getting the type of 'dp' (line 74)
        dp_209384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'dp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 15), tuple_209382, dp_209384)
        
        # Assigning a type to the variable 'stypy_return_type' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type', tuple_209382)
        
        # ################# End of '_scalar_func_3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_scalar_func_3' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_209385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_scalar_func_3'
        return stypy_return_type_209385


    @norecursion
    def _line_func_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_line_func_1'
        module_type_store = module_type_store.open_function_context('_line_func_1', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_function_name', 'TestLineSearch._line_func_1')
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch._line_func_1.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch._line_func_1', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_line_func_1', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_line_func_1(...)' code ##################

        
        # Getting the type of 'self' (line 79)
        self_209386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Obtaining the member 'fcount' of a type (line 79)
        fcount_209387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_209386, 'fcount')
        int_209388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'int')
        # Applying the binary operator '+=' (line 79)
        result_iadd_209389 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 8), '+=', fcount_209387, int_209388)
        # Getting the type of 'self' (line 79)
        self_209390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'fcount' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_209390, 'fcount', result_iadd_209389)
        
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to dot(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'x' (line 80)
        x_209393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'x', False)
        # Getting the type of 'x' (line 80)
        x_209394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'x', False)
        # Processing the call keyword arguments (line 80)
        kwargs_209395 = {}
        # Getting the type of 'np' (line 80)
        np_209391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 80)
        dot_209392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), np_209391, 'dot')
        # Calling dot(args, kwargs) (line 80)
        dot_call_result_209396 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), dot_209392, *[x_209393, x_209394], **kwargs_209395)
        
        # Assigning a type to the variable 'f' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'f', dot_call_result_209396)
        
        # Assigning a BinOp to a Name (line 81):
        
        # Assigning a BinOp to a Name (line 81):
        int_209397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 13), 'int')
        # Getting the type of 'x' (line 81)
        x_209398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'x')
        # Applying the binary operator '*' (line 81)
        result_mul_209399 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 13), '*', int_209397, x_209398)
        
        # Assigning a type to the variable 'df' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'df', result_mul_209399)
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_209400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'f' (line 82)
        f_209401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_209400, f_209401)
        # Adding element type (line 82)
        # Getting the type of 'df' (line 82)
        df_209402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'df')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 15), tuple_209400, df_209402)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', tuple_209400)
        
        # ################# End of '_line_func_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_line_func_1' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_209403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209403)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_line_func_1'
        return stypy_return_type_209403


    @norecursion
    def _line_func_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_line_func_2'
        module_type_store = module_type_store.open_function_context('_line_func_2', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_function_name', 'TestLineSearch._line_func_2')
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch._line_func_2.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch._line_func_2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_line_func_2', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_line_func_2(...)' code ##################

        
        # Getting the type of 'self' (line 85)
        self_209404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Obtaining the member 'fcount' of a type (line 85)
        fcount_209405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_209404, 'fcount')
        int_209406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'int')
        # Applying the binary operator '+=' (line 85)
        result_iadd_209407 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 8), '+=', fcount_209405, int_209406)
        # Getting the type of 'self' (line 85)
        self_209408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'fcount' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_209408, 'fcount', result_iadd_209407)
        
        
        # Assigning a BinOp to a Name (line 86):
        
        # Assigning a BinOp to a Name (line 86):
        
        # Call to dot(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'x' (line 86)
        x_209411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'x', False)
        
        # Call to dot(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'self' (line 86)
        self_209414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'self', False)
        # Obtaining the member 'A' of a type (line 86)
        A_209415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 29), self_209414, 'A')
        # Getting the type of 'x' (line 86)
        x_209416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'x', False)
        # Processing the call keyword arguments (line 86)
        kwargs_209417 = {}
        # Getting the type of 'np' (line 86)
        np_209412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'np', False)
        # Obtaining the member 'dot' of a type (line 86)
        dot_209413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 22), np_209412, 'dot')
        # Calling dot(args, kwargs) (line 86)
        dot_call_result_209418 = invoke(stypy.reporting.localization.Localization(__file__, 86, 22), dot_209413, *[A_209415, x_209416], **kwargs_209417)
        
        # Processing the call keyword arguments (line 86)
        kwargs_209419 = {}
        # Getting the type of 'np' (line 86)
        np_209409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 86)
        dot_209410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), np_209409, 'dot')
        # Calling dot(args, kwargs) (line 86)
        dot_call_result_209420 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), dot_209410, *[x_209411, dot_call_result_209418], **kwargs_209419)
        
        int_209421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 43), 'int')
        # Applying the binary operator '+' (line 86)
        result_add_209422 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 12), '+', dot_call_result_209420, int_209421)
        
        # Assigning a type to the variable 'f' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'f', result_add_209422)
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to dot(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_209425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'self', False)
        # Obtaining the member 'A' of a type (line 87)
        A_209426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 20), self_209425, 'A')
        # Getting the type of 'self' (line 87)
        self_209427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'self', False)
        # Obtaining the member 'A' of a type (line 87)
        A_209428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 29), self_209427, 'A')
        # Obtaining the member 'T' of a type (line 87)
        T_209429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 29), A_209428, 'T')
        # Applying the binary operator '+' (line 87)
        result_add_209430 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 20), '+', A_209426, T_209429)
        
        # Getting the type of 'x' (line 87)
        x_209431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 39), 'x', False)
        # Processing the call keyword arguments (line 87)
        kwargs_209432 = {}
        # Getting the type of 'np' (line 87)
        np_209423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 87)
        dot_209424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), np_209423, 'dot')
        # Calling dot(args, kwargs) (line 87)
        dot_call_result_209433 = invoke(stypy.reporting.localization.Localization(__file__, 87, 13), dot_209424, *[result_add_209430, x_209431], **kwargs_209432)
        
        # Assigning a type to the variable 'df' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'df', dot_call_result_209433)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_209434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        # Getting the type of 'f' (line 88)
        f_209435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), tuple_209434, f_209435)
        # Adding element type (line 88)
        # Getting the type of 'df' (line 88)
        df_209436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'df')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), tuple_209434, df_209436)
        
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', tuple_209434)
        
        # ################# End of '_line_func_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_line_func_2' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_209437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209437)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_line_func_2'
        return stypy_return_type_209437


    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.setup_method')
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a List to a Attribute (line 93):
        
        # Assigning a List to a Attribute (line 93):
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_209438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        
        # Getting the type of 'self' (line 93)
        self_209439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member 'scalar_funcs' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_209439, 'scalar_funcs', list_209438)
        
        # Assigning a List to a Attribute (line 94):
        
        # Assigning a List to a Attribute (line 94):
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_209440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        
        # Getting the type of 'self' (line 94)
        self_209441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'line_funcs' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_209441, 'line_funcs', list_209440)
        
        # Assigning a Num to a Attribute (line 95):
        
        # Assigning a Num to a Attribute (line 95):
        int_209442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 17), 'int')
        # Getting the type of 'self' (line 95)
        self_209443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'N' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_209443, 'N', int_209442)
        
        # Assigning a Num to a Attribute (line 96):
        
        # Assigning a Num to a Attribute (line 96):
        int_209444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'int')
        # Getting the type of 'self' (line 96)
        self_209445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'fcount' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_209445, 'fcount', int_209444)

        @norecursion
        def bind_index(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'bind_index'
            module_type_store = module_type_store.open_function_context('bind_index', 98, 8, False)
            
            # Passed parameters checking function
            bind_index.stypy_localization = localization
            bind_index.stypy_type_of_self = None
            bind_index.stypy_type_store = module_type_store
            bind_index.stypy_function_name = 'bind_index'
            bind_index.stypy_param_names_list = ['func', 'idx']
            bind_index.stypy_varargs_param_name = None
            bind_index.stypy_kwargs_param_name = None
            bind_index.stypy_call_defaults = defaults
            bind_index.stypy_call_varargs = varargs
            bind_index.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'bind_index', ['func', 'idx'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'bind_index', localization, ['func', 'idx'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'bind_index(...)' code ##################


            @norecursion
            def _stypy_temp_lambda_68(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_68'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_68', 100, 19, True)
                # Passed parameters checking function
                _stypy_temp_lambda_68.stypy_localization = localization
                _stypy_temp_lambda_68.stypy_type_of_self = None
                _stypy_temp_lambda_68.stypy_type_store = module_type_store
                _stypy_temp_lambda_68.stypy_function_name = '_stypy_temp_lambda_68'
                _stypy_temp_lambda_68.stypy_param_names_list = []
                _stypy_temp_lambda_68.stypy_varargs_param_name = 'a'
                _stypy_temp_lambda_68.stypy_kwargs_param_name = 'kw'
                _stypy_temp_lambda_68.stypy_call_defaults = defaults
                _stypy_temp_lambda_68.stypy_call_varargs = varargs
                _stypy_temp_lambda_68.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_68', [], 'a', 'kw', defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_68', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Obtaining the type of the subscript
                # Getting the type of 'idx' (line 100)
                idx_209446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 51), 'idx')
                
                # Call to func(...): (line 100)
                # Getting the type of 'a' (line 100)
                a_209448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 42), 'a', False)
                # Processing the call keyword arguments (line 100)
                # Getting the type of 'kw' (line 100)
                kw_209449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 47), 'kw', False)
                kwargs_209450 = {'kw_209449': kw_209449}
                # Getting the type of 'func' (line 100)
                func_209447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'func', False)
                # Calling func(args, kwargs) (line 100)
                func_call_result_209451 = invoke(stypy.reporting.localization.Localization(__file__, 100, 36), func_209447, *[a_209448], **kwargs_209450)
                
                # Obtaining the member '__getitem__' of a type (line 100)
                getitem___209452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 36), func_call_result_209451, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 100)
                subscript_call_result_209453 = invoke(stypy.reporting.localization.Localization(__file__, 100, 36), getitem___209452, idx_209446)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'stypy_return_type', subscript_call_result_209453)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_68' in the type store
                # Getting the type of 'stypy_return_type' (line 100)
                stypy_return_type_209454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_209454)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_68'
                return stypy_return_type_209454

            # Assigning a type to the variable '_stypy_temp_lambda_68' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), '_stypy_temp_lambda_68', _stypy_temp_lambda_68)
            # Getting the type of '_stypy_temp_lambda_68' (line 100)
            _stypy_temp_lambda_68_209455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), '_stypy_temp_lambda_68')
            # Assigning a type to the variable 'stypy_return_type' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'stypy_return_type', _stypy_temp_lambda_68_209455)
            
            # ################# End of 'bind_index(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'bind_index' in the type store
            # Getting the type of 'stypy_return_type' (line 98)
            stypy_return_type_209456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_209456)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'bind_index'
            return stypy_return_type_209456

        # Assigning a type to the variable 'bind_index' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'bind_index', bind_index)
        
        
        # Call to sorted(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to dir(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'self' (line 102)
        self_209459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'self', False)
        # Processing the call keyword arguments (line 102)
        kwargs_209460 = {}
        # Getting the type of 'dir' (line 102)
        dir_209458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'dir', False)
        # Calling dir(args, kwargs) (line 102)
        dir_call_result_209461 = invoke(stypy.reporting.localization.Localization(__file__, 102, 27), dir_209458, *[self_209459], **kwargs_209460)
        
        # Processing the call keyword arguments (line 102)
        kwargs_209462 = {}
        # Getting the type of 'sorted' (line 102)
        sorted_209457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'sorted', False)
        # Calling sorted(args, kwargs) (line 102)
        sorted_call_result_209463 = invoke(stypy.reporting.localization.Localization(__file__, 102, 20), sorted_209457, *[dir_call_result_209461], **kwargs_209462)
        
        # Testing the type of a for loop iterable (line 102)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 8), sorted_call_result_209463)
        # Getting the type of the for loop variable (line 102)
        for_loop_var_209464 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 8), sorted_call_result_209463)
        # Assigning a type to the variable 'name' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'name', for_loop_var_209464)
        # SSA begins for a for statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to startswith(...): (line 103)
        # Processing the call arguments (line 103)
        str_209467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 31), 'str', '_scalar_func_')
        # Processing the call keyword arguments (line 103)
        kwargs_209468 = {}
        # Getting the type of 'name' (line 103)
        name_209465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'name', False)
        # Obtaining the member 'startswith' of a type (line 103)
        startswith_209466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), name_209465, 'startswith')
        # Calling startswith(args, kwargs) (line 103)
        startswith_call_result_209469 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), startswith_209466, *[str_209467], **kwargs_209468)
        
        # Testing the type of an if condition (line 103)
        if_condition_209470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 12), startswith_call_result_209469)
        # Assigning a type to the variable 'if_condition_209470' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'if_condition_209470', if_condition_209470)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 104):
        
        # Assigning a Call to a Name (line 104):
        
        # Call to getattr(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'self' (line 104)
        self_209472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'self', False)
        # Getting the type of 'name' (line 104)
        name_209473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 38), 'name', False)
        # Processing the call keyword arguments (line 104)
        kwargs_209474 = {}
        # Getting the type of 'getattr' (line 104)
        getattr_209471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 104)
        getattr_call_result_209475 = invoke(stypy.reporting.localization.Localization(__file__, 104, 24), getattr_209471, *[self_209472, name_209473], **kwargs_209474)
        
        # Assigning a type to the variable 'value' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'value', getattr_call_result_209475)
        
        # Call to append(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_209479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        # Getting the type of 'name' (line 106)
        name_209480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 21), tuple_209479, name_209480)
        # Adding element type (line 106)
        
        # Call to bind_index(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'value' (line 106)
        value_209482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 38), 'value', False)
        int_209483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 45), 'int')
        # Processing the call keyword arguments (line 106)
        kwargs_209484 = {}
        # Getting the type of 'bind_index' (line 106)
        bind_index_209481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'bind_index', False)
        # Calling bind_index(args, kwargs) (line 106)
        bind_index_call_result_209485 = invoke(stypy.reporting.localization.Localization(__file__, 106, 27), bind_index_209481, *[value_209482, int_209483], **kwargs_209484)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 21), tuple_209479, bind_index_call_result_209485)
        # Adding element type (line 106)
        
        # Call to bind_index(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'value' (line 106)
        value_209487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 60), 'value', False)
        int_209488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 67), 'int')
        # Processing the call keyword arguments (line 106)
        kwargs_209489 = {}
        # Getting the type of 'bind_index' (line 106)
        bind_index_209486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 49), 'bind_index', False)
        # Calling bind_index(args, kwargs) (line 106)
        bind_index_call_result_209490 = invoke(stypy.reporting.localization.Localization(__file__, 106, 49), bind_index_209486, *[value_209487, int_209488], **kwargs_209489)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 21), tuple_209479, bind_index_call_result_209490)
        
        # Processing the call keyword arguments (line 105)
        kwargs_209491 = {}
        # Getting the type of 'self' (line 105)
        self_209476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'self', False)
        # Obtaining the member 'scalar_funcs' of a type (line 105)
        scalar_funcs_209477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), self_209476, 'scalar_funcs')
        # Obtaining the member 'append' of a type (line 105)
        append_209478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), scalar_funcs_209477, 'append')
        # Calling append(args, kwargs) (line 105)
        append_call_result_209492 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), append_209478, *[tuple_209479], **kwargs_209491)
        
        # SSA branch for the else part of an if statement (line 103)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to startswith(...): (line 107)
        # Processing the call arguments (line 107)
        str_209495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 33), 'str', '_line_func_')
        # Processing the call keyword arguments (line 107)
        kwargs_209496 = {}
        # Getting the type of 'name' (line 107)
        name_209493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'name', False)
        # Obtaining the member 'startswith' of a type (line 107)
        startswith_209494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 17), name_209493, 'startswith')
        # Calling startswith(args, kwargs) (line 107)
        startswith_call_result_209497 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), startswith_209494, *[str_209495], **kwargs_209496)
        
        # Testing the type of an if condition (line 107)
        if_condition_209498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 17), startswith_call_result_209497)
        # Assigning a type to the variable 'if_condition_209498' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'if_condition_209498', if_condition_209498)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to getattr(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'self' (line 108)
        self_209500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'self', False)
        # Getting the type of 'name' (line 108)
        name_209501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 38), 'name', False)
        # Processing the call keyword arguments (line 108)
        kwargs_209502 = {}
        # Getting the type of 'getattr' (line 108)
        getattr_209499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 108)
        getattr_call_result_209503 = invoke(stypy.reporting.localization.Localization(__file__, 108, 24), getattr_209499, *[self_209500, name_209501], **kwargs_209502)
        
        # Assigning a type to the variable 'value' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'value', getattr_call_result_209503)
        
        # Call to append(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'tuple' (line 110)
        tuple_209507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 110)
        # Adding element type (line 110)
        # Getting the type of 'name' (line 110)
        name_209508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), tuple_209507, name_209508)
        # Adding element type (line 110)
        
        # Call to bind_index(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'value' (line 110)
        value_209510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 38), 'value', False)
        int_209511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 45), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_209512 = {}
        # Getting the type of 'bind_index' (line 110)
        bind_index_209509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'bind_index', False)
        # Calling bind_index(args, kwargs) (line 110)
        bind_index_call_result_209513 = invoke(stypy.reporting.localization.Localization(__file__, 110, 27), bind_index_209509, *[value_209510, int_209511], **kwargs_209512)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), tuple_209507, bind_index_call_result_209513)
        # Adding element type (line 110)
        
        # Call to bind_index(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'value' (line 110)
        value_209515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 60), 'value', False)
        int_209516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 67), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_209517 = {}
        # Getting the type of 'bind_index' (line 110)
        bind_index_209514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 49), 'bind_index', False)
        # Calling bind_index(args, kwargs) (line 110)
        bind_index_call_result_209518 = invoke(stypy.reporting.localization.Localization(__file__, 110, 49), bind_index_209514, *[value_209515, int_209516], **kwargs_209517)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 21), tuple_209507, bind_index_call_result_209518)
        
        # Processing the call keyword arguments (line 109)
        kwargs_209519 = {}
        # Getting the type of 'self' (line 109)
        self_209504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'self', False)
        # Obtaining the member 'line_funcs' of a type (line 109)
        line_funcs_209505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), self_209504, 'line_funcs')
        # Obtaining the member 'append' of a type (line 109)
        append_209506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), line_funcs_209505, 'append')
        # Calling append(args, kwargs) (line 109)
        append_call_result_209520 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), append_209506, *[tuple_209507], **kwargs_209519)
        
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to seed(...): (line 112)
        # Processing the call arguments (line 112)
        int_209524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 23), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_209525 = {}
        # Getting the type of 'np' (line 112)
        np_209521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 112)
        random_209522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), np_209521, 'random')
        # Obtaining the member 'seed' of a type (line 112)
        seed_209523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), random_209522, 'seed')
        # Calling seed(args, kwargs) (line 112)
        seed_call_result_209526 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), seed_209523, *[int_209524], **kwargs_209525)
        
        
        # Assigning a Call to a Attribute (line 113):
        
        # Assigning a Call to a Attribute (line 113):
        
        # Call to randn(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_209530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'self', False)
        # Obtaining the member 'N' of a type (line 113)
        N_209531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 33), self_209530, 'N')
        # Getting the type of 'self' (line 113)
        self_209532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'self', False)
        # Obtaining the member 'N' of a type (line 113)
        N_209533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 41), self_209532, 'N')
        # Processing the call keyword arguments (line 113)
        kwargs_209534 = {}
        # Getting the type of 'np' (line 113)
        np_209527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'np', False)
        # Obtaining the member 'random' of a type (line 113)
        random_209528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 17), np_209527, 'random')
        # Obtaining the member 'randn' of a type (line 113)
        randn_209529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 17), random_209528, 'randn')
        # Calling randn(args, kwargs) (line 113)
        randn_call_result_209535 = invoke(stypy.reporting.localization.Localization(__file__, 113, 17), randn_209529, *[N_209531, N_209533], **kwargs_209534)
        
        # Getting the type of 'self' (line 113)
        self_209536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'A' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_209536, 'A', randn_call_result_209535)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_209537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209537)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_209537


    @norecursion
    def scalar_iter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'scalar_iter'
        module_type_store = module_type_store.open_function_context('scalar_iter', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.scalar_iter')
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.scalar_iter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.scalar_iter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'scalar_iter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'scalar_iter(...)' code ##################

        
        # Getting the type of 'self' (line 116)
        self_209538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'self')
        # Obtaining the member 'scalar_funcs' of a type (line 116)
        scalar_funcs_209539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 33), self_209538, 'scalar_funcs')
        # Testing the type of a for loop iterable (line 116)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 8), scalar_funcs_209539)
        # Getting the type of the for loop variable (line 116)
        for_loop_var_209540 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 8), scalar_funcs_209539)
        # Assigning a type to the variable 'name' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 8), for_loop_var_209540))
        # Assigning a type to the variable 'phi' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'phi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 8), for_loop_var_209540))
        # Assigning a type to the variable 'derphi' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'derphi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 8), for_loop_var_209540))
        # SSA begins for a for statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to randn(...): (line 117)
        # Processing the call arguments (line 117)
        int_209544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'int')
        # Processing the call keyword arguments (line 117)
        kwargs_209545 = {}
        # Getting the type of 'np' (line 117)
        np_209541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'np', False)
        # Obtaining the member 'random' of a type (line 117)
        random_209542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 28), np_209541, 'random')
        # Obtaining the member 'randn' of a type (line 117)
        randn_209543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 28), random_209542, 'randn')
        # Calling randn(args, kwargs) (line 117)
        randn_call_result_209546 = invoke(stypy.reporting.localization.Localization(__file__, 117, 28), randn_209543, *[int_209544], **kwargs_209545)
        
        # Testing the type of a for loop iterable (line 117)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 12), randn_call_result_209546)
        # Getting the type of the for loop variable (line 117)
        for_loop_var_209547 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 12), randn_call_result_209546)
        # Assigning a type to the variable 'old_phi0' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'old_phi0', for_loop_var_209547)
        # SSA begins for a for statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_209548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        # Getting the type of 'name' (line 118)
        name_209549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 22), tuple_209548, name_209549)
        # Adding element type (line 118)
        # Getting the type of 'phi' (line 118)
        phi_209550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'phi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 22), tuple_209548, phi_209550)
        # Adding element type (line 118)
        # Getting the type of 'derphi' (line 118)
        derphi_209551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'derphi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 22), tuple_209548, derphi_209551)
        # Adding element type (line 118)
        # Getting the type of 'old_phi0' (line 118)
        old_phi0_209552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 41), 'old_phi0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 22), tuple_209548, old_phi0_209552)
        
        GeneratorType_209553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 16), GeneratorType_209553, tuple_209548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'stypy_return_type', GeneratorType_209553)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'scalar_iter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'scalar_iter' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_209554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209554)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'scalar_iter'
        return stypy_return_type_209554


    @norecursion
    def line_iter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'line_iter'
        module_type_store = module_type_store.open_function_context('line_iter', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.line_iter')
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.line_iter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.line_iter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'line_iter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'line_iter(...)' code ##################

        
        # Getting the type of 'self' (line 121)
        self_209555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 31), 'self')
        # Obtaining the member 'line_funcs' of a type (line 121)
        line_funcs_209556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 31), self_209555, 'line_funcs')
        # Testing the type of a for loop iterable (line 121)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 8), line_funcs_209556)
        # Getting the type of the for loop variable (line 121)
        for_loop_var_209557 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 8), line_funcs_209556)
        # Assigning a type to the variable 'name' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), for_loop_var_209557))
        # Assigning a type to the variable 'f' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), for_loop_var_209557))
        # Assigning a type to the variable 'fprime' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'fprime', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), for_loop_var_209557))
        # SSA begins for a for statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Name (line 122):
        
        # Assigning a Num to a Name (line 122):
        int_209558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'int')
        # Assigning a type to the variable 'k' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'k', int_209558)
        
        
        # Getting the type of 'k' (line 123)
        k_209559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'k')
        int_209560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 22), 'int')
        # Applying the binary operator '<' (line 123)
        result_lt_209561 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 18), '<', k_209559, int_209560)
        
        # Testing the type of an if condition (line 123)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 12), result_lt_209561)
        # SSA begins for while statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to randn(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_209565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'self', False)
        # Obtaining the member 'N' of a type (line 124)
        N_209566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 36), self_209565, 'N')
        # Processing the call keyword arguments (line 124)
        kwargs_209567 = {}
        # Getting the type of 'np' (line 124)
        np_209562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 124)
        random_209563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), np_209562, 'random')
        # Obtaining the member 'randn' of a type (line 124)
        randn_209564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 20), random_209563, 'randn')
        # Calling randn(args, kwargs) (line 124)
        randn_call_result_209568 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), randn_209564, *[N_209566], **kwargs_209567)
        
        # Assigning a type to the variable 'x' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'x', randn_call_result_209568)
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to randn(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'self' (line 125)
        self_209572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'self', False)
        # Obtaining the member 'N' of a type (line 125)
        N_209573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 36), self_209572, 'N')
        # Processing the call keyword arguments (line 125)
        kwargs_209574 = {}
        # Getting the type of 'np' (line 125)
        np_209569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 125)
        random_209570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 20), np_209569, 'random')
        # Obtaining the member 'randn' of a type (line 125)
        randn_209571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 20), random_209570, 'randn')
        # Calling randn(args, kwargs) (line 125)
        randn_call_result_209575 = invoke(stypy.reporting.localization.Localization(__file__, 125, 20), randn_209571, *[N_209573], **kwargs_209574)
        
        # Assigning a type to the variable 'p' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'p', randn_call_result_209575)
        
        
        
        # Call to dot(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'p' (line 126)
        p_209578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'p', False)
        
        # Call to fprime(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'x' (line 126)
        x_209580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'x', False)
        # Processing the call keyword arguments (line 126)
        kwargs_209581 = {}
        # Getting the type of 'fprime' (line 126)
        fprime_209579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 29), 'fprime', False)
        # Calling fprime(args, kwargs) (line 126)
        fprime_call_result_209582 = invoke(stypy.reporting.localization.Localization(__file__, 126, 29), fprime_209579, *[x_209580], **kwargs_209581)
        
        # Processing the call keyword arguments (line 126)
        kwargs_209583 = {}
        # Getting the type of 'np' (line 126)
        np_209576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'np', False)
        # Obtaining the member 'dot' of a type (line 126)
        dot_209577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), np_209576, 'dot')
        # Calling dot(args, kwargs) (line 126)
        dot_call_result_209584 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), dot_209577, *[p_209578, fprime_call_result_209582], **kwargs_209583)
        
        int_209585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 43), 'int')
        # Applying the binary operator '>=' (line 126)
        result_ge_209586 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 19), '>=', dot_call_result_209584, int_209585)
        
        # Testing the type of an if condition (line 126)
        if_condition_209587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 16), result_ge_209586)
        # Assigning a type to the variable 'if_condition_209587' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'if_condition_209587', if_condition_209587)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'k' (line 129)
        k_209588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'k')
        int_209589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 21), 'int')
        # Applying the binary operator '+=' (line 129)
        result_iadd_209590 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), '+=', k_209588, int_209589)
        # Assigning a type to the variable 'k' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'k', result_iadd_209590)
        
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to float(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Call to randn(...): (line 130)
        # Processing the call keyword arguments (line 130)
        kwargs_209595 = {}
        # Getting the type of 'np' (line 130)
        np_209592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 31), 'np', False)
        # Obtaining the member 'random' of a type (line 130)
        random_209593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 31), np_209592, 'random')
        # Obtaining the member 'randn' of a type (line 130)
        randn_209594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 31), random_209593, 'randn')
        # Calling randn(args, kwargs) (line 130)
        randn_call_result_209596 = invoke(stypy.reporting.localization.Localization(__file__, 130, 31), randn_209594, *[], **kwargs_209595)
        
        # Processing the call keyword arguments (line 130)
        kwargs_209597 = {}
        # Getting the type of 'float' (line 130)
        float_209591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'float', False)
        # Calling float(args, kwargs) (line 130)
        float_call_result_209598 = invoke(stypy.reporting.localization.Localization(__file__, 130, 25), float_209591, *[randn_call_result_209596], **kwargs_209597)
        
        # Assigning a type to the variable 'old_fv' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'old_fv', float_call_result_209598)
        # Creating a generator
        
        # Obtaining an instance of the builtin type 'tuple' (line 131)
        tuple_209599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 131)
        # Adding element type (line 131)
        # Getting the type of 'name' (line 131)
        name_209600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 22), tuple_209599, name_209600)
        # Adding element type (line 131)
        # Getting the type of 'f' (line 131)
        f_209601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'f')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 22), tuple_209599, f_209601)
        # Adding element type (line 131)
        # Getting the type of 'fprime' (line 131)
        fprime_209602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'fprime')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 22), tuple_209599, fprime_209602)
        # Adding element type (line 131)
        # Getting the type of 'x' (line 131)
        x_209603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 39), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 22), tuple_209599, x_209603)
        # Adding element type (line 131)
        # Getting the type of 'p' (line 131)
        p_209604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 42), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 22), tuple_209599, p_209604)
        # Adding element type (line 131)
        # Getting the type of 'old_fv' (line 131)
        old_fv_209605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 45), 'old_fv')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 22), tuple_209599, old_fv_209605)
        
        GeneratorType_209606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 16), GeneratorType_209606, tuple_209599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'stypy_return_type', GeneratorType_209606)
        # SSA join for while statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'line_iter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'line_iter' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_209607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'line_iter'
        return stypy_return_type_209607


    @norecursion
    def test_scalar_search_wolfe1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_search_wolfe1'
        module_type_store = module_type_store.open_function_context('test_scalar_search_wolfe1', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_scalar_search_wolfe1')
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_scalar_search_wolfe1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_scalar_search_wolfe1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_search_wolfe1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_search_wolfe1(...)' code ##################

        
        # Assigning a Num to a Name (line 136):
        
        # Assigning a Num to a Name (line 136):
        int_209608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 12), 'int')
        # Assigning a type to the variable 'c' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'c', int_209608)
        
        
        # Call to scalar_iter(...): (line 137)
        # Processing the call keyword arguments (line 137)
        kwargs_209611 = {}
        # Getting the type of 'self' (line 137)
        self_209609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 43), 'self', False)
        # Obtaining the member 'scalar_iter' of a type (line 137)
        scalar_iter_209610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 43), self_209609, 'scalar_iter')
        # Calling scalar_iter(args, kwargs) (line 137)
        scalar_iter_call_result_209612 = invoke(stypy.reporting.localization.Localization(__file__, 137, 43), scalar_iter_209610, *[], **kwargs_209611)
        
        # Testing the type of a for loop iterable (line 137)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 8), scalar_iter_call_result_209612)
        # Getting the type of the for loop variable (line 137)
        for_loop_var_209613 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 8), scalar_iter_call_result_209612)
        # Assigning a type to the variable 'name' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_209613))
        # Assigning a type to the variable 'phi' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'phi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_209613))
        # Assigning a type to the variable 'derphi' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'derphi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_209613))
        # Assigning a type to the variable 'old_phi0' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'old_phi0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_209613))
        # SSA begins for a for statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'c' (line 138)
        c_209614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'c')
        int_209615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 17), 'int')
        # Applying the binary operator '+=' (line 138)
        result_iadd_209616 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 12), '+=', c_209614, int_209615)
        # Assigning a type to the variable 'c' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'c', result_iadd_209616)
        
        
        # Assigning a Call to a Tuple (line 139):
        
        # Assigning a Subscript to a Name (line 139):
        
        # Obtaining the type of the subscript
        int_209617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 12), 'int')
        
        # Call to scalar_search_wolfe1(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'phi' (line 139)
        phi_209620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 52), 'phi', False)
        # Getting the type of 'derphi' (line 139)
        derphi_209621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 57), 'derphi', False)
        
        # Call to phi(...): (line 139)
        # Processing the call arguments (line 139)
        int_209623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 69), 'int')
        # Processing the call keyword arguments (line 139)
        kwargs_209624 = {}
        # Getting the type of 'phi' (line 139)
        phi_209622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 65), 'phi', False)
        # Calling phi(args, kwargs) (line 139)
        phi_call_result_209625 = invoke(stypy.reporting.localization.Localization(__file__, 139, 65), phi_209622, *[int_209623], **kwargs_209624)
        
        # Getting the type of 'old_phi0' (line 140)
        old_phi0_209626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 52), 'old_phi0', False)
        
        # Call to derphi(...): (line 140)
        # Processing the call arguments (line 140)
        int_209628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 69), 'int')
        # Processing the call keyword arguments (line 140)
        kwargs_209629 = {}
        # Getting the type of 'derphi' (line 140)
        derphi_209627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 62), 'derphi', False)
        # Calling derphi(args, kwargs) (line 140)
        derphi_call_result_209630 = invoke(stypy.reporting.localization.Localization(__file__, 140, 62), derphi_209627, *[int_209628], **kwargs_209629)
        
        # Processing the call keyword arguments (line 139)
        kwargs_209631 = {}
        # Getting the type of 'ls' (line 139)
        ls_209618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'ls', False)
        # Obtaining the member 'scalar_search_wolfe1' of a type (line 139)
        scalar_search_wolfe1_209619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 28), ls_209618, 'scalar_search_wolfe1')
        # Calling scalar_search_wolfe1(args, kwargs) (line 139)
        scalar_search_wolfe1_call_result_209632 = invoke(stypy.reporting.localization.Localization(__file__, 139, 28), scalar_search_wolfe1_209619, *[phi_209620, derphi_209621, phi_call_result_209625, old_phi0_209626, derphi_call_result_209630], **kwargs_209631)
        
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___209633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), scalar_search_wolfe1_call_result_209632, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_209634 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), getitem___209633, int_209617)
        
        # Assigning a type to the variable 'tuple_var_assignment_209083' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'tuple_var_assignment_209083', subscript_call_result_209634)
        
        # Assigning a Subscript to a Name (line 139):
        
        # Obtaining the type of the subscript
        int_209635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 12), 'int')
        
        # Call to scalar_search_wolfe1(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'phi' (line 139)
        phi_209638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 52), 'phi', False)
        # Getting the type of 'derphi' (line 139)
        derphi_209639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 57), 'derphi', False)
        
        # Call to phi(...): (line 139)
        # Processing the call arguments (line 139)
        int_209641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 69), 'int')
        # Processing the call keyword arguments (line 139)
        kwargs_209642 = {}
        # Getting the type of 'phi' (line 139)
        phi_209640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 65), 'phi', False)
        # Calling phi(args, kwargs) (line 139)
        phi_call_result_209643 = invoke(stypy.reporting.localization.Localization(__file__, 139, 65), phi_209640, *[int_209641], **kwargs_209642)
        
        # Getting the type of 'old_phi0' (line 140)
        old_phi0_209644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 52), 'old_phi0', False)
        
        # Call to derphi(...): (line 140)
        # Processing the call arguments (line 140)
        int_209646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 69), 'int')
        # Processing the call keyword arguments (line 140)
        kwargs_209647 = {}
        # Getting the type of 'derphi' (line 140)
        derphi_209645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 62), 'derphi', False)
        # Calling derphi(args, kwargs) (line 140)
        derphi_call_result_209648 = invoke(stypy.reporting.localization.Localization(__file__, 140, 62), derphi_209645, *[int_209646], **kwargs_209647)
        
        # Processing the call keyword arguments (line 139)
        kwargs_209649 = {}
        # Getting the type of 'ls' (line 139)
        ls_209636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'ls', False)
        # Obtaining the member 'scalar_search_wolfe1' of a type (line 139)
        scalar_search_wolfe1_209637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 28), ls_209636, 'scalar_search_wolfe1')
        # Calling scalar_search_wolfe1(args, kwargs) (line 139)
        scalar_search_wolfe1_call_result_209650 = invoke(stypy.reporting.localization.Localization(__file__, 139, 28), scalar_search_wolfe1_209637, *[phi_209638, derphi_209639, phi_call_result_209643, old_phi0_209644, derphi_call_result_209648], **kwargs_209649)
        
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___209651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), scalar_search_wolfe1_call_result_209650, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_209652 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), getitem___209651, int_209635)
        
        # Assigning a type to the variable 'tuple_var_assignment_209084' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'tuple_var_assignment_209084', subscript_call_result_209652)
        
        # Assigning a Subscript to a Name (line 139):
        
        # Obtaining the type of the subscript
        int_209653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 12), 'int')
        
        # Call to scalar_search_wolfe1(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'phi' (line 139)
        phi_209656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 52), 'phi', False)
        # Getting the type of 'derphi' (line 139)
        derphi_209657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 57), 'derphi', False)
        
        # Call to phi(...): (line 139)
        # Processing the call arguments (line 139)
        int_209659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 69), 'int')
        # Processing the call keyword arguments (line 139)
        kwargs_209660 = {}
        # Getting the type of 'phi' (line 139)
        phi_209658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 65), 'phi', False)
        # Calling phi(args, kwargs) (line 139)
        phi_call_result_209661 = invoke(stypy.reporting.localization.Localization(__file__, 139, 65), phi_209658, *[int_209659], **kwargs_209660)
        
        # Getting the type of 'old_phi0' (line 140)
        old_phi0_209662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 52), 'old_phi0', False)
        
        # Call to derphi(...): (line 140)
        # Processing the call arguments (line 140)
        int_209664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 69), 'int')
        # Processing the call keyword arguments (line 140)
        kwargs_209665 = {}
        # Getting the type of 'derphi' (line 140)
        derphi_209663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 62), 'derphi', False)
        # Calling derphi(args, kwargs) (line 140)
        derphi_call_result_209666 = invoke(stypy.reporting.localization.Localization(__file__, 140, 62), derphi_209663, *[int_209664], **kwargs_209665)
        
        # Processing the call keyword arguments (line 139)
        kwargs_209667 = {}
        # Getting the type of 'ls' (line 139)
        ls_209654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'ls', False)
        # Obtaining the member 'scalar_search_wolfe1' of a type (line 139)
        scalar_search_wolfe1_209655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 28), ls_209654, 'scalar_search_wolfe1')
        # Calling scalar_search_wolfe1(args, kwargs) (line 139)
        scalar_search_wolfe1_call_result_209668 = invoke(stypy.reporting.localization.Localization(__file__, 139, 28), scalar_search_wolfe1_209655, *[phi_209656, derphi_209657, phi_call_result_209661, old_phi0_209662, derphi_call_result_209666], **kwargs_209667)
        
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___209669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), scalar_search_wolfe1_call_result_209668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_209670 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), getitem___209669, int_209653)
        
        # Assigning a type to the variable 'tuple_var_assignment_209085' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'tuple_var_assignment_209085', subscript_call_result_209670)
        
        # Assigning a Name to a Name (line 139):
        # Getting the type of 'tuple_var_assignment_209083' (line 139)
        tuple_var_assignment_209083_209671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'tuple_var_assignment_209083')
        # Assigning a type to the variable 's' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 's', tuple_var_assignment_209083_209671)
        
        # Assigning a Name to a Name (line 139):
        # Getting the type of 'tuple_var_assignment_209084' (line 139)
        tuple_var_assignment_209084_209672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'tuple_var_assignment_209084')
        # Assigning a type to the variable 'phi1' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'phi1', tuple_var_assignment_209084_209672)
        
        # Assigning a Name to a Name (line 139):
        # Getting the type of 'tuple_var_assignment_209085' (line 139)
        tuple_var_assignment_209085_209673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'tuple_var_assignment_209085')
        # Assigning a type to the variable 'phi0' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'phi0', tuple_var_assignment_209085_209673)
        
        # Call to assert_fp_equal(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'phi0' (line 141)
        phi0_209675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 28), 'phi0', False)
        
        # Call to phi(...): (line 141)
        # Processing the call arguments (line 141)
        int_209677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 38), 'int')
        # Processing the call keyword arguments (line 141)
        kwargs_209678 = {}
        # Getting the type of 'phi' (line 141)
        phi_209676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'phi', False)
        # Calling phi(args, kwargs) (line 141)
        phi_call_result_209679 = invoke(stypy.reporting.localization.Localization(__file__, 141, 34), phi_209676, *[int_209677], **kwargs_209678)
        
        # Getting the type of 'name' (line 141)
        name_209680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), 'name', False)
        # Processing the call keyword arguments (line 141)
        kwargs_209681 = {}
        # Getting the type of 'assert_fp_equal' (line 141)
        assert_fp_equal_209674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 141)
        assert_fp_equal_call_result_209682 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), assert_fp_equal_209674, *[phi0_209675, phi_call_result_209679, name_209680], **kwargs_209681)
        
        
        # Call to assert_fp_equal(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'phi1' (line 142)
        phi1_209684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 28), 'phi1', False)
        
        # Call to phi(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 's' (line 142)
        s_209686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 's', False)
        # Processing the call keyword arguments (line 142)
        kwargs_209687 = {}
        # Getting the type of 'phi' (line 142)
        phi_209685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'phi', False)
        # Calling phi(args, kwargs) (line 142)
        phi_call_result_209688 = invoke(stypy.reporting.localization.Localization(__file__, 142, 34), phi_209685, *[s_209686], **kwargs_209687)
        
        # Getting the type of 'name' (line 142)
        name_209689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'name', False)
        # Processing the call keyword arguments (line 142)
        kwargs_209690 = {}
        # Getting the type of 'assert_fp_equal' (line 142)
        assert_fp_equal_209683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 142)
        assert_fp_equal_call_result_209691 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), assert_fp_equal_209683, *[phi1_209684, phi_call_result_209688, name_209689], **kwargs_209690)
        
        
        # Call to assert_wolfe(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 's' (line 143)
        s_209693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 's', False)
        # Getting the type of 'phi' (line 143)
        phi_209694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'phi', False)
        # Getting the type of 'derphi' (line 143)
        derphi_209695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'derphi', False)
        # Processing the call keyword arguments (line 143)
        # Getting the type of 'name' (line 143)
        name_209696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 49), 'name', False)
        keyword_209697 = name_209696
        kwargs_209698 = {'err_msg': keyword_209697}
        # Getting the type of 'assert_wolfe' (line 143)
        assert_wolfe_209692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'assert_wolfe', False)
        # Calling assert_wolfe(args, kwargs) (line 143)
        assert_wolfe_call_result_209699 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), assert_wolfe_209692, *[s_209693, phi_209694, derphi_209695], **kwargs_209698)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Getting the type of 'c' (line 145)
        c_209701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'c', False)
        int_209702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 20), 'int')
        # Applying the binary operator '>' (line 145)
        result_gt_209703 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 16), '>', c_209701, int_209702)
        
        # Processing the call keyword arguments (line 145)
        kwargs_209704 = {}
        # Getting the type of 'assert_' (line 145)
        assert__209700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 145)
        assert__call_result_209705 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), assert__209700, *[result_gt_209703], **kwargs_209704)
        
        
        # ################# End of 'test_scalar_search_wolfe1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_search_wolfe1' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_209706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209706)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_search_wolfe1'
        return stypy_return_type_209706


    @norecursion
    def test_scalar_search_wolfe2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_search_wolfe2'
        module_type_store = module_type_store.open_function_context('test_scalar_search_wolfe2', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_scalar_search_wolfe2')
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_scalar_search_wolfe2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_scalar_search_wolfe2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_search_wolfe2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_search_wolfe2(...)' code ##################

        
        
        # Call to scalar_iter(...): (line 148)
        # Processing the call keyword arguments (line 148)
        kwargs_209709 = {}
        # Getting the type of 'self' (line 148)
        self_209707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 43), 'self', False)
        # Obtaining the member 'scalar_iter' of a type (line 148)
        scalar_iter_209708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 43), self_209707, 'scalar_iter')
        # Calling scalar_iter(args, kwargs) (line 148)
        scalar_iter_call_result_209710 = invoke(stypy.reporting.localization.Localization(__file__, 148, 43), scalar_iter_209708, *[], **kwargs_209709)
        
        # Testing the type of a for loop iterable (line 148)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 8), scalar_iter_call_result_209710)
        # Getting the type of the for loop variable (line 148)
        for_loop_var_209711 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 8), scalar_iter_call_result_209710)
        # Assigning a type to the variable 'name' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 8), for_loop_var_209711))
        # Assigning a type to the variable 'phi' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'phi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 8), for_loop_var_209711))
        # Assigning a type to the variable 'derphi' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'derphi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 8), for_loop_var_209711))
        # Assigning a type to the variable 'old_phi0' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'old_phi0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 8), for_loop_var_209711))
        # SSA begins for a for statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 149):
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_209712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'int')
        
        # Call to scalar_search_wolfe2(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'phi' (line 150)
        phi_209715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'phi', False)
        # Getting the type of 'derphi' (line 150)
        derphi_209716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'derphi', False)
        
        # Call to phi(...): (line 150)
        # Processing the call arguments (line 150)
        int_209718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_209719 = {}
        # Getting the type of 'phi' (line 150)
        phi_209717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'phi', False)
        # Calling phi(args, kwargs) (line 150)
        phi_call_result_209720 = invoke(stypy.reporting.localization.Localization(__file__, 150, 29), phi_209717, *[int_209718], **kwargs_209719)
        
        # Getting the type of 'old_phi0' (line 150)
        old_phi0_209721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'old_phi0', False)
        
        # Call to derphi(...): (line 150)
        # Processing the call arguments (line 150)
        int_209723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 54), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_209724 = {}
        # Getting the type of 'derphi' (line 150)
        derphi_209722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 47), 'derphi', False)
        # Calling derphi(args, kwargs) (line 150)
        derphi_call_result_209725 = invoke(stypy.reporting.localization.Localization(__file__, 150, 47), derphi_209722, *[int_209723], **kwargs_209724)
        
        # Processing the call keyword arguments (line 149)
        kwargs_209726 = {}
        # Getting the type of 'ls' (line 149)
        ls_209713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'ls', False)
        # Obtaining the member 'scalar_search_wolfe2' of a type (line 149)
        scalar_search_wolfe2_209714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 37), ls_209713, 'scalar_search_wolfe2')
        # Calling scalar_search_wolfe2(args, kwargs) (line 149)
        scalar_search_wolfe2_call_result_209727 = invoke(stypy.reporting.localization.Localization(__file__, 149, 37), scalar_search_wolfe2_209714, *[phi_209715, derphi_209716, phi_call_result_209720, old_phi0_209721, derphi_call_result_209725], **kwargs_209726)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___209728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), scalar_search_wolfe2_call_result_209727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_209729 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), getitem___209728, int_209712)
        
        # Assigning a type to the variable 'tuple_var_assignment_209086' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'tuple_var_assignment_209086', subscript_call_result_209729)
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_209730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'int')
        
        # Call to scalar_search_wolfe2(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'phi' (line 150)
        phi_209733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'phi', False)
        # Getting the type of 'derphi' (line 150)
        derphi_209734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'derphi', False)
        
        # Call to phi(...): (line 150)
        # Processing the call arguments (line 150)
        int_209736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_209737 = {}
        # Getting the type of 'phi' (line 150)
        phi_209735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'phi', False)
        # Calling phi(args, kwargs) (line 150)
        phi_call_result_209738 = invoke(stypy.reporting.localization.Localization(__file__, 150, 29), phi_209735, *[int_209736], **kwargs_209737)
        
        # Getting the type of 'old_phi0' (line 150)
        old_phi0_209739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'old_phi0', False)
        
        # Call to derphi(...): (line 150)
        # Processing the call arguments (line 150)
        int_209741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 54), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_209742 = {}
        # Getting the type of 'derphi' (line 150)
        derphi_209740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 47), 'derphi', False)
        # Calling derphi(args, kwargs) (line 150)
        derphi_call_result_209743 = invoke(stypy.reporting.localization.Localization(__file__, 150, 47), derphi_209740, *[int_209741], **kwargs_209742)
        
        # Processing the call keyword arguments (line 149)
        kwargs_209744 = {}
        # Getting the type of 'ls' (line 149)
        ls_209731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'ls', False)
        # Obtaining the member 'scalar_search_wolfe2' of a type (line 149)
        scalar_search_wolfe2_209732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 37), ls_209731, 'scalar_search_wolfe2')
        # Calling scalar_search_wolfe2(args, kwargs) (line 149)
        scalar_search_wolfe2_call_result_209745 = invoke(stypy.reporting.localization.Localization(__file__, 149, 37), scalar_search_wolfe2_209732, *[phi_209733, derphi_209734, phi_call_result_209738, old_phi0_209739, derphi_call_result_209743], **kwargs_209744)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___209746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), scalar_search_wolfe2_call_result_209745, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_209747 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), getitem___209746, int_209730)
        
        # Assigning a type to the variable 'tuple_var_assignment_209087' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'tuple_var_assignment_209087', subscript_call_result_209747)
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_209748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'int')
        
        # Call to scalar_search_wolfe2(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'phi' (line 150)
        phi_209751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'phi', False)
        # Getting the type of 'derphi' (line 150)
        derphi_209752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'derphi', False)
        
        # Call to phi(...): (line 150)
        # Processing the call arguments (line 150)
        int_209754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_209755 = {}
        # Getting the type of 'phi' (line 150)
        phi_209753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'phi', False)
        # Calling phi(args, kwargs) (line 150)
        phi_call_result_209756 = invoke(stypy.reporting.localization.Localization(__file__, 150, 29), phi_209753, *[int_209754], **kwargs_209755)
        
        # Getting the type of 'old_phi0' (line 150)
        old_phi0_209757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'old_phi0', False)
        
        # Call to derphi(...): (line 150)
        # Processing the call arguments (line 150)
        int_209759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 54), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_209760 = {}
        # Getting the type of 'derphi' (line 150)
        derphi_209758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 47), 'derphi', False)
        # Calling derphi(args, kwargs) (line 150)
        derphi_call_result_209761 = invoke(stypy.reporting.localization.Localization(__file__, 150, 47), derphi_209758, *[int_209759], **kwargs_209760)
        
        # Processing the call keyword arguments (line 149)
        kwargs_209762 = {}
        # Getting the type of 'ls' (line 149)
        ls_209749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'ls', False)
        # Obtaining the member 'scalar_search_wolfe2' of a type (line 149)
        scalar_search_wolfe2_209750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 37), ls_209749, 'scalar_search_wolfe2')
        # Calling scalar_search_wolfe2(args, kwargs) (line 149)
        scalar_search_wolfe2_call_result_209763 = invoke(stypy.reporting.localization.Localization(__file__, 149, 37), scalar_search_wolfe2_209750, *[phi_209751, derphi_209752, phi_call_result_209756, old_phi0_209757, derphi_call_result_209761], **kwargs_209762)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___209764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), scalar_search_wolfe2_call_result_209763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_209765 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), getitem___209764, int_209748)
        
        # Assigning a type to the variable 'tuple_var_assignment_209088' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'tuple_var_assignment_209088', subscript_call_result_209765)
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_209766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'int')
        
        # Call to scalar_search_wolfe2(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'phi' (line 150)
        phi_209769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'phi', False)
        # Getting the type of 'derphi' (line 150)
        derphi_209770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'derphi', False)
        
        # Call to phi(...): (line 150)
        # Processing the call arguments (line 150)
        int_209772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_209773 = {}
        # Getting the type of 'phi' (line 150)
        phi_209771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'phi', False)
        # Calling phi(args, kwargs) (line 150)
        phi_call_result_209774 = invoke(stypy.reporting.localization.Localization(__file__, 150, 29), phi_209771, *[int_209772], **kwargs_209773)
        
        # Getting the type of 'old_phi0' (line 150)
        old_phi0_209775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'old_phi0', False)
        
        # Call to derphi(...): (line 150)
        # Processing the call arguments (line 150)
        int_209777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 54), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_209778 = {}
        # Getting the type of 'derphi' (line 150)
        derphi_209776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 47), 'derphi', False)
        # Calling derphi(args, kwargs) (line 150)
        derphi_call_result_209779 = invoke(stypy.reporting.localization.Localization(__file__, 150, 47), derphi_209776, *[int_209777], **kwargs_209778)
        
        # Processing the call keyword arguments (line 149)
        kwargs_209780 = {}
        # Getting the type of 'ls' (line 149)
        ls_209767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'ls', False)
        # Obtaining the member 'scalar_search_wolfe2' of a type (line 149)
        scalar_search_wolfe2_209768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 37), ls_209767, 'scalar_search_wolfe2')
        # Calling scalar_search_wolfe2(args, kwargs) (line 149)
        scalar_search_wolfe2_call_result_209781 = invoke(stypy.reporting.localization.Localization(__file__, 149, 37), scalar_search_wolfe2_209768, *[phi_209769, derphi_209770, phi_call_result_209774, old_phi0_209775, derphi_call_result_209779], **kwargs_209780)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___209782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), scalar_search_wolfe2_call_result_209781, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_209783 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), getitem___209782, int_209766)
        
        # Assigning a type to the variable 'tuple_var_assignment_209089' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'tuple_var_assignment_209089', subscript_call_result_209783)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_209086' (line 149)
        tuple_var_assignment_209086_209784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'tuple_var_assignment_209086')
        # Assigning a type to the variable 's' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 's', tuple_var_assignment_209086_209784)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_209087' (line 149)
        tuple_var_assignment_209087_209785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'tuple_var_assignment_209087')
        # Assigning a type to the variable 'phi1' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'phi1', tuple_var_assignment_209087_209785)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_209088' (line 149)
        tuple_var_assignment_209088_209786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'tuple_var_assignment_209088')
        # Assigning a type to the variable 'phi0' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'phi0', tuple_var_assignment_209088_209786)
        
        # Assigning a Name to a Name (line 149):
        # Getting the type of 'tuple_var_assignment_209089' (line 149)
        tuple_var_assignment_209089_209787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'tuple_var_assignment_209089')
        # Assigning a type to the variable 'derphi1' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'derphi1', tuple_var_assignment_209089_209787)
        
        # Call to assert_fp_equal(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'phi0' (line 151)
        phi0_209789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'phi0', False)
        
        # Call to phi(...): (line 151)
        # Processing the call arguments (line 151)
        int_209791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 38), 'int')
        # Processing the call keyword arguments (line 151)
        kwargs_209792 = {}
        # Getting the type of 'phi' (line 151)
        phi_209790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'phi', False)
        # Calling phi(args, kwargs) (line 151)
        phi_call_result_209793 = invoke(stypy.reporting.localization.Localization(__file__, 151, 34), phi_209790, *[int_209791], **kwargs_209792)
        
        # Getting the type of 'name' (line 151)
        name_209794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 42), 'name', False)
        # Processing the call keyword arguments (line 151)
        kwargs_209795 = {}
        # Getting the type of 'assert_fp_equal' (line 151)
        assert_fp_equal_209788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 151)
        assert_fp_equal_call_result_209796 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), assert_fp_equal_209788, *[phi0_209789, phi_call_result_209793, name_209794], **kwargs_209795)
        
        
        # Call to assert_fp_equal(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'phi1' (line 152)
        phi1_209798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'phi1', False)
        
        # Call to phi(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 's' (line 152)
        s_209800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 38), 's', False)
        # Processing the call keyword arguments (line 152)
        kwargs_209801 = {}
        # Getting the type of 'phi' (line 152)
        phi_209799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 34), 'phi', False)
        # Calling phi(args, kwargs) (line 152)
        phi_call_result_209802 = invoke(stypy.reporting.localization.Localization(__file__, 152, 34), phi_209799, *[s_209800], **kwargs_209801)
        
        # Getting the type of 'name' (line 152)
        name_209803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 42), 'name', False)
        # Processing the call keyword arguments (line 152)
        kwargs_209804 = {}
        # Getting the type of 'assert_fp_equal' (line 152)
        assert_fp_equal_209797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 152)
        assert_fp_equal_call_result_209805 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), assert_fp_equal_209797, *[phi1_209798, phi_call_result_209802, name_209803], **kwargs_209804)
        
        
        # Type idiom detected: calculating its left and rigth part (line 153)
        # Getting the type of 'derphi1' (line 153)
        derphi1_209806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'derphi1')
        # Getting the type of 'None' (line 153)
        None_209807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 30), 'None')
        
        (may_be_209808, more_types_in_union_209809) = may_not_be_none(derphi1_209806, None_209807)

        if may_be_209808:

            if more_types_in_union_209809:
                # Runtime conditional SSA (line 153)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to assert_fp_equal(...): (line 154)
            # Processing the call arguments (line 154)
            # Getting the type of 'derphi1' (line 154)
            derphi1_209811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 32), 'derphi1', False)
            
            # Call to derphi(...): (line 154)
            # Processing the call arguments (line 154)
            # Getting the type of 's' (line 154)
            s_209813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 48), 's', False)
            # Processing the call keyword arguments (line 154)
            kwargs_209814 = {}
            # Getting the type of 'derphi' (line 154)
            derphi_209812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 41), 'derphi', False)
            # Calling derphi(args, kwargs) (line 154)
            derphi_call_result_209815 = invoke(stypy.reporting.localization.Localization(__file__, 154, 41), derphi_209812, *[s_209813], **kwargs_209814)
            
            # Getting the type of 'name' (line 154)
            name_209816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 52), 'name', False)
            # Processing the call keyword arguments (line 154)
            kwargs_209817 = {}
            # Getting the type of 'assert_fp_equal' (line 154)
            assert_fp_equal_209810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'assert_fp_equal', False)
            # Calling assert_fp_equal(args, kwargs) (line 154)
            assert_fp_equal_call_result_209818 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), assert_fp_equal_209810, *[derphi1_209811, derphi_call_result_209815, name_209816], **kwargs_209817)
            

            if more_types_in_union_209809:
                # SSA join for if statement (line 153)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to assert_wolfe(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 's' (line 155)
        s_209820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), 's', False)
        # Getting the type of 'phi' (line 155)
        phi_209821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'phi', False)
        # Getting the type of 'derphi' (line 155)
        derphi_209822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'derphi', False)
        # Processing the call keyword arguments (line 155)
        str_209823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 49), 'str', '%s %g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 155)
        tuple_209824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 155)
        # Adding element type (line 155)
        # Getting the type of 'name' (line 155)
        name_209825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 60), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 60), tuple_209824, name_209825)
        # Adding element type (line 155)
        # Getting the type of 'old_phi0' (line 155)
        old_phi0_209826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 66), 'old_phi0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 60), tuple_209824, old_phi0_209826)
        
        # Applying the binary operator '%' (line 155)
        result_mod_209827 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 49), '%', str_209823, tuple_209824)
        
        keyword_209828 = result_mod_209827
        kwargs_209829 = {'err_msg': keyword_209828}
        # Getting the type of 'assert_wolfe' (line 155)
        assert_wolfe_209819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'assert_wolfe', False)
        # Calling assert_wolfe(args, kwargs) (line 155)
        assert_wolfe_call_result_209830 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), assert_wolfe_209819, *[s_209820, phi_209821, derphi_209822], **kwargs_209829)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_scalar_search_wolfe2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_search_wolfe2' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_209831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209831)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_search_wolfe2'
        return stypy_return_type_209831


    @norecursion
    def test_scalar_search_armijo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar_search_armijo'
        module_type_store = module_type_store.open_function_context('test_scalar_search_armijo', 157, 4, False)
        # Assigning a type to the variable 'self' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_scalar_search_armijo')
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_scalar_search_armijo.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_scalar_search_armijo', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar_search_armijo', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar_search_armijo(...)' code ##################

        
        
        # Call to scalar_iter(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_209834 = {}
        # Getting the type of 'self' (line 158)
        self_209832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 43), 'self', False)
        # Obtaining the member 'scalar_iter' of a type (line 158)
        scalar_iter_209833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 43), self_209832, 'scalar_iter')
        # Calling scalar_iter(args, kwargs) (line 158)
        scalar_iter_call_result_209835 = invoke(stypy.reporting.localization.Localization(__file__, 158, 43), scalar_iter_209833, *[], **kwargs_209834)
        
        # Testing the type of a for loop iterable (line 158)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 8), scalar_iter_call_result_209835)
        # Getting the type of the for loop variable (line 158)
        for_loop_var_209836 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 8), scalar_iter_call_result_209835)
        # Assigning a type to the variable 'name' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), for_loop_var_209836))
        # Assigning a type to the variable 'phi' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'phi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), for_loop_var_209836))
        # Assigning a type to the variable 'derphi' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'derphi', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), for_loop_var_209836))
        # Assigning a type to the variable 'old_phi0' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'old_phi0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), for_loop_var_209836))
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 159):
        
        # Assigning a Subscript to a Name (line 159):
        
        # Obtaining the type of the subscript
        int_209837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'int')
        
        # Call to scalar_search_armijo(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'phi' (line 159)
        phi_209840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'phi', False)
        
        # Call to phi(...): (line 159)
        # Processing the call arguments (line 159)
        int_209842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 55), 'int')
        # Processing the call keyword arguments (line 159)
        kwargs_209843 = {}
        # Getting the type of 'phi' (line 159)
        phi_209841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'phi', False)
        # Calling phi(args, kwargs) (line 159)
        phi_call_result_209844 = invoke(stypy.reporting.localization.Localization(__file__, 159, 51), phi_209841, *[int_209842], **kwargs_209843)
        
        
        # Call to derphi(...): (line 159)
        # Processing the call arguments (line 159)
        int_209846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 66), 'int')
        # Processing the call keyword arguments (line 159)
        kwargs_209847 = {}
        # Getting the type of 'derphi' (line 159)
        derphi_209845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'derphi', False)
        # Calling derphi(args, kwargs) (line 159)
        derphi_call_result_209848 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), derphi_209845, *[int_209846], **kwargs_209847)
        
        # Processing the call keyword arguments (line 159)
        kwargs_209849 = {}
        # Getting the type of 'ls' (line 159)
        ls_209838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'ls', False)
        # Obtaining the member 'scalar_search_armijo' of a type (line 159)
        scalar_search_armijo_209839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), ls_209838, 'scalar_search_armijo')
        # Calling scalar_search_armijo(args, kwargs) (line 159)
        scalar_search_armijo_call_result_209850 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), scalar_search_armijo_209839, *[phi_209840, phi_call_result_209844, derphi_call_result_209848], **kwargs_209849)
        
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___209851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), scalar_search_armijo_call_result_209850, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_209852 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), getitem___209851, int_209837)
        
        # Assigning a type to the variable 'tuple_var_assignment_209090' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'tuple_var_assignment_209090', subscript_call_result_209852)
        
        # Assigning a Subscript to a Name (line 159):
        
        # Obtaining the type of the subscript
        int_209853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'int')
        
        # Call to scalar_search_armijo(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'phi' (line 159)
        phi_209856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'phi', False)
        
        # Call to phi(...): (line 159)
        # Processing the call arguments (line 159)
        int_209858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 55), 'int')
        # Processing the call keyword arguments (line 159)
        kwargs_209859 = {}
        # Getting the type of 'phi' (line 159)
        phi_209857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 51), 'phi', False)
        # Calling phi(args, kwargs) (line 159)
        phi_call_result_209860 = invoke(stypy.reporting.localization.Localization(__file__, 159, 51), phi_209857, *[int_209858], **kwargs_209859)
        
        
        # Call to derphi(...): (line 159)
        # Processing the call arguments (line 159)
        int_209862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 66), 'int')
        # Processing the call keyword arguments (line 159)
        kwargs_209863 = {}
        # Getting the type of 'derphi' (line 159)
        derphi_209861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 59), 'derphi', False)
        # Calling derphi(args, kwargs) (line 159)
        derphi_call_result_209864 = invoke(stypy.reporting.localization.Localization(__file__, 159, 59), derphi_209861, *[int_209862], **kwargs_209863)
        
        # Processing the call keyword arguments (line 159)
        kwargs_209865 = {}
        # Getting the type of 'ls' (line 159)
        ls_209854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'ls', False)
        # Obtaining the member 'scalar_search_armijo' of a type (line 159)
        scalar_search_armijo_209855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), ls_209854, 'scalar_search_armijo')
        # Calling scalar_search_armijo(args, kwargs) (line 159)
        scalar_search_armijo_call_result_209866 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), scalar_search_armijo_209855, *[phi_209856, phi_call_result_209860, derphi_call_result_209864], **kwargs_209865)
        
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___209867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), scalar_search_armijo_call_result_209866, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_209868 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), getitem___209867, int_209853)
        
        # Assigning a type to the variable 'tuple_var_assignment_209091' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'tuple_var_assignment_209091', subscript_call_result_209868)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'tuple_var_assignment_209090' (line 159)
        tuple_var_assignment_209090_209869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'tuple_var_assignment_209090')
        # Assigning a type to the variable 's' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 's', tuple_var_assignment_209090_209869)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'tuple_var_assignment_209091' (line 159)
        tuple_var_assignment_209091_209870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'tuple_var_assignment_209091')
        # Assigning a type to the variable 'phi1' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'phi1', tuple_var_assignment_209091_209870)
        
        # Call to assert_fp_equal(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'phi1' (line 160)
        phi1_209872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'phi1', False)
        
        # Call to phi(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 's' (line 160)
        s_209874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 's', False)
        # Processing the call keyword arguments (line 160)
        kwargs_209875 = {}
        # Getting the type of 'phi' (line 160)
        phi_209873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'phi', False)
        # Calling phi(args, kwargs) (line 160)
        phi_call_result_209876 = invoke(stypy.reporting.localization.Localization(__file__, 160, 34), phi_209873, *[s_209874], **kwargs_209875)
        
        # Getting the type of 'name' (line 160)
        name_209877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 42), 'name', False)
        # Processing the call keyword arguments (line 160)
        kwargs_209878 = {}
        # Getting the type of 'assert_fp_equal' (line 160)
        assert_fp_equal_209871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 160)
        assert_fp_equal_call_result_209879 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), assert_fp_equal_209871, *[phi1_209872, phi_call_result_209876, name_209877], **kwargs_209878)
        
        
        # Call to assert_armijo(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 's' (line 161)
        s_209881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 's', False)
        # Getting the type of 'phi' (line 161)
        phi_209882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 29), 'phi', False)
        # Processing the call keyword arguments (line 161)
        str_209883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 42), 'str', '%s %g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_209884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        # Getting the type of 'name' (line 161)
        name_209885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 53), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 53), tuple_209884, name_209885)
        # Adding element type (line 161)
        # Getting the type of 'old_phi0' (line 161)
        old_phi0_209886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 59), 'old_phi0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 53), tuple_209884, old_phi0_209886)
        
        # Applying the binary operator '%' (line 161)
        result_mod_209887 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 42), '%', str_209883, tuple_209884)
        
        keyword_209888 = result_mod_209887
        kwargs_209889 = {'err_msg': keyword_209888}
        # Getting the type of 'assert_armijo' (line 161)
        assert_armijo_209880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'assert_armijo', False)
        # Calling assert_armijo(args, kwargs) (line 161)
        assert_armijo_call_result_209890 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), assert_armijo_209880, *[s_209881, phi_209882], **kwargs_209889)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_scalar_search_armijo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar_search_armijo' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_209891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209891)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar_search_armijo'
        return stypy_return_type_209891


    @norecursion
    def test_line_search_wolfe1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_line_search_wolfe1'
        module_type_store = module_type_store.open_function_context('test_line_search_wolfe1', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_line_search_wolfe1')
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_line_search_wolfe1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_line_search_wolfe1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_line_search_wolfe1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_line_search_wolfe1(...)' code ##################

        
        # Assigning a Num to a Name (line 166):
        
        # Assigning a Num to a Name (line 166):
        int_209892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 12), 'int')
        # Assigning a type to the variable 'c' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'c', int_209892)
        
        # Assigning a Num to a Name (line 167):
        
        # Assigning a Num to a Name (line 167):
        int_209893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'int')
        # Assigning a type to the variable 'smax' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'smax', int_209893)
        
        
        # Call to line_iter(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_209896 = {}
        # Getting the type of 'self' (line 168)
        self_209894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 44), 'self', False)
        # Obtaining the member 'line_iter' of a type (line 168)
        line_iter_209895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 44), self_209894, 'line_iter')
        # Calling line_iter(args, kwargs) (line 168)
        line_iter_call_result_209897 = invoke(stypy.reporting.localization.Localization(__file__, 168, 44), line_iter_209895, *[], **kwargs_209896)
        
        # Testing the type of a for loop iterable (line 168)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 8), line_iter_call_result_209897)
        # Getting the type of the for loop variable (line 168)
        for_loop_var_209898 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 8), line_iter_call_result_209897)
        # Assigning a type to the variable 'name' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_209898))
        # Assigning a type to the variable 'f' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_209898))
        # Assigning a type to the variable 'fprime' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'fprime', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_209898))
        # Assigning a type to the variable 'x' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_209898))
        # Assigning a type to the variable 'p' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_209898))
        # Assigning a type to the variable 'old_f' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'old_f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 8), for_loop_var_209898))
        # SSA begins for a for statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to f(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'x' (line 169)
        x_209900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'x', False)
        # Processing the call keyword arguments (line 169)
        kwargs_209901 = {}
        # Getting the type of 'f' (line 169)
        f_209899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'f', False)
        # Calling f(args, kwargs) (line 169)
        f_call_result_209902 = invoke(stypy.reporting.localization.Localization(__file__, 169, 17), f_209899, *[x_209900], **kwargs_209901)
        
        # Assigning a type to the variable 'f0' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'f0', f_call_result_209902)
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to fprime(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'x' (line 170)
        x_209904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'x', False)
        # Processing the call keyword arguments (line 170)
        kwargs_209905 = {}
        # Getting the type of 'fprime' (line 170)
        fprime_209903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'fprime', False)
        # Calling fprime(args, kwargs) (line 170)
        fprime_call_result_209906 = invoke(stypy.reporting.localization.Localization(__file__, 170, 17), fprime_209903, *[x_209904], **kwargs_209905)
        
        # Assigning a type to the variable 'g0' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'g0', fprime_call_result_209906)
        
        # Assigning a Num to a Attribute (line 171):
        
        # Assigning a Num to a Attribute (line 171):
        int_209907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 26), 'int')
        # Getting the type of 'self' (line 171)
        self_209908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'self')
        # Setting the type of the member 'fcount' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), self_209908, 'fcount', int_209907)
        
        # Assigning a Call to a Tuple (line 172):
        
        # Assigning a Subscript to a Name (line 172):
        
        # Obtaining the type of the subscript
        int_209909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'int')
        
        # Call to line_search_wolfe1(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'f' (line 172)
        f_209912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'f', False)
        # Getting the type of 'fprime' (line 172)
        fprime_209913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 62), 'fprime', False)
        # Getting the type of 'x' (line 172)
        x_209914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 70), 'x', False)
        # Getting the type of 'p' (line 172)
        p_209915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 73), 'p', False)
        # Getting the type of 'g0' (line 173)
        g0_209916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 59), 'g0', False)
        # Getting the type of 'f0' (line 173)
        f0_209917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 63), 'f0', False)
        # Getting the type of 'old_f' (line 173)
        old_f_209918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 67), 'old_f', False)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'smax' (line 174)
        smax_209919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 64), 'smax', False)
        keyword_209920 = smax_209919
        kwargs_209921 = {'amax': keyword_209920}
        # Getting the type of 'ls' (line 172)
        ls_209910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'ls', False)
        # Obtaining the member 'line_search_wolfe1' of a type (line 172)
        line_search_wolfe1_209911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 37), ls_209910, 'line_search_wolfe1')
        # Calling line_search_wolfe1(args, kwargs) (line 172)
        line_search_wolfe1_call_result_209922 = invoke(stypy.reporting.localization.Localization(__file__, 172, 37), line_search_wolfe1_209911, *[f_209912, fprime_209913, x_209914, p_209915, g0_209916, f0_209917, old_f_209918], **kwargs_209921)
        
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___209923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), line_search_wolfe1_call_result_209922, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_209924 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), getitem___209923, int_209909)
        
        # Assigning a type to the variable 'tuple_var_assignment_209092' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209092', subscript_call_result_209924)
        
        # Assigning a Subscript to a Name (line 172):
        
        # Obtaining the type of the subscript
        int_209925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'int')
        
        # Call to line_search_wolfe1(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'f' (line 172)
        f_209928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'f', False)
        # Getting the type of 'fprime' (line 172)
        fprime_209929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 62), 'fprime', False)
        # Getting the type of 'x' (line 172)
        x_209930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 70), 'x', False)
        # Getting the type of 'p' (line 172)
        p_209931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 73), 'p', False)
        # Getting the type of 'g0' (line 173)
        g0_209932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 59), 'g0', False)
        # Getting the type of 'f0' (line 173)
        f0_209933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 63), 'f0', False)
        # Getting the type of 'old_f' (line 173)
        old_f_209934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 67), 'old_f', False)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'smax' (line 174)
        smax_209935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 64), 'smax', False)
        keyword_209936 = smax_209935
        kwargs_209937 = {'amax': keyword_209936}
        # Getting the type of 'ls' (line 172)
        ls_209926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'ls', False)
        # Obtaining the member 'line_search_wolfe1' of a type (line 172)
        line_search_wolfe1_209927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 37), ls_209926, 'line_search_wolfe1')
        # Calling line_search_wolfe1(args, kwargs) (line 172)
        line_search_wolfe1_call_result_209938 = invoke(stypy.reporting.localization.Localization(__file__, 172, 37), line_search_wolfe1_209927, *[f_209928, fprime_209929, x_209930, p_209931, g0_209932, f0_209933, old_f_209934], **kwargs_209937)
        
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___209939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), line_search_wolfe1_call_result_209938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_209940 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), getitem___209939, int_209925)
        
        # Assigning a type to the variable 'tuple_var_assignment_209093' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209093', subscript_call_result_209940)
        
        # Assigning a Subscript to a Name (line 172):
        
        # Obtaining the type of the subscript
        int_209941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'int')
        
        # Call to line_search_wolfe1(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'f' (line 172)
        f_209944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'f', False)
        # Getting the type of 'fprime' (line 172)
        fprime_209945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 62), 'fprime', False)
        # Getting the type of 'x' (line 172)
        x_209946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 70), 'x', False)
        # Getting the type of 'p' (line 172)
        p_209947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 73), 'p', False)
        # Getting the type of 'g0' (line 173)
        g0_209948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 59), 'g0', False)
        # Getting the type of 'f0' (line 173)
        f0_209949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 63), 'f0', False)
        # Getting the type of 'old_f' (line 173)
        old_f_209950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 67), 'old_f', False)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'smax' (line 174)
        smax_209951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 64), 'smax', False)
        keyword_209952 = smax_209951
        kwargs_209953 = {'amax': keyword_209952}
        # Getting the type of 'ls' (line 172)
        ls_209942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'ls', False)
        # Obtaining the member 'line_search_wolfe1' of a type (line 172)
        line_search_wolfe1_209943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 37), ls_209942, 'line_search_wolfe1')
        # Calling line_search_wolfe1(args, kwargs) (line 172)
        line_search_wolfe1_call_result_209954 = invoke(stypy.reporting.localization.Localization(__file__, 172, 37), line_search_wolfe1_209943, *[f_209944, fprime_209945, x_209946, p_209947, g0_209948, f0_209949, old_f_209950], **kwargs_209953)
        
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___209955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), line_search_wolfe1_call_result_209954, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_209956 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), getitem___209955, int_209941)
        
        # Assigning a type to the variable 'tuple_var_assignment_209094' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209094', subscript_call_result_209956)
        
        # Assigning a Subscript to a Name (line 172):
        
        # Obtaining the type of the subscript
        int_209957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'int')
        
        # Call to line_search_wolfe1(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'f' (line 172)
        f_209960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'f', False)
        # Getting the type of 'fprime' (line 172)
        fprime_209961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 62), 'fprime', False)
        # Getting the type of 'x' (line 172)
        x_209962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 70), 'x', False)
        # Getting the type of 'p' (line 172)
        p_209963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 73), 'p', False)
        # Getting the type of 'g0' (line 173)
        g0_209964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 59), 'g0', False)
        # Getting the type of 'f0' (line 173)
        f0_209965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 63), 'f0', False)
        # Getting the type of 'old_f' (line 173)
        old_f_209966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 67), 'old_f', False)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'smax' (line 174)
        smax_209967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 64), 'smax', False)
        keyword_209968 = smax_209967
        kwargs_209969 = {'amax': keyword_209968}
        # Getting the type of 'ls' (line 172)
        ls_209958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'ls', False)
        # Obtaining the member 'line_search_wolfe1' of a type (line 172)
        line_search_wolfe1_209959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 37), ls_209958, 'line_search_wolfe1')
        # Calling line_search_wolfe1(args, kwargs) (line 172)
        line_search_wolfe1_call_result_209970 = invoke(stypy.reporting.localization.Localization(__file__, 172, 37), line_search_wolfe1_209959, *[f_209960, fprime_209961, x_209962, p_209963, g0_209964, f0_209965, old_f_209966], **kwargs_209969)
        
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___209971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), line_search_wolfe1_call_result_209970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_209972 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), getitem___209971, int_209957)
        
        # Assigning a type to the variable 'tuple_var_assignment_209095' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209095', subscript_call_result_209972)
        
        # Assigning a Subscript to a Name (line 172):
        
        # Obtaining the type of the subscript
        int_209973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'int')
        
        # Call to line_search_wolfe1(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'f' (line 172)
        f_209976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'f', False)
        # Getting the type of 'fprime' (line 172)
        fprime_209977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 62), 'fprime', False)
        # Getting the type of 'x' (line 172)
        x_209978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 70), 'x', False)
        # Getting the type of 'p' (line 172)
        p_209979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 73), 'p', False)
        # Getting the type of 'g0' (line 173)
        g0_209980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 59), 'g0', False)
        # Getting the type of 'f0' (line 173)
        f0_209981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 63), 'f0', False)
        # Getting the type of 'old_f' (line 173)
        old_f_209982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 67), 'old_f', False)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'smax' (line 174)
        smax_209983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 64), 'smax', False)
        keyword_209984 = smax_209983
        kwargs_209985 = {'amax': keyword_209984}
        # Getting the type of 'ls' (line 172)
        ls_209974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'ls', False)
        # Obtaining the member 'line_search_wolfe1' of a type (line 172)
        line_search_wolfe1_209975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 37), ls_209974, 'line_search_wolfe1')
        # Calling line_search_wolfe1(args, kwargs) (line 172)
        line_search_wolfe1_call_result_209986 = invoke(stypy.reporting.localization.Localization(__file__, 172, 37), line_search_wolfe1_209975, *[f_209976, fprime_209977, x_209978, p_209979, g0_209980, f0_209981, old_f_209982], **kwargs_209985)
        
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___209987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), line_search_wolfe1_call_result_209986, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_209988 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), getitem___209987, int_209973)
        
        # Assigning a type to the variable 'tuple_var_assignment_209096' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209096', subscript_call_result_209988)
        
        # Assigning a Subscript to a Name (line 172):
        
        # Obtaining the type of the subscript
        int_209989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'int')
        
        # Call to line_search_wolfe1(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'f' (line 172)
        f_209992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'f', False)
        # Getting the type of 'fprime' (line 172)
        fprime_209993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 62), 'fprime', False)
        # Getting the type of 'x' (line 172)
        x_209994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 70), 'x', False)
        # Getting the type of 'p' (line 172)
        p_209995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 73), 'p', False)
        # Getting the type of 'g0' (line 173)
        g0_209996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 59), 'g0', False)
        # Getting the type of 'f0' (line 173)
        f0_209997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 63), 'f0', False)
        # Getting the type of 'old_f' (line 173)
        old_f_209998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 67), 'old_f', False)
        # Processing the call keyword arguments (line 172)
        # Getting the type of 'smax' (line 174)
        smax_209999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 64), 'smax', False)
        keyword_210000 = smax_209999
        kwargs_210001 = {'amax': keyword_210000}
        # Getting the type of 'ls' (line 172)
        ls_209990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'ls', False)
        # Obtaining the member 'line_search_wolfe1' of a type (line 172)
        line_search_wolfe1_209991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 37), ls_209990, 'line_search_wolfe1')
        # Calling line_search_wolfe1(args, kwargs) (line 172)
        line_search_wolfe1_call_result_210002 = invoke(stypy.reporting.localization.Localization(__file__, 172, 37), line_search_wolfe1_209991, *[f_209992, fprime_209993, x_209994, p_209995, g0_209996, f0_209997, old_f_209998], **kwargs_210001)
        
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___210003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), line_search_wolfe1_call_result_210002, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_210004 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), getitem___210003, int_209989)
        
        # Assigning a type to the variable 'tuple_var_assignment_209097' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209097', subscript_call_result_210004)
        
        # Assigning a Name to a Name (line 172):
        # Getting the type of 'tuple_var_assignment_209092' (line 172)
        tuple_var_assignment_209092_210005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209092')
        # Assigning a type to the variable 's' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 's', tuple_var_assignment_209092_210005)
        
        # Assigning a Name to a Name (line 172):
        # Getting the type of 'tuple_var_assignment_209093' (line 172)
        tuple_var_assignment_209093_210006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209093')
        # Assigning a type to the variable 'fc' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'fc', tuple_var_assignment_209093_210006)
        
        # Assigning a Name to a Name (line 172):
        # Getting the type of 'tuple_var_assignment_209094' (line 172)
        tuple_var_assignment_209094_210007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209094')
        # Assigning a type to the variable 'gc' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'gc', tuple_var_assignment_209094_210007)
        
        # Assigning a Name to a Name (line 172):
        # Getting the type of 'tuple_var_assignment_209095' (line 172)
        tuple_var_assignment_209095_210008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209095')
        # Assigning a type to the variable 'fv' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'fv', tuple_var_assignment_209095_210008)
        
        # Assigning a Name to a Name (line 172):
        # Getting the type of 'tuple_var_assignment_209096' (line 172)
        tuple_var_assignment_209096_210009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209096')
        # Assigning a type to the variable 'ofv' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'ofv', tuple_var_assignment_209096_210009)
        
        # Assigning a Name to a Name (line 172):
        # Getting the type of 'tuple_var_assignment_209097' (line 172)
        tuple_var_assignment_209097_210010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'tuple_var_assignment_209097')
        # Assigning a type to the variable 'gv' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'gv', tuple_var_assignment_209097_210010)
        
        # Call to assert_equal(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'self' (line 175)
        self_210012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'self', False)
        # Obtaining the member 'fcount' of a type (line 175)
        fcount_210013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 25), self_210012, 'fcount')
        # Getting the type of 'fc' (line 175)
        fc_210014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 38), 'fc', False)
        # Getting the type of 'gc' (line 175)
        gc_210015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 41), 'gc', False)
        # Applying the binary operator '+' (line 175)
        result_add_210016 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 38), '+', fc_210014, gc_210015)
        
        # Processing the call keyword arguments (line 175)
        kwargs_210017 = {}
        # Getting the type of 'assert_equal' (line 175)
        assert_equal_210011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 175)
        assert_equal_call_result_210018 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), assert_equal_210011, *[fcount_210013, result_add_210016], **kwargs_210017)
        
        
        # Call to assert_fp_equal(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'ofv' (line 176)
        ofv_210020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'ofv', False)
        
        # Call to f(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'x' (line 176)
        x_210022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'x', False)
        # Processing the call keyword arguments (line 176)
        kwargs_210023 = {}
        # Getting the type of 'f' (line 176)
        f_210021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'f', False)
        # Calling f(args, kwargs) (line 176)
        f_call_result_210024 = invoke(stypy.reporting.localization.Localization(__file__, 176, 33), f_210021, *[x_210022], **kwargs_210023)
        
        # Processing the call keyword arguments (line 176)
        kwargs_210025 = {}
        # Getting the type of 'assert_fp_equal' (line 176)
        assert_fp_equal_210019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 176)
        assert_fp_equal_call_result_210026 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), assert_fp_equal_210019, *[ofv_210020, f_call_result_210024], **kwargs_210025)
        
        
        # Type idiom detected: calculating its left and rigth part (line 177)
        # Getting the type of 's' (line 177)
        s_210027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 's')
        # Getting the type of 'None' (line 177)
        None_210028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'None')
        
        (may_be_210029, more_types_in_union_210030) = may_be_none(s_210027, None_210028)

        if may_be_210029:

            if more_types_in_union_210030:
                # Runtime conditional SSA (line 177)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            if more_types_in_union_210030:
                # SSA join for if statement (line 177)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to assert_fp_equal(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'fv' (line 179)
        fv_210032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 'fv', False)
        
        # Call to f(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'x' (line 179)
        x_210034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'x', False)
        # Getting the type of 's' (line 179)
        s_210035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 38), 's', False)
        # Getting the type of 'p' (line 179)
        p_210036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 40), 'p', False)
        # Applying the binary operator '*' (line 179)
        result_mul_210037 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 38), '*', s_210035, p_210036)
        
        # Applying the binary operator '+' (line 179)
        result_add_210038 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 34), '+', x_210034, result_mul_210037)
        
        # Processing the call keyword arguments (line 179)
        kwargs_210039 = {}
        # Getting the type of 'f' (line 179)
        f_210033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'f', False)
        # Calling f(args, kwargs) (line 179)
        f_call_result_210040 = invoke(stypy.reporting.localization.Localization(__file__, 179, 32), f_210033, *[result_add_210038], **kwargs_210039)
        
        # Processing the call keyword arguments (line 179)
        kwargs_210041 = {}
        # Getting the type of 'assert_fp_equal' (line 179)
        assert_fp_equal_210031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 179)
        assert_fp_equal_call_result_210042 = invoke(stypy.reporting.localization.Localization(__file__, 179, 12), assert_fp_equal_210031, *[fv_210032, f_call_result_210040], **kwargs_210041)
        
        
        # Call to assert_array_almost_equal(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'gv' (line 180)
        gv_210044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 38), 'gv', False)
        
        # Call to fprime(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'x' (line 180)
        x_210046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 49), 'x', False)
        # Getting the type of 's' (line 180)
        s_210047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 53), 's', False)
        # Getting the type of 'p' (line 180)
        p_210048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 55), 'p', False)
        # Applying the binary operator '*' (line 180)
        result_mul_210049 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 53), '*', s_210047, p_210048)
        
        # Applying the binary operator '+' (line 180)
        result_add_210050 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 49), '+', x_210046, result_mul_210049)
        
        # Processing the call keyword arguments (line 180)
        kwargs_210051 = {}
        # Getting the type of 'fprime' (line 180)
        fprime_210045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 42), 'fprime', False)
        # Calling fprime(args, kwargs) (line 180)
        fprime_call_result_210052 = invoke(stypy.reporting.localization.Localization(__file__, 180, 42), fprime_210045, *[result_add_210050], **kwargs_210051)
        
        # Processing the call keyword arguments (line 180)
        int_210053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 67), 'int')
        keyword_210054 = int_210053
        kwargs_210055 = {'decimal': keyword_210054}
        # Getting the type of 'assert_array_almost_equal' (line 180)
        assert_array_almost_equal_210043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 180)
        assert_array_almost_equal_call_result_210056 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), assert_array_almost_equal_210043, *[gv_210044, fprime_call_result_210052], **kwargs_210055)
        
        
        
        # Getting the type of 's' (line 181)
        s_210057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 's')
        # Getting the type of 'smax' (line 181)
        smax_210058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'smax')
        # Applying the binary operator '<' (line 181)
        result_lt_210059 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 15), '<', s_210057, smax_210058)
        
        # Testing the type of an if condition (line 181)
        if_condition_210060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 12), result_lt_210059)
        # Assigning a type to the variable 'if_condition_210060' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'if_condition_210060', if_condition_210060)
        # SSA begins for if statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'c' (line 182)
        c_210061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'c')
        int_210062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 21), 'int')
        # Applying the binary operator '+=' (line 182)
        result_iadd_210063 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 16), '+=', c_210061, int_210062)
        # Assigning a type to the variable 'c' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'c', result_iadd_210063)
        
        
        # Call to assert_line_wolfe(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'x' (line 183)
        x_210065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 34), 'x', False)
        # Getting the type of 'p' (line 183)
        p_210066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 37), 'p', False)
        # Getting the type of 's' (line 183)
        s_210067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 40), 's', False)
        # Getting the type of 'f' (line 183)
        f_210068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 43), 'f', False)
        # Getting the type of 'fprime' (line 183)
        fprime_210069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 46), 'fprime', False)
        # Processing the call keyword arguments (line 183)
        # Getting the type of 'name' (line 183)
        name_210070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 62), 'name', False)
        keyword_210071 = name_210070
        kwargs_210072 = {'err_msg': keyword_210071}
        # Getting the type of 'assert_line_wolfe' (line 183)
        assert_line_wolfe_210064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'assert_line_wolfe', False)
        # Calling assert_line_wolfe(args, kwargs) (line 183)
        assert_line_wolfe_call_result_210073 = invoke(stypy.reporting.localization.Localization(__file__, 183, 16), assert_line_wolfe_210064, *[x_210065, p_210066, s_210067, f_210068, fprime_210069], **kwargs_210072)
        
        # SSA join for if statement (line 181)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Getting the type of 'c' (line 185)
        c_210075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'c', False)
        int_210076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'int')
        # Applying the binary operator '>' (line 185)
        result_gt_210077 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 16), '>', c_210075, int_210076)
        
        # Processing the call keyword arguments (line 185)
        kwargs_210078 = {}
        # Getting the type of 'assert_' (line 185)
        assert__210074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 185)
        assert__call_result_210079 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert__210074, *[result_gt_210077], **kwargs_210078)
        
        
        # ################# End of 'test_line_search_wolfe1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_line_search_wolfe1' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_210080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_210080)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_line_search_wolfe1'
        return stypy_return_type_210080


    @norecursion
    def test_line_search_wolfe2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_line_search_wolfe2'
        module_type_store = module_type_store.open_function_context('test_line_search_wolfe2', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_line_search_wolfe2')
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_line_search_wolfe2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_line_search_wolfe2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_line_search_wolfe2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_line_search_wolfe2(...)' code ##################

        
        # Assigning a Num to a Name (line 188):
        
        # Assigning a Num to a Name (line 188):
        int_210081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 12), 'int')
        # Assigning a type to the variable 'c' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'c', int_210081)
        
        # Assigning a Num to a Name (line 189):
        
        # Assigning a Num to a Name (line 189):
        int_210082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'int')
        # Assigning a type to the variable 'smax' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'smax', int_210082)
        
        
        # Call to line_iter(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_210085 = {}
        # Getting the type of 'self' (line 190)
        self_210083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 44), 'self', False)
        # Obtaining the member 'line_iter' of a type (line 190)
        line_iter_210084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 44), self_210083, 'line_iter')
        # Calling line_iter(args, kwargs) (line 190)
        line_iter_call_result_210086 = invoke(stypy.reporting.localization.Localization(__file__, 190, 44), line_iter_210084, *[], **kwargs_210085)
        
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), line_iter_call_result_210086)
        # Getting the type of the for loop variable (line 190)
        for_loop_var_210087 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), line_iter_call_result_210086)
        # Assigning a type to the variable 'name' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_210087))
        # Assigning a type to the variable 'f' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_210087))
        # Assigning a type to the variable 'fprime' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'fprime', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_210087))
        # Assigning a type to the variable 'x' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_210087))
        # Assigning a type to the variable 'p' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_210087))
        # Assigning a type to the variable 'old_f' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'old_f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_210087))
        # SSA begins for a for statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to f(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'x' (line 191)
        x_210089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'x', False)
        # Processing the call keyword arguments (line 191)
        kwargs_210090 = {}
        # Getting the type of 'f' (line 191)
        f_210088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 17), 'f', False)
        # Calling f(args, kwargs) (line 191)
        f_call_result_210091 = invoke(stypy.reporting.localization.Localization(__file__, 191, 17), f_210088, *[x_210089], **kwargs_210090)
        
        # Assigning a type to the variable 'f0' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'f0', f_call_result_210091)
        
        # Assigning a Call to a Name (line 192):
        
        # Assigning a Call to a Name (line 192):
        
        # Call to fprime(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'x' (line 192)
        x_210093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'x', False)
        # Processing the call keyword arguments (line 192)
        kwargs_210094 = {}
        # Getting the type of 'fprime' (line 192)
        fprime_210092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'fprime', False)
        # Calling fprime(args, kwargs) (line 192)
        fprime_call_result_210095 = invoke(stypy.reporting.localization.Localization(__file__, 192, 17), fprime_210092, *[x_210093], **kwargs_210094)
        
        # Assigning a type to the variable 'g0' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'g0', fprime_call_result_210095)
        
        # Assigning a Num to a Attribute (line 193):
        
        # Assigning a Num to a Attribute (line 193):
        int_210096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 26), 'int')
        # Getting the type of 'self' (line 193)
        self_210097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'self')
        # Setting the type of the member 'fcount' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), self_210097, 'fcount', int_210096)
        
        # Call to suppress_warnings(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_210099 = {}
        # Getting the type of 'suppress_warnings' (line 194)
        suppress_warnings_210098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 194)
        suppress_warnings_call_result_210100 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), suppress_warnings_210098, *[], **kwargs_210099)
        
        with_210101 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 194, 17), suppress_warnings_call_result_210100, 'with parameter', '__enter__', '__exit__')

        if with_210101:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 194)
            enter___210102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), suppress_warnings_call_result_210100, '__enter__')
            with_enter_210103 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), enter___210102)
            # Assigning a type to the variable 'sup' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'sup', with_enter_210103)
            
            # Call to filter(...): (line 195)
            # Processing the call arguments (line 195)
            # Getting the type of 'LineSearchWarning' (line 195)
            LineSearchWarning_210106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'LineSearchWarning', False)
            str_210107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 27), 'str', 'The line search algorithm could not find a solution')
            # Processing the call keyword arguments (line 195)
            kwargs_210108 = {}
            # Getting the type of 'sup' (line 195)
            sup_210104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 195)
            filter_210105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 16), sup_210104, 'filter')
            # Calling filter(args, kwargs) (line 195)
            filter_call_result_210109 = invoke(stypy.reporting.localization.Localization(__file__, 195, 16), filter_210105, *[LineSearchWarning_210106, str_210107], **kwargs_210108)
            
            
            # Call to filter(...): (line 197)
            # Processing the call arguments (line 197)
            # Getting the type of 'LineSearchWarning' (line 197)
            LineSearchWarning_210112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'LineSearchWarning', False)
            str_210113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 27), 'str', 'The line search algorithm did not converge')
            # Processing the call keyword arguments (line 197)
            kwargs_210114 = {}
            # Getting the type of 'sup' (line 197)
            sup_210110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 197)
            filter_210111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 16), sup_210110, 'filter')
            # Calling filter(args, kwargs) (line 197)
            filter_call_result_210115 = invoke(stypy.reporting.localization.Localization(__file__, 197, 16), filter_210111, *[LineSearchWarning_210112, str_210113], **kwargs_210114)
            
            
            # Assigning a Call to a Tuple (line 199):
            
            # Assigning a Subscript to a Name (line 199):
            
            # Obtaining the type of the subscript
            int_210116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
            
            # Call to line_search_wolfe2(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'f' (line 199)
            f_210119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 63), 'f', False)
            # Getting the type of 'fprime' (line 199)
            fprime_210120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 66), 'fprime', False)
            # Getting the type of 'x' (line 199)
            x_210121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 74), 'x', False)
            # Getting the type of 'p' (line 199)
            p_210122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 77), 'p', False)
            # Getting the type of 'g0' (line 200)
            g0_210123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 63), 'g0', False)
            # Getting the type of 'f0' (line 200)
            f0_210124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 67), 'f0', False)
            # Getting the type of 'old_f' (line 200)
            old_f_210125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 71), 'old_f', False)
            # Processing the call keyword arguments (line 199)
            # Getting the type of 'smax' (line 201)
            smax_210126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'smax', False)
            keyword_210127 = smax_210126
            kwargs_210128 = {'amax': keyword_210127}
            # Getting the type of 'ls' (line 199)
            ls_210117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'ls', False)
            # Obtaining the member 'line_search_wolfe2' of a type (line 199)
            line_search_wolfe2_210118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 41), ls_210117, 'line_search_wolfe2')
            # Calling line_search_wolfe2(args, kwargs) (line 199)
            line_search_wolfe2_call_result_210129 = invoke(stypy.reporting.localization.Localization(__file__, 199, 41), line_search_wolfe2_210118, *[f_210119, fprime_210120, x_210121, p_210122, g0_210123, f0_210124, old_f_210125], **kwargs_210128)
            
            # Obtaining the member '__getitem__' of a type (line 199)
            getitem___210130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), line_search_wolfe2_call_result_210129, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 199)
            subscript_call_result_210131 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), getitem___210130, int_210116)
            
            # Assigning a type to the variable 'tuple_var_assignment_209098' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209098', subscript_call_result_210131)
            
            # Assigning a Subscript to a Name (line 199):
            
            # Obtaining the type of the subscript
            int_210132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
            
            # Call to line_search_wolfe2(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'f' (line 199)
            f_210135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 63), 'f', False)
            # Getting the type of 'fprime' (line 199)
            fprime_210136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 66), 'fprime', False)
            # Getting the type of 'x' (line 199)
            x_210137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 74), 'x', False)
            # Getting the type of 'p' (line 199)
            p_210138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 77), 'p', False)
            # Getting the type of 'g0' (line 200)
            g0_210139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 63), 'g0', False)
            # Getting the type of 'f0' (line 200)
            f0_210140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 67), 'f0', False)
            # Getting the type of 'old_f' (line 200)
            old_f_210141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 71), 'old_f', False)
            # Processing the call keyword arguments (line 199)
            # Getting the type of 'smax' (line 201)
            smax_210142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'smax', False)
            keyword_210143 = smax_210142
            kwargs_210144 = {'amax': keyword_210143}
            # Getting the type of 'ls' (line 199)
            ls_210133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'ls', False)
            # Obtaining the member 'line_search_wolfe2' of a type (line 199)
            line_search_wolfe2_210134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 41), ls_210133, 'line_search_wolfe2')
            # Calling line_search_wolfe2(args, kwargs) (line 199)
            line_search_wolfe2_call_result_210145 = invoke(stypy.reporting.localization.Localization(__file__, 199, 41), line_search_wolfe2_210134, *[f_210135, fprime_210136, x_210137, p_210138, g0_210139, f0_210140, old_f_210141], **kwargs_210144)
            
            # Obtaining the member '__getitem__' of a type (line 199)
            getitem___210146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), line_search_wolfe2_call_result_210145, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 199)
            subscript_call_result_210147 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), getitem___210146, int_210132)
            
            # Assigning a type to the variable 'tuple_var_assignment_209099' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209099', subscript_call_result_210147)
            
            # Assigning a Subscript to a Name (line 199):
            
            # Obtaining the type of the subscript
            int_210148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
            
            # Call to line_search_wolfe2(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'f' (line 199)
            f_210151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 63), 'f', False)
            # Getting the type of 'fprime' (line 199)
            fprime_210152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 66), 'fprime', False)
            # Getting the type of 'x' (line 199)
            x_210153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 74), 'x', False)
            # Getting the type of 'p' (line 199)
            p_210154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 77), 'p', False)
            # Getting the type of 'g0' (line 200)
            g0_210155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 63), 'g0', False)
            # Getting the type of 'f0' (line 200)
            f0_210156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 67), 'f0', False)
            # Getting the type of 'old_f' (line 200)
            old_f_210157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 71), 'old_f', False)
            # Processing the call keyword arguments (line 199)
            # Getting the type of 'smax' (line 201)
            smax_210158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'smax', False)
            keyword_210159 = smax_210158
            kwargs_210160 = {'amax': keyword_210159}
            # Getting the type of 'ls' (line 199)
            ls_210149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'ls', False)
            # Obtaining the member 'line_search_wolfe2' of a type (line 199)
            line_search_wolfe2_210150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 41), ls_210149, 'line_search_wolfe2')
            # Calling line_search_wolfe2(args, kwargs) (line 199)
            line_search_wolfe2_call_result_210161 = invoke(stypy.reporting.localization.Localization(__file__, 199, 41), line_search_wolfe2_210150, *[f_210151, fprime_210152, x_210153, p_210154, g0_210155, f0_210156, old_f_210157], **kwargs_210160)
            
            # Obtaining the member '__getitem__' of a type (line 199)
            getitem___210162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), line_search_wolfe2_call_result_210161, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 199)
            subscript_call_result_210163 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), getitem___210162, int_210148)
            
            # Assigning a type to the variable 'tuple_var_assignment_209100' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209100', subscript_call_result_210163)
            
            # Assigning a Subscript to a Name (line 199):
            
            # Obtaining the type of the subscript
            int_210164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
            
            # Call to line_search_wolfe2(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'f' (line 199)
            f_210167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 63), 'f', False)
            # Getting the type of 'fprime' (line 199)
            fprime_210168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 66), 'fprime', False)
            # Getting the type of 'x' (line 199)
            x_210169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 74), 'x', False)
            # Getting the type of 'p' (line 199)
            p_210170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 77), 'p', False)
            # Getting the type of 'g0' (line 200)
            g0_210171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 63), 'g0', False)
            # Getting the type of 'f0' (line 200)
            f0_210172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 67), 'f0', False)
            # Getting the type of 'old_f' (line 200)
            old_f_210173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 71), 'old_f', False)
            # Processing the call keyword arguments (line 199)
            # Getting the type of 'smax' (line 201)
            smax_210174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'smax', False)
            keyword_210175 = smax_210174
            kwargs_210176 = {'amax': keyword_210175}
            # Getting the type of 'ls' (line 199)
            ls_210165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'ls', False)
            # Obtaining the member 'line_search_wolfe2' of a type (line 199)
            line_search_wolfe2_210166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 41), ls_210165, 'line_search_wolfe2')
            # Calling line_search_wolfe2(args, kwargs) (line 199)
            line_search_wolfe2_call_result_210177 = invoke(stypy.reporting.localization.Localization(__file__, 199, 41), line_search_wolfe2_210166, *[f_210167, fprime_210168, x_210169, p_210170, g0_210171, f0_210172, old_f_210173], **kwargs_210176)
            
            # Obtaining the member '__getitem__' of a type (line 199)
            getitem___210178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), line_search_wolfe2_call_result_210177, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 199)
            subscript_call_result_210179 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), getitem___210178, int_210164)
            
            # Assigning a type to the variable 'tuple_var_assignment_209101' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209101', subscript_call_result_210179)
            
            # Assigning a Subscript to a Name (line 199):
            
            # Obtaining the type of the subscript
            int_210180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
            
            # Call to line_search_wolfe2(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'f' (line 199)
            f_210183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 63), 'f', False)
            # Getting the type of 'fprime' (line 199)
            fprime_210184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 66), 'fprime', False)
            # Getting the type of 'x' (line 199)
            x_210185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 74), 'x', False)
            # Getting the type of 'p' (line 199)
            p_210186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 77), 'p', False)
            # Getting the type of 'g0' (line 200)
            g0_210187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 63), 'g0', False)
            # Getting the type of 'f0' (line 200)
            f0_210188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 67), 'f0', False)
            # Getting the type of 'old_f' (line 200)
            old_f_210189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 71), 'old_f', False)
            # Processing the call keyword arguments (line 199)
            # Getting the type of 'smax' (line 201)
            smax_210190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'smax', False)
            keyword_210191 = smax_210190
            kwargs_210192 = {'amax': keyword_210191}
            # Getting the type of 'ls' (line 199)
            ls_210181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'ls', False)
            # Obtaining the member 'line_search_wolfe2' of a type (line 199)
            line_search_wolfe2_210182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 41), ls_210181, 'line_search_wolfe2')
            # Calling line_search_wolfe2(args, kwargs) (line 199)
            line_search_wolfe2_call_result_210193 = invoke(stypy.reporting.localization.Localization(__file__, 199, 41), line_search_wolfe2_210182, *[f_210183, fprime_210184, x_210185, p_210186, g0_210187, f0_210188, old_f_210189], **kwargs_210192)
            
            # Obtaining the member '__getitem__' of a type (line 199)
            getitem___210194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), line_search_wolfe2_call_result_210193, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 199)
            subscript_call_result_210195 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), getitem___210194, int_210180)
            
            # Assigning a type to the variable 'tuple_var_assignment_209102' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209102', subscript_call_result_210195)
            
            # Assigning a Subscript to a Name (line 199):
            
            # Obtaining the type of the subscript
            int_210196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
            
            # Call to line_search_wolfe2(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'f' (line 199)
            f_210199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 63), 'f', False)
            # Getting the type of 'fprime' (line 199)
            fprime_210200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 66), 'fprime', False)
            # Getting the type of 'x' (line 199)
            x_210201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 74), 'x', False)
            # Getting the type of 'p' (line 199)
            p_210202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 77), 'p', False)
            # Getting the type of 'g0' (line 200)
            g0_210203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 63), 'g0', False)
            # Getting the type of 'f0' (line 200)
            f0_210204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 67), 'f0', False)
            # Getting the type of 'old_f' (line 200)
            old_f_210205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 71), 'old_f', False)
            # Processing the call keyword arguments (line 199)
            # Getting the type of 'smax' (line 201)
            smax_210206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 68), 'smax', False)
            keyword_210207 = smax_210206
            kwargs_210208 = {'amax': keyword_210207}
            # Getting the type of 'ls' (line 199)
            ls_210197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'ls', False)
            # Obtaining the member 'line_search_wolfe2' of a type (line 199)
            line_search_wolfe2_210198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 41), ls_210197, 'line_search_wolfe2')
            # Calling line_search_wolfe2(args, kwargs) (line 199)
            line_search_wolfe2_call_result_210209 = invoke(stypy.reporting.localization.Localization(__file__, 199, 41), line_search_wolfe2_210198, *[f_210199, fprime_210200, x_210201, p_210202, g0_210203, f0_210204, old_f_210205], **kwargs_210208)
            
            # Obtaining the member '__getitem__' of a type (line 199)
            getitem___210210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), line_search_wolfe2_call_result_210209, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 199)
            subscript_call_result_210211 = invoke(stypy.reporting.localization.Localization(__file__, 199, 16), getitem___210210, int_210196)
            
            # Assigning a type to the variable 'tuple_var_assignment_209103' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209103', subscript_call_result_210211)
            
            # Assigning a Name to a Name (line 199):
            # Getting the type of 'tuple_var_assignment_209098' (line 199)
            tuple_var_assignment_209098_210212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209098')
            # Assigning a type to the variable 's' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 's', tuple_var_assignment_209098_210212)
            
            # Assigning a Name to a Name (line 199):
            # Getting the type of 'tuple_var_assignment_209099' (line 199)
            tuple_var_assignment_209099_210213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209099')
            # Assigning a type to the variable 'fc' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'fc', tuple_var_assignment_209099_210213)
            
            # Assigning a Name to a Name (line 199):
            # Getting the type of 'tuple_var_assignment_209100' (line 199)
            tuple_var_assignment_209100_210214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209100')
            # Assigning a type to the variable 'gc' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'gc', tuple_var_assignment_209100_210214)
            
            # Assigning a Name to a Name (line 199):
            # Getting the type of 'tuple_var_assignment_209101' (line 199)
            tuple_var_assignment_209101_210215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209101')
            # Assigning a type to the variable 'fv' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'fv', tuple_var_assignment_209101_210215)
            
            # Assigning a Name to a Name (line 199):
            # Getting the type of 'tuple_var_assignment_209102' (line 199)
            tuple_var_assignment_209102_210216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209102')
            # Assigning a type to the variable 'ofv' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'ofv', tuple_var_assignment_209102_210216)
            
            # Assigning a Name to a Name (line 199):
            # Getting the type of 'tuple_var_assignment_209103' (line 199)
            tuple_var_assignment_209103_210217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'tuple_var_assignment_209103')
            # Assigning a type to the variable 'gv' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 36), 'gv', tuple_var_assignment_209103_210217)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 194)
            exit___210218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), suppress_warnings_call_result_210100, '__exit__')
            with_exit_210219 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), exit___210218, None, None, None)

        
        # Call to assert_equal(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'self' (line 202)
        self_210221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'self', False)
        # Obtaining the member 'fcount' of a type (line 202)
        fcount_210222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 25), self_210221, 'fcount')
        # Getting the type of 'fc' (line 202)
        fc_210223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'fc', False)
        # Getting the type of 'gc' (line 202)
        gc_210224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 41), 'gc', False)
        # Applying the binary operator '+' (line 202)
        result_add_210225 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 38), '+', fc_210223, gc_210224)
        
        # Processing the call keyword arguments (line 202)
        kwargs_210226 = {}
        # Getting the type of 'assert_equal' (line 202)
        assert_equal_210220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 202)
        assert_equal_call_result_210227 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), assert_equal_210220, *[fcount_210222, result_add_210225], **kwargs_210226)
        
        
        # Call to assert_fp_equal(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'ofv' (line 203)
        ofv_210229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 28), 'ofv', False)
        
        # Call to f(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'x' (line 203)
        x_210231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 35), 'x', False)
        # Processing the call keyword arguments (line 203)
        kwargs_210232 = {}
        # Getting the type of 'f' (line 203)
        f_210230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 33), 'f', False)
        # Calling f(args, kwargs) (line 203)
        f_call_result_210233 = invoke(stypy.reporting.localization.Localization(__file__, 203, 33), f_210230, *[x_210231], **kwargs_210232)
        
        # Processing the call keyword arguments (line 203)
        kwargs_210234 = {}
        # Getting the type of 'assert_fp_equal' (line 203)
        assert_fp_equal_210228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 203)
        assert_fp_equal_call_result_210235 = invoke(stypy.reporting.localization.Localization(__file__, 203, 12), assert_fp_equal_210228, *[ofv_210229, f_call_result_210233], **kwargs_210234)
        
        
        # Call to assert_fp_equal(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'fv' (line 204)
        fv_210237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 28), 'fv', False)
        
        # Call to f(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'x' (line 204)
        x_210239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 34), 'x', False)
        # Getting the type of 's' (line 204)
        s_210240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 38), 's', False)
        # Getting the type of 'p' (line 204)
        p_210241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 40), 'p', False)
        # Applying the binary operator '*' (line 204)
        result_mul_210242 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 38), '*', s_210240, p_210241)
        
        # Applying the binary operator '+' (line 204)
        result_add_210243 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 34), '+', x_210239, result_mul_210242)
        
        # Processing the call keyword arguments (line 204)
        kwargs_210244 = {}
        # Getting the type of 'f' (line 204)
        f_210238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 32), 'f', False)
        # Calling f(args, kwargs) (line 204)
        f_call_result_210245 = invoke(stypy.reporting.localization.Localization(__file__, 204, 32), f_210238, *[result_add_210243], **kwargs_210244)
        
        # Processing the call keyword arguments (line 204)
        kwargs_210246 = {}
        # Getting the type of 'assert_fp_equal' (line 204)
        assert_fp_equal_210236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 204)
        assert_fp_equal_call_result_210247 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), assert_fp_equal_210236, *[fv_210237, f_call_result_210245], **kwargs_210246)
        
        
        # Type idiom detected: calculating its left and rigth part (line 205)
        # Getting the type of 'gv' (line 205)
        gv_210248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'gv')
        # Getting the type of 'None' (line 205)
        None_210249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 25), 'None')
        
        (may_be_210250, more_types_in_union_210251) = may_not_be_none(gv_210248, None_210249)

        if may_be_210250:

            if more_types_in_union_210251:
                # Runtime conditional SSA (line 205)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to assert_array_almost_equal(...): (line 206)
            # Processing the call arguments (line 206)
            # Getting the type of 'gv' (line 206)
            gv_210253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 42), 'gv', False)
            
            # Call to fprime(...): (line 206)
            # Processing the call arguments (line 206)
            # Getting the type of 'x' (line 206)
            x_210255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 53), 'x', False)
            # Getting the type of 's' (line 206)
            s_210256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 57), 's', False)
            # Getting the type of 'p' (line 206)
            p_210257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 59), 'p', False)
            # Applying the binary operator '*' (line 206)
            result_mul_210258 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 57), '*', s_210256, p_210257)
            
            # Applying the binary operator '+' (line 206)
            result_add_210259 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 53), '+', x_210255, result_mul_210258)
            
            # Processing the call keyword arguments (line 206)
            kwargs_210260 = {}
            # Getting the type of 'fprime' (line 206)
            fprime_210254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 46), 'fprime', False)
            # Calling fprime(args, kwargs) (line 206)
            fprime_call_result_210261 = invoke(stypy.reporting.localization.Localization(__file__, 206, 46), fprime_210254, *[result_add_210259], **kwargs_210260)
            
            # Processing the call keyword arguments (line 206)
            int_210262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 71), 'int')
            keyword_210263 = int_210262
            kwargs_210264 = {'decimal': keyword_210263}
            # Getting the type of 'assert_array_almost_equal' (line 206)
            assert_array_almost_equal_210252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'assert_array_almost_equal', False)
            # Calling assert_array_almost_equal(args, kwargs) (line 206)
            assert_array_almost_equal_call_result_210265 = invoke(stypy.reporting.localization.Localization(__file__, 206, 16), assert_array_almost_equal_210252, *[gv_210253, fprime_call_result_210261], **kwargs_210264)
            

            if more_types_in_union_210251:
                # SSA join for if statement (line 205)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 's' (line 207)
        s_210266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 's')
        # Getting the type of 'smax' (line 207)
        smax_210267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'smax')
        # Applying the binary operator '<' (line 207)
        result_lt_210268 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 15), '<', s_210266, smax_210267)
        
        # Testing the type of an if condition (line 207)
        if_condition_210269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 12), result_lt_210268)
        # Assigning a type to the variable 'if_condition_210269' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'if_condition_210269', if_condition_210269)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'c' (line 208)
        c_210270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'c')
        int_210271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'int')
        # Applying the binary operator '+=' (line 208)
        result_iadd_210272 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 16), '+=', c_210270, int_210271)
        # Assigning a type to the variable 'c' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'c', result_iadd_210272)
        
        
        # Call to assert_line_wolfe(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'x' (line 209)
        x_210274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 34), 'x', False)
        # Getting the type of 'p' (line 209)
        p_210275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 37), 'p', False)
        # Getting the type of 's' (line 209)
        s_210276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 's', False)
        # Getting the type of 'f' (line 209)
        f_210277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 43), 'f', False)
        # Getting the type of 'fprime' (line 209)
        fprime_210278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 46), 'fprime', False)
        # Processing the call keyword arguments (line 209)
        # Getting the type of 'name' (line 209)
        name_210279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 62), 'name', False)
        keyword_210280 = name_210279
        kwargs_210281 = {'err_msg': keyword_210280}
        # Getting the type of 'assert_line_wolfe' (line 209)
        assert_line_wolfe_210273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 'assert_line_wolfe', False)
        # Calling assert_line_wolfe(args, kwargs) (line 209)
        assert_line_wolfe_call_result_210282 = invoke(stypy.reporting.localization.Localization(__file__, 209, 16), assert_line_wolfe_210273, *[x_210274, p_210275, s_210276, f_210277, fprime_210278], **kwargs_210281)
        
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Getting the type of 'c' (line 210)
        c_210284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'c', False)
        int_210285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 20), 'int')
        # Applying the binary operator '>' (line 210)
        result_gt_210286 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 16), '>', c_210284, int_210285)
        
        # Processing the call keyword arguments (line 210)
        kwargs_210287 = {}
        # Getting the type of 'assert_' (line 210)
        assert__210283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 210)
        assert__call_result_210288 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), assert__210283, *[result_gt_210286], **kwargs_210287)
        
        
        # ################# End of 'test_line_search_wolfe2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_line_search_wolfe2' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_210289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_210289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_line_search_wolfe2'
        return stypy_return_type_210289


    @norecursion
    def test_line_search_wolfe2_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_line_search_wolfe2_bounds'
        module_type_store = module_type_store.open_function_context('test_line_search_wolfe2_bounds', 212, 4, False)
        # Assigning a type to the variable 'self' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_line_search_wolfe2_bounds')
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_line_search_wolfe2_bounds.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_line_search_wolfe2_bounds', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_line_search_wolfe2_bounds', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_line_search_wolfe2_bounds(...)' code ##################

        
        # Assigning a Lambda to a Name (line 218):
        
        # Assigning a Lambda to a Name (line 218):

        @norecursion
        def _stypy_temp_lambda_69(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_69'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_69', 218, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_69.stypy_localization = localization
            _stypy_temp_lambda_69.stypy_type_of_self = None
            _stypy_temp_lambda_69.stypy_type_store = module_type_store
            _stypy_temp_lambda_69.stypy_function_name = '_stypy_temp_lambda_69'
            _stypy_temp_lambda_69.stypy_param_names_list = ['x']
            _stypy_temp_lambda_69.stypy_varargs_param_name = None
            _stypy_temp_lambda_69.stypy_kwargs_param_name = None
            _stypy_temp_lambda_69.stypy_call_defaults = defaults
            _stypy_temp_lambda_69.stypy_call_varargs = varargs
            _stypy_temp_lambda_69.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_69', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_69', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to dot(...): (line 218)
            # Processing the call arguments (line 218)
            # Getting the type of 'x' (line 218)
            x_210292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 29), 'x', False)
            # Getting the type of 'x' (line 218)
            x_210293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 32), 'x', False)
            # Processing the call keyword arguments (line 218)
            kwargs_210294 = {}
            # Getting the type of 'np' (line 218)
            np_210290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'np', False)
            # Obtaining the member 'dot' of a type (line 218)
            dot_210291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 22), np_210290, 'dot')
            # Calling dot(args, kwargs) (line 218)
            dot_call_result_210295 = invoke(stypy.reporting.localization.Localization(__file__, 218, 22), dot_210291, *[x_210292, x_210293], **kwargs_210294)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'stypy_return_type', dot_call_result_210295)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_69' in the type store
            # Getting the type of 'stypy_return_type' (line 218)
            stypy_return_type_210296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_210296)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_69'
            return stypy_return_type_210296

        # Assigning a type to the variable '_stypy_temp_lambda_69' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), '_stypy_temp_lambda_69', _stypy_temp_lambda_69)
        # Getting the type of '_stypy_temp_lambda_69' (line 218)
        _stypy_temp_lambda_69_210297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), '_stypy_temp_lambda_69')
        # Assigning a type to the variable 'f' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'f', _stypy_temp_lambda_69_210297)
        
        # Assigning a Lambda to a Name (line 219):
        
        # Assigning a Lambda to a Name (line 219):

        @norecursion
        def _stypy_temp_lambda_70(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_70'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_70', 219, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_70.stypy_localization = localization
            _stypy_temp_lambda_70.stypy_type_of_self = None
            _stypy_temp_lambda_70.stypy_type_store = module_type_store
            _stypy_temp_lambda_70.stypy_function_name = '_stypy_temp_lambda_70'
            _stypy_temp_lambda_70.stypy_param_names_list = ['x']
            _stypy_temp_lambda_70.stypy_varargs_param_name = None
            _stypy_temp_lambda_70.stypy_kwargs_param_name = None
            _stypy_temp_lambda_70.stypy_call_defaults = defaults
            _stypy_temp_lambda_70.stypy_call_varargs = varargs
            _stypy_temp_lambda_70.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_70', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_70', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_210298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 23), 'int')
            # Getting the type of 'x' (line 219)
            x_210299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 27), 'x')
            # Applying the binary operator '*' (line 219)
            result_mul_210300 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 23), '*', int_210298, x_210299)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 219)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 13), 'stypy_return_type', result_mul_210300)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_70' in the type store
            # Getting the type of 'stypy_return_type' (line 219)
            stypy_return_type_210301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_210301)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_70'
            return stypy_return_type_210301

        # Assigning a type to the variable '_stypy_temp_lambda_70' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 13), '_stypy_temp_lambda_70', _stypy_temp_lambda_70)
        # Getting the type of '_stypy_temp_lambda_70' (line 219)
        _stypy_temp_lambda_70_210302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 13), '_stypy_temp_lambda_70')
        # Assigning a type to the variable 'fp' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'fp', _stypy_temp_lambda_70_210302)
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to array(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Obtaining an instance of the builtin type 'list' (line 220)
        list_210305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 220)
        # Adding element type (line 220)
        int_210306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 21), list_210305, int_210306)
        # Adding element type (line 220)
        int_210307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 21), list_210305, int_210307)
        
        # Processing the call keyword arguments (line 220)
        kwargs_210308 = {}
        # Getting the type of 'np' (line 220)
        np_210303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 220)
        array_210304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), np_210303, 'array')
        # Calling array(args, kwargs) (line 220)
        array_call_result_210309 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), array_210304, *[list_210305], **kwargs_210308)
        
        # Assigning a type to the variable 'p' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'p', array_call_result_210309)
        
        # Assigning a BinOp to a Name (line 223):
        
        # Assigning a BinOp to a Name (line 223):
        int_210310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 12), 'int')
        # Getting the type of 'p' (line 223)
        p_210311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 18), 'p')
        # Applying the binary operator '*' (line 223)
        result_mul_210312 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 12), '*', int_210310, p_210311)
        
        # Assigning a type to the variable 'x' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'x', result_mul_210312)
        
        # Assigning a Num to a Name (line 224):
        
        # Assigning a Num to a Name (line 224):
        float_210313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 13), 'float')
        # Assigning a type to the variable 'c2' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'c2', float_210313)
        
        # Assigning a Call to a Tuple (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_210314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to line_search_wolfe2(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'f' (line 226)
        f_210317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'f', False)
        # Getting the type of 'fp' (line 226)
        fp_210318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 52), 'fp', False)
        # Getting the type of 'x' (line 226)
        x_210319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 56), 'x', False)
        # Getting the type of 'p' (line 226)
        p_210320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 59), 'p', False)
        # Processing the call keyword arguments (line 226)
        int_210321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 67), 'int')
        keyword_210322 = int_210321
        # Getting the type of 'c2' (line 226)
        c2_210323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 74), 'c2', False)
        keyword_210324 = c2_210323
        kwargs_210325 = {'c2': keyword_210324, 'amax': keyword_210322}
        # Getting the type of 'ls' (line 226)
        ls_210315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 226)
        line_search_wolfe2_210316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), ls_210315, 'line_search_wolfe2')
        # Calling line_search_wolfe2(args, kwargs) (line 226)
        line_search_wolfe2_call_result_210326 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), line_search_wolfe2_210316, *[f_210317, fp_210318, x_210319, p_210320], **kwargs_210325)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___210327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), line_search_wolfe2_call_result_210326, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_210328 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___210327, int_210314)
        
        # Assigning a type to the variable 'tuple_var_assignment_209104' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209104', subscript_call_result_210328)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_210329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to line_search_wolfe2(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'f' (line 226)
        f_210332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'f', False)
        # Getting the type of 'fp' (line 226)
        fp_210333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 52), 'fp', False)
        # Getting the type of 'x' (line 226)
        x_210334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 56), 'x', False)
        # Getting the type of 'p' (line 226)
        p_210335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 59), 'p', False)
        # Processing the call keyword arguments (line 226)
        int_210336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 67), 'int')
        keyword_210337 = int_210336
        # Getting the type of 'c2' (line 226)
        c2_210338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 74), 'c2', False)
        keyword_210339 = c2_210338
        kwargs_210340 = {'c2': keyword_210339, 'amax': keyword_210337}
        # Getting the type of 'ls' (line 226)
        ls_210330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 226)
        line_search_wolfe2_210331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), ls_210330, 'line_search_wolfe2')
        # Calling line_search_wolfe2(args, kwargs) (line 226)
        line_search_wolfe2_call_result_210341 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), line_search_wolfe2_210331, *[f_210332, fp_210333, x_210334, p_210335], **kwargs_210340)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___210342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), line_search_wolfe2_call_result_210341, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_210343 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___210342, int_210329)
        
        # Assigning a type to the variable 'tuple_var_assignment_209105' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209105', subscript_call_result_210343)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_210344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to line_search_wolfe2(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'f' (line 226)
        f_210347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'f', False)
        # Getting the type of 'fp' (line 226)
        fp_210348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 52), 'fp', False)
        # Getting the type of 'x' (line 226)
        x_210349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 56), 'x', False)
        # Getting the type of 'p' (line 226)
        p_210350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 59), 'p', False)
        # Processing the call keyword arguments (line 226)
        int_210351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 67), 'int')
        keyword_210352 = int_210351
        # Getting the type of 'c2' (line 226)
        c2_210353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 74), 'c2', False)
        keyword_210354 = c2_210353
        kwargs_210355 = {'c2': keyword_210354, 'amax': keyword_210352}
        # Getting the type of 'ls' (line 226)
        ls_210345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 226)
        line_search_wolfe2_210346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), ls_210345, 'line_search_wolfe2')
        # Calling line_search_wolfe2(args, kwargs) (line 226)
        line_search_wolfe2_call_result_210356 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), line_search_wolfe2_210346, *[f_210347, fp_210348, x_210349, p_210350], **kwargs_210355)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___210357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), line_search_wolfe2_call_result_210356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_210358 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___210357, int_210344)
        
        # Assigning a type to the variable 'tuple_var_assignment_209106' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209106', subscript_call_result_210358)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_210359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to line_search_wolfe2(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'f' (line 226)
        f_210362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'f', False)
        # Getting the type of 'fp' (line 226)
        fp_210363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 52), 'fp', False)
        # Getting the type of 'x' (line 226)
        x_210364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 56), 'x', False)
        # Getting the type of 'p' (line 226)
        p_210365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 59), 'p', False)
        # Processing the call keyword arguments (line 226)
        int_210366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 67), 'int')
        keyword_210367 = int_210366
        # Getting the type of 'c2' (line 226)
        c2_210368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 74), 'c2', False)
        keyword_210369 = c2_210368
        kwargs_210370 = {'c2': keyword_210369, 'amax': keyword_210367}
        # Getting the type of 'ls' (line 226)
        ls_210360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 226)
        line_search_wolfe2_210361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), ls_210360, 'line_search_wolfe2')
        # Calling line_search_wolfe2(args, kwargs) (line 226)
        line_search_wolfe2_call_result_210371 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), line_search_wolfe2_210361, *[f_210362, fp_210363, x_210364, p_210365], **kwargs_210370)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___210372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), line_search_wolfe2_call_result_210371, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_210373 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___210372, int_210359)
        
        # Assigning a type to the variable 'tuple_var_assignment_209107' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209107', subscript_call_result_210373)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_210374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to line_search_wolfe2(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'f' (line 226)
        f_210377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'f', False)
        # Getting the type of 'fp' (line 226)
        fp_210378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 52), 'fp', False)
        # Getting the type of 'x' (line 226)
        x_210379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 56), 'x', False)
        # Getting the type of 'p' (line 226)
        p_210380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 59), 'p', False)
        # Processing the call keyword arguments (line 226)
        int_210381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 67), 'int')
        keyword_210382 = int_210381
        # Getting the type of 'c2' (line 226)
        c2_210383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 74), 'c2', False)
        keyword_210384 = c2_210383
        kwargs_210385 = {'c2': keyword_210384, 'amax': keyword_210382}
        # Getting the type of 'ls' (line 226)
        ls_210375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 226)
        line_search_wolfe2_210376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), ls_210375, 'line_search_wolfe2')
        # Calling line_search_wolfe2(args, kwargs) (line 226)
        line_search_wolfe2_call_result_210386 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), line_search_wolfe2_210376, *[f_210377, fp_210378, x_210379, p_210380], **kwargs_210385)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___210387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), line_search_wolfe2_call_result_210386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_210388 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___210387, int_210374)
        
        # Assigning a type to the variable 'tuple_var_assignment_209108' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209108', subscript_call_result_210388)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_210389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to line_search_wolfe2(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'f' (line 226)
        f_210392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'f', False)
        # Getting the type of 'fp' (line 226)
        fp_210393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 52), 'fp', False)
        # Getting the type of 'x' (line 226)
        x_210394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 56), 'x', False)
        # Getting the type of 'p' (line 226)
        p_210395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 59), 'p', False)
        # Processing the call keyword arguments (line 226)
        int_210396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 67), 'int')
        keyword_210397 = int_210396
        # Getting the type of 'c2' (line 226)
        c2_210398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 74), 'c2', False)
        keyword_210399 = c2_210398
        kwargs_210400 = {'c2': keyword_210399, 'amax': keyword_210397}
        # Getting the type of 'ls' (line 226)
        ls_210390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 226)
        line_search_wolfe2_210391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), ls_210390, 'line_search_wolfe2')
        # Calling line_search_wolfe2(args, kwargs) (line 226)
        line_search_wolfe2_call_result_210401 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), line_search_wolfe2_210391, *[f_210392, fp_210393, x_210394, p_210395], **kwargs_210400)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___210402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), line_search_wolfe2_call_result_210401, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_210403 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___210402, int_210389)
        
        # Assigning a type to the variable 'tuple_var_assignment_209109' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209109', subscript_call_result_210403)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_209104' (line 226)
        tuple_var_assignment_209104_210404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209104')
        # Assigning a type to the variable 's' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 's', tuple_var_assignment_209104_210404)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_209105' (line 226)
        tuple_var_assignment_209105_210405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209105')
        # Assigning a type to the variable '_' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), '_', tuple_var_assignment_209105_210405)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_209106' (line 226)
        tuple_var_assignment_209106_210406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209106')
        # Assigning a type to the variable '_' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 14), '_', tuple_var_assignment_209106_210406)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_209107' (line 226)
        tuple_var_assignment_209107_210407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209107')
        # Assigning a type to the variable '_' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), '_', tuple_var_assignment_209107_210407)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_209108' (line 226)
        tuple_var_assignment_209108_210408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209108')
        # Assigning a type to the variable '_' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), '_', tuple_var_assignment_209108_210408)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_209109' (line 226)
        tuple_var_assignment_209109_210409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_209109')
        # Assigning a type to the variable '_' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), '_', tuple_var_assignment_209109_210409)
        
        # Call to assert_line_wolfe(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'x' (line 227)
        x_210411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'x', False)
        # Getting the type of 'p' (line 227)
        p_210412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'p', False)
        # Getting the type of 's' (line 227)
        s_210413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 32), 's', False)
        # Getting the type of 'f' (line 227)
        f_210414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 35), 'f', False)
        # Getting the type of 'fp' (line 227)
        fp_210415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 38), 'fp', False)
        # Processing the call keyword arguments (line 227)
        kwargs_210416 = {}
        # Getting the type of 'assert_line_wolfe' (line 227)
        assert_line_wolfe_210410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'assert_line_wolfe', False)
        # Calling assert_line_wolfe(args, kwargs) (line 227)
        assert_line_wolfe_call_result_210417 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assert_line_wolfe_210410, *[x_210411, p_210412, s_210413, f_210414, fp_210415], **kwargs_210416)
        
        
        # Assigning a Call to a Tuple (line 229):
        
        # Assigning a Subscript to a Name (line 229):
        
        # Obtaining the type of the subscript
        int_210418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'int')
        
        # Call to assert_warns(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'LineSearchWarning' (line 229)
        LineSearchWarning_210420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 40), 'LineSearchWarning', False)
        # Getting the type of 'ls' (line 230)
        ls_210421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 230)
        line_search_wolfe2_210422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 40), ls_210421, 'line_search_wolfe2')
        # Getting the type of 'f' (line 230)
        f_210423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 63), 'f', False)
        # Getting the type of 'fp' (line 230)
        fp_210424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 66), 'fp', False)
        # Getting the type of 'x' (line 230)
        x_210425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 70), 'x', False)
        # Getting the type of 'p' (line 230)
        p_210426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 73), 'p', False)
        # Processing the call keyword arguments (line 229)
        int_210427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'int')
        keyword_210428 = int_210427
        # Getting the type of 'c2' (line 231)
        c2_210429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 52), 'c2', False)
        keyword_210430 = c2_210429
        kwargs_210431 = {'c2': keyword_210430, 'amax': keyword_210428}
        # Getting the type of 'assert_warns' (line 229)
        assert_warns_210419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 229)
        assert_warns_call_result_210432 = invoke(stypy.reporting.localization.Localization(__file__, 229, 27), assert_warns_210419, *[LineSearchWarning_210420, line_search_wolfe2_210422, f_210423, fp_210424, x_210425, p_210426], **kwargs_210431)
        
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___210433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), assert_warns_call_result_210432, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_210434 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), getitem___210433, int_210418)
        
        # Assigning a type to the variable 'tuple_var_assignment_209110' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209110', subscript_call_result_210434)
        
        # Assigning a Subscript to a Name (line 229):
        
        # Obtaining the type of the subscript
        int_210435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'int')
        
        # Call to assert_warns(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'LineSearchWarning' (line 229)
        LineSearchWarning_210437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 40), 'LineSearchWarning', False)
        # Getting the type of 'ls' (line 230)
        ls_210438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 230)
        line_search_wolfe2_210439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 40), ls_210438, 'line_search_wolfe2')
        # Getting the type of 'f' (line 230)
        f_210440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 63), 'f', False)
        # Getting the type of 'fp' (line 230)
        fp_210441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 66), 'fp', False)
        # Getting the type of 'x' (line 230)
        x_210442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 70), 'x', False)
        # Getting the type of 'p' (line 230)
        p_210443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 73), 'p', False)
        # Processing the call keyword arguments (line 229)
        int_210444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'int')
        keyword_210445 = int_210444
        # Getting the type of 'c2' (line 231)
        c2_210446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 52), 'c2', False)
        keyword_210447 = c2_210446
        kwargs_210448 = {'c2': keyword_210447, 'amax': keyword_210445}
        # Getting the type of 'assert_warns' (line 229)
        assert_warns_210436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 229)
        assert_warns_call_result_210449 = invoke(stypy.reporting.localization.Localization(__file__, 229, 27), assert_warns_210436, *[LineSearchWarning_210437, line_search_wolfe2_210439, f_210440, fp_210441, x_210442, p_210443], **kwargs_210448)
        
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___210450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), assert_warns_call_result_210449, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_210451 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), getitem___210450, int_210435)
        
        # Assigning a type to the variable 'tuple_var_assignment_209111' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209111', subscript_call_result_210451)
        
        # Assigning a Subscript to a Name (line 229):
        
        # Obtaining the type of the subscript
        int_210452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'int')
        
        # Call to assert_warns(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'LineSearchWarning' (line 229)
        LineSearchWarning_210454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 40), 'LineSearchWarning', False)
        # Getting the type of 'ls' (line 230)
        ls_210455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 230)
        line_search_wolfe2_210456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 40), ls_210455, 'line_search_wolfe2')
        # Getting the type of 'f' (line 230)
        f_210457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 63), 'f', False)
        # Getting the type of 'fp' (line 230)
        fp_210458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 66), 'fp', False)
        # Getting the type of 'x' (line 230)
        x_210459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 70), 'x', False)
        # Getting the type of 'p' (line 230)
        p_210460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 73), 'p', False)
        # Processing the call keyword arguments (line 229)
        int_210461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'int')
        keyword_210462 = int_210461
        # Getting the type of 'c2' (line 231)
        c2_210463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 52), 'c2', False)
        keyword_210464 = c2_210463
        kwargs_210465 = {'c2': keyword_210464, 'amax': keyword_210462}
        # Getting the type of 'assert_warns' (line 229)
        assert_warns_210453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 229)
        assert_warns_call_result_210466 = invoke(stypy.reporting.localization.Localization(__file__, 229, 27), assert_warns_210453, *[LineSearchWarning_210454, line_search_wolfe2_210456, f_210457, fp_210458, x_210459, p_210460], **kwargs_210465)
        
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___210467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), assert_warns_call_result_210466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_210468 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), getitem___210467, int_210452)
        
        # Assigning a type to the variable 'tuple_var_assignment_209112' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209112', subscript_call_result_210468)
        
        # Assigning a Subscript to a Name (line 229):
        
        # Obtaining the type of the subscript
        int_210469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'int')
        
        # Call to assert_warns(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'LineSearchWarning' (line 229)
        LineSearchWarning_210471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 40), 'LineSearchWarning', False)
        # Getting the type of 'ls' (line 230)
        ls_210472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 230)
        line_search_wolfe2_210473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 40), ls_210472, 'line_search_wolfe2')
        # Getting the type of 'f' (line 230)
        f_210474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 63), 'f', False)
        # Getting the type of 'fp' (line 230)
        fp_210475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 66), 'fp', False)
        # Getting the type of 'x' (line 230)
        x_210476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 70), 'x', False)
        # Getting the type of 'p' (line 230)
        p_210477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 73), 'p', False)
        # Processing the call keyword arguments (line 229)
        int_210478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'int')
        keyword_210479 = int_210478
        # Getting the type of 'c2' (line 231)
        c2_210480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 52), 'c2', False)
        keyword_210481 = c2_210480
        kwargs_210482 = {'c2': keyword_210481, 'amax': keyword_210479}
        # Getting the type of 'assert_warns' (line 229)
        assert_warns_210470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 229)
        assert_warns_call_result_210483 = invoke(stypy.reporting.localization.Localization(__file__, 229, 27), assert_warns_210470, *[LineSearchWarning_210471, line_search_wolfe2_210473, f_210474, fp_210475, x_210476, p_210477], **kwargs_210482)
        
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___210484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), assert_warns_call_result_210483, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_210485 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), getitem___210484, int_210469)
        
        # Assigning a type to the variable 'tuple_var_assignment_209113' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209113', subscript_call_result_210485)
        
        # Assigning a Subscript to a Name (line 229):
        
        # Obtaining the type of the subscript
        int_210486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'int')
        
        # Call to assert_warns(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'LineSearchWarning' (line 229)
        LineSearchWarning_210488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 40), 'LineSearchWarning', False)
        # Getting the type of 'ls' (line 230)
        ls_210489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 230)
        line_search_wolfe2_210490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 40), ls_210489, 'line_search_wolfe2')
        # Getting the type of 'f' (line 230)
        f_210491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 63), 'f', False)
        # Getting the type of 'fp' (line 230)
        fp_210492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 66), 'fp', False)
        # Getting the type of 'x' (line 230)
        x_210493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 70), 'x', False)
        # Getting the type of 'p' (line 230)
        p_210494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 73), 'p', False)
        # Processing the call keyword arguments (line 229)
        int_210495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'int')
        keyword_210496 = int_210495
        # Getting the type of 'c2' (line 231)
        c2_210497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 52), 'c2', False)
        keyword_210498 = c2_210497
        kwargs_210499 = {'c2': keyword_210498, 'amax': keyword_210496}
        # Getting the type of 'assert_warns' (line 229)
        assert_warns_210487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 229)
        assert_warns_call_result_210500 = invoke(stypy.reporting.localization.Localization(__file__, 229, 27), assert_warns_210487, *[LineSearchWarning_210488, line_search_wolfe2_210490, f_210491, fp_210492, x_210493, p_210494], **kwargs_210499)
        
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___210501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), assert_warns_call_result_210500, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_210502 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), getitem___210501, int_210486)
        
        # Assigning a type to the variable 'tuple_var_assignment_209114' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209114', subscript_call_result_210502)
        
        # Assigning a Subscript to a Name (line 229):
        
        # Obtaining the type of the subscript
        int_210503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'int')
        
        # Call to assert_warns(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'LineSearchWarning' (line 229)
        LineSearchWarning_210505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 40), 'LineSearchWarning', False)
        # Getting the type of 'ls' (line 230)
        ls_210506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 230)
        line_search_wolfe2_210507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 40), ls_210506, 'line_search_wolfe2')
        # Getting the type of 'f' (line 230)
        f_210508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 63), 'f', False)
        # Getting the type of 'fp' (line 230)
        fp_210509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 66), 'fp', False)
        # Getting the type of 'x' (line 230)
        x_210510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 70), 'x', False)
        # Getting the type of 'p' (line 230)
        p_210511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 73), 'p', False)
        # Processing the call keyword arguments (line 229)
        int_210512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'int')
        keyword_210513 = int_210512
        # Getting the type of 'c2' (line 231)
        c2_210514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 52), 'c2', False)
        keyword_210515 = c2_210514
        kwargs_210516 = {'c2': keyword_210515, 'amax': keyword_210513}
        # Getting the type of 'assert_warns' (line 229)
        assert_warns_210504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 229)
        assert_warns_call_result_210517 = invoke(stypy.reporting.localization.Localization(__file__, 229, 27), assert_warns_210504, *[LineSearchWarning_210505, line_search_wolfe2_210507, f_210508, fp_210509, x_210510, p_210511], **kwargs_210516)
        
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___210518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 8), assert_warns_call_result_210517, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_210519 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), getitem___210518, int_210503)
        
        # Assigning a type to the variable 'tuple_var_assignment_209115' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209115', subscript_call_result_210519)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'tuple_var_assignment_209110' (line 229)
        tuple_var_assignment_209110_210520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209110')
        # Assigning a type to the variable 's' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 's', tuple_var_assignment_209110_210520)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'tuple_var_assignment_209111' (line 229)
        tuple_var_assignment_209111_210521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209111')
        # Assigning a type to the variable '_' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), '_', tuple_var_assignment_209111_210521)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'tuple_var_assignment_209112' (line 229)
        tuple_var_assignment_209112_210522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209112')
        # Assigning a type to the variable '_' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 14), '_', tuple_var_assignment_209112_210522)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'tuple_var_assignment_209113' (line 229)
        tuple_var_assignment_209113_210523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209113')
        # Assigning a type to the variable '_' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), '_', tuple_var_assignment_209113_210523)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'tuple_var_assignment_209114' (line 229)
        tuple_var_assignment_209114_210524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209114')
        # Assigning a type to the variable '_' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), '_', tuple_var_assignment_209114_210524)
        
        # Assigning a Name to a Name (line 229):
        # Getting the type of 'tuple_var_assignment_209115' (line 229)
        tuple_var_assignment_209115_210525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'tuple_var_assignment_209115')
        # Assigning a type to the variable '_' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 23), '_', tuple_var_assignment_209115_210525)
        
        # Call to assert_(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Getting the type of 's' (line 232)
        s_210527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 's', False)
        # Getting the type of 'None' (line 232)
        None_210528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 21), 'None', False)
        # Applying the binary operator 'is' (line 232)
        result_is__210529 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 16), 'is', s_210527, None_210528)
        
        # Processing the call keyword arguments (line 232)
        kwargs_210530 = {}
        # Getting the type of 'assert_' (line 232)
        assert__210526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 232)
        assert__call_result_210531 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assert__210526, *[result_is__210529], **kwargs_210530)
        
        
        # Call to assert_warns(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'LineSearchWarning' (line 235)
        LineSearchWarning_210533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 21), 'LineSearchWarning', False)
        # Getting the type of 'ls' (line 235)
        ls_210534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 40), 'ls', False)
        # Obtaining the member 'line_search_wolfe2' of a type (line 235)
        line_search_wolfe2_210535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 40), ls_210534, 'line_search_wolfe2')
        # Getting the type of 'f' (line 235)
        f_210536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 63), 'f', False)
        # Getting the type of 'fp' (line 235)
        fp_210537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 66), 'fp', False)
        # Getting the type of 'x' (line 235)
        x_210538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 70), 'x', False)
        # Getting the type of 'p' (line 235)
        p_210539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 73), 'p', False)
        # Processing the call keyword arguments (line 235)
        # Getting the type of 'c2' (line 236)
        c2_210540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'c2', False)
        keyword_210541 = c2_210540
        int_210542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 36), 'int')
        keyword_210543 = int_210542
        kwargs_210544 = {'c2': keyword_210541, 'maxiter': keyword_210543}
        # Getting the type of 'assert_warns' (line 235)
        assert_warns_210532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 235)
        assert_warns_call_result_210545 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), assert_warns_210532, *[LineSearchWarning_210533, line_search_wolfe2_210535, f_210536, fp_210537, x_210538, p_210539], **kwargs_210544)
        
        
        # ################# End of 'test_line_search_wolfe2_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_line_search_wolfe2_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 212)
        stypy_return_type_210546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_210546)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_line_search_wolfe2_bounds'
        return stypy_return_type_210546


    @norecursion
    def test_line_search_armijo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_line_search_armijo'
        module_type_store = module_type_store.open_function_context('test_line_search_armijo', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_line_search_armijo')
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_line_search_armijo.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_line_search_armijo', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_line_search_armijo', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_line_search_armijo(...)' code ##################

        
        # Assigning a Num to a Name (line 239):
        
        # Assigning a Num to a Name (line 239):
        int_210547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 12), 'int')
        # Assigning a type to the variable 'c' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'c', int_210547)
        
        
        # Call to line_iter(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_210550 = {}
        # Getting the type of 'self' (line 240)
        self_210548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'self', False)
        # Obtaining the member 'line_iter' of a type (line 240)
        line_iter_210549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 44), self_210548, 'line_iter')
        # Calling line_iter(args, kwargs) (line 240)
        line_iter_call_result_210551 = invoke(stypy.reporting.localization.Localization(__file__, 240, 44), line_iter_210549, *[], **kwargs_210550)
        
        # Testing the type of a for loop iterable (line 240)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 240, 8), line_iter_call_result_210551)
        # Getting the type of the for loop variable (line 240)
        for_loop_var_210552 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 240, 8), line_iter_call_result_210551)
        # Assigning a type to the variable 'name' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), for_loop_var_210552))
        # Assigning a type to the variable 'f' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), for_loop_var_210552))
        # Assigning a type to the variable 'fprime' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'fprime', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), for_loop_var_210552))
        # Assigning a type to the variable 'x' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), for_loop_var_210552))
        # Assigning a type to the variable 'p' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), for_loop_var_210552))
        # Assigning a type to the variable 'old_f' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'old_f', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 8), for_loop_var_210552))
        # SSA begins for a for statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to f(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'x' (line 241)
        x_210554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 19), 'x', False)
        # Processing the call keyword arguments (line 241)
        kwargs_210555 = {}
        # Getting the type of 'f' (line 241)
        f_210553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'f', False)
        # Calling f(args, kwargs) (line 241)
        f_call_result_210556 = invoke(stypy.reporting.localization.Localization(__file__, 241, 17), f_210553, *[x_210554], **kwargs_210555)
        
        # Assigning a type to the variable 'f0' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'f0', f_call_result_210556)
        
        # Assigning a Call to a Name (line 242):
        
        # Assigning a Call to a Name (line 242):
        
        # Call to fprime(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'x' (line 242)
        x_210558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'x', False)
        # Processing the call keyword arguments (line 242)
        kwargs_210559 = {}
        # Getting the type of 'fprime' (line 242)
        fprime_210557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 17), 'fprime', False)
        # Calling fprime(args, kwargs) (line 242)
        fprime_call_result_210560 = invoke(stypy.reporting.localization.Localization(__file__, 242, 17), fprime_210557, *[x_210558], **kwargs_210559)
        
        # Assigning a type to the variable 'g0' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'g0', fprime_call_result_210560)
        
        # Assigning a Num to a Attribute (line 243):
        
        # Assigning a Num to a Attribute (line 243):
        int_210561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 26), 'int')
        # Getting the type of 'self' (line 243)
        self_210562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self')
        # Setting the type of the member 'fcount' of a type (line 243)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_210562, 'fcount', int_210561)
        
        # Assigning a Call to a Tuple (line 244):
        
        # Assigning a Subscript to a Name (line 244):
        
        # Obtaining the type of the subscript
        int_210563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 12), 'int')
        
        # Call to line_search_armijo(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'f' (line 244)
        f_210566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 46), 'f', False)
        # Getting the type of 'x' (line 244)
        x_210567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 49), 'x', False)
        # Getting the type of 'p' (line 244)
        p_210568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 52), 'p', False)
        # Getting the type of 'g0' (line 244)
        g0_210569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 55), 'g0', False)
        # Getting the type of 'f0' (line 244)
        f0_210570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 59), 'f0', False)
        # Processing the call keyword arguments (line 244)
        kwargs_210571 = {}
        # Getting the type of 'ls' (line 244)
        ls_210564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'ls', False)
        # Obtaining the member 'line_search_armijo' of a type (line 244)
        line_search_armijo_210565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), ls_210564, 'line_search_armijo')
        # Calling line_search_armijo(args, kwargs) (line 244)
        line_search_armijo_call_result_210572 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), line_search_armijo_210565, *[f_210566, x_210567, p_210568, g0_210569, f0_210570], **kwargs_210571)
        
        # Obtaining the member '__getitem__' of a type (line 244)
        getitem___210573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), line_search_armijo_call_result_210572, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 244)
        subscript_call_result_210574 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), getitem___210573, int_210563)
        
        # Assigning a type to the variable 'tuple_var_assignment_209116' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'tuple_var_assignment_209116', subscript_call_result_210574)
        
        # Assigning a Subscript to a Name (line 244):
        
        # Obtaining the type of the subscript
        int_210575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 12), 'int')
        
        # Call to line_search_armijo(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'f' (line 244)
        f_210578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 46), 'f', False)
        # Getting the type of 'x' (line 244)
        x_210579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 49), 'x', False)
        # Getting the type of 'p' (line 244)
        p_210580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 52), 'p', False)
        # Getting the type of 'g0' (line 244)
        g0_210581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 55), 'g0', False)
        # Getting the type of 'f0' (line 244)
        f0_210582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 59), 'f0', False)
        # Processing the call keyword arguments (line 244)
        kwargs_210583 = {}
        # Getting the type of 'ls' (line 244)
        ls_210576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'ls', False)
        # Obtaining the member 'line_search_armijo' of a type (line 244)
        line_search_armijo_210577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), ls_210576, 'line_search_armijo')
        # Calling line_search_armijo(args, kwargs) (line 244)
        line_search_armijo_call_result_210584 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), line_search_armijo_210577, *[f_210578, x_210579, p_210580, g0_210581, f0_210582], **kwargs_210583)
        
        # Obtaining the member '__getitem__' of a type (line 244)
        getitem___210585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), line_search_armijo_call_result_210584, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 244)
        subscript_call_result_210586 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), getitem___210585, int_210575)
        
        # Assigning a type to the variable 'tuple_var_assignment_209117' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'tuple_var_assignment_209117', subscript_call_result_210586)
        
        # Assigning a Subscript to a Name (line 244):
        
        # Obtaining the type of the subscript
        int_210587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 12), 'int')
        
        # Call to line_search_armijo(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'f' (line 244)
        f_210590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 46), 'f', False)
        # Getting the type of 'x' (line 244)
        x_210591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 49), 'x', False)
        # Getting the type of 'p' (line 244)
        p_210592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 52), 'p', False)
        # Getting the type of 'g0' (line 244)
        g0_210593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 55), 'g0', False)
        # Getting the type of 'f0' (line 244)
        f0_210594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 59), 'f0', False)
        # Processing the call keyword arguments (line 244)
        kwargs_210595 = {}
        # Getting the type of 'ls' (line 244)
        ls_210588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'ls', False)
        # Obtaining the member 'line_search_armijo' of a type (line 244)
        line_search_armijo_210589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), ls_210588, 'line_search_armijo')
        # Calling line_search_armijo(args, kwargs) (line 244)
        line_search_armijo_call_result_210596 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), line_search_armijo_210589, *[f_210590, x_210591, p_210592, g0_210593, f0_210594], **kwargs_210595)
        
        # Obtaining the member '__getitem__' of a type (line 244)
        getitem___210597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), line_search_armijo_call_result_210596, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 244)
        subscript_call_result_210598 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), getitem___210597, int_210587)
        
        # Assigning a type to the variable 'tuple_var_assignment_209118' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'tuple_var_assignment_209118', subscript_call_result_210598)
        
        # Assigning a Name to a Name (line 244):
        # Getting the type of 'tuple_var_assignment_209116' (line 244)
        tuple_var_assignment_209116_210599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'tuple_var_assignment_209116')
        # Assigning a type to the variable 's' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 's', tuple_var_assignment_209116_210599)
        
        # Assigning a Name to a Name (line 244):
        # Getting the type of 'tuple_var_assignment_209117' (line 244)
        tuple_var_assignment_209117_210600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'tuple_var_assignment_209117')
        # Assigning a type to the variable 'fc' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'fc', tuple_var_assignment_209117_210600)
        
        # Assigning a Name to a Name (line 244):
        # Getting the type of 'tuple_var_assignment_209118' (line 244)
        tuple_var_assignment_209118_210601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'tuple_var_assignment_209118')
        # Assigning a type to the variable 'fv' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'fv', tuple_var_assignment_209118_210601)
        
        # Getting the type of 'c' (line 245)
        c_210602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'c')
        int_210603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 17), 'int')
        # Applying the binary operator '+=' (line 245)
        result_iadd_210604 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 12), '+=', c_210602, int_210603)
        # Assigning a type to the variable 'c' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'c', result_iadd_210604)
        
        
        # Call to assert_equal(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'self' (line 246)
        self_210606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'self', False)
        # Obtaining the member 'fcount' of a type (line 246)
        fcount_210607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 25), self_210606, 'fcount')
        # Getting the type of 'fc' (line 246)
        fc_210608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'fc', False)
        # Processing the call keyword arguments (line 246)
        kwargs_210609 = {}
        # Getting the type of 'assert_equal' (line 246)
        assert_equal_210605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 246)
        assert_equal_call_result_210610 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), assert_equal_210605, *[fcount_210607, fc_210608], **kwargs_210609)
        
        
        # Call to assert_fp_equal(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'fv' (line 247)
        fv_210612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'fv', False)
        
        # Call to f(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'x' (line 247)
        x_210614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 34), 'x', False)
        # Getting the type of 's' (line 247)
        s_210615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 38), 's', False)
        # Getting the type of 'p' (line 247)
        p_210616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 40), 'p', False)
        # Applying the binary operator '*' (line 247)
        result_mul_210617 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 38), '*', s_210615, p_210616)
        
        # Applying the binary operator '+' (line 247)
        result_add_210618 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 34), '+', x_210614, result_mul_210617)
        
        # Processing the call keyword arguments (line 247)
        kwargs_210619 = {}
        # Getting the type of 'f' (line 247)
        f_210613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 32), 'f', False)
        # Calling f(args, kwargs) (line 247)
        f_call_result_210620 = invoke(stypy.reporting.localization.Localization(__file__, 247, 32), f_210613, *[result_add_210618], **kwargs_210619)
        
        # Processing the call keyword arguments (line 247)
        kwargs_210621 = {}
        # Getting the type of 'assert_fp_equal' (line 247)
        assert_fp_equal_210611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'assert_fp_equal', False)
        # Calling assert_fp_equal(args, kwargs) (line 247)
        assert_fp_equal_call_result_210622 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), assert_fp_equal_210611, *[fv_210612, f_call_result_210620], **kwargs_210621)
        
        
        # Call to assert_line_armijo(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'x' (line 248)
        x_210624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 31), 'x', False)
        # Getting the type of 'p' (line 248)
        p_210625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 34), 'p', False)
        # Getting the type of 's' (line 248)
        s_210626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 37), 's', False)
        # Getting the type of 'f' (line 248)
        f_210627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 40), 'f', False)
        # Processing the call keyword arguments (line 248)
        # Getting the type of 'name' (line 248)
        name_210628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 51), 'name', False)
        keyword_210629 = name_210628
        kwargs_210630 = {'err_msg': keyword_210629}
        # Getting the type of 'assert_line_armijo' (line 248)
        assert_line_armijo_210623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'assert_line_armijo', False)
        # Calling assert_line_armijo(args, kwargs) (line 248)
        assert_line_armijo_call_result_210631 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), assert_line_armijo_210623, *[x_210624, p_210625, s_210626, f_210627], **kwargs_210630)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Getting the type of 'c' (line 249)
        c_210633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'c', False)
        int_210634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 21), 'int')
        # Applying the binary operator '>=' (line 249)
        result_ge_210635 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 16), '>=', c_210633, int_210634)
        
        # Processing the call keyword arguments (line 249)
        kwargs_210636 = {}
        # Getting the type of 'assert_' (line 249)
        assert__210632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 249)
        assert__call_result_210637 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), assert__210632, *[result_ge_210635], **kwargs_210636)
        
        
        # ################# End of 'test_line_search_armijo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_line_search_armijo' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_210638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_210638)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_line_search_armijo'
        return stypy_return_type_210638


    @norecursion
    def test_armijo_terminate_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_armijo_terminate_1'
        module_type_store = module_type_store.open_function_context('test_armijo_terminate_1', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_armijo_terminate_1')
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_armijo_terminate_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_armijo_terminate_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_armijo_terminate_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_armijo_terminate_1(...)' code ##################

        
        # Assigning a List to a Name (line 256):
        
        # Assigning a List to a Name (line 256):
        
        # Obtaining an instance of the builtin type 'list' (line 256)
        list_210639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 256)
        # Adding element type (line 256)
        int_210640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 16), list_210639, int_210640)
        
        # Assigning a type to the variable 'count' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'count', list_210639)

        @norecursion
        def phi(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'phi'
            module_type_store = module_type_store.open_function_context('phi', 258, 8, False)
            
            # Passed parameters checking function
            phi.stypy_localization = localization
            phi.stypy_type_of_self = None
            phi.stypy_type_store = module_type_store
            phi.stypy_function_name = 'phi'
            phi.stypy_param_names_list = ['s']
            phi.stypy_varargs_param_name = None
            phi.stypy_kwargs_param_name = None
            phi.stypy_call_defaults = defaults
            phi.stypy_call_varargs = varargs
            phi.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'phi', ['s'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'phi', localization, ['s'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'phi(...)' code ##################

            
            # Getting the type of 'count' (line 259)
            count_210641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'count')
            
            # Obtaining the type of the subscript
            int_210642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 18), 'int')
            # Getting the type of 'count' (line 259)
            count_210643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'count')
            # Obtaining the member '__getitem__' of a type (line 259)
            getitem___210644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), count_210643, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 259)
            subscript_call_result_210645 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), getitem___210644, int_210642)
            
            int_210646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 24), 'int')
            # Applying the binary operator '+=' (line 259)
            result_iadd_210647 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 12), '+=', subscript_call_result_210645, int_210646)
            # Getting the type of 'count' (line 259)
            count_210648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'count')
            int_210649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 18), 'int')
            # Storing an element on a container (line 259)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 12), count_210648, (int_210649, result_iadd_210647))
            
            
            # Getting the type of 's' (line 260)
            s_210650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 's')
            # Applying the 'usub' unary operator (line 260)
            result___neg___210651 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 19), 'usub', s_210650)
            
            float_210652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 24), 'float')
            # Getting the type of 's' (line 260)
            s_210653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 's')
            int_210654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'int')
            # Applying the binary operator '**' (line 260)
            result_pow_210655 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 29), '**', s_210653, int_210654)
            
            # Applying the binary operator '*' (line 260)
            result_mul_210656 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 24), '*', float_210652, result_pow_210655)
            
            # Applying the binary operator '+' (line 260)
            result_add_210657 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 19), '+', result___neg___210651, result_mul_210656)
            
            # Assigning a type to the variable 'stypy_return_type' (line 260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'stypy_return_type', result_add_210657)
            
            # ################# End of 'phi(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'phi' in the type store
            # Getting the type of 'stypy_return_type' (line 258)
            stypy_return_type_210658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_210658)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'phi'
            return stypy_return_type_210658

        # Assigning a type to the variable 'phi' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'phi', phi)
        
        # Assigning a Call to a Tuple (line 261):
        
        # Assigning a Subscript to a Name (line 261):
        
        # Obtaining the type of the subscript
        int_210659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'int')
        
        # Call to scalar_search_armijo(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'phi' (line 261)
        phi_210662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 42), 'phi', False)
        
        # Call to phi(...): (line 261)
        # Processing the call arguments (line 261)
        int_210664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 51), 'int')
        # Processing the call keyword arguments (line 261)
        kwargs_210665 = {}
        # Getting the type of 'phi' (line 261)
        phi_210663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 47), 'phi', False)
        # Calling phi(args, kwargs) (line 261)
        phi_call_result_210666 = invoke(stypy.reporting.localization.Localization(__file__, 261, 47), phi_210663, *[int_210664], **kwargs_210665)
        
        int_210667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 55), 'int')
        # Processing the call keyword arguments (line 261)
        int_210668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 66), 'int')
        keyword_210669 = int_210668
        kwargs_210670 = {'alpha0': keyword_210669}
        # Getting the type of 'ls' (line 261)
        ls_210660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 18), 'ls', False)
        # Obtaining the member 'scalar_search_armijo' of a type (line 261)
        scalar_search_armijo_210661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 18), ls_210660, 'scalar_search_armijo')
        # Calling scalar_search_armijo(args, kwargs) (line 261)
        scalar_search_armijo_call_result_210671 = invoke(stypy.reporting.localization.Localization(__file__, 261, 18), scalar_search_armijo_210661, *[phi_210662, phi_call_result_210666, int_210667], **kwargs_210670)
        
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___210672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), scalar_search_armijo_call_result_210671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_210673 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), getitem___210672, int_210659)
        
        # Assigning a type to the variable 'tuple_var_assignment_209119' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_209119', subscript_call_result_210673)
        
        # Assigning a Subscript to a Name (line 261):
        
        # Obtaining the type of the subscript
        int_210674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'int')
        
        # Call to scalar_search_armijo(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'phi' (line 261)
        phi_210677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 42), 'phi', False)
        
        # Call to phi(...): (line 261)
        # Processing the call arguments (line 261)
        int_210679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 51), 'int')
        # Processing the call keyword arguments (line 261)
        kwargs_210680 = {}
        # Getting the type of 'phi' (line 261)
        phi_210678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 47), 'phi', False)
        # Calling phi(args, kwargs) (line 261)
        phi_call_result_210681 = invoke(stypy.reporting.localization.Localization(__file__, 261, 47), phi_210678, *[int_210679], **kwargs_210680)
        
        int_210682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 55), 'int')
        # Processing the call keyword arguments (line 261)
        int_210683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 66), 'int')
        keyword_210684 = int_210683
        kwargs_210685 = {'alpha0': keyword_210684}
        # Getting the type of 'ls' (line 261)
        ls_210675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 18), 'ls', False)
        # Obtaining the member 'scalar_search_armijo' of a type (line 261)
        scalar_search_armijo_210676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 18), ls_210675, 'scalar_search_armijo')
        # Calling scalar_search_armijo(args, kwargs) (line 261)
        scalar_search_armijo_call_result_210686 = invoke(stypy.reporting.localization.Localization(__file__, 261, 18), scalar_search_armijo_210676, *[phi_210677, phi_call_result_210681, int_210682], **kwargs_210685)
        
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___210687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), scalar_search_armijo_call_result_210686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_210688 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), getitem___210687, int_210674)
        
        # Assigning a type to the variable 'tuple_var_assignment_209120' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_209120', subscript_call_result_210688)
        
        # Assigning a Name to a Name (line 261):
        # Getting the type of 'tuple_var_assignment_209119' (line 261)
        tuple_var_assignment_209119_210689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_209119')
        # Assigning a type to the variable 's' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 's', tuple_var_assignment_209119_210689)
        
        # Assigning a Name to a Name (line 261):
        # Getting the type of 'tuple_var_assignment_209120' (line 261)
        tuple_var_assignment_209120_210690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_209120')
        # Assigning a type to the variable 'phi1' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'phi1', tuple_var_assignment_209120_210690)
        
        # Call to assert_equal(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 's' (line 262)
        s_210692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 21), 's', False)
        int_210693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 24), 'int')
        # Processing the call keyword arguments (line 262)
        kwargs_210694 = {}
        # Getting the type of 'assert_equal' (line 262)
        assert_equal_210691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 262)
        assert_equal_call_result_210695 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assert_equal_210691, *[s_210692, int_210693], **kwargs_210694)
        
        
        # Call to assert_equal(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Obtaining the type of the subscript
        int_210697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'int')
        # Getting the type of 'count' (line 263)
        count_210698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 21), 'count', False)
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___210699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 21), count_210698, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_210700 = invoke(stypy.reporting.localization.Localization(__file__, 263, 21), getitem___210699, int_210697)
        
        int_210701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 31), 'int')
        # Processing the call keyword arguments (line 263)
        kwargs_210702 = {}
        # Getting the type of 'assert_equal' (line 263)
        assert_equal_210696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 263)
        assert_equal_call_result_210703 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), assert_equal_210696, *[subscript_call_result_210700, int_210701], **kwargs_210702)
        
        
        # Call to assert_armijo(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 's' (line 264)
        s_210705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 22), 's', False)
        # Getting the type of 'phi' (line 264)
        phi_210706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 25), 'phi', False)
        # Processing the call keyword arguments (line 264)
        kwargs_210707 = {}
        # Getting the type of 'assert_armijo' (line 264)
        assert_armijo_210704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'assert_armijo', False)
        # Calling assert_armijo(args, kwargs) (line 264)
        assert_armijo_call_result_210708 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), assert_armijo_210704, *[s_210705, phi_210706], **kwargs_210707)
        
        
        # ################# End of 'test_armijo_terminate_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_armijo_terminate_1' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_210709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_210709)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_armijo_terminate_1'
        return stypy_return_type_210709


    @norecursion
    def test_wolfe_terminate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wolfe_terminate'
        module_type_store = module_type_store.open_function_context('test_wolfe_terminate', 266, 4, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_localization', localization)
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_function_name', 'TestLineSearch.test_wolfe_terminate')
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_param_names_list', [])
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLineSearch.test_wolfe_terminate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.test_wolfe_terminate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wolfe_terminate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wolfe_terminate(...)' code ##################


        @norecursion
        def phi(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'phi'
            module_type_store = module_type_store.open_function_context('phi', 270, 8, False)
            
            # Passed parameters checking function
            phi.stypy_localization = localization
            phi.stypy_type_of_self = None
            phi.stypy_type_store = module_type_store
            phi.stypy_function_name = 'phi'
            phi.stypy_param_names_list = ['s']
            phi.stypy_varargs_param_name = None
            phi.stypy_kwargs_param_name = None
            phi.stypy_call_defaults = defaults
            phi.stypy_call_varargs = varargs
            phi.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'phi', ['s'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'phi', localization, ['s'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'phi(...)' code ##################

            
            # Getting the type of 'count' (line 271)
            count_210710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'count')
            
            # Obtaining the type of the subscript
            int_210711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 18), 'int')
            # Getting the type of 'count' (line 271)
            count_210712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'count')
            # Obtaining the member '__getitem__' of a type (line 271)
            getitem___210713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), count_210712, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 271)
            subscript_call_result_210714 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), getitem___210713, int_210711)
            
            int_210715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 24), 'int')
            # Applying the binary operator '+=' (line 271)
            result_iadd_210716 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 12), '+=', subscript_call_result_210714, int_210715)
            # Getting the type of 'count' (line 271)
            count_210717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'count')
            int_210718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 18), 'int')
            # Storing an element on a container (line 271)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 12), count_210717, (int_210718, result_iadd_210716))
            
            
            # Getting the type of 's' (line 272)
            s_210719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 's')
            # Applying the 'usub' unary operator (line 272)
            result___neg___210720 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 19), 'usub', s_210719)
            
            float_210721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 24), 'float')
            # Getting the type of 's' (line 272)
            s_210722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 's')
            int_210723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 32), 'int')
            # Applying the binary operator '**' (line 272)
            result_pow_210724 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 29), '**', s_210722, int_210723)
            
            # Applying the binary operator '*' (line 272)
            result_mul_210725 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 24), '*', float_210721, result_pow_210724)
            
            # Applying the binary operator '+' (line 272)
            result_add_210726 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 19), '+', result___neg___210720, result_mul_210725)
            
            # Assigning a type to the variable 'stypy_return_type' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'stypy_return_type', result_add_210726)
            
            # ################# End of 'phi(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'phi' in the type store
            # Getting the type of 'stypy_return_type' (line 270)
            stypy_return_type_210727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_210727)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'phi'
            return stypy_return_type_210727

        # Assigning a type to the variable 'phi' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'phi', phi)

        @norecursion
        def derphi(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'derphi'
            module_type_store = module_type_store.open_function_context('derphi', 274, 8, False)
            
            # Passed parameters checking function
            derphi.stypy_localization = localization
            derphi.stypy_type_of_self = None
            derphi.stypy_type_store = module_type_store
            derphi.stypy_function_name = 'derphi'
            derphi.stypy_param_names_list = ['s']
            derphi.stypy_varargs_param_name = None
            derphi.stypy_kwargs_param_name = None
            derphi.stypy_call_defaults = defaults
            derphi.stypy_call_varargs = varargs
            derphi.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'derphi', ['s'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'derphi', localization, ['s'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'derphi(...)' code ##################

            
            # Getting the type of 'count' (line 275)
            count_210728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'count')
            
            # Obtaining the type of the subscript
            int_210729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 18), 'int')
            # Getting the type of 'count' (line 275)
            count_210730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'count')
            # Obtaining the member '__getitem__' of a type (line 275)
            getitem___210731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), count_210730, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 275)
            subscript_call_result_210732 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), getitem___210731, int_210729)
            
            int_210733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 24), 'int')
            # Applying the binary operator '+=' (line 275)
            result_iadd_210734 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 12), '+=', subscript_call_result_210732, int_210733)
            # Getting the type of 'count' (line 275)
            count_210735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'count')
            int_210736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 18), 'int')
            # Storing an element on a container (line 275)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), count_210735, (int_210736, result_iadd_210734))
            
            int_210737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 19), 'int')
            float_210738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 24), 'float')
            int_210739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 29), 'int')
            # Applying the binary operator '*' (line 276)
            result_mul_210740 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 24), '*', float_210738, int_210739)
            
            # Getting the type of 's' (line 276)
            s_210741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 31), 's')
            # Applying the binary operator '*' (line 276)
            result_mul_210742 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 30), '*', result_mul_210740, s_210741)
            
            # Applying the binary operator '+' (line 276)
            result_add_210743 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 19), '+', int_210737, result_mul_210742)
            
            # Assigning a type to the variable 'stypy_return_type' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'stypy_return_type', result_add_210743)
            
            # ################# End of 'derphi(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'derphi' in the type store
            # Getting the type of 'stypy_return_type' (line 274)
            stypy_return_type_210744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_210744)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'derphi'
            return stypy_return_type_210744

        # Assigning a type to the variable 'derphi' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'derphi', derphi)
        
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_210745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        # Getting the type of 'ls' (line 278)
        ls_210746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'ls')
        # Obtaining the member 'scalar_search_wolfe1' of a type (line 278)
        scalar_search_wolfe1_210747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 21), ls_210746, 'scalar_search_wolfe1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 20), list_210745, scalar_search_wolfe1_210747)
        # Adding element type (line 278)
        # Getting the type of 'ls' (line 278)
        ls_210748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 46), 'ls')
        # Obtaining the member 'scalar_search_wolfe2' of a type (line 278)
        scalar_search_wolfe2_210749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 46), ls_210748, 'scalar_search_wolfe2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 20), list_210745, scalar_search_wolfe2_210749)
        
        # Testing the type of a for loop iterable (line 278)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 278, 8), list_210745)
        # Getting the type of the for loop variable (line 278)
        for_loop_var_210750 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 278, 8), list_210745)
        # Assigning a type to the variable 'func' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'func', for_loop_var_210750)
        # SSA begins for a for statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 279):
        
        # Assigning a List to a Name (line 279):
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_210751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        int_210752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 20), list_210751, int_210752)
        
        # Assigning a type to the variable 'count' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'count', list_210751)
        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to func(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'phi' (line 280)
        phi_210754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 21), 'phi', False)
        # Getting the type of 'derphi' (line 280)
        derphi_210755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 26), 'derphi', False)
        
        # Call to phi(...): (line 280)
        # Processing the call arguments (line 280)
        int_210757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 38), 'int')
        # Processing the call keyword arguments (line 280)
        kwargs_210758 = {}
        # Getting the type of 'phi' (line 280)
        phi_210756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 34), 'phi', False)
        # Calling phi(args, kwargs) (line 280)
        phi_call_result_210759 = invoke(stypy.reporting.localization.Localization(__file__, 280, 34), phi_210756, *[int_210757], **kwargs_210758)
        
        # Getting the type of 'None' (line 280)
        None_210760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 42), 'None', False)
        
        # Call to derphi(...): (line 280)
        # Processing the call arguments (line 280)
        int_210762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 55), 'int')
        # Processing the call keyword arguments (line 280)
        kwargs_210763 = {}
        # Getting the type of 'derphi' (line 280)
        derphi_210761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 48), 'derphi', False)
        # Calling derphi(args, kwargs) (line 280)
        derphi_call_result_210764 = invoke(stypy.reporting.localization.Localization(__file__, 280, 48), derphi_210761, *[int_210762], **kwargs_210763)
        
        # Processing the call keyword arguments (line 280)
        kwargs_210765 = {}
        # Getting the type of 'func' (line 280)
        func_210753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'func', False)
        # Calling func(args, kwargs) (line 280)
        func_call_result_210766 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), func_210753, *[phi_210754, derphi_210755, phi_call_result_210759, None_210760, derphi_call_result_210764], **kwargs_210765)
        
        # Assigning a type to the variable 'r' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'r', func_call_result_210766)
        
        # Call to assert_(...): (line 281)
        # Processing the call arguments (line 281)
        
        
        # Obtaining the type of the subscript
        int_210768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 22), 'int')
        # Getting the type of 'r' (line 281)
        r_210769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___210770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 20), r_210769, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_210771 = invoke(stypy.reporting.localization.Localization(__file__, 281, 20), getitem___210770, int_210768)
        
        # Getting the type of 'None' (line 281)
        None_210772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'None', False)
        # Applying the binary operator 'isnot' (line 281)
        result_is_not_210773 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), 'isnot', subscript_call_result_210771, None_210772)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 281)
        tuple_210774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 281)
        # Adding element type (line 281)
        # Getting the type of 'r' (line 281)
        r_210775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 39), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 39), tuple_210774, r_210775)
        # Adding element type (line 281)
        # Getting the type of 'func' (line 281)
        func_210776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 42), 'func', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 39), tuple_210774, func_210776)
        
        # Processing the call keyword arguments (line 281)
        kwargs_210777 = {}
        # Getting the type of 'assert_' (line 281)
        assert__210767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 281)
        assert__call_result_210778 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), assert__210767, *[result_is_not_210773, tuple_210774], **kwargs_210777)
        
        
        # Call to assert_(...): (line 282)
        # Processing the call arguments (line 282)
        
        
        # Obtaining the type of the subscript
        int_210780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 26), 'int')
        # Getting the type of 'count' (line 282)
        count_210781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'count', False)
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___210782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 20), count_210781, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_210783 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), getitem___210782, int_210780)
        
        int_210784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 32), 'int')
        int_210785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 36), 'int')
        # Applying the binary operator '+' (line 282)
        result_add_210786 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 32), '+', int_210784, int_210785)
        
        # Applying the binary operator '<=' (line 282)
        result_le_210787 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 20), '<=', subscript_call_result_210783, result_add_210786)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 282)
        tuple_210788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 282)
        # Adding element type (line 282)
        # Getting the type of 'count' (line 282)
        count_210789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 40), 'count', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 40), tuple_210788, count_210789)
        # Adding element type (line 282)
        # Getting the type of 'func' (line 282)
        func_210790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 47), 'func', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 40), tuple_210788, func_210790)
        
        # Processing the call keyword arguments (line 282)
        kwargs_210791 = {}
        # Getting the type of 'assert_' (line 282)
        assert__210779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 282)
        assert__call_result_210792 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), assert__210779, *[result_le_210787, tuple_210788], **kwargs_210791)
        
        
        # Call to assert_wolfe(...): (line 283)
        # Processing the call arguments (line 283)
        
        # Obtaining the type of the subscript
        int_210794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 27), 'int')
        # Getting the type of 'r' (line 283)
        r_210795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 283)
        getitem___210796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 25), r_210795, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 283)
        subscript_call_result_210797 = invoke(stypy.reporting.localization.Localization(__file__, 283, 25), getitem___210796, int_210794)
        
        # Getting the type of 'phi' (line 283)
        phi_210798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 31), 'phi', False)
        # Getting the type of 'derphi' (line 283)
        derphi_210799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 36), 'derphi', False)
        # Processing the call keyword arguments (line 283)
        
        # Call to str(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'func' (line 283)
        func_210801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 56), 'func', False)
        # Processing the call keyword arguments (line 283)
        kwargs_210802 = {}
        # Getting the type of 'str' (line 283)
        str_210800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 52), 'str', False)
        # Calling str(args, kwargs) (line 283)
        str_call_result_210803 = invoke(stypy.reporting.localization.Localization(__file__, 283, 52), str_210800, *[func_210801], **kwargs_210802)
        
        keyword_210804 = str_call_result_210803
        kwargs_210805 = {'err_msg': keyword_210804}
        # Getting the type of 'assert_wolfe' (line 283)
        assert_wolfe_210793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'assert_wolfe', False)
        # Calling assert_wolfe(args, kwargs) (line 283)
        assert_wolfe_call_result_210806 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), assert_wolfe_210793, *[subscript_call_result_210797, phi_210798, derphi_210799], **kwargs_210805)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_wolfe_terminate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wolfe_terminate' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_210807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_210807)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wolfe_terminate'
        return stypy_return_type_210807


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 56, 0, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLineSearch.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLineSearch' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'TestLineSearch', TestLineSearch)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

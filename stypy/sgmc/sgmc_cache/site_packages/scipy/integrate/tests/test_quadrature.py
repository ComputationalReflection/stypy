
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy import cos, sin, pi
5: from numpy.testing import assert_equal, \
6:     assert_almost_equal, assert_allclose, assert_
7: from scipy._lib._numpy_compat import suppress_warnings
8: 
9: from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
10:                              cumtrapz, quad, simps, fixed_quad)
11: from scipy.integrate.quadrature import AccuracyWarning
12: 
13: 
14: class TestFixedQuad(object):
15:     def test_scalar(self):
16:         n = 4
17:         func = lambda x: x**(2*n - 1)
18:         expected = 1/(2*n)
19:         got, _ = fixed_quad(func, 0, 1, n=n)
20:         # quadrature exact for this input
21:         assert_allclose(got, expected, rtol=1e-12)
22: 
23:     def test_vector(self):
24:         n = 4
25:         p = np.arange(1, 2*n)
26:         func = lambda x: x**p[:,None]
27:         expected = 1/(p + 1)
28:         got, _ = fixed_quad(func, 0, 1, n=n)
29:         assert_allclose(got, expected, rtol=1e-12)
30: 
31: 
32: class TestQuadrature(object):
33:     def quad(self, x, a, b, args):
34:         raise NotImplementedError
35: 
36:     def test_quadrature(self):
37:         # Typical function with two extra arguments:
38:         def myfunc(x, n, z):       # Bessel function integrand
39:             return cos(n*x-z*sin(x))/pi
40:         val, err = quadrature(myfunc, 0, pi, (2, 1.8))
41:         table_val = 0.30614353532540296487
42:         assert_almost_equal(val, table_val, decimal=7)
43: 
44:     def test_quadrature_rtol(self):
45:         def myfunc(x, n, z):       # Bessel function integrand
46:             return 1e90 * cos(n*x-z*sin(x))/pi
47:         val, err = quadrature(myfunc, 0, pi, (2, 1.8), rtol=1e-10)
48:         table_val = 1e90 * 0.30614353532540296487
49:         assert_allclose(val, table_val, rtol=1e-10)
50: 
51:     def test_quadrature_miniter(self):
52:         # Typical function with two extra arguments:
53:         def myfunc(x, n, z):       # Bessel function integrand
54:             return cos(n*x-z*sin(x))/pi
55:         table_val = 0.30614353532540296487
56:         for miniter in [5, 52]:
57:             val, err = quadrature(myfunc, 0, pi, (2, 1.8), miniter=miniter)
58:             assert_almost_equal(val, table_val, decimal=7)
59:             assert_(err < 1.0)
60: 
61:     def test_quadrature_single_args(self):
62:         def myfunc(x, n):
63:             return 1e90 * cos(n*x-1.8*sin(x))/pi
64:         val, err = quadrature(myfunc, 0, pi, args=2, rtol=1e-10)
65:         table_val = 1e90 * 0.30614353532540296487
66:         assert_allclose(val, table_val, rtol=1e-10)
67: 
68:     def test_romberg(self):
69:         # Typical function with two extra arguments:
70:         def myfunc(x, n, z):       # Bessel function integrand
71:             return cos(n*x-z*sin(x))/pi
72:         val = romberg(myfunc, 0, pi, args=(2, 1.8))
73:         table_val = 0.30614353532540296487
74:         assert_almost_equal(val, table_val, decimal=7)
75: 
76:     def test_romberg_rtol(self):
77:         # Typical function with two extra arguments:
78:         def myfunc(x, n, z):       # Bessel function integrand
79:             return 1e19*cos(n*x-z*sin(x))/pi
80:         val = romberg(myfunc, 0, pi, args=(2, 1.8), rtol=1e-10)
81:         table_val = 1e19*0.30614353532540296487
82:         assert_allclose(val, table_val, rtol=1e-10)
83: 
84:     def test_romb(self):
85:         assert_equal(romb(np.arange(17)), 128)
86: 
87:     def test_romb_gh_3731(self):
88:         # Check that romb makes maximal use of data points
89:         x = np.arange(2**4+1)
90:         y = np.cos(0.2*x)
91:         val = romb(y)
92:         val2, err = quad(lambda x: np.cos(0.2*x), x.min(), x.max())
93:         assert_allclose(val, val2, rtol=1e-8, atol=0)
94: 
95:         # should be equal to romb with 2**k+1 samples
96:         with suppress_warnings() as sup:
97:             sup.filter(AccuracyWarning, "divmax .4. exceeded")
98:             val3 = romberg(lambda x: np.cos(0.2*x), x.min(), x.max(), divmax=4)
99:         assert_allclose(val, val3, rtol=1e-12, atol=0)
100: 
101:     def test_non_dtype(self):
102:         # Check that we work fine with functions returning float
103:         import math
104:         valmath = romberg(math.sin, 0, 1)
105:         expected_val = 0.45969769413185085
106:         assert_almost_equal(valmath, expected_val, decimal=7)
107: 
108:     def test_newton_cotes(self):
109:         '''Test the first few degrees, for evenly spaced points.'''
110:         n = 1
111:         wts, errcoff = newton_cotes(n, 1)
112:         assert_equal(wts, n*np.array([0.5, 0.5]))
113:         assert_almost_equal(errcoff, -n**3/12.0)
114: 
115:         n = 2
116:         wts, errcoff = newton_cotes(n, 1)
117:         assert_almost_equal(wts, n*np.array([1.0, 4.0, 1.0])/6.0)
118:         assert_almost_equal(errcoff, -n**5/2880.0)
119: 
120:         n = 3
121:         wts, errcoff = newton_cotes(n, 1)
122:         assert_almost_equal(wts, n*np.array([1.0, 3.0, 3.0, 1.0])/8.0)
123:         assert_almost_equal(errcoff, -n**5/6480.0)
124: 
125:         n = 4
126:         wts, errcoff = newton_cotes(n, 1)
127:         assert_almost_equal(wts, n*np.array([7.0, 32.0, 12.0, 32.0, 7.0])/90.0)
128:         assert_almost_equal(errcoff, -n**7/1935360.0)
129: 
130:     def test_newton_cotes2(self):
131:         '''Test newton_cotes with points that are not evenly spaced.'''
132: 
133:         x = np.array([0.0, 1.5, 2.0])
134:         y = x**2
135:         wts, errcoff = newton_cotes(x)
136:         exact_integral = 8.0/3
137:         numeric_integral = np.dot(wts, y)
138:         assert_almost_equal(numeric_integral, exact_integral)
139: 
140:         x = np.array([0.0, 1.4, 2.1, 3.0])
141:         y = x**2
142:         wts, errcoff = newton_cotes(x)
143:         exact_integral = 9.0
144:         numeric_integral = np.dot(wts, y)
145:         assert_almost_equal(numeric_integral, exact_integral)
146: 
147:     def test_simps(self):
148:         y = np.arange(17)
149:         assert_equal(simps(y), 128)
150:         assert_equal(simps(y, dx=0.5), 64)
151:         assert_equal(simps(y, x=np.linspace(0, 4, 17)), 32)
152: 
153:         y = np.arange(4)
154:         x = 2**y
155:         assert_equal(simps(y, x=x, even='avg'), 13.875)
156:         assert_equal(simps(y, x=x, even='first'), 13.75)
157:         assert_equal(simps(y, x=x, even='last'), 14)
158: 
159: 
160: class TestCumtrapz(object):
161:     def test_1d(self):
162:         x = np.linspace(-2, 2, num=5)
163:         y = x
164:         y_int = cumtrapz(y, x, initial=0)
165:         y_expected = [0., -1.5, -2., -1.5, 0.]
166:         assert_allclose(y_int, y_expected)
167: 
168:         y_int = cumtrapz(y, x, initial=None)
169:         assert_allclose(y_int, y_expected[1:])
170: 
171:     def test_y_nd_x_nd(self):
172:         x = np.arange(3 * 2 * 4).reshape(3, 2, 4)
173:         y = x
174:         y_int = cumtrapz(y, x, initial=0)
175:         y_expected = np.array([[[0., 0.5, 2., 4.5],
176:                                 [0., 4.5, 10., 16.5]],
177:                                [[0., 8.5, 18., 28.5],
178:                                 [0., 12.5, 26., 40.5]],
179:                                [[0., 16.5, 34., 52.5],
180:                                 [0., 20.5, 42., 64.5]]])
181: 
182:         assert_allclose(y_int, y_expected)
183: 
184:         # Try with all axes
185:         shapes = [(2, 2, 4), (3, 1, 4), (3, 2, 3)]
186:         for axis, shape in zip([0, 1, 2], shapes):
187:             y_int = cumtrapz(y, x, initial=3.45, axis=axis)
188:             assert_equal(y_int.shape, (3, 2, 4))
189:             y_int = cumtrapz(y, x, initial=None, axis=axis)
190:             assert_equal(y_int.shape, shape)
191: 
192:     def test_y_nd_x_1d(self):
193:         y = np.arange(3 * 2 * 4).reshape(3, 2, 4)
194:         x = np.arange(4)**2
195:         # Try with all axes
196:         ys_expected = (
197:             np.array([[[4., 5., 6., 7.],
198:                        [8., 9., 10., 11.]],
199:                       [[40., 44., 48., 52.],
200:                        [56., 60., 64., 68.]]]),
201:             np.array([[[2., 3., 4., 5.]],
202:                       [[10., 11., 12., 13.]],
203:                       [[18., 19., 20., 21.]]]),
204:             np.array([[[0.5, 5., 17.5],
205:                        [4.5, 21., 53.5]],
206:                       [[8.5, 37., 89.5],
207:                        [12.5, 53., 125.5]],
208:                       [[16.5, 69., 161.5],
209:                        [20.5, 85., 197.5]]]))
210: 
211:         for axis, y_expected in zip([0, 1, 2], ys_expected):
212:             y_int = cumtrapz(y, x=x[:y.shape[axis]], axis=axis, initial=None)
213:             assert_allclose(y_int, y_expected)
214: 
215:     def test_x_none(self):
216:         y = np.linspace(-2, 2, num=5)
217: 
218:         y_int = cumtrapz(y)
219:         y_expected = [-1.5, -2., -1.5, 0.]
220:         assert_allclose(y_int, y_expected)
221: 
222:         y_int = cumtrapz(y, initial=1.23)
223:         y_expected = [1.23, -1.5, -2., -1.5, 0.]
224:         assert_allclose(y_int, y_expected)
225: 
226:         y_int = cumtrapz(y, dx=3)
227:         y_expected = [-4.5, -6., -4.5, 0.]
228:         assert_allclose(y_int, y_expected)
229: 
230:         y_int = cumtrapz(y, dx=3, initial=1.23)
231:         y_expected = [1.23, -4.5, -6., -4.5, 0.]
232:         assert_allclose(y_int, y_expected)
233: 
234: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_50880 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_50880) is not StypyTypeError):

    if (import_50880 != 'pyd_module'):
        __import__(import_50880)
        sys_modules_50881 = sys.modules[import_50880]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_50881.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_50880)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy import cos, sin, pi' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_50882 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_50882) is not StypyTypeError):

    if (import_50882 != 'pyd_module'):
        __import__(import_50882)
        sys_modules_50883 = sys.modules[import_50882]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_50883.module_type_store, module_type_store, ['cos', 'sin', 'pi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_50883, sys_modules_50883.module_type_store, module_type_store)
    else:
        from numpy import cos, sin, pi

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', None, module_type_store, ['cos', 'sin', 'pi'], [cos, sin, pi])

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_50882)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_equal, assert_almost_equal, assert_allclose, assert_' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_50884 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_50884) is not StypyTypeError):

    if (import_50884 != 'pyd_module'):
        __import__(import_50884)
        sys_modules_50885 = sys.modules[import_50884]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_50885.module_type_store, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_allclose', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_50885, sys_modules_50885.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_almost_equal, assert_allclose, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_allclose', 'assert_'], [assert_equal, assert_almost_equal, assert_allclose, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_50884)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_50886 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat')

if (type(import_50886) is not StypyTypeError):

    if (import_50886 != 'pyd_module'):
        __import__(import_50886)
        sys_modules_50887 = sys.modules[import_50886]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', sys_modules_50887.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_50887, sys_modules_50887.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', import_50886)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.integrate import quadrature, romberg, romb, newton_cotes, cumtrapz, quad, simps, fixed_quad' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_50888 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate')

if (type(import_50888) is not StypyTypeError):

    if (import_50888 != 'pyd_module'):
        __import__(import_50888)
        sys_modules_50889 = sys.modules[import_50888]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate', sys_modules_50889.module_type_store, module_type_store, ['quadrature', 'romberg', 'romb', 'newton_cotes', 'cumtrapz', 'quad', 'simps', 'fixed_quad'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_50889, sys_modules_50889.module_type_store, module_type_store)
    else:
        from scipy.integrate import quadrature, romberg, romb, newton_cotes, cumtrapz, quad, simps, fixed_quad

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate', None, module_type_store, ['quadrature', 'romberg', 'romb', 'newton_cotes', 'cumtrapz', 'quad', 'simps', 'fixed_quad'], [quadrature, romberg, romb, newton_cotes, cumtrapz, quad, simps, fixed_quad])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate', import_50888)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.integrate.quadrature import AccuracyWarning' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/tests/')
import_50890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.integrate.quadrature')

if (type(import_50890) is not StypyTypeError):

    if (import_50890 != 'pyd_module'):
        __import__(import_50890)
        sys_modules_50891 = sys.modules[import_50890]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.integrate.quadrature', sys_modules_50891.module_type_store, module_type_store, ['AccuracyWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_50891, sys_modules_50891.module_type_store, module_type_store)
    else:
        from scipy.integrate.quadrature import AccuracyWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.integrate.quadrature', None, module_type_store, ['AccuracyWarning'], [AccuracyWarning])

else:
    # Assigning a type to the variable 'scipy.integrate.quadrature' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.integrate.quadrature', import_50890)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/tests/')

# Declaration of the 'TestFixedQuad' class

class TestFixedQuad(object, ):

    @norecursion
    def test_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalar'
        module_type_store = module_type_store.open_function_context('test_scalar', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_localization', localization)
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_function_name', 'TestFixedQuad.test_scalar')
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_param_names_list', [])
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFixedQuad.test_scalar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFixedQuad.test_scalar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalar(...)' code ##################

        
        # Assigning a Num to a Name (line 16):
        
        # Assigning a Num to a Name (line 16):
        int_50892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
        # Assigning a type to the variable 'n' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'n', int_50892)
        
        # Assigning a Lambda to a Name (line 17):
        
        # Assigning a Lambda to a Name (line 17):

        @norecursion
        def _stypy_temp_lambda_43(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_43'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_43', 17, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_43.stypy_localization = localization
            _stypy_temp_lambda_43.stypy_type_of_self = None
            _stypy_temp_lambda_43.stypy_type_store = module_type_store
            _stypy_temp_lambda_43.stypy_function_name = '_stypy_temp_lambda_43'
            _stypy_temp_lambda_43.stypy_param_names_list = ['x']
            _stypy_temp_lambda_43.stypy_varargs_param_name = None
            _stypy_temp_lambda_43.stypy_kwargs_param_name = None
            _stypy_temp_lambda_43.stypy_call_defaults = defaults
            _stypy_temp_lambda_43.stypy_call_varargs = varargs
            _stypy_temp_lambda_43.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_43', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_43', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 17)
            x_50893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'x')
            int_50894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 29), 'int')
            # Getting the type of 'n' (line 17)
            n_50895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 31), 'n')
            # Applying the binary operator '*' (line 17)
            result_mul_50896 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 29), '*', int_50894, n_50895)
            
            int_50897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'int')
            # Applying the binary operator '-' (line 17)
            result_sub_50898 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 29), '-', result_mul_50896, int_50897)
            
            # Applying the binary operator '**' (line 17)
            result_pow_50899 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 25), '**', x_50893, result_sub_50898)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'stypy_return_type', result_pow_50899)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_43' in the type store
            # Getting the type of 'stypy_return_type' (line 17)
            stypy_return_type_50900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50900)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_43'
            return stypy_return_type_50900

        # Assigning a type to the variable '_stypy_temp_lambda_43' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), '_stypy_temp_lambda_43', _stypy_temp_lambda_43)
        # Getting the type of '_stypy_temp_lambda_43' (line 17)
        _stypy_temp_lambda_43_50901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), '_stypy_temp_lambda_43')
        # Assigning a type to the variable 'func' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'func', _stypy_temp_lambda_43_50901)
        
        # Assigning a BinOp to a Name (line 18):
        
        # Assigning a BinOp to a Name (line 18):
        int_50902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'int')
        int_50903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 22), 'int')
        # Getting the type of 'n' (line 18)
        n_50904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'n')
        # Applying the binary operator '*' (line 18)
        result_mul_50905 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 22), '*', int_50903, n_50904)
        
        # Applying the binary operator 'div' (line 18)
        result_div_50906 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 19), 'div', int_50902, result_mul_50905)
        
        # Assigning a type to the variable 'expected' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'expected', result_div_50906)
        
        # Assigning a Call to a Tuple (line 19):
        
        # Assigning a Subscript to a Name (line 19):
        
        # Obtaining the type of the subscript
        int_50907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'int')
        
        # Call to fixed_quad(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'func' (line 19)
        func_50909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), 'func', False)
        int_50910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'int')
        int_50911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 37), 'int')
        # Processing the call keyword arguments (line 19)
        # Getting the type of 'n' (line 19)
        n_50912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 42), 'n', False)
        keyword_50913 = n_50912
        kwargs_50914 = {'n': keyword_50913}
        # Getting the type of 'fixed_quad' (line 19)
        fixed_quad_50908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'fixed_quad', False)
        # Calling fixed_quad(args, kwargs) (line 19)
        fixed_quad_call_result_50915 = invoke(stypy.reporting.localization.Localization(__file__, 19, 17), fixed_quad_50908, *[func_50909, int_50910, int_50911], **kwargs_50914)
        
        # Obtaining the member '__getitem__' of a type (line 19)
        getitem___50916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), fixed_quad_call_result_50915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 19)
        subscript_call_result_50917 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), getitem___50916, int_50907)
        
        # Assigning a type to the variable 'tuple_var_assignment_50854' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_var_assignment_50854', subscript_call_result_50917)
        
        # Assigning a Subscript to a Name (line 19):
        
        # Obtaining the type of the subscript
        int_50918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'int')
        
        # Call to fixed_quad(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'func' (line 19)
        func_50920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), 'func', False)
        int_50921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'int')
        int_50922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 37), 'int')
        # Processing the call keyword arguments (line 19)
        # Getting the type of 'n' (line 19)
        n_50923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 42), 'n', False)
        keyword_50924 = n_50923
        kwargs_50925 = {'n': keyword_50924}
        # Getting the type of 'fixed_quad' (line 19)
        fixed_quad_50919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'fixed_quad', False)
        # Calling fixed_quad(args, kwargs) (line 19)
        fixed_quad_call_result_50926 = invoke(stypy.reporting.localization.Localization(__file__, 19, 17), fixed_quad_50919, *[func_50920, int_50921, int_50922], **kwargs_50925)
        
        # Obtaining the member '__getitem__' of a type (line 19)
        getitem___50927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), fixed_quad_call_result_50926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 19)
        subscript_call_result_50928 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), getitem___50927, int_50918)
        
        # Assigning a type to the variable 'tuple_var_assignment_50855' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_var_assignment_50855', subscript_call_result_50928)
        
        # Assigning a Name to a Name (line 19):
        # Getting the type of 'tuple_var_assignment_50854' (line 19)
        tuple_var_assignment_50854_50929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_var_assignment_50854')
        # Assigning a type to the variable 'got' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'got', tuple_var_assignment_50854_50929)
        
        # Assigning a Name to a Name (line 19):
        # Getting the type of 'tuple_var_assignment_50855' (line 19)
        tuple_var_assignment_50855_50930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_var_assignment_50855')
        # Assigning a type to the variable '_' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), '_', tuple_var_assignment_50855_50930)
        
        # Call to assert_allclose(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'got' (line 21)
        got_50932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'got', False)
        # Getting the type of 'expected' (line 21)
        expected_50933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 29), 'expected', False)
        # Processing the call keyword arguments (line 21)
        float_50934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'float')
        keyword_50935 = float_50934
        kwargs_50936 = {'rtol': keyword_50935}
        # Getting the type of 'assert_allclose' (line 21)
        assert_allclose_50931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 21)
        assert_allclose_call_result_50937 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert_allclose_50931, *[got_50932, expected_50933], **kwargs_50936)
        
        
        # ################# End of 'test_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_50938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalar'
        return stypy_return_type_50938


    @norecursion
    def test_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vector'
        module_type_store = module_type_store.open_function_context('test_vector', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_localization', localization)
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_function_name', 'TestFixedQuad.test_vector')
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_param_names_list', [])
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFixedQuad.test_vector.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFixedQuad.test_vector', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vector', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vector(...)' code ##################

        
        # Assigning a Num to a Name (line 24):
        
        # Assigning a Num to a Name (line 24):
        int_50939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        # Assigning a type to the variable 'n' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'n', int_50939)
        
        # Assigning a Call to a Name (line 25):
        
        # Assigning a Call to a Name (line 25):
        
        # Call to arange(...): (line 25)
        # Processing the call arguments (line 25)
        int_50942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'int')
        int_50943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
        # Getting the type of 'n' (line 25)
        n_50944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'n', False)
        # Applying the binary operator '*' (line 25)
        result_mul_50945 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '*', int_50943, n_50944)
        
        # Processing the call keyword arguments (line 25)
        kwargs_50946 = {}
        # Getting the type of 'np' (line 25)
        np_50940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 25)
        arange_50941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), np_50940, 'arange')
        # Calling arange(args, kwargs) (line 25)
        arange_call_result_50947 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), arange_50941, *[int_50942, result_mul_50945], **kwargs_50946)
        
        # Assigning a type to the variable 'p' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'p', arange_call_result_50947)
        
        # Assigning a Lambda to a Name (line 26):
        
        # Assigning a Lambda to a Name (line 26):

        @norecursion
        def _stypy_temp_lambda_44(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_44'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_44', 26, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_44.stypy_localization = localization
            _stypy_temp_lambda_44.stypy_type_of_self = None
            _stypy_temp_lambda_44.stypy_type_store = module_type_store
            _stypy_temp_lambda_44.stypy_function_name = '_stypy_temp_lambda_44'
            _stypy_temp_lambda_44.stypy_param_names_list = ['x']
            _stypy_temp_lambda_44.stypy_varargs_param_name = None
            _stypy_temp_lambda_44.stypy_kwargs_param_name = None
            _stypy_temp_lambda_44.stypy_call_defaults = defaults
            _stypy_temp_lambda_44.stypy_call_varargs = varargs
            _stypy_temp_lambda_44.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_44', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_44', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 26)
            x_50948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'x')
            
            # Obtaining the type of the subscript
            slice_50949 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 26, 28), None, None, None)
            # Getting the type of 'None' (line 26)
            None_50950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 32), 'None')
            # Getting the type of 'p' (line 26)
            p_50951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 28), 'p')
            # Obtaining the member '__getitem__' of a type (line 26)
            getitem___50952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 28), p_50951, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 26)
            subscript_call_result_50953 = invoke(stypy.reporting.localization.Localization(__file__, 26, 28), getitem___50952, (slice_50949, None_50950))
            
            # Applying the binary operator '**' (line 26)
            result_pow_50954 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 25), '**', x_50948, subscript_call_result_50953)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'stypy_return_type', result_pow_50954)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_44' in the type store
            # Getting the type of 'stypy_return_type' (line 26)
            stypy_return_type_50955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_50955)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_44'
            return stypy_return_type_50955

        # Assigning a type to the variable '_stypy_temp_lambda_44' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), '_stypy_temp_lambda_44', _stypy_temp_lambda_44)
        # Getting the type of '_stypy_temp_lambda_44' (line 26)
        _stypy_temp_lambda_44_50956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), '_stypy_temp_lambda_44')
        # Assigning a type to the variable 'func' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'func', _stypy_temp_lambda_44_50956)
        
        # Assigning a BinOp to a Name (line 27):
        
        # Assigning a BinOp to a Name (line 27):
        int_50957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'int')
        # Getting the type of 'p' (line 27)
        p_50958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'p')
        int_50959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'int')
        # Applying the binary operator '+' (line 27)
        result_add_50960 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 22), '+', p_50958, int_50959)
        
        # Applying the binary operator 'div' (line 27)
        result_div_50961 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 19), 'div', int_50957, result_add_50960)
        
        # Assigning a type to the variable 'expected' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'expected', result_div_50961)
        
        # Assigning a Call to a Tuple (line 28):
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_50962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to fixed_quad(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'func' (line 28)
        func_50964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'func', False)
        int_50965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'int')
        int_50966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'int')
        # Processing the call keyword arguments (line 28)
        # Getting the type of 'n' (line 28)
        n_50967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'n', False)
        keyword_50968 = n_50967
        kwargs_50969 = {'n': keyword_50968}
        # Getting the type of 'fixed_quad' (line 28)
        fixed_quad_50963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'fixed_quad', False)
        # Calling fixed_quad(args, kwargs) (line 28)
        fixed_quad_call_result_50970 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), fixed_quad_50963, *[func_50964, int_50965, int_50966], **kwargs_50969)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___50971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), fixed_quad_call_result_50970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_50972 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___50971, int_50962)
        
        # Assigning a type to the variable 'tuple_var_assignment_50856' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_50856', subscript_call_result_50972)
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_50973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to fixed_quad(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'func' (line 28)
        func_50975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'func', False)
        int_50976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'int')
        int_50977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'int')
        # Processing the call keyword arguments (line 28)
        # Getting the type of 'n' (line 28)
        n_50978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'n', False)
        keyword_50979 = n_50978
        kwargs_50980 = {'n': keyword_50979}
        # Getting the type of 'fixed_quad' (line 28)
        fixed_quad_50974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'fixed_quad', False)
        # Calling fixed_quad(args, kwargs) (line 28)
        fixed_quad_call_result_50981 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), fixed_quad_50974, *[func_50975, int_50976, int_50977], **kwargs_50980)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___50982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), fixed_quad_call_result_50981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_50983 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___50982, int_50973)
        
        # Assigning a type to the variable 'tuple_var_assignment_50857' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_50857', subscript_call_result_50983)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_50856' (line 28)
        tuple_var_assignment_50856_50984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_50856')
        # Assigning a type to the variable 'got' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'got', tuple_var_assignment_50856_50984)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_50857' (line 28)
        tuple_var_assignment_50857_50985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_50857')
        # Assigning a type to the variable '_' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), '_', tuple_var_assignment_50857_50985)
        
        # Call to assert_allclose(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'got' (line 29)
        got_50987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'got', False)
        # Getting the type of 'expected' (line 29)
        expected_50988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'expected', False)
        # Processing the call keyword arguments (line 29)
        float_50989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 44), 'float')
        keyword_50990 = float_50989
        kwargs_50991 = {'rtol': keyword_50990}
        # Getting the type of 'assert_allclose' (line 29)
        assert_allclose_50986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 29)
        assert_allclose_call_result_50992 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert_allclose_50986, *[got_50987, expected_50988], **kwargs_50991)
        
        
        # ################# End of 'test_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_50993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50993)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vector'
        return stypy_return_type_50993


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFixedQuad.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFixedQuad' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TestFixedQuad', TestFixedQuad)
# Declaration of the 'TestQuadrature' class

class TestQuadrature(object, ):

    @norecursion
    def quad(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'quad'
        module_type_store = module_type_store.open_function_context('quad', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.quad.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.quad.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.quad.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.quad.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.quad')
        TestQuadrature.quad.__dict__.__setitem__('stypy_param_names_list', ['x', 'a', 'b', 'args'])
        TestQuadrature.quad.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.quad.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.quad.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.quad.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.quad.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.quad.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.quad', ['x', 'a', 'b', 'args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'quad', localization, ['x', 'a', 'b', 'args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'quad(...)' code ##################

        # Getting the type of 'NotImplementedError' (line 34)
        NotImplementedError_50994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 34, 8), NotImplementedError_50994, 'raise parameter', BaseException)
        
        # ################# End of 'quad(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'quad' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_50995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50995)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'quad'
        return stypy_return_type_50995


    @norecursion
    def test_quadrature(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadrature'
        module_type_store = module_type_store.open_function_context('test_quadrature', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_quadrature')
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_quadrature.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_quadrature', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadrature', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadrature(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 38, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'n', 'z']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'n', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'n', 'z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            # Call to cos(...): (line 39)
            # Processing the call arguments (line 39)
            # Getting the type of 'n' (line 39)
            n_50997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'n', False)
            # Getting the type of 'x' (line 39)
            x_50998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'x', False)
            # Applying the binary operator '*' (line 39)
            result_mul_50999 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '*', n_50997, x_50998)
            
            # Getting the type of 'z' (line 39)
            z_51000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'z', False)
            
            # Call to sin(...): (line 39)
            # Processing the call arguments (line 39)
            # Getting the type of 'x' (line 39)
            x_51002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'x', False)
            # Processing the call keyword arguments (line 39)
            kwargs_51003 = {}
            # Getting the type of 'sin' (line 39)
            sin_51001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'sin', False)
            # Calling sin(args, kwargs) (line 39)
            sin_call_result_51004 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), sin_51001, *[x_51002], **kwargs_51003)
            
            # Applying the binary operator '*' (line 39)
            result_mul_51005 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 27), '*', z_51000, sin_call_result_51004)
            
            # Applying the binary operator '-' (line 39)
            result_sub_51006 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '-', result_mul_50999, result_mul_51005)
            
            # Processing the call keyword arguments (line 39)
            kwargs_51007 = {}
            # Getting the type of 'cos' (line 39)
            cos_50996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'cos', False)
            # Calling cos(args, kwargs) (line 39)
            cos_call_result_51008 = invoke(stypy.reporting.localization.Localization(__file__, 39, 19), cos_50996, *[result_sub_51006], **kwargs_51007)
            
            # Getting the type of 'pi' (line 39)
            pi_51009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'pi')
            # Applying the binary operator 'div' (line 39)
            result_div_51010 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 19), 'div', cos_call_result_51008, pi_51009)
            
            # Assigning a type to the variable 'stypy_return_type' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'stypy_return_type', result_div_51010)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 38)
            stypy_return_type_51011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_51011)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_51011

        # Assigning a type to the variable 'myfunc' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'myfunc', myfunc)
        
        # Assigning a Call to a Tuple (line 40):
        
        # Assigning a Subscript to a Name (line 40):
        
        # Obtaining the type of the subscript
        int_51012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'int')
        
        # Call to quadrature(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'myfunc' (line 40)
        myfunc_51014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'myfunc', False)
        int_51015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'int')
        # Getting the type of 'pi' (line 40)
        pi_51016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 'pi', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_51017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        int_51018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 46), tuple_51017, int_51018)
        # Adding element type (line 40)
        float_51019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 46), tuple_51017, float_51019)
        
        # Processing the call keyword arguments (line 40)
        kwargs_51020 = {}
        # Getting the type of 'quadrature' (line 40)
        quadrature_51013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'quadrature', False)
        # Calling quadrature(args, kwargs) (line 40)
        quadrature_call_result_51021 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), quadrature_51013, *[myfunc_51014, int_51015, pi_51016, tuple_51017], **kwargs_51020)
        
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___51022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), quadrature_call_result_51021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_51023 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), getitem___51022, int_51012)
        
        # Assigning a type to the variable 'tuple_var_assignment_50858' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_50858', subscript_call_result_51023)
        
        # Assigning a Subscript to a Name (line 40):
        
        # Obtaining the type of the subscript
        int_51024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'int')
        
        # Call to quadrature(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'myfunc' (line 40)
        myfunc_51026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'myfunc', False)
        int_51027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'int')
        # Getting the type of 'pi' (line 40)
        pi_51028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 'pi', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_51029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        int_51030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 46), tuple_51029, int_51030)
        # Adding element type (line 40)
        float_51031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 46), tuple_51029, float_51031)
        
        # Processing the call keyword arguments (line 40)
        kwargs_51032 = {}
        # Getting the type of 'quadrature' (line 40)
        quadrature_51025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'quadrature', False)
        # Calling quadrature(args, kwargs) (line 40)
        quadrature_call_result_51033 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), quadrature_51025, *[myfunc_51026, int_51027, pi_51028, tuple_51029], **kwargs_51032)
        
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___51034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), quadrature_call_result_51033, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_51035 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), getitem___51034, int_51024)
        
        # Assigning a type to the variable 'tuple_var_assignment_50859' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_50859', subscript_call_result_51035)
        
        # Assigning a Name to a Name (line 40):
        # Getting the type of 'tuple_var_assignment_50858' (line 40)
        tuple_var_assignment_50858_51036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_50858')
        # Assigning a type to the variable 'val' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'val', tuple_var_assignment_50858_51036)
        
        # Assigning a Name to a Name (line 40):
        # Getting the type of 'tuple_var_assignment_50859' (line 40)
        tuple_var_assignment_50859_51037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'tuple_var_assignment_50859')
        # Assigning a type to the variable 'err' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'err', tuple_var_assignment_50859_51037)
        
        # Assigning a Num to a Name (line 41):
        
        # Assigning a Num to a Name (line 41):
        float_51038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'float')
        # Assigning a type to the variable 'table_val' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'table_val', float_51038)
        
        # Call to assert_almost_equal(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'val' (line 42)
        val_51040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'val', False)
        # Getting the type of 'table_val' (line 42)
        table_val_51041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'table_val', False)
        # Processing the call keyword arguments (line 42)
        int_51042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 52), 'int')
        keyword_51043 = int_51042
        kwargs_51044 = {'decimal': keyword_51043}
        # Getting the type of 'assert_almost_equal' (line 42)
        assert_almost_equal_51039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 42)
        assert_almost_equal_call_result_51045 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_almost_equal_51039, *[val_51040, table_val_51041], **kwargs_51044)
        
        
        # ################# End of 'test_quadrature(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadrature' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_51046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51046)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadrature'
        return stypy_return_type_51046


    @norecursion
    def test_quadrature_rtol(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadrature_rtol'
        module_type_store = module_type_store.open_function_context('test_quadrature_rtol', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_quadrature_rtol')
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_quadrature_rtol.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_quadrature_rtol', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadrature_rtol', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadrature_rtol(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 45, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'n', 'z']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'n', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'n', 'z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            float_51047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'float')
            
            # Call to cos(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'n' (line 46)
            n_51049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 'n', False)
            # Getting the type of 'x' (line 46)
            x_51050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'x', False)
            # Applying the binary operator '*' (line 46)
            result_mul_51051 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 30), '*', n_51049, x_51050)
            
            # Getting the type of 'z' (line 46)
            z_51052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'z', False)
            
            # Call to sin(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'x' (line 46)
            x_51054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'x', False)
            # Processing the call keyword arguments (line 46)
            kwargs_51055 = {}
            # Getting the type of 'sin' (line 46)
            sin_51053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'sin', False)
            # Calling sin(args, kwargs) (line 46)
            sin_call_result_51056 = invoke(stypy.reporting.localization.Localization(__file__, 46, 36), sin_51053, *[x_51054], **kwargs_51055)
            
            # Applying the binary operator '*' (line 46)
            result_mul_51057 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 34), '*', z_51052, sin_call_result_51056)
            
            # Applying the binary operator '-' (line 46)
            result_sub_51058 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 30), '-', result_mul_51051, result_mul_51057)
            
            # Processing the call keyword arguments (line 46)
            kwargs_51059 = {}
            # Getting the type of 'cos' (line 46)
            cos_51048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'cos', False)
            # Calling cos(args, kwargs) (line 46)
            cos_call_result_51060 = invoke(stypy.reporting.localization.Localization(__file__, 46, 26), cos_51048, *[result_sub_51058], **kwargs_51059)
            
            # Applying the binary operator '*' (line 46)
            result_mul_51061 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 19), '*', float_51047, cos_call_result_51060)
            
            # Getting the type of 'pi' (line 46)
            pi_51062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 44), 'pi')
            # Applying the binary operator 'div' (line 46)
            result_div_51063 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 43), 'div', result_mul_51061, pi_51062)
            
            # Assigning a type to the variable 'stypy_return_type' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', result_div_51063)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 45)
            stypy_return_type_51064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_51064)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_51064

        # Assigning a type to the variable 'myfunc' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'myfunc', myfunc)
        
        # Assigning a Call to a Tuple (line 47):
        
        # Assigning a Subscript to a Name (line 47):
        
        # Obtaining the type of the subscript
        int_51065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'int')
        
        # Call to quadrature(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'myfunc' (line 47)
        myfunc_51067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'myfunc', False)
        int_51068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'int')
        # Getting the type of 'pi' (line 47)
        pi_51069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'pi', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_51070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        int_51071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 46), tuple_51070, int_51071)
        # Adding element type (line 47)
        float_51072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 46), tuple_51070, float_51072)
        
        # Processing the call keyword arguments (line 47)
        float_51073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 60), 'float')
        keyword_51074 = float_51073
        kwargs_51075 = {'rtol': keyword_51074}
        # Getting the type of 'quadrature' (line 47)
        quadrature_51066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'quadrature', False)
        # Calling quadrature(args, kwargs) (line 47)
        quadrature_call_result_51076 = invoke(stypy.reporting.localization.Localization(__file__, 47, 19), quadrature_51066, *[myfunc_51067, int_51068, pi_51069, tuple_51070], **kwargs_51075)
        
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___51077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), quadrature_call_result_51076, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_51078 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), getitem___51077, int_51065)
        
        # Assigning a type to the variable 'tuple_var_assignment_50860' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_var_assignment_50860', subscript_call_result_51078)
        
        # Assigning a Subscript to a Name (line 47):
        
        # Obtaining the type of the subscript
        int_51079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'int')
        
        # Call to quadrature(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'myfunc' (line 47)
        myfunc_51081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'myfunc', False)
        int_51082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'int')
        # Getting the type of 'pi' (line 47)
        pi_51083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'pi', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_51084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        int_51085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 46), tuple_51084, int_51085)
        # Adding element type (line 47)
        float_51086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 46), tuple_51084, float_51086)
        
        # Processing the call keyword arguments (line 47)
        float_51087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 60), 'float')
        keyword_51088 = float_51087
        kwargs_51089 = {'rtol': keyword_51088}
        # Getting the type of 'quadrature' (line 47)
        quadrature_51080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'quadrature', False)
        # Calling quadrature(args, kwargs) (line 47)
        quadrature_call_result_51090 = invoke(stypy.reporting.localization.Localization(__file__, 47, 19), quadrature_51080, *[myfunc_51081, int_51082, pi_51083, tuple_51084], **kwargs_51089)
        
        # Obtaining the member '__getitem__' of a type (line 47)
        getitem___51091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), quadrature_call_result_51090, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 47)
        subscript_call_result_51092 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), getitem___51091, int_51079)
        
        # Assigning a type to the variable 'tuple_var_assignment_50861' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_var_assignment_50861', subscript_call_result_51092)
        
        # Assigning a Name to a Name (line 47):
        # Getting the type of 'tuple_var_assignment_50860' (line 47)
        tuple_var_assignment_50860_51093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_var_assignment_50860')
        # Assigning a type to the variable 'val' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'val', tuple_var_assignment_50860_51093)
        
        # Assigning a Name to a Name (line 47):
        # Getting the type of 'tuple_var_assignment_50861' (line 47)
        tuple_var_assignment_50861_51094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_var_assignment_50861')
        # Assigning a type to the variable 'err' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'err', tuple_var_assignment_50861_51094)
        
        # Assigning a BinOp to a Name (line 48):
        
        # Assigning a BinOp to a Name (line 48):
        float_51095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'float')
        float_51096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'float')
        # Applying the binary operator '*' (line 48)
        result_mul_51097 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 20), '*', float_51095, float_51096)
        
        # Assigning a type to the variable 'table_val' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'table_val', result_mul_51097)
        
        # Call to assert_allclose(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'val' (line 49)
        val_51099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'val', False)
        # Getting the type of 'table_val' (line 49)
        table_val_51100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 29), 'table_val', False)
        # Processing the call keyword arguments (line 49)
        float_51101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 45), 'float')
        keyword_51102 = float_51101
        kwargs_51103 = {'rtol': keyword_51102}
        # Getting the type of 'assert_allclose' (line 49)
        assert_allclose_51098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 49)
        assert_allclose_call_result_51104 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assert_allclose_51098, *[val_51099, table_val_51100], **kwargs_51103)
        
        
        # ################# End of 'test_quadrature_rtol(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadrature_rtol' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_51105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51105)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadrature_rtol'
        return stypy_return_type_51105


    @norecursion
    def test_quadrature_miniter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadrature_miniter'
        module_type_store = module_type_store.open_function_context('test_quadrature_miniter', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_quadrature_miniter')
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_quadrature_miniter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_quadrature_miniter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadrature_miniter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadrature_miniter(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 53, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'n', 'z']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'n', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'n', 'z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            # Call to cos(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'n' (line 54)
            n_51107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'n', False)
            # Getting the type of 'x' (line 54)
            x_51108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'x', False)
            # Applying the binary operator '*' (line 54)
            result_mul_51109 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 23), '*', n_51107, x_51108)
            
            # Getting the type of 'z' (line 54)
            z_51110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'z', False)
            
            # Call to sin(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'x' (line 54)
            x_51112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'x', False)
            # Processing the call keyword arguments (line 54)
            kwargs_51113 = {}
            # Getting the type of 'sin' (line 54)
            sin_51111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'sin', False)
            # Calling sin(args, kwargs) (line 54)
            sin_call_result_51114 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), sin_51111, *[x_51112], **kwargs_51113)
            
            # Applying the binary operator '*' (line 54)
            result_mul_51115 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 27), '*', z_51110, sin_call_result_51114)
            
            # Applying the binary operator '-' (line 54)
            result_sub_51116 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 23), '-', result_mul_51109, result_mul_51115)
            
            # Processing the call keyword arguments (line 54)
            kwargs_51117 = {}
            # Getting the type of 'cos' (line 54)
            cos_51106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'cos', False)
            # Calling cos(args, kwargs) (line 54)
            cos_call_result_51118 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), cos_51106, *[result_sub_51116], **kwargs_51117)
            
            # Getting the type of 'pi' (line 54)
            pi_51119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'pi')
            # Applying the binary operator 'div' (line 54)
            result_div_51120 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 19), 'div', cos_call_result_51118, pi_51119)
            
            # Assigning a type to the variable 'stypy_return_type' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'stypy_return_type', result_div_51120)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 53)
            stypy_return_type_51121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_51121)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_51121

        # Assigning a type to the variable 'myfunc' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'myfunc', myfunc)
        
        # Assigning a Num to a Name (line 55):
        
        # Assigning a Num to a Name (line 55):
        float_51122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'float')
        # Assigning a type to the variable 'table_val' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'table_val', float_51122)
        
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_51123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_51124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 23), list_51123, int_51124)
        # Adding element type (line 56)
        int_51125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 23), list_51123, int_51125)
        
        # Testing the type of a for loop iterable (line 56)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 8), list_51123)
        # Getting the type of the for loop variable (line 56)
        for_loop_var_51126 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 8), list_51123)
        # Assigning a type to the variable 'miniter' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'miniter', for_loop_var_51126)
        # SSA begins for a for statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 57):
        
        # Assigning a Subscript to a Name (line 57):
        
        # Obtaining the type of the subscript
        int_51127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'int')
        
        # Call to quadrature(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'myfunc' (line 57)
        myfunc_51129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'myfunc', False)
        int_51130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 42), 'int')
        # Getting the type of 'pi' (line 57)
        pi_51131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'pi', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 57)
        tuple_51132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 57)
        # Adding element type (line 57)
        int_51133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 50), tuple_51132, int_51133)
        # Adding element type (line 57)
        float_51134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 50), tuple_51132, float_51134)
        
        # Processing the call keyword arguments (line 57)
        # Getting the type of 'miniter' (line 57)
        miniter_51135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 67), 'miniter', False)
        keyword_51136 = miniter_51135
        kwargs_51137 = {'miniter': keyword_51136}
        # Getting the type of 'quadrature' (line 57)
        quadrature_51128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'quadrature', False)
        # Calling quadrature(args, kwargs) (line 57)
        quadrature_call_result_51138 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), quadrature_51128, *[myfunc_51129, int_51130, pi_51131, tuple_51132], **kwargs_51137)
        
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___51139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), quadrature_call_result_51138, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_51140 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), getitem___51139, int_51127)
        
        # Assigning a type to the variable 'tuple_var_assignment_50862' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'tuple_var_assignment_50862', subscript_call_result_51140)
        
        # Assigning a Subscript to a Name (line 57):
        
        # Obtaining the type of the subscript
        int_51141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'int')
        
        # Call to quadrature(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'myfunc' (line 57)
        myfunc_51143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'myfunc', False)
        int_51144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 42), 'int')
        # Getting the type of 'pi' (line 57)
        pi_51145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'pi', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 57)
        tuple_51146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 57)
        # Adding element type (line 57)
        int_51147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 50), tuple_51146, int_51147)
        # Adding element type (line 57)
        float_51148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 50), tuple_51146, float_51148)
        
        # Processing the call keyword arguments (line 57)
        # Getting the type of 'miniter' (line 57)
        miniter_51149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 67), 'miniter', False)
        keyword_51150 = miniter_51149
        kwargs_51151 = {'miniter': keyword_51150}
        # Getting the type of 'quadrature' (line 57)
        quadrature_51142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'quadrature', False)
        # Calling quadrature(args, kwargs) (line 57)
        quadrature_call_result_51152 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), quadrature_51142, *[myfunc_51143, int_51144, pi_51145, tuple_51146], **kwargs_51151)
        
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___51153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), quadrature_call_result_51152, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_51154 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), getitem___51153, int_51141)
        
        # Assigning a type to the variable 'tuple_var_assignment_50863' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'tuple_var_assignment_50863', subscript_call_result_51154)
        
        # Assigning a Name to a Name (line 57):
        # Getting the type of 'tuple_var_assignment_50862' (line 57)
        tuple_var_assignment_50862_51155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'tuple_var_assignment_50862')
        # Assigning a type to the variable 'val' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'val', tuple_var_assignment_50862_51155)
        
        # Assigning a Name to a Name (line 57):
        # Getting the type of 'tuple_var_assignment_50863' (line 57)
        tuple_var_assignment_50863_51156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'tuple_var_assignment_50863')
        # Assigning a type to the variable 'err' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'err', tuple_var_assignment_50863_51156)
        
        # Call to assert_almost_equal(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'val' (line 58)
        val_51158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'val', False)
        # Getting the type of 'table_val' (line 58)
        table_val_51159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'table_val', False)
        # Processing the call keyword arguments (line 58)
        int_51160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 56), 'int')
        keyword_51161 = int_51160
        kwargs_51162 = {'decimal': keyword_51161}
        # Getting the type of 'assert_almost_equal' (line 58)
        assert_almost_equal_51157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 58)
        assert_almost_equal_call_result_51163 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), assert_almost_equal_51157, *[val_51158, table_val_51159], **kwargs_51162)
        
        
        # Call to assert_(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Getting the type of 'err' (line 59)
        err_51165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'err', False)
        float_51166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'float')
        # Applying the binary operator '<' (line 59)
        result_lt_51167 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 20), '<', err_51165, float_51166)
        
        # Processing the call keyword arguments (line 59)
        kwargs_51168 = {}
        # Getting the type of 'assert_' (line 59)
        assert__51164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 59)
        assert__call_result_51169 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), assert__51164, *[result_lt_51167], **kwargs_51168)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_quadrature_miniter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadrature_miniter' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_51170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51170)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadrature_miniter'
        return stypy_return_type_51170


    @norecursion
    def test_quadrature_single_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadrature_single_args'
        module_type_store = module_type_store.open_function_context('test_quadrature_single_args', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_quadrature_single_args')
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_quadrature_single_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_quadrature_single_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadrature_single_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadrature_single_args(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 62, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'n']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'n'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'n'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            float_51171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'float')
            
            # Call to cos(...): (line 63)
            # Processing the call arguments (line 63)
            # Getting the type of 'n' (line 63)
            n_51173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'n', False)
            # Getting the type of 'x' (line 63)
            x_51174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'x', False)
            # Applying the binary operator '*' (line 63)
            result_mul_51175 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 30), '*', n_51173, x_51174)
            
            float_51176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'float')
            
            # Call to sin(...): (line 63)
            # Processing the call arguments (line 63)
            # Getting the type of 'x' (line 63)
            x_51178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 42), 'x', False)
            # Processing the call keyword arguments (line 63)
            kwargs_51179 = {}
            # Getting the type of 'sin' (line 63)
            sin_51177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 38), 'sin', False)
            # Calling sin(args, kwargs) (line 63)
            sin_call_result_51180 = invoke(stypy.reporting.localization.Localization(__file__, 63, 38), sin_51177, *[x_51178], **kwargs_51179)
            
            # Applying the binary operator '*' (line 63)
            result_mul_51181 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 34), '*', float_51176, sin_call_result_51180)
            
            # Applying the binary operator '-' (line 63)
            result_sub_51182 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 30), '-', result_mul_51175, result_mul_51181)
            
            # Processing the call keyword arguments (line 63)
            kwargs_51183 = {}
            # Getting the type of 'cos' (line 63)
            cos_51172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'cos', False)
            # Calling cos(args, kwargs) (line 63)
            cos_call_result_51184 = invoke(stypy.reporting.localization.Localization(__file__, 63, 26), cos_51172, *[result_sub_51182], **kwargs_51183)
            
            # Applying the binary operator '*' (line 63)
            result_mul_51185 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 19), '*', float_51171, cos_call_result_51184)
            
            # Getting the type of 'pi' (line 63)
            pi_51186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 46), 'pi')
            # Applying the binary operator 'div' (line 63)
            result_div_51187 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 45), 'div', result_mul_51185, pi_51186)
            
            # Assigning a type to the variable 'stypy_return_type' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', result_div_51187)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 62)
            stypy_return_type_51188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_51188)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_51188

        # Assigning a type to the variable 'myfunc' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'myfunc', myfunc)
        
        # Assigning a Call to a Tuple (line 64):
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_51189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to quadrature(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'myfunc' (line 64)
        myfunc_51191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'myfunc', False)
        int_51192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 38), 'int')
        # Getting the type of 'pi' (line 64)
        pi_51193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'pi', False)
        # Processing the call keyword arguments (line 64)
        int_51194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 50), 'int')
        keyword_51195 = int_51194
        float_51196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 58), 'float')
        keyword_51197 = float_51196
        kwargs_51198 = {'rtol': keyword_51197, 'args': keyword_51195}
        # Getting the type of 'quadrature' (line 64)
        quadrature_51190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'quadrature', False)
        # Calling quadrature(args, kwargs) (line 64)
        quadrature_call_result_51199 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), quadrature_51190, *[myfunc_51191, int_51192, pi_51193], **kwargs_51198)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___51200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), quadrature_call_result_51199, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_51201 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___51200, int_51189)
        
        # Assigning a type to the variable 'tuple_var_assignment_50864' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_50864', subscript_call_result_51201)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_51202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to quadrature(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'myfunc' (line 64)
        myfunc_51204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'myfunc', False)
        int_51205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 38), 'int')
        # Getting the type of 'pi' (line 64)
        pi_51206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'pi', False)
        # Processing the call keyword arguments (line 64)
        int_51207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 50), 'int')
        keyword_51208 = int_51207
        float_51209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 58), 'float')
        keyword_51210 = float_51209
        kwargs_51211 = {'rtol': keyword_51210, 'args': keyword_51208}
        # Getting the type of 'quadrature' (line 64)
        quadrature_51203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'quadrature', False)
        # Calling quadrature(args, kwargs) (line 64)
        quadrature_call_result_51212 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), quadrature_51203, *[myfunc_51204, int_51205, pi_51206], **kwargs_51211)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___51213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), quadrature_call_result_51212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_51214 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___51213, int_51202)
        
        # Assigning a type to the variable 'tuple_var_assignment_50865' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_50865', subscript_call_result_51214)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_50864' (line 64)
        tuple_var_assignment_50864_51215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_50864')
        # Assigning a type to the variable 'val' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'val', tuple_var_assignment_50864_51215)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_50865' (line 64)
        tuple_var_assignment_50865_51216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_50865')
        # Assigning a type to the variable 'err' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'err', tuple_var_assignment_50865_51216)
        
        # Assigning a BinOp to a Name (line 65):
        
        # Assigning a BinOp to a Name (line 65):
        float_51217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'float')
        float_51218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 27), 'float')
        # Applying the binary operator '*' (line 65)
        result_mul_51219 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 20), '*', float_51217, float_51218)
        
        # Assigning a type to the variable 'table_val' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'table_val', result_mul_51219)
        
        # Call to assert_allclose(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'val' (line 66)
        val_51221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'val', False)
        # Getting the type of 'table_val' (line 66)
        table_val_51222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'table_val', False)
        # Processing the call keyword arguments (line 66)
        float_51223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 45), 'float')
        keyword_51224 = float_51223
        kwargs_51225 = {'rtol': keyword_51224}
        # Getting the type of 'assert_allclose' (line 66)
        assert_allclose_51220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 66)
        assert_allclose_call_result_51226 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_allclose_51220, *[val_51221, table_val_51222], **kwargs_51225)
        
        
        # ################# End of 'test_quadrature_single_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadrature_single_args' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_51227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadrature_single_args'
        return stypy_return_type_51227


    @norecursion
    def test_romberg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_romberg'
        module_type_store = module_type_store.open_function_context('test_romberg', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_romberg')
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_romberg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_romberg', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_romberg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_romberg(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 70, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'n', 'z']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'n', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'n', 'z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            
            # Call to cos(...): (line 71)
            # Processing the call arguments (line 71)
            # Getting the type of 'n' (line 71)
            n_51229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'n', False)
            # Getting the type of 'x' (line 71)
            x_51230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'x', False)
            # Applying the binary operator '*' (line 71)
            result_mul_51231 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 23), '*', n_51229, x_51230)
            
            # Getting the type of 'z' (line 71)
            z_51232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'z', False)
            
            # Call to sin(...): (line 71)
            # Processing the call arguments (line 71)
            # Getting the type of 'x' (line 71)
            x_51234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 33), 'x', False)
            # Processing the call keyword arguments (line 71)
            kwargs_51235 = {}
            # Getting the type of 'sin' (line 71)
            sin_51233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 29), 'sin', False)
            # Calling sin(args, kwargs) (line 71)
            sin_call_result_51236 = invoke(stypy.reporting.localization.Localization(__file__, 71, 29), sin_51233, *[x_51234], **kwargs_51235)
            
            # Applying the binary operator '*' (line 71)
            result_mul_51237 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 27), '*', z_51232, sin_call_result_51236)
            
            # Applying the binary operator '-' (line 71)
            result_sub_51238 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 23), '-', result_mul_51231, result_mul_51237)
            
            # Processing the call keyword arguments (line 71)
            kwargs_51239 = {}
            # Getting the type of 'cos' (line 71)
            cos_51228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'cos', False)
            # Calling cos(args, kwargs) (line 71)
            cos_call_result_51240 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), cos_51228, *[result_sub_51238], **kwargs_51239)
            
            # Getting the type of 'pi' (line 71)
            pi_51241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'pi')
            # Applying the binary operator 'div' (line 71)
            result_div_51242 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 19), 'div', cos_call_result_51240, pi_51241)
            
            # Assigning a type to the variable 'stypy_return_type' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'stypy_return_type', result_div_51242)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 70)
            stypy_return_type_51243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_51243)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_51243

        # Assigning a type to the variable 'myfunc' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'myfunc', myfunc)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to romberg(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'myfunc' (line 72)
        myfunc_51245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'myfunc', False)
        int_51246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 30), 'int')
        # Getting the type of 'pi' (line 72)
        pi_51247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 33), 'pi', False)
        # Processing the call keyword arguments (line 72)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_51248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        int_51249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 43), tuple_51248, int_51249)
        # Adding element type (line 72)
        float_51250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 43), tuple_51248, float_51250)
        
        keyword_51251 = tuple_51248
        kwargs_51252 = {'args': keyword_51251}
        # Getting the type of 'romberg' (line 72)
        romberg_51244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'romberg', False)
        # Calling romberg(args, kwargs) (line 72)
        romberg_call_result_51253 = invoke(stypy.reporting.localization.Localization(__file__, 72, 14), romberg_51244, *[myfunc_51245, int_51246, pi_51247], **kwargs_51252)
        
        # Assigning a type to the variable 'val' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'val', romberg_call_result_51253)
        
        # Assigning a Num to a Name (line 73):
        
        # Assigning a Num to a Name (line 73):
        float_51254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'float')
        # Assigning a type to the variable 'table_val' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'table_val', float_51254)
        
        # Call to assert_almost_equal(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'val' (line 74)
        val_51256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'val', False)
        # Getting the type of 'table_val' (line 74)
        table_val_51257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 33), 'table_val', False)
        # Processing the call keyword arguments (line 74)
        int_51258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 52), 'int')
        keyword_51259 = int_51258
        kwargs_51260 = {'decimal': keyword_51259}
        # Getting the type of 'assert_almost_equal' (line 74)
        assert_almost_equal_51255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 74)
        assert_almost_equal_call_result_51261 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), assert_almost_equal_51255, *[val_51256, table_val_51257], **kwargs_51260)
        
        
        # ################# End of 'test_romberg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_romberg' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_51262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51262)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_romberg'
        return stypy_return_type_51262


    @norecursion
    def test_romberg_rtol(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_romberg_rtol'
        module_type_store = module_type_store.open_function_context('test_romberg_rtol', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_romberg_rtol')
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_romberg_rtol.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_romberg_rtol', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_romberg_rtol', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_romberg_rtol(...)' code ##################


        @norecursion
        def myfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'myfunc'
            module_type_store = module_type_store.open_function_context('myfunc', 78, 8, False)
            
            # Passed parameters checking function
            myfunc.stypy_localization = localization
            myfunc.stypy_type_of_self = None
            myfunc.stypy_type_store = module_type_store
            myfunc.stypy_function_name = 'myfunc'
            myfunc.stypy_param_names_list = ['x', 'n', 'z']
            myfunc.stypy_varargs_param_name = None
            myfunc.stypy_kwargs_param_name = None
            myfunc.stypy_call_defaults = defaults
            myfunc.stypy_call_varargs = varargs
            myfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'myfunc', ['x', 'n', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'myfunc', localization, ['x', 'n', 'z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'myfunc(...)' code ##################

            float_51263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'float')
            
            # Call to cos(...): (line 79)
            # Processing the call arguments (line 79)
            # Getting the type of 'n' (line 79)
            n_51265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'n', False)
            # Getting the type of 'x' (line 79)
            x_51266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'x', False)
            # Applying the binary operator '*' (line 79)
            result_mul_51267 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 28), '*', n_51265, x_51266)
            
            # Getting the type of 'z' (line 79)
            z_51268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'z', False)
            
            # Call to sin(...): (line 79)
            # Processing the call arguments (line 79)
            # Getting the type of 'x' (line 79)
            x_51270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 38), 'x', False)
            # Processing the call keyword arguments (line 79)
            kwargs_51271 = {}
            # Getting the type of 'sin' (line 79)
            sin_51269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 34), 'sin', False)
            # Calling sin(args, kwargs) (line 79)
            sin_call_result_51272 = invoke(stypy.reporting.localization.Localization(__file__, 79, 34), sin_51269, *[x_51270], **kwargs_51271)
            
            # Applying the binary operator '*' (line 79)
            result_mul_51273 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 32), '*', z_51268, sin_call_result_51272)
            
            # Applying the binary operator '-' (line 79)
            result_sub_51274 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 28), '-', result_mul_51267, result_mul_51273)
            
            # Processing the call keyword arguments (line 79)
            kwargs_51275 = {}
            # Getting the type of 'cos' (line 79)
            cos_51264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'cos', False)
            # Calling cos(args, kwargs) (line 79)
            cos_call_result_51276 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), cos_51264, *[result_sub_51274], **kwargs_51275)
            
            # Applying the binary operator '*' (line 79)
            result_mul_51277 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 19), '*', float_51263, cos_call_result_51276)
            
            # Getting the type of 'pi' (line 79)
            pi_51278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'pi')
            # Applying the binary operator 'div' (line 79)
            result_div_51279 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 41), 'div', result_mul_51277, pi_51278)
            
            # Assigning a type to the variable 'stypy_return_type' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type', result_div_51279)
            
            # ################# End of 'myfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'myfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 78)
            stypy_return_type_51280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_51280)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'myfunc'
            return stypy_return_type_51280

        # Assigning a type to the variable 'myfunc' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'myfunc', myfunc)
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to romberg(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'myfunc' (line 80)
        myfunc_51282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'myfunc', False)
        int_51283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 30), 'int')
        # Getting the type of 'pi' (line 80)
        pi_51284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 'pi', False)
        # Processing the call keyword arguments (line 80)
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_51285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        int_51286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 43), tuple_51285, int_51286)
        # Adding element type (line 80)
        float_51287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 43), tuple_51285, float_51287)
        
        keyword_51288 = tuple_51285
        float_51289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 57), 'float')
        keyword_51290 = float_51289
        kwargs_51291 = {'rtol': keyword_51290, 'args': keyword_51288}
        # Getting the type of 'romberg' (line 80)
        romberg_51281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 14), 'romberg', False)
        # Calling romberg(args, kwargs) (line 80)
        romberg_call_result_51292 = invoke(stypy.reporting.localization.Localization(__file__, 80, 14), romberg_51281, *[myfunc_51282, int_51283, pi_51284], **kwargs_51291)
        
        # Assigning a type to the variable 'val' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'val', romberg_call_result_51292)
        
        # Assigning a BinOp to a Name (line 81):
        
        # Assigning a BinOp to a Name (line 81):
        float_51293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'float')
        float_51294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'float')
        # Applying the binary operator '*' (line 81)
        result_mul_51295 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 20), '*', float_51293, float_51294)
        
        # Assigning a type to the variable 'table_val' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'table_val', result_mul_51295)
        
        # Call to assert_allclose(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'val' (line 82)
        val_51297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'val', False)
        # Getting the type of 'table_val' (line 82)
        table_val_51298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'table_val', False)
        # Processing the call keyword arguments (line 82)
        float_51299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 45), 'float')
        keyword_51300 = float_51299
        kwargs_51301 = {'rtol': keyword_51300}
        # Getting the type of 'assert_allclose' (line 82)
        assert_allclose_51296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 82)
        assert_allclose_call_result_51302 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assert_allclose_51296, *[val_51297, table_val_51298], **kwargs_51301)
        
        
        # ################# End of 'test_romberg_rtol(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_romberg_rtol' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_51303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51303)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_romberg_rtol'
        return stypy_return_type_51303


    @norecursion
    def test_romb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_romb'
        module_type_store = module_type_store.open_function_context('test_romb', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_romb')
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_romb.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_romb', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_romb', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_romb(...)' code ##################

        
        # Call to assert_equal(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to romb(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to arange(...): (line 85)
        # Processing the call arguments (line 85)
        int_51308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 36), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_51309 = {}
        # Getting the type of 'np' (line 85)
        np_51306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'np', False)
        # Obtaining the member 'arange' of a type (line 85)
        arange_51307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 26), np_51306, 'arange')
        # Calling arange(args, kwargs) (line 85)
        arange_call_result_51310 = invoke(stypy.reporting.localization.Localization(__file__, 85, 26), arange_51307, *[int_51308], **kwargs_51309)
        
        # Processing the call keyword arguments (line 85)
        kwargs_51311 = {}
        # Getting the type of 'romb' (line 85)
        romb_51305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'romb', False)
        # Calling romb(args, kwargs) (line 85)
        romb_call_result_51312 = invoke(stypy.reporting.localization.Localization(__file__, 85, 21), romb_51305, *[arange_call_result_51310], **kwargs_51311)
        
        int_51313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 42), 'int')
        # Processing the call keyword arguments (line 85)
        kwargs_51314 = {}
        # Getting the type of 'assert_equal' (line 85)
        assert_equal_51304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 85)
        assert_equal_call_result_51315 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), assert_equal_51304, *[romb_call_result_51312, int_51313], **kwargs_51314)
        
        
        # ################# End of 'test_romb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_romb' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_51316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51316)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_romb'
        return stypy_return_type_51316


    @norecursion
    def test_romb_gh_3731(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_romb_gh_3731'
        module_type_store = module_type_store.open_function_context('test_romb_gh_3731', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_romb_gh_3731')
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_romb_gh_3731.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_romb_gh_3731', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_romb_gh_3731', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_romb_gh_3731(...)' code ##################

        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to arange(...): (line 89)
        # Processing the call arguments (line 89)
        int_51319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'int')
        int_51320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 25), 'int')
        # Applying the binary operator '**' (line 89)
        result_pow_51321 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 22), '**', int_51319, int_51320)
        
        int_51322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 27), 'int')
        # Applying the binary operator '+' (line 89)
        result_add_51323 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 22), '+', result_pow_51321, int_51322)
        
        # Processing the call keyword arguments (line 89)
        kwargs_51324 = {}
        # Getting the type of 'np' (line 89)
        np_51317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 89)
        arange_51318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), np_51317, 'arange')
        # Calling arange(args, kwargs) (line 89)
        arange_call_result_51325 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), arange_51318, *[result_add_51323], **kwargs_51324)
        
        # Assigning a type to the variable 'x' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'x', arange_call_result_51325)
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to cos(...): (line 90)
        # Processing the call arguments (line 90)
        float_51328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'float')
        # Getting the type of 'x' (line 90)
        x_51329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'x', False)
        # Applying the binary operator '*' (line 90)
        result_mul_51330 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), '*', float_51328, x_51329)
        
        # Processing the call keyword arguments (line 90)
        kwargs_51331 = {}
        # Getting the type of 'np' (line 90)
        np_51326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'np', False)
        # Obtaining the member 'cos' of a type (line 90)
        cos_51327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), np_51326, 'cos')
        # Calling cos(args, kwargs) (line 90)
        cos_call_result_51332 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), cos_51327, *[result_mul_51330], **kwargs_51331)
        
        # Assigning a type to the variable 'y' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'y', cos_call_result_51332)
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to romb(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'y' (line 91)
        y_51334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'y', False)
        # Processing the call keyword arguments (line 91)
        kwargs_51335 = {}
        # Getting the type of 'romb' (line 91)
        romb_51333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'romb', False)
        # Calling romb(args, kwargs) (line 91)
        romb_call_result_51336 = invoke(stypy.reporting.localization.Localization(__file__, 91, 14), romb_51333, *[y_51334], **kwargs_51335)
        
        # Assigning a type to the variable 'val' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'val', romb_call_result_51336)
        
        # Assigning a Call to a Tuple (line 92):
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_51337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'int')
        
        # Call to quad(...): (line 92)
        # Processing the call arguments (line 92)

        @norecursion
        def _stypy_temp_lambda_45(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_45'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_45', 92, 25, True)
            # Passed parameters checking function
            _stypy_temp_lambda_45.stypy_localization = localization
            _stypy_temp_lambda_45.stypy_type_of_self = None
            _stypy_temp_lambda_45.stypy_type_store = module_type_store
            _stypy_temp_lambda_45.stypy_function_name = '_stypy_temp_lambda_45'
            _stypy_temp_lambda_45.stypy_param_names_list = ['x']
            _stypy_temp_lambda_45.stypy_varargs_param_name = None
            _stypy_temp_lambda_45.stypy_kwargs_param_name = None
            _stypy_temp_lambda_45.stypy_call_defaults = defaults
            _stypy_temp_lambda_45.stypy_call_varargs = varargs
            _stypy_temp_lambda_45.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_45', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_45', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to cos(...): (line 92)
            # Processing the call arguments (line 92)
            float_51341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 42), 'float')
            # Getting the type of 'x' (line 92)
            x_51342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'x', False)
            # Applying the binary operator '*' (line 92)
            result_mul_51343 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 42), '*', float_51341, x_51342)
            
            # Processing the call keyword arguments (line 92)
            kwargs_51344 = {}
            # Getting the type of 'np' (line 92)
            np_51339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'np', False)
            # Obtaining the member 'cos' of a type (line 92)
            cos_51340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 35), np_51339, 'cos')
            # Calling cos(args, kwargs) (line 92)
            cos_call_result_51345 = invoke(stypy.reporting.localization.Localization(__file__, 92, 35), cos_51340, *[result_mul_51343], **kwargs_51344)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'stypy_return_type', cos_call_result_51345)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_45' in the type store
            # Getting the type of 'stypy_return_type' (line 92)
            stypy_return_type_51346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_51346)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_45'
            return stypy_return_type_51346

        # Assigning a type to the variable '_stypy_temp_lambda_45' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), '_stypy_temp_lambda_45', _stypy_temp_lambda_45)
        # Getting the type of '_stypy_temp_lambda_45' (line 92)
        _stypy_temp_lambda_45_51347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), '_stypy_temp_lambda_45')
        
        # Call to min(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_51350 = {}
        # Getting the type of 'x' (line 92)
        x_51348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 50), 'x', False)
        # Obtaining the member 'min' of a type (line 92)
        min_51349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 50), x_51348, 'min')
        # Calling min(args, kwargs) (line 92)
        min_call_result_51351 = invoke(stypy.reporting.localization.Localization(__file__, 92, 50), min_51349, *[], **kwargs_51350)
        
        
        # Call to max(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_51354 = {}
        # Getting the type of 'x' (line 92)
        x_51352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 59), 'x', False)
        # Obtaining the member 'max' of a type (line 92)
        max_51353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 59), x_51352, 'max')
        # Calling max(args, kwargs) (line 92)
        max_call_result_51355 = invoke(stypy.reporting.localization.Localization(__file__, 92, 59), max_51353, *[], **kwargs_51354)
        
        # Processing the call keyword arguments (line 92)
        kwargs_51356 = {}
        # Getting the type of 'quad' (line 92)
        quad_51338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 92)
        quad_call_result_51357 = invoke(stypy.reporting.localization.Localization(__file__, 92, 20), quad_51338, *[_stypy_temp_lambda_45_51347, min_call_result_51351, max_call_result_51355], **kwargs_51356)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___51358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), quad_call_result_51357, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_51359 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), getitem___51358, int_51337)
        
        # Assigning a type to the variable 'tuple_var_assignment_50866' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_50866', subscript_call_result_51359)
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_51360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'int')
        
        # Call to quad(...): (line 92)
        # Processing the call arguments (line 92)

        @norecursion
        def _stypy_temp_lambda_46(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_46'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_46', 92, 25, True)
            # Passed parameters checking function
            _stypy_temp_lambda_46.stypy_localization = localization
            _stypy_temp_lambda_46.stypy_type_of_self = None
            _stypy_temp_lambda_46.stypy_type_store = module_type_store
            _stypy_temp_lambda_46.stypy_function_name = '_stypy_temp_lambda_46'
            _stypy_temp_lambda_46.stypy_param_names_list = ['x']
            _stypy_temp_lambda_46.stypy_varargs_param_name = None
            _stypy_temp_lambda_46.stypy_kwargs_param_name = None
            _stypy_temp_lambda_46.stypy_call_defaults = defaults
            _stypy_temp_lambda_46.stypy_call_varargs = varargs
            _stypy_temp_lambda_46.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_46', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_46', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to cos(...): (line 92)
            # Processing the call arguments (line 92)
            float_51364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 42), 'float')
            # Getting the type of 'x' (line 92)
            x_51365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 'x', False)
            # Applying the binary operator '*' (line 92)
            result_mul_51366 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 42), '*', float_51364, x_51365)
            
            # Processing the call keyword arguments (line 92)
            kwargs_51367 = {}
            # Getting the type of 'np' (line 92)
            np_51362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'np', False)
            # Obtaining the member 'cos' of a type (line 92)
            cos_51363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 35), np_51362, 'cos')
            # Calling cos(args, kwargs) (line 92)
            cos_call_result_51368 = invoke(stypy.reporting.localization.Localization(__file__, 92, 35), cos_51363, *[result_mul_51366], **kwargs_51367)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'stypy_return_type', cos_call_result_51368)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_46' in the type store
            # Getting the type of 'stypy_return_type' (line 92)
            stypy_return_type_51369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_51369)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_46'
            return stypy_return_type_51369

        # Assigning a type to the variable '_stypy_temp_lambda_46' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), '_stypy_temp_lambda_46', _stypy_temp_lambda_46)
        # Getting the type of '_stypy_temp_lambda_46' (line 92)
        _stypy_temp_lambda_46_51370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), '_stypy_temp_lambda_46')
        
        # Call to min(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_51373 = {}
        # Getting the type of 'x' (line 92)
        x_51371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 50), 'x', False)
        # Obtaining the member 'min' of a type (line 92)
        min_51372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 50), x_51371, 'min')
        # Calling min(args, kwargs) (line 92)
        min_call_result_51374 = invoke(stypy.reporting.localization.Localization(__file__, 92, 50), min_51372, *[], **kwargs_51373)
        
        
        # Call to max(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_51377 = {}
        # Getting the type of 'x' (line 92)
        x_51375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 59), 'x', False)
        # Obtaining the member 'max' of a type (line 92)
        max_51376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 59), x_51375, 'max')
        # Calling max(args, kwargs) (line 92)
        max_call_result_51378 = invoke(stypy.reporting.localization.Localization(__file__, 92, 59), max_51376, *[], **kwargs_51377)
        
        # Processing the call keyword arguments (line 92)
        kwargs_51379 = {}
        # Getting the type of 'quad' (line 92)
        quad_51361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'quad', False)
        # Calling quad(args, kwargs) (line 92)
        quad_call_result_51380 = invoke(stypy.reporting.localization.Localization(__file__, 92, 20), quad_51361, *[_stypy_temp_lambda_46_51370, min_call_result_51374, max_call_result_51378], **kwargs_51379)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___51381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), quad_call_result_51380, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_51382 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), getitem___51381, int_51360)
        
        # Assigning a type to the variable 'tuple_var_assignment_50867' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_50867', subscript_call_result_51382)
        
        # Assigning a Name to a Name (line 92):
        # Getting the type of 'tuple_var_assignment_50866' (line 92)
        tuple_var_assignment_50866_51383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_50866')
        # Assigning a type to the variable 'val2' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'val2', tuple_var_assignment_50866_51383)
        
        # Assigning a Name to a Name (line 92):
        # Getting the type of 'tuple_var_assignment_50867' (line 92)
        tuple_var_assignment_50867_51384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'tuple_var_assignment_50867')
        # Assigning a type to the variable 'err' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'err', tuple_var_assignment_50867_51384)
        
        # Call to assert_allclose(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'val' (line 93)
        val_51386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'val', False)
        # Getting the type of 'val2' (line 93)
        val2_51387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 29), 'val2', False)
        # Processing the call keyword arguments (line 93)
        float_51388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 40), 'float')
        keyword_51389 = float_51388
        int_51390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 51), 'int')
        keyword_51391 = int_51390
        kwargs_51392 = {'rtol': keyword_51389, 'atol': keyword_51391}
        # Getting the type of 'assert_allclose' (line 93)
        assert_allclose_51385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 93)
        assert_allclose_call_result_51393 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), assert_allclose_51385, *[val_51386, val2_51387], **kwargs_51392)
        
        
        # Call to suppress_warnings(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_51395 = {}
        # Getting the type of 'suppress_warnings' (line 96)
        suppress_warnings_51394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 96)
        suppress_warnings_call_result_51396 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), suppress_warnings_51394, *[], **kwargs_51395)
        
        with_51397 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 96, 13), suppress_warnings_call_result_51396, 'with parameter', '__enter__', '__exit__')

        if with_51397:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 96)
            enter___51398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), suppress_warnings_call_result_51396, '__enter__')
            with_enter_51399 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), enter___51398)
            # Assigning a type to the variable 'sup' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'sup', with_enter_51399)
            
            # Call to filter(...): (line 97)
            # Processing the call arguments (line 97)
            # Getting the type of 'AccuracyWarning' (line 97)
            AccuracyWarning_51402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'AccuracyWarning', False)
            str_51403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'str', 'divmax .4. exceeded')
            # Processing the call keyword arguments (line 97)
            kwargs_51404 = {}
            # Getting the type of 'sup' (line 97)
            sup_51400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 97)
            filter_51401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), sup_51400, 'filter')
            # Calling filter(args, kwargs) (line 97)
            filter_call_result_51405 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), filter_51401, *[AccuracyWarning_51402, str_51403], **kwargs_51404)
            
            
            # Assigning a Call to a Name (line 98):
            
            # Assigning a Call to a Name (line 98):
            
            # Call to romberg(...): (line 98)
            # Processing the call arguments (line 98)

            @norecursion
            def _stypy_temp_lambda_47(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_47'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_47', 98, 27, True)
                # Passed parameters checking function
                _stypy_temp_lambda_47.stypy_localization = localization
                _stypy_temp_lambda_47.stypy_type_of_self = None
                _stypy_temp_lambda_47.stypy_type_store = module_type_store
                _stypy_temp_lambda_47.stypy_function_name = '_stypy_temp_lambda_47'
                _stypy_temp_lambda_47.stypy_param_names_list = ['x']
                _stypy_temp_lambda_47.stypy_varargs_param_name = None
                _stypy_temp_lambda_47.stypy_kwargs_param_name = None
                _stypy_temp_lambda_47.stypy_call_defaults = defaults
                _stypy_temp_lambda_47.stypy_call_varargs = varargs
                _stypy_temp_lambda_47.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_47', ['x'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_47', ['x'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to cos(...): (line 98)
                # Processing the call arguments (line 98)
                float_51409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 44), 'float')
                # Getting the type of 'x' (line 98)
                x_51410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 48), 'x', False)
                # Applying the binary operator '*' (line 98)
                result_mul_51411 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 44), '*', float_51409, x_51410)
                
                # Processing the call keyword arguments (line 98)
                kwargs_51412 = {}
                # Getting the type of 'np' (line 98)
                np_51407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'np', False)
                # Obtaining the member 'cos' of a type (line 98)
                cos_51408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 37), np_51407, 'cos')
                # Calling cos(args, kwargs) (line 98)
                cos_call_result_51413 = invoke(stypy.reporting.localization.Localization(__file__, 98, 37), cos_51408, *[result_mul_51411], **kwargs_51412)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 98)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'stypy_return_type', cos_call_result_51413)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_47' in the type store
                # Getting the type of 'stypy_return_type' (line 98)
                stypy_return_type_51414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_51414)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_47'
                return stypy_return_type_51414

            # Assigning a type to the variable '_stypy_temp_lambda_47' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), '_stypy_temp_lambda_47', _stypy_temp_lambda_47)
            # Getting the type of '_stypy_temp_lambda_47' (line 98)
            _stypy_temp_lambda_47_51415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), '_stypy_temp_lambda_47')
            
            # Call to min(...): (line 98)
            # Processing the call keyword arguments (line 98)
            kwargs_51418 = {}
            # Getting the type of 'x' (line 98)
            x_51416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 52), 'x', False)
            # Obtaining the member 'min' of a type (line 98)
            min_51417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 52), x_51416, 'min')
            # Calling min(args, kwargs) (line 98)
            min_call_result_51419 = invoke(stypy.reporting.localization.Localization(__file__, 98, 52), min_51417, *[], **kwargs_51418)
            
            
            # Call to max(...): (line 98)
            # Processing the call keyword arguments (line 98)
            kwargs_51422 = {}
            # Getting the type of 'x' (line 98)
            x_51420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 61), 'x', False)
            # Obtaining the member 'max' of a type (line 98)
            max_51421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 61), x_51420, 'max')
            # Calling max(args, kwargs) (line 98)
            max_call_result_51423 = invoke(stypy.reporting.localization.Localization(__file__, 98, 61), max_51421, *[], **kwargs_51422)
            
            # Processing the call keyword arguments (line 98)
            int_51424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 77), 'int')
            keyword_51425 = int_51424
            kwargs_51426 = {'divmax': keyword_51425}
            # Getting the type of 'romberg' (line 98)
            romberg_51406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 19), 'romberg', False)
            # Calling romberg(args, kwargs) (line 98)
            romberg_call_result_51427 = invoke(stypy.reporting.localization.Localization(__file__, 98, 19), romberg_51406, *[_stypy_temp_lambda_47_51415, min_call_result_51419, max_call_result_51423], **kwargs_51426)
            
            # Assigning a type to the variable 'val3' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'val3', romberg_call_result_51427)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 96)
            exit___51428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), suppress_warnings_call_result_51396, '__exit__')
            with_exit_51429 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), exit___51428, None, None, None)

        
        # Call to assert_allclose(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'val' (line 99)
        val_51431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'val', False)
        # Getting the type of 'val3' (line 99)
        val3_51432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'val3', False)
        # Processing the call keyword arguments (line 99)
        float_51433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'float')
        keyword_51434 = float_51433
        int_51435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 52), 'int')
        keyword_51436 = int_51435
        kwargs_51437 = {'rtol': keyword_51434, 'atol': keyword_51436}
        # Getting the type of 'assert_allclose' (line 99)
        assert_allclose_51430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 99)
        assert_allclose_call_result_51438 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assert_allclose_51430, *[val_51431, val3_51432], **kwargs_51437)
        
        
        # ################# End of 'test_romb_gh_3731(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_romb_gh_3731' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_51439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51439)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_romb_gh_3731'
        return stypy_return_type_51439


    @norecursion
    def test_non_dtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_non_dtype'
        module_type_store = module_type_store.open_function_context('test_non_dtype', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_non_dtype')
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_non_dtype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_non_dtype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_non_dtype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_non_dtype(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 103, 8))
        
        # 'import math' statement (line 103)
        import math

        import_module(stypy.reporting.localization.Localization(__file__, 103, 8), 'math', math, module_type_store)
        
        
        # Assigning a Call to a Name (line 104):
        
        # Assigning a Call to a Name (line 104):
        
        # Call to romberg(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'math' (line 104)
        math_51441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'math', False)
        # Obtaining the member 'sin' of a type (line 104)
        sin_51442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 26), math_51441, 'sin')
        int_51443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
        int_51444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 39), 'int')
        # Processing the call keyword arguments (line 104)
        kwargs_51445 = {}
        # Getting the type of 'romberg' (line 104)
        romberg_51440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'romberg', False)
        # Calling romberg(args, kwargs) (line 104)
        romberg_call_result_51446 = invoke(stypy.reporting.localization.Localization(__file__, 104, 18), romberg_51440, *[sin_51442, int_51443, int_51444], **kwargs_51445)
        
        # Assigning a type to the variable 'valmath' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'valmath', romberg_call_result_51446)
        
        # Assigning a Num to a Name (line 105):
        
        # Assigning a Num to a Name (line 105):
        float_51447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'float')
        # Assigning a type to the variable 'expected_val' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'expected_val', float_51447)
        
        # Call to assert_almost_equal(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'valmath' (line 106)
        valmath_51449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'valmath', False)
        # Getting the type of 'expected_val' (line 106)
        expected_val_51450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 37), 'expected_val', False)
        # Processing the call keyword arguments (line 106)
        int_51451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 59), 'int')
        keyword_51452 = int_51451
        kwargs_51453 = {'decimal': keyword_51452}
        # Getting the type of 'assert_almost_equal' (line 106)
        assert_almost_equal_51448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 106)
        assert_almost_equal_call_result_51454 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_almost_equal_51448, *[valmath_51449, expected_val_51450], **kwargs_51453)
        
        
        # ################# End of 'test_non_dtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_non_dtype' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_51455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51455)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_non_dtype'
        return stypy_return_type_51455


    @norecursion
    def test_newton_cotes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_newton_cotes'
        module_type_store = module_type_store.open_function_context('test_newton_cotes', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_newton_cotes')
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_newton_cotes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_newton_cotes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_newton_cotes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_newton_cotes(...)' code ##################

        str_51456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'str', 'Test the first few degrees, for evenly spaced points.')
        
        # Assigning a Num to a Name (line 110):
        
        # Assigning a Num to a Name (line 110):
        int_51457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 12), 'int')
        # Assigning a type to the variable 'n' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'n', int_51457)
        
        # Assigning a Call to a Tuple (line 111):
        
        # Assigning a Subscript to a Name (line 111):
        
        # Obtaining the type of the subscript
        int_51458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 8), 'int')
        
        # Call to newton_cotes(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'n' (line 111)
        n_51460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'n', False)
        int_51461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 39), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_51462 = {}
        # Getting the type of 'newton_cotes' (line 111)
        newton_cotes_51459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 111)
        newton_cotes_call_result_51463 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), newton_cotes_51459, *[n_51460, int_51461], **kwargs_51462)
        
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___51464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), newton_cotes_call_result_51463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_51465 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), getitem___51464, int_51458)
        
        # Assigning a type to the variable 'tuple_var_assignment_50868' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'tuple_var_assignment_50868', subscript_call_result_51465)
        
        # Assigning a Subscript to a Name (line 111):
        
        # Obtaining the type of the subscript
        int_51466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 8), 'int')
        
        # Call to newton_cotes(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'n' (line 111)
        n_51468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'n', False)
        int_51469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 39), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_51470 = {}
        # Getting the type of 'newton_cotes' (line 111)
        newton_cotes_51467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 111)
        newton_cotes_call_result_51471 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), newton_cotes_51467, *[n_51468, int_51469], **kwargs_51470)
        
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___51472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), newton_cotes_call_result_51471, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_51473 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), getitem___51472, int_51466)
        
        # Assigning a type to the variable 'tuple_var_assignment_50869' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'tuple_var_assignment_50869', subscript_call_result_51473)
        
        # Assigning a Name to a Name (line 111):
        # Getting the type of 'tuple_var_assignment_50868' (line 111)
        tuple_var_assignment_50868_51474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'tuple_var_assignment_50868')
        # Assigning a type to the variable 'wts' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'wts', tuple_var_assignment_50868_51474)
        
        # Assigning a Name to a Name (line 111):
        # Getting the type of 'tuple_var_assignment_50869' (line 111)
        tuple_var_assignment_50869_51475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'tuple_var_assignment_50869')
        # Assigning a type to the variable 'errcoff' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'errcoff', tuple_var_assignment_50869_51475)
        
        # Call to assert_equal(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'wts' (line 112)
        wts_51477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'wts', False)
        # Getting the type of 'n' (line 112)
        n_51478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'n', False)
        
        # Call to array(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_51481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        float_51482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 37), list_51481, float_51482)
        # Adding element type (line 112)
        float_51483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 37), list_51481, float_51483)
        
        # Processing the call keyword arguments (line 112)
        kwargs_51484 = {}
        # Getting the type of 'np' (line 112)
        np_51479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'np', False)
        # Obtaining the member 'array' of a type (line 112)
        array_51480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), np_51479, 'array')
        # Calling array(args, kwargs) (line 112)
        array_call_result_51485 = invoke(stypy.reporting.localization.Localization(__file__, 112, 28), array_51480, *[list_51481], **kwargs_51484)
        
        # Applying the binary operator '*' (line 112)
        result_mul_51486 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 26), '*', n_51478, array_call_result_51485)
        
        # Processing the call keyword arguments (line 112)
        kwargs_51487 = {}
        # Getting the type of 'assert_equal' (line 112)
        assert_equal_51476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 112)
        assert_equal_call_result_51488 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert_equal_51476, *[wts_51477, result_mul_51486], **kwargs_51487)
        
        
        # Call to assert_almost_equal(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'errcoff' (line 113)
        errcoff_51490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'errcoff', False)
        
        # Getting the type of 'n' (line 113)
        n_51491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 38), 'n', False)
        int_51492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 41), 'int')
        # Applying the binary operator '**' (line 113)
        result_pow_51493 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 38), '**', n_51491, int_51492)
        
        # Applying the 'usub' unary operator (line 113)
        result___neg___51494 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 37), 'usub', result_pow_51493)
        
        float_51495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 43), 'float')
        # Applying the binary operator 'div' (line 113)
        result_div_51496 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 37), 'div', result___neg___51494, float_51495)
        
        # Processing the call keyword arguments (line 113)
        kwargs_51497 = {}
        # Getting the type of 'assert_almost_equal' (line 113)
        assert_almost_equal_51489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 113)
        assert_almost_equal_call_result_51498 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_almost_equal_51489, *[errcoff_51490, result_div_51496], **kwargs_51497)
        
        
        # Assigning a Num to a Name (line 115):
        
        # Assigning a Num to a Name (line 115):
        int_51499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
        # Assigning a type to the variable 'n' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'n', int_51499)
        
        # Assigning a Call to a Tuple (line 116):
        
        # Assigning a Subscript to a Name (line 116):
        
        # Obtaining the type of the subscript
        int_51500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'int')
        
        # Call to newton_cotes(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'n' (line 116)
        n_51502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 36), 'n', False)
        int_51503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 39), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_51504 = {}
        # Getting the type of 'newton_cotes' (line 116)
        newton_cotes_51501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 116)
        newton_cotes_call_result_51505 = invoke(stypy.reporting.localization.Localization(__file__, 116, 23), newton_cotes_51501, *[n_51502, int_51503], **kwargs_51504)
        
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___51506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), newton_cotes_call_result_51505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_51507 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), getitem___51506, int_51500)
        
        # Assigning a type to the variable 'tuple_var_assignment_50870' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_50870', subscript_call_result_51507)
        
        # Assigning a Subscript to a Name (line 116):
        
        # Obtaining the type of the subscript
        int_51508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'int')
        
        # Call to newton_cotes(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'n' (line 116)
        n_51510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 36), 'n', False)
        int_51511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 39), 'int')
        # Processing the call keyword arguments (line 116)
        kwargs_51512 = {}
        # Getting the type of 'newton_cotes' (line 116)
        newton_cotes_51509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 116)
        newton_cotes_call_result_51513 = invoke(stypy.reporting.localization.Localization(__file__, 116, 23), newton_cotes_51509, *[n_51510, int_51511], **kwargs_51512)
        
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___51514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), newton_cotes_call_result_51513, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_51515 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), getitem___51514, int_51508)
        
        # Assigning a type to the variable 'tuple_var_assignment_50871' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_50871', subscript_call_result_51515)
        
        # Assigning a Name to a Name (line 116):
        # Getting the type of 'tuple_var_assignment_50870' (line 116)
        tuple_var_assignment_50870_51516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_50870')
        # Assigning a type to the variable 'wts' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'wts', tuple_var_assignment_50870_51516)
        
        # Assigning a Name to a Name (line 116):
        # Getting the type of 'tuple_var_assignment_50871' (line 116)
        tuple_var_assignment_50871_51517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'tuple_var_assignment_50871')
        # Assigning a type to the variable 'errcoff' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'errcoff', tuple_var_assignment_50871_51517)
        
        # Call to assert_almost_equal(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'wts' (line 117)
        wts_51519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'wts', False)
        # Getting the type of 'n' (line 117)
        n_51520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'n', False)
        
        # Call to array(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_51523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        float_51524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 44), list_51523, float_51524)
        # Adding element type (line 117)
        float_51525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 44), list_51523, float_51525)
        # Adding element type (line 117)
        float_51526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 44), list_51523, float_51526)
        
        # Processing the call keyword arguments (line 117)
        kwargs_51527 = {}
        # Getting the type of 'np' (line 117)
        np_51521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 117)
        array_51522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 35), np_51521, 'array')
        # Calling array(args, kwargs) (line 117)
        array_call_result_51528 = invoke(stypy.reporting.localization.Localization(__file__, 117, 35), array_51522, *[list_51523], **kwargs_51527)
        
        # Applying the binary operator '*' (line 117)
        result_mul_51529 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 33), '*', n_51520, array_call_result_51528)
        
        float_51530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 61), 'float')
        # Applying the binary operator 'div' (line 117)
        result_div_51531 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 60), 'div', result_mul_51529, float_51530)
        
        # Processing the call keyword arguments (line 117)
        kwargs_51532 = {}
        # Getting the type of 'assert_almost_equal' (line 117)
        assert_almost_equal_51518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 117)
        assert_almost_equal_call_result_51533 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), assert_almost_equal_51518, *[wts_51519, result_div_51531], **kwargs_51532)
        
        
        # Call to assert_almost_equal(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'errcoff' (line 118)
        errcoff_51535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'errcoff', False)
        
        # Getting the type of 'n' (line 118)
        n_51536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'n', False)
        int_51537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 41), 'int')
        # Applying the binary operator '**' (line 118)
        result_pow_51538 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 38), '**', n_51536, int_51537)
        
        # Applying the 'usub' unary operator (line 118)
        result___neg___51539 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 37), 'usub', result_pow_51538)
        
        float_51540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 43), 'float')
        # Applying the binary operator 'div' (line 118)
        result_div_51541 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 37), 'div', result___neg___51539, float_51540)
        
        # Processing the call keyword arguments (line 118)
        kwargs_51542 = {}
        # Getting the type of 'assert_almost_equal' (line 118)
        assert_almost_equal_51534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 118)
        assert_almost_equal_call_result_51543 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), assert_almost_equal_51534, *[errcoff_51535, result_div_51541], **kwargs_51542)
        
        
        # Assigning a Num to a Name (line 120):
        
        # Assigning a Num to a Name (line 120):
        int_51544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
        # Assigning a type to the variable 'n' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'n', int_51544)
        
        # Assigning a Call to a Tuple (line 121):
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_51545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'int')
        
        # Call to newton_cotes(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'n' (line 121)
        n_51547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 36), 'n', False)
        int_51548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'int')
        # Processing the call keyword arguments (line 121)
        kwargs_51549 = {}
        # Getting the type of 'newton_cotes' (line 121)
        newton_cotes_51546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 121)
        newton_cotes_call_result_51550 = invoke(stypy.reporting.localization.Localization(__file__, 121, 23), newton_cotes_51546, *[n_51547, int_51548], **kwargs_51549)
        
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___51551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), newton_cotes_call_result_51550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_51552 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), getitem___51551, int_51545)
        
        # Assigning a type to the variable 'tuple_var_assignment_50872' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_50872', subscript_call_result_51552)
        
        # Assigning a Subscript to a Name (line 121):
        
        # Obtaining the type of the subscript
        int_51553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 8), 'int')
        
        # Call to newton_cotes(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'n' (line 121)
        n_51555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 36), 'n', False)
        int_51556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'int')
        # Processing the call keyword arguments (line 121)
        kwargs_51557 = {}
        # Getting the type of 'newton_cotes' (line 121)
        newton_cotes_51554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 121)
        newton_cotes_call_result_51558 = invoke(stypy.reporting.localization.Localization(__file__, 121, 23), newton_cotes_51554, *[n_51555, int_51556], **kwargs_51557)
        
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___51559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), newton_cotes_call_result_51558, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_51560 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), getitem___51559, int_51553)
        
        # Assigning a type to the variable 'tuple_var_assignment_50873' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_50873', subscript_call_result_51560)
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'tuple_var_assignment_50872' (line 121)
        tuple_var_assignment_50872_51561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_50872')
        # Assigning a type to the variable 'wts' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'wts', tuple_var_assignment_50872_51561)
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'tuple_var_assignment_50873' (line 121)
        tuple_var_assignment_50873_51562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'tuple_var_assignment_50873')
        # Assigning a type to the variable 'errcoff' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'errcoff', tuple_var_assignment_50873_51562)
        
        # Call to assert_almost_equal(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'wts' (line 122)
        wts_51564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'wts', False)
        # Getting the type of 'n' (line 122)
        n_51565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 33), 'n', False)
        
        # Call to array(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_51568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        float_51569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 44), list_51568, float_51569)
        # Adding element type (line 122)
        float_51570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 44), list_51568, float_51570)
        # Adding element type (line 122)
        float_51571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 44), list_51568, float_51571)
        # Adding element type (line 122)
        float_51572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 44), list_51568, float_51572)
        
        # Processing the call keyword arguments (line 122)
        kwargs_51573 = {}
        # Getting the type of 'np' (line 122)
        np_51566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 122)
        array_51567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 35), np_51566, 'array')
        # Calling array(args, kwargs) (line 122)
        array_call_result_51574 = invoke(stypy.reporting.localization.Localization(__file__, 122, 35), array_51567, *[list_51568], **kwargs_51573)
        
        # Applying the binary operator '*' (line 122)
        result_mul_51575 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 33), '*', n_51565, array_call_result_51574)
        
        float_51576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 66), 'float')
        # Applying the binary operator 'div' (line 122)
        result_div_51577 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 65), 'div', result_mul_51575, float_51576)
        
        # Processing the call keyword arguments (line 122)
        kwargs_51578 = {}
        # Getting the type of 'assert_almost_equal' (line 122)
        assert_almost_equal_51563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 122)
        assert_almost_equal_call_result_51579 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assert_almost_equal_51563, *[wts_51564, result_div_51577], **kwargs_51578)
        
        
        # Call to assert_almost_equal(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'errcoff' (line 123)
        errcoff_51581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'errcoff', False)
        
        # Getting the type of 'n' (line 123)
        n_51582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'n', False)
        int_51583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'int')
        # Applying the binary operator '**' (line 123)
        result_pow_51584 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 38), '**', n_51582, int_51583)
        
        # Applying the 'usub' unary operator (line 123)
        result___neg___51585 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 37), 'usub', result_pow_51584)
        
        float_51586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 43), 'float')
        # Applying the binary operator 'div' (line 123)
        result_div_51587 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 37), 'div', result___neg___51585, float_51586)
        
        # Processing the call keyword arguments (line 123)
        kwargs_51588 = {}
        # Getting the type of 'assert_almost_equal' (line 123)
        assert_almost_equal_51580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 123)
        assert_almost_equal_call_result_51589 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), assert_almost_equal_51580, *[errcoff_51581, result_div_51587], **kwargs_51588)
        
        
        # Assigning a Num to a Name (line 125):
        
        # Assigning a Num to a Name (line 125):
        int_51590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 12), 'int')
        # Assigning a type to the variable 'n' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'n', int_51590)
        
        # Assigning a Call to a Tuple (line 126):
        
        # Assigning a Subscript to a Name (line 126):
        
        # Obtaining the type of the subscript
        int_51591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 8), 'int')
        
        # Call to newton_cotes(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'n' (line 126)
        n_51593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'n', False)
        int_51594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 39), 'int')
        # Processing the call keyword arguments (line 126)
        kwargs_51595 = {}
        # Getting the type of 'newton_cotes' (line 126)
        newton_cotes_51592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 126)
        newton_cotes_call_result_51596 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), newton_cotes_51592, *[n_51593, int_51594], **kwargs_51595)
        
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___51597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), newton_cotes_call_result_51596, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_51598 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), getitem___51597, int_51591)
        
        # Assigning a type to the variable 'tuple_var_assignment_50874' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'tuple_var_assignment_50874', subscript_call_result_51598)
        
        # Assigning a Subscript to a Name (line 126):
        
        # Obtaining the type of the subscript
        int_51599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 8), 'int')
        
        # Call to newton_cotes(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'n' (line 126)
        n_51601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'n', False)
        int_51602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 39), 'int')
        # Processing the call keyword arguments (line 126)
        kwargs_51603 = {}
        # Getting the type of 'newton_cotes' (line 126)
        newton_cotes_51600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 126)
        newton_cotes_call_result_51604 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), newton_cotes_51600, *[n_51601, int_51602], **kwargs_51603)
        
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___51605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), newton_cotes_call_result_51604, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_51606 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), getitem___51605, int_51599)
        
        # Assigning a type to the variable 'tuple_var_assignment_50875' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'tuple_var_assignment_50875', subscript_call_result_51606)
        
        # Assigning a Name to a Name (line 126):
        # Getting the type of 'tuple_var_assignment_50874' (line 126)
        tuple_var_assignment_50874_51607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'tuple_var_assignment_50874')
        # Assigning a type to the variable 'wts' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'wts', tuple_var_assignment_50874_51607)
        
        # Assigning a Name to a Name (line 126):
        # Getting the type of 'tuple_var_assignment_50875' (line 126)
        tuple_var_assignment_50875_51608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'tuple_var_assignment_50875')
        # Assigning a type to the variable 'errcoff' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'errcoff', tuple_var_assignment_50875_51608)
        
        # Call to assert_almost_equal(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'wts' (line 127)
        wts_51610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'wts', False)
        # Getting the type of 'n' (line 127)
        n_51611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'n', False)
        
        # Call to array(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_51614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        float_51615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), list_51614, float_51615)
        # Adding element type (line 127)
        float_51616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), list_51614, float_51616)
        # Adding element type (line 127)
        float_51617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), list_51614, float_51617)
        # Adding element type (line 127)
        float_51618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), list_51614, float_51618)
        # Adding element type (line 127)
        float_51619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), list_51614, float_51619)
        
        # Processing the call keyword arguments (line 127)
        kwargs_51620 = {}
        # Getting the type of 'np' (line 127)
        np_51612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 127)
        array_51613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 35), np_51612, 'array')
        # Calling array(args, kwargs) (line 127)
        array_call_result_51621 = invoke(stypy.reporting.localization.Localization(__file__, 127, 35), array_51613, *[list_51614], **kwargs_51620)
        
        # Applying the binary operator '*' (line 127)
        result_mul_51622 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 33), '*', n_51611, array_call_result_51621)
        
        float_51623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 74), 'float')
        # Applying the binary operator 'div' (line 127)
        result_div_51624 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 73), 'div', result_mul_51622, float_51623)
        
        # Processing the call keyword arguments (line 127)
        kwargs_51625 = {}
        # Getting the type of 'assert_almost_equal' (line 127)
        assert_almost_equal_51609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 127)
        assert_almost_equal_call_result_51626 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), assert_almost_equal_51609, *[wts_51610, result_div_51624], **kwargs_51625)
        
        
        # Call to assert_almost_equal(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'errcoff' (line 128)
        errcoff_51628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'errcoff', False)
        
        # Getting the type of 'n' (line 128)
        n_51629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'n', False)
        int_51630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 41), 'int')
        # Applying the binary operator '**' (line 128)
        result_pow_51631 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 38), '**', n_51629, int_51630)
        
        # Applying the 'usub' unary operator (line 128)
        result___neg___51632 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 37), 'usub', result_pow_51631)
        
        float_51633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 43), 'float')
        # Applying the binary operator 'div' (line 128)
        result_div_51634 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 37), 'div', result___neg___51632, float_51633)
        
        # Processing the call keyword arguments (line 128)
        kwargs_51635 = {}
        # Getting the type of 'assert_almost_equal' (line 128)
        assert_almost_equal_51627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 128)
        assert_almost_equal_call_result_51636 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), assert_almost_equal_51627, *[errcoff_51628, result_div_51634], **kwargs_51635)
        
        
        # ################# End of 'test_newton_cotes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_newton_cotes' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_51637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51637)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_newton_cotes'
        return stypy_return_type_51637


    @norecursion
    def test_newton_cotes2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_newton_cotes2'
        module_type_store = module_type_store.open_function_context('test_newton_cotes2', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_newton_cotes2')
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_newton_cotes2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_newton_cotes2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_newton_cotes2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_newton_cotes2(...)' code ##################

        str_51638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 8), 'str', 'Test newton_cotes with points that are not evenly spaced.')
        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to array(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_51641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        float_51642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_51641, float_51642)
        # Adding element type (line 133)
        float_51643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_51641, float_51643)
        # Adding element type (line 133)
        float_51644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_51641, float_51644)
        
        # Processing the call keyword arguments (line 133)
        kwargs_51645 = {}
        # Getting the type of 'np' (line 133)
        np_51639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 133)
        array_51640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), np_51639, 'array')
        # Calling array(args, kwargs) (line 133)
        array_call_result_51646 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), array_51640, *[list_51641], **kwargs_51645)
        
        # Assigning a type to the variable 'x' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'x', array_call_result_51646)
        
        # Assigning a BinOp to a Name (line 134):
        
        # Assigning a BinOp to a Name (line 134):
        # Getting the type of 'x' (line 134)
        x_51647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'x')
        int_51648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 15), 'int')
        # Applying the binary operator '**' (line 134)
        result_pow_51649 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 12), '**', x_51647, int_51648)
        
        # Assigning a type to the variable 'y' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'y', result_pow_51649)
        
        # Assigning a Call to a Tuple (line 135):
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_51650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to newton_cotes(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'x' (line 135)
        x_51652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'x', False)
        # Processing the call keyword arguments (line 135)
        kwargs_51653 = {}
        # Getting the type of 'newton_cotes' (line 135)
        newton_cotes_51651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 135)
        newton_cotes_call_result_51654 = invoke(stypy.reporting.localization.Localization(__file__, 135, 23), newton_cotes_51651, *[x_51652], **kwargs_51653)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___51655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), newton_cotes_call_result_51654, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_51656 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___51655, int_51650)
        
        # Assigning a type to the variable 'tuple_var_assignment_50876' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_50876', subscript_call_result_51656)
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        int_51657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
        
        # Call to newton_cotes(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'x' (line 135)
        x_51659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'x', False)
        # Processing the call keyword arguments (line 135)
        kwargs_51660 = {}
        # Getting the type of 'newton_cotes' (line 135)
        newton_cotes_51658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 135)
        newton_cotes_call_result_51661 = invoke(stypy.reporting.localization.Localization(__file__, 135, 23), newton_cotes_51658, *[x_51659], **kwargs_51660)
        
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___51662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), newton_cotes_call_result_51661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_51663 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___51662, int_51657)
        
        # Assigning a type to the variable 'tuple_var_assignment_50877' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_50877', subscript_call_result_51663)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_50876' (line 135)
        tuple_var_assignment_50876_51664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_50876')
        # Assigning a type to the variable 'wts' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'wts', tuple_var_assignment_50876_51664)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'tuple_var_assignment_50877' (line 135)
        tuple_var_assignment_50877_51665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_50877')
        # Assigning a type to the variable 'errcoff' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'errcoff', tuple_var_assignment_50877_51665)
        
        # Assigning a BinOp to a Name (line 136):
        
        # Assigning a BinOp to a Name (line 136):
        float_51666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 25), 'float')
        int_51667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'int')
        # Applying the binary operator 'div' (line 136)
        result_div_51668 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 25), 'div', float_51666, int_51667)
        
        # Assigning a type to the variable 'exact_integral' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'exact_integral', result_div_51668)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to dot(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'wts' (line 137)
        wts_51671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'wts', False)
        # Getting the type of 'y' (line 137)
        y_51672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'y', False)
        # Processing the call keyword arguments (line 137)
        kwargs_51673 = {}
        # Getting the type of 'np' (line 137)
        np_51669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'np', False)
        # Obtaining the member 'dot' of a type (line 137)
        dot_51670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 27), np_51669, 'dot')
        # Calling dot(args, kwargs) (line 137)
        dot_call_result_51674 = invoke(stypy.reporting.localization.Localization(__file__, 137, 27), dot_51670, *[wts_51671, y_51672], **kwargs_51673)
        
        # Assigning a type to the variable 'numeric_integral' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'numeric_integral', dot_call_result_51674)
        
        # Call to assert_almost_equal(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'numeric_integral' (line 138)
        numeric_integral_51676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'numeric_integral', False)
        # Getting the type of 'exact_integral' (line 138)
        exact_integral_51677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 46), 'exact_integral', False)
        # Processing the call keyword arguments (line 138)
        kwargs_51678 = {}
        # Getting the type of 'assert_almost_equal' (line 138)
        assert_almost_equal_51675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 138)
        assert_almost_equal_call_result_51679 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), assert_almost_equal_51675, *[numeric_integral_51676, exact_integral_51677], **kwargs_51678)
        
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to array(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_51682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        float_51683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), list_51682, float_51683)
        # Adding element type (line 140)
        float_51684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), list_51682, float_51684)
        # Adding element type (line 140)
        float_51685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), list_51682, float_51685)
        # Adding element type (line 140)
        float_51686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), list_51682, float_51686)
        
        # Processing the call keyword arguments (line 140)
        kwargs_51687 = {}
        # Getting the type of 'np' (line 140)
        np_51680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 140)
        array_51681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), np_51680, 'array')
        # Calling array(args, kwargs) (line 140)
        array_call_result_51688 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), array_51681, *[list_51682], **kwargs_51687)
        
        # Assigning a type to the variable 'x' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'x', array_call_result_51688)
        
        # Assigning a BinOp to a Name (line 141):
        
        # Assigning a BinOp to a Name (line 141):
        # Getting the type of 'x' (line 141)
        x_51689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'x')
        int_51690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'int')
        # Applying the binary operator '**' (line 141)
        result_pow_51691 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 12), '**', x_51689, int_51690)
        
        # Assigning a type to the variable 'y' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'y', result_pow_51691)
        
        # Assigning a Call to a Tuple (line 142):
        
        # Assigning a Subscript to a Name (line 142):
        
        # Obtaining the type of the subscript
        int_51692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 8), 'int')
        
        # Call to newton_cotes(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'x' (line 142)
        x_51694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 36), 'x', False)
        # Processing the call keyword arguments (line 142)
        kwargs_51695 = {}
        # Getting the type of 'newton_cotes' (line 142)
        newton_cotes_51693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 142)
        newton_cotes_call_result_51696 = invoke(stypy.reporting.localization.Localization(__file__, 142, 23), newton_cotes_51693, *[x_51694], **kwargs_51695)
        
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___51697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), newton_cotes_call_result_51696, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_51698 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), getitem___51697, int_51692)
        
        # Assigning a type to the variable 'tuple_var_assignment_50878' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'tuple_var_assignment_50878', subscript_call_result_51698)
        
        # Assigning a Subscript to a Name (line 142):
        
        # Obtaining the type of the subscript
        int_51699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 8), 'int')
        
        # Call to newton_cotes(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'x' (line 142)
        x_51701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 36), 'x', False)
        # Processing the call keyword arguments (line 142)
        kwargs_51702 = {}
        # Getting the type of 'newton_cotes' (line 142)
        newton_cotes_51700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 23), 'newton_cotes', False)
        # Calling newton_cotes(args, kwargs) (line 142)
        newton_cotes_call_result_51703 = invoke(stypy.reporting.localization.Localization(__file__, 142, 23), newton_cotes_51700, *[x_51701], **kwargs_51702)
        
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___51704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), newton_cotes_call_result_51703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_51705 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), getitem___51704, int_51699)
        
        # Assigning a type to the variable 'tuple_var_assignment_50879' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'tuple_var_assignment_50879', subscript_call_result_51705)
        
        # Assigning a Name to a Name (line 142):
        # Getting the type of 'tuple_var_assignment_50878' (line 142)
        tuple_var_assignment_50878_51706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'tuple_var_assignment_50878')
        # Assigning a type to the variable 'wts' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'wts', tuple_var_assignment_50878_51706)
        
        # Assigning a Name to a Name (line 142):
        # Getting the type of 'tuple_var_assignment_50879' (line 142)
        tuple_var_assignment_50879_51707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'tuple_var_assignment_50879')
        # Assigning a type to the variable 'errcoff' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'errcoff', tuple_var_assignment_50879_51707)
        
        # Assigning a Num to a Name (line 143):
        
        # Assigning a Num to a Name (line 143):
        float_51708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 25), 'float')
        # Assigning a type to the variable 'exact_integral' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'exact_integral', float_51708)
        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to dot(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'wts' (line 144)
        wts_51711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'wts', False)
        # Getting the type of 'y' (line 144)
        y_51712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 39), 'y', False)
        # Processing the call keyword arguments (line 144)
        kwargs_51713 = {}
        # Getting the type of 'np' (line 144)
        np_51709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'np', False)
        # Obtaining the member 'dot' of a type (line 144)
        dot_51710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 27), np_51709, 'dot')
        # Calling dot(args, kwargs) (line 144)
        dot_call_result_51714 = invoke(stypy.reporting.localization.Localization(__file__, 144, 27), dot_51710, *[wts_51711, y_51712], **kwargs_51713)
        
        # Assigning a type to the variable 'numeric_integral' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'numeric_integral', dot_call_result_51714)
        
        # Call to assert_almost_equal(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'numeric_integral' (line 145)
        numeric_integral_51716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'numeric_integral', False)
        # Getting the type of 'exact_integral' (line 145)
        exact_integral_51717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 46), 'exact_integral', False)
        # Processing the call keyword arguments (line 145)
        kwargs_51718 = {}
        # Getting the type of 'assert_almost_equal' (line 145)
        assert_almost_equal_51715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 145)
        assert_almost_equal_call_result_51719 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), assert_almost_equal_51715, *[numeric_integral_51716, exact_integral_51717], **kwargs_51718)
        
        
        # ################# End of 'test_newton_cotes2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_newton_cotes2' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_51720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51720)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_newton_cotes2'
        return stypy_return_type_51720


    @norecursion
    def test_simps(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simps'
        module_type_store = module_type_store.open_function_context('test_simps', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_localization', localization)
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_function_name', 'TestQuadrature.test_simps')
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_param_names_list', [])
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestQuadrature.test_simps.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.test_simps', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simps', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simps(...)' code ##################

        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to arange(...): (line 148)
        # Processing the call arguments (line 148)
        int_51723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 22), 'int')
        # Processing the call keyword arguments (line 148)
        kwargs_51724 = {}
        # Getting the type of 'np' (line 148)
        np_51721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 148)
        arange_51722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), np_51721, 'arange')
        # Calling arange(args, kwargs) (line 148)
        arange_call_result_51725 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), arange_51722, *[int_51723], **kwargs_51724)
        
        # Assigning a type to the variable 'y' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'y', arange_call_result_51725)
        
        # Call to assert_equal(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to simps(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'y' (line 149)
        y_51728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'y', False)
        # Processing the call keyword arguments (line 149)
        kwargs_51729 = {}
        # Getting the type of 'simps' (line 149)
        simps_51727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'simps', False)
        # Calling simps(args, kwargs) (line 149)
        simps_call_result_51730 = invoke(stypy.reporting.localization.Localization(__file__, 149, 21), simps_51727, *[y_51728], **kwargs_51729)
        
        int_51731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'int')
        # Processing the call keyword arguments (line 149)
        kwargs_51732 = {}
        # Getting the type of 'assert_equal' (line 149)
        assert_equal_51726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 149)
        assert_equal_call_result_51733 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assert_equal_51726, *[simps_call_result_51730, int_51731], **kwargs_51732)
        
        
        # Call to assert_equal(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Call to simps(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'y' (line 150)
        y_51736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'y', False)
        # Processing the call keyword arguments (line 150)
        float_51737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'float')
        keyword_51738 = float_51737
        kwargs_51739 = {'dx': keyword_51738}
        # Getting the type of 'simps' (line 150)
        simps_51735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'simps', False)
        # Calling simps(args, kwargs) (line 150)
        simps_call_result_51740 = invoke(stypy.reporting.localization.Localization(__file__, 150, 21), simps_51735, *[y_51736], **kwargs_51739)
        
        int_51741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 39), 'int')
        # Processing the call keyword arguments (line 150)
        kwargs_51742 = {}
        # Getting the type of 'assert_equal' (line 150)
        assert_equal_51734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 150)
        assert_equal_call_result_51743 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), assert_equal_51734, *[simps_call_result_51740, int_51741], **kwargs_51742)
        
        
        # Call to assert_equal(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Call to simps(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'y' (line 151)
        y_51746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'y', False)
        # Processing the call keyword arguments (line 151)
        
        # Call to linspace(...): (line 151)
        # Processing the call arguments (line 151)
        int_51749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 44), 'int')
        int_51750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 47), 'int')
        int_51751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 50), 'int')
        # Processing the call keyword arguments (line 151)
        kwargs_51752 = {}
        # Getting the type of 'np' (line 151)
        np_51747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 32), 'np', False)
        # Obtaining the member 'linspace' of a type (line 151)
        linspace_51748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 32), np_51747, 'linspace')
        # Calling linspace(args, kwargs) (line 151)
        linspace_call_result_51753 = invoke(stypy.reporting.localization.Localization(__file__, 151, 32), linspace_51748, *[int_51749, int_51750, int_51751], **kwargs_51752)
        
        keyword_51754 = linspace_call_result_51753
        kwargs_51755 = {'x': keyword_51754}
        # Getting the type of 'simps' (line 151)
        simps_51745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'simps', False)
        # Calling simps(args, kwargs) (line 151)
        simps_call_result_51756 = invoke(stypy.reporting.localization.Localization(__file__, 151, 21), simps_51745, *[y_51746], **kwargs_51755)
        
        int_51757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 56), 'int')
        # Processing the call keyword arguments (line 151)
        kwargs_51758 = {}
        # Getting the type of 'assert_equal' (line 151)
        assert_equal_51744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 151)
        assert_equal_call_result_51759 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), assert_equal_51744, *[simps_call_result_51756, int_51757], **kwargs_51758)
        
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to arange(...): (line 153)
        # Processing the call arguments (line 153)
        int_51762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'int')
        # Processing the call keyword arguments (line 153)
        kwargs_51763 = {}
        # Getting the type of 'np' (line 153)
        np_51760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 153)
        arange_51761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), np_51760, 'arange')
        # Calling arange(args, kwargs) (line 153)
        arange_call_result_51764 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), arange_51761, *[int_51762], **kwargs_51763)
        
        # Assigning a type to the variable 'y' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'y', arange_call_result_51764)
        
        # Assigning a BinOp to a Name (line 154):
        
        # Assigning a BinOp to a Name (line 154):
        int_51765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 12), 'int')
        # Getting the type of 'y' (line 154)
        y_51766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'y')
        # Applying the binary operator '**' (line 154)
        result_pow_51767 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 12), '**', int_51765, y_51766)
        
        # Assigning a type to the variable 'x' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'x', result_pow_51767)
        
        # Call to assert_equal(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Call to simps(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'y' (line 155)
        y_51770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'y', False)
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'x' (line 155)
        x_51771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'x', False)
        keyword_51772 = x_51771
        str_51773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 40), 'str', 'avg')
        keyword_51774 = str_51773
        kwargs_51775 = {'even': keyword_51774, 'x': keyword_51772}
        # Getting the type of 'simps' (line 155)
        simps_51769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'simps', False)
        # Calling simps(args, kwargs) (line 155)
        simps_call_result_51776 = invoke(stypy.reporting.localization.Localization(__file__, 155, 21), simps_51769, *[y_51770], **kwargs_51775)
        
        float_51777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 48), 'float')
        # Processing the call keyword arguments (line 155)
        kwargs_51778 = {}
        # Getting the type of 'assert_equal' (line 155)
        assert_equal_51768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 155)
        assert_equal_call_result_51779 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), assert_equal_51768, *[simps_call_result_51776, float_51777], **kwargs_51778)
        
        
        # Call to assert_equal(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Call to simps(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'y' (line 156)
        y_51782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'y', False)
        # Processing the call keyword arguments (line 156)
        # Getting the type of 'x' (line 156)
        x_51783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'x', False)
        keyword_51784 = x_51783
        str_51785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 40), 'str', 'first')
        keyword_51786 = str_51785
        kwargs_51787 = {'even': keyword_51786, 'x': keyword_51784}
        # Getting the type of 'simps' (line 156)
        simps_51781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'simps', False)
        # Calling simps(args, kwargs) (line 156)
        simps_call_result_51788 = invoke(stypy.reporting.localization.Localization(__file__, 156, 21), simps_51781, *[y_51782], **kwargs_51787)
        
        float_51789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 50), 'float')
        # Processing the call keyword arguments (line 156)
        kwargs_51790 = {}
        # Getting the type of 'assert_equal' (line 156)
        assert_equal_51780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 156)
        assert_equal_call_result_51791 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), assert_equal_51780, *[simps_call_result_51788, float_51789], **kwargs_51790)
        
        
        # Call to assert_equal(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Call to simps(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'y' (line 157)
        y_51794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'y', False)
        # Processing the call keyword arguments (line 157)
        # Getting the type of 'x' (line 157)
        x_51795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'x', False)
        keyword_51796 = x_51795
        str_51797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 40), 'str', 'last')
        keyword_51798 = str_51797
        kwargs_51799 = {'even': keyword_51798, 'x': keyword_51796}
        # Getting the type of 'simps' (line 157)
        simps_51793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'simps', False)
        # Calling simps(args, kwargs) (line 157)
        simps_call_result_51800 = invoke(stypy.reporting.localization.Localization(__file__, 157, 21), simps_51793, *[y_51794], **kwargs_51799)
        
        int_51801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 49), 'int')
        # Processing the call keyword arguments (line 157)
        kwargs_51802 = {}
        # Getting the type of 'assert_equal' (line 157)
        assert_equal_51792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 157)
        assert_equal_call_result_51803 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), assert_equal_51792, *[simps_call_result_51800, int_51801], **kwargs_51802)
        
        
        # ################# End of 'test_simps(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simps' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_51804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51804)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simps'
        return stypy_return_type_51804


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 32, 0, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestQuadrature.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestQuadrature' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'TestQuadrature', TestQuadrature)
# Declaration of the 'TestCumtrapz' class

class TestCumtrapz(object, ):

    @norecursion
    def test_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1d'
        module_type_store = module_type_store.open_function_context('test_1d', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_localization', localization)
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_function_name', 'TestCumtrapz.test_1d')
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCumtrapz.test_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCumtrapz.test_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1d(...)' code ##################

        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to linspace(...): (line 162)
        # Processing the call arguments (line 162)
        int_51807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 24), 'int')
        int_51808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 28), 'int')
        # Processing the call keyword arguments (line 162)
        int_51809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 35), 'int')
        keyword_51810 = int_51809
        kwargs_51811 = {'num': keyword_51810}
        # Getting the type of 'np' (line 162)
        np_51805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 162)
        linspace_51806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), np_51805, 'linspace')
        # Calling linspace(args, kwargs) (line 162)
        linspace_call_result_51812 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), linspace_51806, *[int_51807, int_51808], **kwargs_51811)
        
        # Assigning a type to the variable 'x' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'x', linspace_call_result_51812)
        
        # Assigning a Name to a Name (line 163):
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'x' (line 163)
        x_51813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'x')
        # Assigning a type to the variable 'y' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'y', x_51813)
        
        # Assigning a Call to a Name (line 164):
        
        # Assigning a Call to a Name (line 164):
        
        # Call to cumtrapz(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'y' (line 164)
        y_51815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'y', False)
        # Getting the type of 'x' (line 164)
        x_51816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'x', False)
        # Processing the call keyword arguments (line 164)
        int_51817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 39), 'int')
        keyword_51818 = int_51817
        kwargs_51819 = {'initial': keyword_51818}
        # Getting the type of 'cumtrapz' (line 164)
        cumtrapz_51814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 164)
        cumtrapz_call_result_51820 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), cumtrapz_51814, *[y_51815, x_51816], **kwargs_51819)
        
        # Assigning a type to the variable 'y_int' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'y_int', cumtrapz_call_result_51820)
        
        # Assigning a List to a Name (line 165):
        
        # Assigning a List to a Name (line 165):
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_51821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        float_51822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 21), list_51821, float_51822)
        # Adding element type (line 165)
        float_51823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 21), list_51821, float_51823)
        # Adding element type (line 165)
        float_51824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 21), list_51821, float_51824)
        # Adding element type (line 165)
        float_51825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 21), list_51821, float_51825)
        # Adding element type (line 165)
        float_51826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 21), list_51821, float_51826)
        
        # Assigning a type to the variable 'y_expected' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'y_expected', list_51821)
        
        # Call to assert_allclose(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'y_int' (line 166)
        y_int_51828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'y_int', False)
        # Getting the type of 'y_expected' (line 166)
        y_expected_51829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'y_expected', False)
        # Processing the call keyword arguments (line 166)
        kwargs_51830 = {}
        # Getting the type of 'assert_allclose' (line 166)
        assert_allclose_51827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 166)
        assert_allclose_call_result_51831 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), assert_allclose_51827, *[y_int_51828, y_expected_51829], **kwargs_51830)
        
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to cumtrapz(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'y' (line 168)
        y_51833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'y', False)
        # Getting the type of 'x' (line 168)
        x_51834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'x', False)
        # Processing the call keyword arguments (line 168)
        # Getting the type of 'None' (line 168)
        None_51835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'None', False)
        keyword_51836 = None_51835
        kwargs_51837 = {'initial': keyword_51836}
        # Getting the type of 'cumtrapz' (line 168)
        cumtrapz_51832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 168)
        cumtrapz_call_result_51838 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), cumtrapz_51832, *[y_51833, x_51834], **kwargs_51837)
        
        # Assigning a type to the variable 'y_int' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'y_int', cumtrapz_call_result_51838)
        
        # Call to assert_allclose(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'y_int' (line 169)
        y_int_51840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'y_int', False)
        
        # Obtaining the type of the subscript
        int_51841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 42), 'int')
        slice_51842 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 31), int_51841, None, None)
        # Getting the type of 'y_expected' (line 169)
        y_expected_51843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'y_expected', False)
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___51844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 31), y_expected_51843, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_51845 = invoke(stypy.reporting.localization.Localization(__file__, 169, 31), getitem___51844, slice_51842)
        
        # Processing the call keyword arguments (line 169)
        kwargs_51846 = {}
        # Getting the type of 'assert_allclose' (line 169)
        assert_allclose_51839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 169)
        assert_allclose_call_result_51847 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), assert_allclose_51839, *[y_int_51840, subscript_call_result_51845], **kwargs_51846)
        
        
        # ################# End of 'test_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_51848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1d'
        return stypy_return_type_51848


    @norecursion
    def test_y_nd_x_nd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_nd_x_nd'
        module_type_store = module_type_store.open_function_context('test_y_nd_x_nd', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_localization', localization)
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_function_name', 'TestCumtrapz.test_y_nd_x_nd')
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_param_names_list', [])
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCumtrapz.test_y_nd_x_nd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCumtrapz.test_y_nd_x_nd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_nd_x_nd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_nd_x_nd(...)' code ##################

        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to reshape(...): (line 172)
        # Processing the call arguments (line 172)
        int_51859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 41), 'int')
        int_51860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 44), 'int')
        int_51861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 47), 'int')
        # Processing the call keyword arguments (line 172)
        kwargs_51862 = {}
        
        # Call to arange(...): (line 172)
        # Processing the call arguments (line 172)
        int_51851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 22), 'int')
        int_51852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 26), 'int')
        # Applying the binary operator '*' (line 172)
        result_mul_51853 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 22), '*', int_51851, int_51852)
        
        int_51854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 30), 'int')
        # Applying the binary operator '*' (line 172)
        result_mul_51855 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 28), '*', result_mul_51853, int_51854)
        
        # Processing the call keyword arguments (line 172)
        kwargs_51856 = {}
        # Getting the type of 'np' (line 172)
        np_51849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 172)
        arange_51850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), np_51849, 'arange')
        # Calling arange(args, kwargs) (line 172)
        arange_call_result_51857 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), arange_51850, *[result_mul_51855], **kwargs_51856)
        
        # Obtaining the member 'reshape' of a type (line 172)
        reshape_51858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 12), arange_call_result_51857, 'reshape')
        # Calling reshape(args, kwargs) (line 172)
        reshape_call_result_51863 = invoke(stypy.reporting.localization.Localization(__file__, 172, 12), reshape_51858, *[int_51859, int_51860, int_51861], **kwargs_51862)
        
        # Assigning a type to the variable 'x' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'x', reshape_call_result_51863)
        
        # Assigning a Name to a Name (line 173):
        
        # Assigning a Name to a Name (line 173):
        # Getting the type of 'x' (line 173)
        x_51864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'x')
        # Assigning a type to the variable 'y' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'y', x_51864)
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to cumtrapz(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'y' (line 174)
        y_51866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'y', False)
        # Getting the type of 'x' (line 174)
        x_51867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 28), 'x', False)
        # Processing the call keyword arguments (line 174)
        int_51868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 39), 'int')
        keyword_51869 = int_51868
        kwargs_51870 = {'initial': keyword_51869}
        # Getting the type of 'cumtrapz' (line 174)
        cumtrapz_51865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 174)
        cumtrapz_call_result_51871 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), cumtrapz_51865, *[y_51866, x_51867], **kwargs_51870)
        
        # Assigning a type to the variable 'y_int' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'y_int', cumtrapz_call_result_51871)
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to array(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_51874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_51875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_51876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        float_51877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 32), list_51876, float_51877)
        # Adding element type (line 175)
        float_51878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 32), list_51876, float_51878)
        # Adding element type (line 175)
        float_51879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 32), list_51876, float_51879)
        # Adding element type (line 175)
        float_51880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 32), list_51876, float_51880)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 31), list_51875, list_51876)
        # Adding element type (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_51881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        float_51882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), list_51881, float_51882)
        # Adding element type (line 176)
        float_51883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), list_51881, float_51883)
        # Adding element type (line 176)
        float_51884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), list_51881, float_51884)
        # Adding element type (line 176)
        float_51885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 32), list_51881, float_51885)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 31), list_51875, list_51881)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 30), list_51874, list_51875)
        # Adding element type (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_51886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_51887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        float_51888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 32), list_51887, float_51888)
        # Adding element type (line 177)
        float_51889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 32), list_51887, float_51889)
        # Adding element type (line 177)
        float_51890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 32), list_51887, float_51890)
        # Adding element type (line 177)
        float_51891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 32), list_51887, float_51891)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 31), list_51886, list_51887)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_51892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        float_51893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 32), list_51892, float_51893)
        # Adding element type (line 178)
        float_51894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 32), list_51892, float_51894)
        # Adding element type (line 178)
        float_51895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 32), list_51892, float_51895)
        # Adding element type (line 178)
        float_51896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 32), list_51892, float_51896)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 31), list_51886, list_51892)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 30), list_51874, list_51886)
        # Adding element type (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_51897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_51898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        float_51899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 32), list_51898, float_51899)
        # Adding element type (line 179)
        float_51900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 32), list_51898, float_51900)
        # Adding element type (line 179)
        float_51901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 32), list_51898, float_51901)
        # Adding element type (line 179)
        float_51902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 32), list_51898, float_51902)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 31), list_51897, list_51898)
        # Adding element type (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_51903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        # Adding element type (line 180)
        float_51904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 32), list_51903, float_51904)
        # Adding element type (line 180)
        float_51905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 32), list_51903, float_51905)
        # Adding element type (line 180)
        float_51906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 32), list_51903, float_51906)
        # Adding element type (line 180)
        float_51907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 32), list_51903, float_51907)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 31), list_51897, list_51903)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 30), list_51874, list_51897)
        
        # Processing the call keyword arguments (line 175)
        kwargs_51908 = {}
        # Getting the type of 'np' (line 175)
        np_51872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 175)
        array_51873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 21), np_51872, 'array')
        # Calling array(args, kwargs) (line 175)
        array_call_result_51909 = invoke(stypy.reporting.localization.Localization(__file__, 175, 21), array_51873, *[list_51874], **kwargs_51908)
        
        # Assigning a type to the variable 'y_expected' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'y_expected', array_call_result_51909)
        
        # Call to assert_allclose(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'y_int' (line 182)
        y_int_51911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'y_int', False)
        # Getting the type of 'y_expected' (line 182)
        y_expected_51912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'y_expected', False)
        # Processing the call keyword arguments (line 182)
        kwargs_51913 = {}
        # Getting the type of 'assert_allclose' (line 182)
        assert_allclose_51910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 182)
        assert_allclose_call_result_51914 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), assert_allclose_51910, *[y_int_51911, y_expected_51912], **kwargs_51913)
        
        
        # Assigning a List to a Name (line 185):
        
        # Assigning a List to a Name (line 185):
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_51915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_51916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_51917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 19), tuple_51916, int_51917)
        # Adding element type (line 185)
        int_51918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 19), tuple_51916, int_51918)
        # Adding element type (line 185)
        int_51919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 19), tuple_51916, int_51919)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 17), list_51915, tuple_51916)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_51920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_51921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), tuple_51920, int_51921)
        # Adding element type (line 185)
        int_51922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), tuple_51920, int_51922)
        # Adding element type (line 185)
        int_51923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), tuple_51920, int_51923)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 17), list_51915, tuple_51920)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_51924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_51925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 41), tuple_51924, int_51925)
        # Adding element type (line 185)
        int_51926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 41), tuple_51924, int_51926)
        # Adding element type (line 185)
        int_51927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 41), tuple_51924, int_51927)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 17), list_51915, tuple_51924)
        
        # Assigning a type to the variable 'shapes' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'shapes', list_51915)
        
        
        # Call to zip(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_51929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        # Adding element type (line 186)
        int_51930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 31), list_51929, int_51930)
        # Adding element type (line 186)
        int_51931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 31), list_51929, int_51931)
        # Adding element type (line 186)
        int_51932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 31), list_51929, int_51932)
        
        # Getting the type of 'shapes' (line 186)
        shapes_51933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 42), 'shapes', False)
        # Processing the call keyword arguments (line 186)
        kwargs_51934 = {}
        # Getting the type of 'zip' (line 186)
        zip_51928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'zip', False)
        # Calling zip(args, kwargs) (line 186)
        zip_call_result_51935 = invoke(stypy.reporting.localization.Localization(__file__, 186, 27), zip_51928, *[list_51929, shapes_51933], **kwargs_51934)
        
        # Testing the type of a for loop iterable (line 186)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 186, 8), zip_call_result_51935)
        # Getting the type of the for loop variable (line 186)
        for_loop_var_51936 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 186, 8), zip_call_result_51935)
        # Assigning a type to the variable 'axis' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 8), for_loop_var_51936))
        # Assigning a type to the variable 'shape' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'shape', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 8), for_loop_var_51936))
        # SSA begins for a for statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to cumtrapz(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'y' (line 187)
        y_51938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 29), 'y', False)
        # Getting the type of 'x' (line 187)
        x_51939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 32), 'x', False)
        # Processing the call keyword arguments (line 187)
        float_51940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 43), 'float')
        keyword_51941 = float_51940
        # Getting the type of 'axis' (line 187)
        axis_51942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 54), 'axis', False)
        keyword_51943 = axis_51942
        kwargs_51944 = {'initial': keyword_51941, 'axis': keyword_51943}
        # Getting the type of 'cumtrapz' (line 187)
        cumtrapz_51937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 187)
        cumtrapz_call_result_51945 = invoke(stypy.reporting.localization.Localization(__file__, 187, 20), cumtrapz_51937, *[y_51938, x_51939], **kwargs_51944)
        
        # Assigning a type to the variable 'y_int' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'y_int', cumtrapz_call_result_51945)
        
        # Call to assert_equal(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'y_int' (line 188)
        y_int_51947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'y_int', False)
        # Obtaining the member 'shape' of a type (line 188)
        shape_51948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 25), y_int_51947, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 188)
        tuple_51949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 188)
        # Adding element type (line 188)
        int_51950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 39), tuple_51949, int_51950)
        # Adding element type (line 188)
        int_51951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 39), tuple_51949, int_51951)
        # Adding element type (line 188)
        int_51952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 39), tuple_51949, int_51952)
        
        # Processing the call keyword arguments (line 188)
        kwargs_51953 = {}
        # Getting the type of 'assert_equal' (line 188)
        assert_equal_51946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 188)
        assert_equal_call_result_51954 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), assert_equal_51946, *[shape_51948, tuple_51949], **kwargs_51953)
        
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Call to cumtrapz(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'y' (line 189)
        y_51956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'y', False)
        # Getting the type of 'x' (line 189)
        x_51957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'x', False)
        # Processing the call keyword arguments (line 189)
        # Getting the type of 'None' (line 189)
        None_51958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'None', False)
        keyword_51959 = None_51958
        # Getting the type of 'axis' (line 189)
        axis_51960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 54), 'axis', False)
        keyword_51961 = axis_51960
        kwargs_51962 = {'initial': keyword_51959, 'axis': keyword_51961}
        # Getting the type of 'cumtrapz' (line 189)
        cumtrapz_51955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 189)
        cumtrapz_call_result_51963 = invoke(stypy.reporting.localization.Localization(__file__, 189, 20), cumtrapz_51955, *[y_51956, x_51957], **kwargs_51962)
        
        # Assigning a type to the variable 'y_int' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'y_int', cumtrapz_call_result_51963)
        
        # Call to assert_equal(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'y_int' (line 190)
        y_int_51965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'y_int', False)
        # Obtaining the member 'shape' of a type (line 190)
        shape_51966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 25), y_int_51965, 'shape')
        # Getting the type of 'shape' (line 190)
        shape_51967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 38), 'shape', False)
        # Processing the call keyword arguments (line 190)
        kwargs_51968 = {}
        # Getting the type of 'assert_equal' (line 190)
        assert_equal_51964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 190)
        assert_equal_call_result_51969 = invoke(stypy.reporting.localization.Localization(__file__, 190, 12), assert_equal_51964, *[shape_51966, shape_51967], **kwargs_51968)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_y_nd_x_nd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_nd_x_nd' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_51970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_51970)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_nd_x_nd'
        return stypy_return_type_51970


    @norecursion
    def test_y_nd_x_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_y_nd_x_1d'
        module_type_store = module_type_store.open_function_context('test_y_nd_x_1d', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_localization', localization)
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_function_name', 'TestCumtrapz.test_y_nd_x_1d')
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCumtrapz.test_y_nd_x_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCumtrapz.test_y_nd_x_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_y_nd_x_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_y_nd_x_1d(...)' code ##################

        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to reshape(...): (line 193)
        # Processing the call arguments (line 193)
        int_51981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 41), 'int')
        int_51982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 44), 'int')
        int_51983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 47), 'int')
        # Processing the call keyword arguments (line 193)
        kwargs_51984 = {}
        
        # Call to arange(...): (line 193)
        # Processing the call arguments (line 193)
        int_51973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 22), 'int')
        int_51974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 26), 'int')
        # Applying the binary operator '*' (line 193)
        result_mul_51975 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 22), '*', int_51973, int_51974)
        
        int_51976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 30), 'int')
        # Applying the binary operator '*' (line 193)
        result_mul_51977 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 28), '*', result_mul_51975, int_51976)
        
        # Processing the call keyword arguments (line 193)
        kwargs_51978 = {}
        # Getting the type of 'np' (line 193)
        np_51971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 193)
        arange_51972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), np_51971, 'arange')
        # Calling arange(args, kwargs) (line 193)
        arange_call_result_51979 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), arange_51972, *[result_mul_51977], **kwargs_51978)
        
        # Obtaining the member 'reshape' of a type (line 193)
        reshape_51980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), arange_call_result_51979, 'reshape')
        # Calling reshape(args, kwargs) (line 193)
        reshape_call_result_51985 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), reshape_51980, *[int_51981, int_51982, int_51983], **kwargs_51984)
        
        # Assigning a type to the variable 'y' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'y', reshape_call_result_51985)
        
        # Assigning a BinOp to a Name (line 194):
        
        # Assigning a BinOp to a Name (line 194):
        
        # Call to arange(...): (line 194)
        # Processing the call arguments (line 194)
        int_51988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 22), 'int')
        # Processing the call keyword arguments (line 194)
        kwargs_51989 = {}
        # Getting the type of 'np' (line 194)
        np_51986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 194)
        arange_51987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), np_51986, 'arange')
        # Calling arange(args, kwargs) (line 194)
        arange_call_result_51990 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), arange_51987, *[int_51988], **kwargs_51989)
        
        int_51991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 26), 'int')
        # Applying the binary operator '**' (line 194)
        result_pow_51992 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 12), '**', arange_call_result_51990, int_51991)
        
        # Assigning a type to the variable 'x' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'x', result_pow_51992)
        
        # Assigning a Tuple to a Name (line 196):
        
        # Assigning a Tuple to a Name (line 196):
        
        # Obtaining an instance of the builtin type 'tuple' (line 197)
        tuple_51993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 197)
        # Adding element type (line 197)
        
        # Call to array(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_51996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_51997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_51998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        float_51999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 23), list_51998, float_51999)
        # Adding element type (line 197)
        float_52000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 23), list_51998, float_52000)
        # Adding element type (line 197)
        float_52001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 23), list_51998, float_52001)
        # Adding element type (line 197)
        float_52002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 23), list_51998, float_52002)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 22), list_51997, list_51998)
        # Adding element type (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_52003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        # Adding element type (line 198)
        float_52004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 23), list_52003, float_52004)
        # Adding element type (line 198)
        float_52005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 23), list_52003, float_52005)
        # Adding element type (line 198)
        float_52006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 23), list_52003, float_52006)
        # Adding element type (line 198)
        float_52007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 23), list_52003, float_52007)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 22), list_51997, list_52003)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_51996, list_51997)
        # Adding element type (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_52008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_52009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        float_52010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 23), list_52009, float_52010)
        # Adding element type (line 199)
        float_52011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 23), list_52009, float_52011)
        # Adding element type (line 199)
        float_52012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 23), list_52009, float_52012)
        # Adding element type (line 199)
        float_52013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 23), list_52009, float_52013)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), list_52008, list_52009)
        # Adding element type (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_52014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        float_52015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 23), list_52014, float_52015)
        # Adding element type (line 200)
        float_52016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 23), list_52014, float_52016)
        # Adding element type (line 200)
        float_52017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 23), list_52014, float_52017)
        # Adding element type (line 200)
        float_52018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 23), list_52014, float_52018)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), list_52008, list_52014)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_51996, list_52008)
        
        # Processing the call keyword arguments (line 197)
        kwargs_52019 = {}
        # Getting the type of 'np' (line 197)
        np_51994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 197)
        array_51995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), np_51994, 'array')
        # Calling array(args, kwargs) (line 197)
        array_call_result_52020 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), array_51995, *[list_51996], **kwargs_52019)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 12), tuple_51993, array_call_result_52020)
        # Adding element type (line 197)
        
        # Call to array(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_52023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_52024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_52025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        float_52026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 23), list_52025, float_52026)
        # Adding element type (line 201)
        float_52027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 23), list_52025, float_52027)
        # Adding element type (line 201)
        float_52028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 23), list_52025, float_52028)
        # Adding element type (line 201)
        float_52029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 23), list_52025, float_52029)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 22), list_52024, list_52025)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 21), list_52023, list_52024)
        # Adding element type (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_52030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_52031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        float_52032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 23), list_52031, float_52032)
        # Adding element type (line 202)
        float_52033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 23), list_52031, float_52033)
        # Adding element type (line 202)
        float_52034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 23), list_52031, float_52034)
        # Adding element type (line 202)
        float_52035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 23), list_52031, float_52035)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 22), list_52030, list_52031)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 21), list_52023, list_52030)
        # Adding element type (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_52036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_52037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        float_52038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 23), list_52037, float_52038)
        # Adding element type (line 203)
        float_52039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 23), list_52037, float_52039)
        # Adding element type (line 203)
        float_52040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 23), list_52037, float_52040)
        # Adding element type (line 203)
        float_52041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 23), list_52037, float_52041)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 22), list_52036, list_52037)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 21), list_52023, list_52036)
        
        # Processing the call keyword arguments (line 201)
        kwargs_52042 = {}
        # Getting the type of 'np' (line 201)
        np_52021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 201)
        array_52022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), np_52021, 'array')
        # Calling array(args, kwargs) (line 201)
        array_call_result_52043 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), array_52022, *[list_52023], **kwargs_52042)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 12), tuple_51993, array_call_result_52043)
        # Adding element type (line 197)
        
        # Call to array(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_52046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_52047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_52048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        float_52049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 23), list_52048, float_52049)
        # Adding element type (line 204)
        float_52050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 23), list_52048, float_52050)
        # Adding element type (line 204)
        float_52051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 23), list_52048, float_52051)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 22), list_52047, list_52048)
        # Adding element type (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_52052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        float_52053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 23), list_52052, float_52053)
        # Adding element type (line 205)
        float_52054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 23), list_52052, float_52054)
        # Adding element type (line 205)
        float_52055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 23), list_52052, float_52055)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 22), list_52047, list_52052)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 21), list_52046, list_52047)
        # Adding element type (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 206)
        list_52056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 206)
        # Adding element type (line 206)
        
        # Obtaining an instance of the builtin type 'list' (line 206)
        list_52057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 206)
        # Adding element type (line 206)
        float_52058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 23), list_52057, float_52058)
        # Adding element type (line 206)
        float_52059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 23), list_52057, float_52059)
        # Adding element type (line 206)
        float_52060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 23), list_52057, float_52060)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 22), list_52056, list_52057)
        # Adding element type (line 206)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_52061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        float_52062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 23), list_52061, float_52062)
        # Adding element type (line 207)
        float_52063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 23), list_52061, float_52063)
        # Adding element type (line 207)
        float_52064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 23), list_52061, float_52064)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 22), list_52056, list_52061)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 21), list_52046, list_52056)
        # Adding element type (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_52065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_52066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        float_52067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 23), list_52066, float_52067)
        # Adding element type (line 208)
        float_52068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 23), list_52066, float_52068)
        # Adding element type (line 208)
        float_52069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 23), list_52066, float_52069)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 22), list_52065, list_52066)
        # Adding element type (line 208)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_52070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        float_52071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 23), list_52070, float_52071)
        # Adding element type (line 209)
        float_52072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 23), list_52070, float_52072)
        # Adding element type (line 209)
        float_52073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 23), list_52070, float_52073)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 22), list_52065, list_52070)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 21), list_52046, list_52065)
        
        # Processing the call keyword arguments (line 204)
        kwargs_52074 = {}
        # Getting the type of 'np' (line 204)
        np_52044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 204)
        array_52045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), np_52044, 'array')
        # Calling array(args, kwargs) (line 204)
        array_call_result_52075 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), array_52045, *[list_52046], **kwargs_52074)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 12), tuple_51993, array_call_result_52075)
        
        # Assigning a type to the variable 'ys_expected' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'ys_expected', tuple_51993)
        
        
        # Call to zip(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_52077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        int_52078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 36), list_52077, int_52078)
        # Adding element type (line 211)
        int_52079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 36), list_52077, int_52079)
        # Adding element type (line 211)
        int_52080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 36), list_52077, int_52080)
        
        # Getting the type of 'ys_expected' (line 211)
        ys_expected_52081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 47), 'ys_expected', False)
        # Processing the call keyword arguments (line 211)
        kwargs_52082 = {}
        # Getting the type of 'zip' (line 211)
        zip_52076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 32), 'zip', False)
        # Calling zip(args, kwargs) (line 211)
        zip_call_result_52083 = invoke(stypy.reporting.localization.Localization(__file__, 211, 32), zip_52076, *[list_52077, ys_expected_52081], **kwargs_52082)
        
        # Testing the type of a for loop iterable (line 211)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 211, 8), zip_call_result_52083)
        # Getting the type of the for loop variable (line 211)
        for_loop_var_52084 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 211, 8), zip_call_result_52083)
        # Assigning a type to the variable 'axis' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 8), for_loop_var_52084))
        # Assigning a type to the variable 'y_expected' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'y_expected', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 8), for_loop_var_52084))
        # SSA begins for a for statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to cumtrapz(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'y' (line 212)
        y_52086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'y', False)
        # Processing the call keyword arguments (line 212)
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 212)
        axis_52087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 45), 'axis', False)
        # Getting the type of 'y' (line 212)
        y_52088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'y', False)
        # Obtaining the member 'shape' of a type (line 212)
        shape_52089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 37), y_52088, 'shape')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___52090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 37), shape_52089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_52091 = invoke(stypy.reporting.localization.Localization(__file__, 212, 37), getitem___52090, axis_52087)
        
        slice_52092 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 212, 34), None, subscript_call_result_52091, None)
        # Getting the type of 'x' (line 212)
        x_52093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___52094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 34), x_52093, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_52095 = invoke(stypy.reporting.localization.Localization(__file__, 212, 34), getitem___52094, slice_52092)
        
        keyword_52096 = subscript_call_result_52095
        # Getting the type of 'axis' (line 212)
        axis_52097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 58), 'axis', False)
        keyword_52098 = axis_52097
        # Getting the type of 'None' (line 212)
        None_52099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 72), 'None', False)
        keyword_52100 = None_52099
        kwargs_52101 = {'x': keyword_52096, 'initial': keyword_52100, 'axis': keyword_52098}
        # Getting the type of 'cumtrapz' (line 212)
        cumtrapz_52085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 212)
        cumtrapz_call_result_52102 = invoke(stypy.reporting.localization.Localization(__file__, 212, 20), cumtrapz_52085, *[y_52086], **kwargs_52101)
        
        # Assigning a type to the variable 'y_int' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'y_int', cumtrapz_call_result_52102)
        
        # Call to assert_allclose(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'y_int' (line 213)
        y_int_52104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'y_int', False)
        # Getting the type of 'y_expected' (line 213)
        y_expected_52105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 35), 'y_expected', False)
        # Processing the call keyword arguments (line 213)
        kwargs_52106 = {}
        # Getting the type of 'assert_allclose' (line 213)
        assert_allclose_52103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 213)
        assert_allclose_call_result_52107 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), assert_allclose_52103, *[y_int_52104, y_expected_52105], **kwargs_52106)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_y_nd_x_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_y_nd_x_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_52108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52108)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_y_nd_x_1d'
        return stypy_return_type_52108


    @norecursion
    def test_x_none(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_x_none'
        module_type_store = module_type_store.open_function_context('test_x_none', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_localization', localization)
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_function_name', 'TestCumtrapz.test_x_none')
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_param_names_list', [])
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCumtrapz.test_x_none.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCumtrapz.test_x_none', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_x_none', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_x_none(...)' code ##################

        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to linspace(...): (line 216)
        # Processing the call arguments (line 216)
        int_52111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 24), 'int')
        int_52112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'int')
        # Processing the call keyword arguments (line 216)
        int_52113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 35), 'int')
        keyword_52114 = int_52113
        kwargs_52115 = {'num': keyword_52114}
        # Getting the type of 'np' (line 216)
        np_52109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 216)
        linspace_52110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), np_52109, 'linspace')
        # Calling linspace(args, kwargs) (line 216)
        linspace_call_result_52116 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), linspace_52110, *[int_52111, int_52112], **kwargs_52115)
        
        # Assigning a type to the variable 'y' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'y', linspace_call_result_52116)
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to cumtrapz(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'y' (line 218)
        y_52118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 25), 'y', False)
        # Processing the call keyword arguments (line 218)
        kwargs_52119 = {}
        # Getting the type of 'cumtrapz' (line 218)
        cumtrapz_52117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 218)
        cumtrapz_call_result_52120 = invoke(stypy.reporting.localization.Localization(__file__, 218, 16), cumtrapz_52117, *[y_52118], **kwargs_52119)
        
        # Assigning a type to the variable 'y_int' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'y_int', cumtrapz_call_result_52120)
        
        # Assigning a List to a Name (line 219):
        
        # Assigning a List to a Name (line 219):
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_52121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        float_52122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 21), list_52121, float_52122)
        # Adding element type (line 219)
        float_52123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 21), list_52121, float_52123)
        # Adding element type (line 219)
        float_52124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 21), list_52121, float_52124)
        # Adding element type (line 219)
        float_52125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 21), list_52121, float_52125)
        
        # Assigning a type to the variable 'y_expected' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'y_expected', list_52121)
        
        # Call to assert_allclose(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'y_int' (line 220)
        y_int_52127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'y_int', False)
        # Getting the type of 'y_expected' (line 220)
        y_expected_52128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 31), 'y_expected', False)
        # Processing the call keyword arguments (line 220)
        kwargs_52129 = {}
        # Getting the type of 'assert_allclose' (line 220)
        assert_allclose_52126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 220)
        assert_allclose_call_result_52130 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), assert_allclose_52126, *[y_int_52127, y_expected_52128], **kwargs_52129)
        
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to cumtrapz(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'y' (line 222)
        y_52132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'y', False)
        # Processing the call keyword arguments (line 222)
        float_52133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 36), 'float')
        keyword_52134 = float_52133
        kwargs_52135 = {'initial': keyword_52134}
        # Getting the type of 'cumtrapz' (line 222)
        cumtrapz_52131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 222)
        cumtrapz_call_result_52136 = invoke(stypy.reporting.localization.Localization(__file__, 222, 16), cumtrapz_52131, *[y_52132], **kwargs_52135)
        
        # Assigning a type to the variable 'y_int' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'y_int', cumtrapz_call_result_52136)
        
        # Assigning a List to a Name (line 223):
        
        # Assigning a List to a Name (line 223):
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_52137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        float_52138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 21), list_52137, float_52138)
        # Adding element type (line 223)
        float_52139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 21), list_52137, float_52139)
        # Adding element type (line 223)
        float_52140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 21), list_52137, float_52140)
        # Adding element type (line 223)
        float_52141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 21), list_52137, float_52141)
        # Adding element type (line 223)
        float_52142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 21), list_52137, float_52142)
        
        # Assigning a type to the variable 'y_expected' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'y_expected', list_52137)
        
        # Call to assert_allclose(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'y_int' (line 224)
        y_int_52144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'y_int', False)
        # Getting the type of 'y_expected' (line 224)
        y_expected_52145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'y_expected', False)
        # Processing the call keyword arguments (line 224)
        kwargs_52146 = {}
        # Getting the type of 'assert_allclose' (line 224)
        assert_allclose_52143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 224)
        assert_allclose_call_result_52147 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), assert_allclose_52143, *[y_int_52144, y_expected_52145], **kwargs_52146)
        
        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to cumtrapz(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'y' (line 226)
        y_52149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 25), 'y', False)
        # Processing the call keyword arguments (line 226)
        int_52150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 31), 'int')
        keyword_52151 = int_52150
        kwargs_52152 = {'dx': keyword_52151}
        # Getting the type of 'cumtrapz' (line 226)
        cumtrapz_52148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 226)
        cumtrapz_call_result_52153 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), cumtrapz_52148, *[y_52149], **kwargs_52152)
        
        # Assigning a type to the variable 'y_int' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'y_int', cumtrapz_call_result_52153)
        
        # Assigning a List to a Name (line 227):
        
        # Assigning a List to a Name (line 227):
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_52154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        float_52155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 21), list_52154, float_52155)
        # Adding element type (line 227)
        float_52156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 21), list_52154, float_52156)
        # Adding element type (line 227)
        float_52157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 21), list_52154, float_52157)
        # Adding element type (line 227)
        float_52158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 21), list_52154, float_52158)
        
        # Assigning a type to the variable 'y_expected' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'y_expected', list_52154)
        
        # Call to assert_allclose(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'y_int' (line 228)
        y_int_52160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'y_int', False)
        # Getting the type of 'y_expected' (line 228)
        y_expected_52161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 31), 'y_expected', False)
        # Processing the call keyword arguments (line 228)
        kwargs_52162 = {}
        # Getting the type of 'assert_allclose' (line 228)
        assert_allclose_52159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 228)
        assert_allclose_call_result_52163 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), assert_allclose_52159, *[y_int_52160, y_expected_52161], **kwargs_52162)
        
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to cumtrapz(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'y' (line 230)
        y_52165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'y', False)
        # Processing the call keyword arguments (line 230)
        int_52166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 31), 'int')
        keyword_52167 = int_52166
        float_52168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 42), 'float')
        keyword_52169 = float_52168
        kwargs_52170 = {'initial': keyword_52169, 'dx': keyword_52167}
        # Getting the type of 'cumtrapz' (line 230)
        cumtrapz_52164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'cumtrapz', False)
        # Calling cumtrapz(args, kwargs) (line 230)
        cumtrapz_call_result_52171 = invoke(stypy.reporting.localization.Localization(__file__, 230, 16), cumtrapz_52164, *[y_52165], **kwargs_52170)
        
        # Assigning a type to the variable 'y_int' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'y_int', cumtrapz_call_result_52171)
        
        # Assigning a List to a Name (line 231):
        
        # Assigning a List to a Name (line 231):
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_52172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        float_52173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_52172, float_52173)
        # Adding element type (line 231)
        float_52174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_52172, float_52174)
        # Adding element type (line 231)
        float_52175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_52172, float_52175)
        # Adding element type (line 231)
        float_52176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_52172, float_52176)
        # Adding element type (line 231)
        float_52177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_52172, float_52177)
        
        # Assigning a type to the variable 'y_expected' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'y_expected', list_52172)
        
        # Call to assert_allclose(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'y_int' (line 232)
        y_int_52179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'y_int', False)
        # Getting the type of 'y_expected' (line 232)
        y_expected_52180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'y_expected', False)
        # Processing the call keyword arguments (line 232)
        kwargs_52181 = {}
        # Getting the type of 'assert_allclose' (line 232)
        assert_allclose_52178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 232)
        assert_allclose_call_result_52182 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assert_allclose_52178, *[y_int_52179, y_expected_52180], **kwargs_52181)
        
        
        # ################# End of 'test_x_none(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_x_none' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_52183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52183)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_x_none'
        return stypy_return_type_52183


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 160, 0, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCumtrapz.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCumtrapz' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'TestCumtrapz', TestCumtrapz)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

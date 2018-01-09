
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit tests for TNC optimization routine from tnc.py
3: '''
4: 
5: from numpy.testing import assert_allclose, assert_equal
6: 
7: from scipy import optimize
8: import numpy as np
9: from math import pow
10: 
11: 
12: class TestTnc(object):
13:     '''TNC non-linear optimization.
14: 
15:     These tests are taken from Prof. K. Schittkowski's test examples
16:     for constrained non-linear programming.
17: 
18:     http://www.uni-bayreuth.de/departments/math/~kschittkowski/home.htm
19: 
20:     '''
21:     def setup_method(self):
22:         # options for minimize
23:         self.opts = {'disp': False, 'maxiter': 200}
24: 
25:     # objective functions and jacobian for each test
26:     def f1(self, x, a=100.0):
27:         return a * pow((x[1] - pow(x[0], 2)), 2) + pow(1.0 - x[0], 2)
28: 
29:     def g1(self, x, a=100.0):
30:         dif = [0, 0]
31:         dif[1] = 2 * a * (x[1] - pow(x[0], 2))
32:         dif[0] = -2.0 * (x[0] * (dif[1] - 1.0) + 1.0)
33:         return dif
34: 
35:     def fg1(self, x, a=100.0):
36:         return self.f1(x, a), self.g1(x, a)
37: 
38:     def f3(self, x):
39:         return x[1] + pow(x[1] - x[0], 2) * 1.0e-5
40: 
41:     def g3(self, x):
42:         dif = [0, 0]
43:         dif[0] = -2.0 * (x[1] - x[0]) * 1.0e-5
44:         dif[1] = 1.0 - dif[0]
45:         return dif
46: 
47:     def fg3(self, x):
48:         return self.f3(x), self.g3(x)
49: 
50:     def f4(self, x):
51:         return pow(x[0] + 1.0, 3) / 3.0 + x[1]
52: 
53:     def g4(self, x):
54:         dif = [0, 0]
55:         dif[0] = pow(x[0] + 1.0, 2)
56:         dif[1] = 1.0
57:         return dif
58: 
59:     def fg4(self, x):
60:         return self.f4(x), self.g4(x)
61: 
62:     def f5(self, x):
63:         return np.sin(x[0] + x[1]) + pow(x[0] - x[1], 2) - \
64:                 1.5 * x[0] + 2.5 * x[1] + 1.0
65: 
66:     def g5(self, x):
67:         dif = [0, 0]
68:         v1 = np.cos(x[0] + x[1])
69:         v2 = 2.0*(x[0] - x[1])
70: 
71:         dif[0] = v1 + v2 - 1.5
72:         dif[1] = v1 - v2 + 2.5
73:         return dif
74: 
75:     def fg5(self, x):
76:         return self.f5(x), self.g5(x)
77: 
78:     def f38(self, x):
79:         return (100.0 * pow(x[1] - pow(x[0], 2), 2) +
80:                 pow(1.0 - x[0], 2) + 90.0 * pow(x[3] - pow(x[2], 2), 2) +
81:                 pow(1.0 - x[2], 2) + 10.1 * (pow(x[1] - 1.0, 2) +
82:                                              pow(x[3] - 1.0, 2)) +
83:                 19.8 * (x[1] - 1.0) * (x[3] - 1.0)) * 1.0e-5
84: 
85:     def g38(self, x):
86:         dif = [0, 0, 0, 0]
87:         dif[0] = (-400.0 * x[0] * (x[1] - pow(x[0], 2)) -
88:                   2.0 * (1.0 - x[0])) * 1.0e-5
89:         dif[1] = (200.0 * (x[1] - pow(x[0], 2)) + 20.2 * (x[1] - 1.0) +
90:                   19.8 * (x[3] - 1.0)) * 1.0e-5
91:         dif[2] = (- 360.0 * x[2] * (x[3] - pow(x[2], 2)) -
92:                   2.0 * (1.0 - x[2])) * 1.0e-5
93:         dif[3] = (180.0 * (x[3] - pow(x[2], 2)) + 20.2 * (x[3] - 1.0) +
94:                   19.8 * (x[1] - 1.0)) * 1.0e-5
95:         return dif
96: 
97:     def fg38(self, x):
98:         return self.f38(x), self.g38(x)
99: 
100:     def f45(self, x):
101:         return 2.0 - x[0] * x[1] * x[2] * x[3] * x[4] / 120.0
102: 
103:     def g45(self, x):
104:         dif = [0] * 5
105:         dif[0] = - x[1] * x[2] * x[3] * x[4] / 120.0
106:         dif[1] = - x[0] * x[2] * x[3] * x[4] / 120.0
107:         dif[2] = - x[0] * x[1] * x[3] * x[4] / 120.0
108:         dif[3] = - x[0] * x[1] * x[2] * x[4] / 120.0
109:         dif[4] = - x[0] * x[1] * x[2] * x[3] / 120.0
110:         return dif
111: 
112:     def fg45(self, x):
113:         return self.f45(x), self.g45(x)
114: 
115:     # tests
116:     # minimize with method=TNC
117:     def test_minimize_tnc1(self):
118:         x0, bnds = [-2, 1], ([-np.inf, None], [-1.5, None])
119:         xopt = [1, 1]
120:         iterx = []  # to test callback
121: 
122:         res = optimize.minimize(self.f1, x0, method='TNC', jac=self.g1,
123:                                 bounds=bnds, options=self.opts,
124:                                 callback=iterx.append)
125:         assert_allclose(res.fun, self.f1(xopt), atol=1e-8)
126:         assert_equal(len(iterx), res.nit)
127: 
128:     def test_minimize_tnc1b(self):
129:         x0, bnds = np.matrix([-2, 1]), ([-np.inf, None],[-1.5, None])
130:         xopt = [1, 1]
131:         x = optimize.minimize(self.f1, x0, method='TNC',
132:                               bounds=bnds, options=self.opts).x
133:         assert_allclose(self.f1(x), self.f1(xopt), atol=1e-4)
134: 
135:     def test_minimize_tnc1c(self):
136:         x0, bnds = [-2, 1], ([-np.inf, None],[-1.5, None])
137:         xopt = [1, 1]
138:         x = optimize.minimize(self.fg1, x0, method='TNC',
139:                               jac=True, bounds=bnds,
140:                               options=self.opts).x
141:         assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8)
142: 
143:     def test_minimize_tnc2(self):
144:         x0, bnds = [-2, 1], ([-np.inf, None], [1.5, None])
145:         xopt = [-1.2210262419616387, 1.5]
146:         x = optimize.minimize(self.f1, x0, method='TNC',
147:                               jac=self.g1, bounds=bnds,
148:                               options=self.opts).x
149:         assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8)
150: 
151:     def test_minimize_tnc3(self):
152:         x0, bnds = [10, 1], ([-np.inf, None], [0.0, None])
153:         xopt = [0, 0]
154:         x = optimize.minimize(self.f3, x0, method='TNC',
155:                               jac=self.g3, bounds=bnds,
156:                               options=self.opts).x
157:         assert_allclose(self.f3(x), self.f3(xopt), atol=1e-8)
158: 
159:     def test_minimize_tnc4(self):
160:         x0,bnds = [1.125, 0.125], [(1, None), (0, None)]
161:         xopt = [1, 0]
162:         x = optimize.minimize(self.f4, x0, method='TNC',
163:                               jac=self.g4, bounds=bnds,
164:                               options=self.opts).x
165:         assert_allclose(self.f4(x), self.f4(xopt), atol=1e-8)
166: 
167:     def test_minimize_tnc5(self):
168:         x0, bnds = [0, 0], [(-1.5, 4),(-3, 3)]
169:         xopt = [-0.54719755119659763, -1.5471975511965976]
170:         x = optimize.minimize(self.f5, x0, method='TNC',
171:                               jac=self.g5, bounds=bnds,
172:                               options=self.opts).x
173:         assert_allclose(self.f5(x), self.f5(xopt), atol=1e-8)
174: 
175:     def test_minimize_tnc38(self):
176:         x0, bnds = np.array([-3, -1, -3, -1]), [(-10, 10)]*4
177:         xopt = [1]*4
178:         x = optimize.minimize(self.f38, x0, method='TNC',
179:                               jac=self.g38, bounds=bnds,
180:                               options=self.opts).x
181:         assert_allclose(self.f38(x), self.f38(xopt), atol=1e-8)
182: 
183:     def test_minimize_tnc45(self):
184:         x0, bnds = [2] * 5, [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
185:         xopt = [1, 2, 3, 4, 5]
186:         x = optimize.minimize(self.f45, x0, method='TNC',
187:                               jac=self.g45, bounds=bnds,
188:                               options=self.opts).x
189:         assert_allclose(self.f45(x), self.f45(xopt), atol=1e-8)
190: 
191:     # fmin_tnc
192:     def test_tnc1(self):
193:         fg, x, bounds = self.fg1, [-2, 1], ([-np.inf, None], [-1.5, None])
194:         xopt = [1, 1]
195: 
196:         x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds, args=(100.0, ),
197:                                       messages=optimize.tnc.MSG_NONE,
198:                                       maxfun=200)
199: 
200:         assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
201:                         err_msg="TNC failed with status: " +
202:                                 optimize.tnc.RCSTRINGS[rc])
203: 
204:     def test_tnc1b(self):
205:         x, bounds = [-2, 1], ([-np.inf, None], [-1.5, None])
206:         xopt = [1, 1]
207: 
208:         x, nf, rc = optimize.fmin_tnc(self.f1, x, approx_grad=True,
209:                                       bounds=bounds,
210:                                       messages=optimize.tnc.MSG_NONE,
211:                                       maxfun=200)
212: 
213:         assert_allclose(self.f1(x), self.f1(xopt), atol=1e-4,
214:                         err_msg="TNC failed with status: " +
215:                                 optimize.tnc.RCSTRINGS[rc])
216: 
217:     def test_tnc1c(self):
218:         x, bounds = [-2, 1], ([-np.inf, None], [-1.5, None])
219:         xopt = [1, 1]
220: 
221:         x, nf, rc = optimize.fmin_tnc(self.f1, x, fprime=self.g1,
222:                                       bounds=bounds,
223:                                       messages=optimize.tnc.MSG_NONE,
224:                                       maxfun=200)
225: 
226:         assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
227:                         err_msg="TNC failed with status: " +
228:                                 optimize.tnc.RCSTRINGS[rc])
229: 
230:     def test_tnc2(self):
231:         fg, x, bounds = self.fg1, [-2, 1], ([-np.inf, None], [1.5, None])
232:         xopt = [-1.2210262419616387, 1.5]
233: 
234:         x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
235:                                       messages=optimize.tnc.MSG_NONE,
236:                                       maxfun=200)
237: 
238:         assert_allclose(self.f1(x), self.f1(xopt), atol=1e-8,
239:                         err_msg="TNC failed with status: " +
240:                                 optimize.tnc.RCSTRINGS[rc])
241: 
242:     def test_tnc3(self):
243:         fg, x, bounds = self.fg3, [10, 1], ([-np.inf, None], [0.0, None])
244:         xopt = [0, 0]
245: 
246:         x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
247:                                       messages=optimize.tnc.MSG_NONE,
248:                                       maxfun=200)
249: 
250:         assert_allclose(self.f3(x), self.f3(xopt), atol=1e-8,
251:                         err_msg="TNC failed with status: " +
252:                                 optimize.tnc.RCSTRINGS[rc])
253: 
254:     def test_tnc4(self):
255:         fg, x, bounds = self.fg4, [1.125, 0.125], [(1, None), (0, None)]
256:         xopt = [1, 0]
257: 
258:         x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
259:                                       messages=optimize.tnc.MSG_NONE,
260:                                       maxfun=200)
261: 
262:         assert_allclose(self.f4(x), self.f4(xopt), atol=1e-8,
263:                         err_msg="TNC failed with status: " +
264:                                 optimize.tnc.RCSTRINGS[rc])
265: 
266:     def test_tnc5(self):
267:         fg, x, bounds = self.fg5, [0, 0], [(-1.5, 4),(-3, 3)]
268:         xopt = [-0.54719755119659763, -1.5471975511965976]
269: 
270:         x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
271:                                       messages=optimize.tnc.MSG_NONE,
272:                                       maxfun=200)
273: 
274:         assert_allclose(self.f5(x), self.f5(xopt), atol=1e-8,
275:                         err_msg="TNC failed with status: " +
276:                                 optimize.tnc.RCSTRINGS[rc])
277: 
278:     def test_tnc38(self):
279:         fg, x, bounds = self.fg38, np.array([-3, -1, -3, -1]), [(-10, 10)]*4
280:         xopt = [1]*4
281: 
282:         x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
283:                                       messages=optimize.tnc.MSG_NONE,
284:                                       maxfun=200)
285: 
286:         assert_allclose(self.f38(x), self.f38(xopt), atol=1e-8,
287:                         err_msg="TNC failed with status: " +
288:                                 optimize.tnc.RCSTRINGS[rc])
289: 
290:     def test_tnc45(self):
291:         fg, x, bounds = self.fg45, [2] * 5, [(0, 1), (0, 2), (0, 3),
292:                                              (0, 4), (0, 5)]
293:         xopt = [1, 2, 3, 4, 5]
294: 
295:         x, nf, rc = optimize.fmin_tnc(fg, x, bounds=bounds,
296:                                       messages=optimize.tnc.MSG_NONE,
297:                                       maxfun=200)
298: 
299:         assert_allclose(self.f45(x), self.f45(xopt), atol=1e-8,
300:                         err_msg="TNC failed with status: " +
301:                                 optimize.tnc.RCSTRINGS[rc])
302: 
303: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_231746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nUnit tests for TNC optimization routine from tnc.py\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_allclose, assert_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_231747 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_231747) is not StypyTypeError):

    if (import_231747 != 'pyd_module'):
        __import__(import_231747)
        sys_modules_231748 = sys.modules[import_231747]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_231748.module_type_store, module_type_store, ['assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_231748, sys_modules_231748.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal'], [assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_231747)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy import optimize' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_231749 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy')

if (type(import_231749) is not StypyTypeError):

    if (import_231749 != 'pyd_module'):
        __import__(import_231749)
        sys_modules_231750 = sys.modules[import_231749]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', sys_modules_231750.module_type_store, module_type_store, ['optimize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_231750, sys_modules_231750.module_type_store, module_type_store)
    else:
        from scipy import optimize

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', None, module_type_store, ['optimize'], [optimize])

else:
    # Assigning a type to the variable 'scipy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', import_231749)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_231751 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_231751) is not StypyTypeError):

    if (import_231751 != 'pyd_module'):
        __import__(import_231751)
        sys_modules_231752 = sys.modules[import_231751]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_231752.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_231751)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from math import pow' statement (line 9)
try:
    from math import pow

except:
    pow = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'math', None, module_type_store, ['pow'], [pow])

# Declaration of the 'TestTnc' class

class TestTnc(object, ):
    str_231753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', "TNC non-linear optimization.\n\n    These tests are taken from Prof. K. Schittkowski's test examples\n    for constrained non-linear programming.\n\n    http://www.uni-bayreuth.de/departments/math/~kschittkowski/home.htm\n\n    ")

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.setup_method.__dict__.__setitem__('stypy_function_name', 'TestTnc.setup_method')
        TestTnc.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Dict to a Attribute (line 23):
        
        # Assigning a Dict to a Attribute (line 23):
        
        # Obtaining an instance of the builtin type 'dict' (line 23)
        dict_231754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 23)
        # Adding element type (key, value) (line 23)
        str_231755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'str', 'disp')
        # Getting the type of 'False' (line 23)
        False_231756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'False')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 20), dict_231754, (str_231755, False_231756))
        # Adding element type (key, value) (line 23)
        str_231757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 36), 'str', 'maxiter')
        int_231758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 47), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 20), dict_231754, (str_231757, int_231758))
        
        # Getting the type of 'self' (line 23)
        self_231759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'opts' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_231759, 'opts', dict_231754)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_231760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231760)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_231760


    @norecursion
    def f1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_231761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'float')
        defaults = [float_231761]
        # Create a new context for function 'f1'
        module_type_store = module_type_store.open_function_context('f1', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.f1.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.f1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.f1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.f1.__dict__.__setitem__('stypy_function_name', 'TestTnc.f1')
        TestTnc.f1.__dict__.__setitem__('stypy_param_names_list', ['x', 'a'])
        TestTnc.f1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.f1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.f1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.f1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.f1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.f1.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.f1', ['x', 'a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f1', localization, ['x', 'a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f1(...)' code ##################

        # Getting the type of 'a' (line 27)
        a_231762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'a')
        
        # Call to pow(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Obtaining the type of the subscript
        int_231764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'int')
        # Getting the type of 'x' (line 27)
        x_231765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___231766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 24), x_231765, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_231767 = invoke(stypy.reporting.localization.Localization(__file__, 27, 24), getitem___231766, int_231764)
        
        
        # Call to pow(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Obtaining the type of the subscript
        int_231769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 37), 'int')
        # Getting the type of 'x' (line 27)
        x_231770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 35), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___231771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 35), x_231770, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_231772 = invoke(stypy.reporting.localization.Localization(__file__, 27, 35), getitem___231771, int_231769)
        
        int_231773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 41), 'int')
        # Processing the call keyword arguments (line 27)
        kwargs_231774 = {}
        # Getting the type of 'pow' (line 27)
        pow_231768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'pow', False)
        # Calling pow(args, kwargs) (line 27)
        pow_call_result_231775 = invoke(stypy.reporting.localization.Localization(__file__, 27, 31), pow_231768, *[subscript_call_result_231772, int_231773], **kwargs_231774)
        
        # Applying the binary operator '-' (line 27)
        result_sub_231776 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 24), '-', subscript_call_result_231767, pow_call_result_231775)
        
        int_231777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 46), 'int')
        # Processing the call keyword arguments (line 27)
        kwargs_231778 = {}
        # Getting the type of 'pow' (line 27)
        pow_231763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'pow', False)
        # Calling pow(args, kwargs) (line 27)
        pow_call_result_231779 = invoke(stypy.reporting.localization.Localization(__file__, 27, 19), pow_231763, *[result_sub_231776, int_231777], **kwargs_231778)
        
        # Applying the binary operator '*' (line 27)
        result_mul_231780 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 15), '*', a_231762, pow_call_result_231779)
        
        
        # Call to pow(...): (line 27)
        # Processing the call arguments (line 27)
        float_231782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 55), 'float')
        
        # Obtaining the type of the subscript
        int_231783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 63), 'int')
        # Getting the type of 'x' (line 27)
        x_231784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 61), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___231785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 61), x_231784, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_231786 = invoke(stypy.reporting.localization.Localization(__file__, 27, 61), getitem___231785, int_231783)
        
        # Applying the binary operator '-' (line 27)
        result_sub_231787 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 55), '-', float_231782, subscript_call_result_231786)
        
        int_231788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 67), 'int')
        # Processing the call keyword arguments (line 27)
        kwargs_231789 = {}
        # Getting the type of 'pow' (line 27)
        pow_231781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 51), 'pow', False)
        # Calling pow(args, kwargs) (line 27)
        pow_call_result_231790 = invoke(stypy.reporting.localization.Localization(__file__, 27, 51), pow_231781, *[result_sub_231787, int_231788], **kwargs_231789)
        
        # Applying the binary operator '+' (line 27)
        result_add_231791 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 15), '+', result_mul_231780, pow_call_result_231790)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', result_add_231791)
        
        # ################# End of 'f1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f1' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_231792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231792)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f1'
        return stypy_return_type_231792


    @norecursion
    def g1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_231793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'float')
        defaults = [float_231793]
        # Create a new context for function 'g1'
        module_type_store = module_type_store.open_function_context('g1', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.g1.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.g1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.g1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.g1.__dict__.__setitem__('stypy_function_name', 'TestTnc.g1')
        TestTnc.g1.__dict__.__setitem__('stypy_param_names_list', ['x', 'a'])
        TestTnc.g1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.g1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.g1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.g1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.g1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.g1.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.g1', ['x', 'a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'g1', localization, ['x', 'a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'g1(...)' code ##################

        
        # Assigning a List to a Name (line 30):
        
        # Assigning a List to a Name (line 30):
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_231794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        int_231795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 14), list_231794, int_231795)
        # Adding element type (line 30)
        int_231796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 14), list_231794, int_231796)
        
        # Assigning a type to the variable 'dif' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'dif', list_231794)
        
        # Assigning a BinOp to a Subscript (line 31):
        
        # Assigning a BinOp to a Subscript (line 31):
        int_231797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'int')
        # Getting the type of 'a' (line 31)
        a_231798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'a')
        # Applying the binary operator '*' (line 31)
        result_mul_231799 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 17), '*', int_231797, a_231798)
        
        
        # Obtaining the type of the subscript
        int_231800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'int')
        # Getting the type of 'x' (line 31)
        x_231801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___231802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 26), x_231801, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_231803 = invoke(stypy.reporting.localization.Localization(__file__, 31, 26), getitem___231802, int_231800)
        
        
        # Call to pow(...): (line 31)
        # Processing the call arguments (line 31)
        
        # Obtaining the type of the subscript
        int_231805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 39), 'int')
        # Getting the type of 'x' (line 31)
        x_231806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___231807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 37), x_231806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_231808 = invoke(stypy.reporting.localization.Localization(__file__, 31, 37), getitem___231807, int_231805)
        
        int_231809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 43), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_231810 = {}
        # Getting the type of 'pow' (line 31)
        pow_231804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'pow', False)
        # Calling pow(args, kwargs) (line 31)
        pow_call_result_231811 = invoke(stypy.reporting.localization.Localization(__file__, 31, 33), pow_231804, *[subscript_call_result_231808, int_231809], **kwargs_231810)
        
        # Applying the binary operator '-' (line 31)
        result_sub_231812 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 26), '-', subscript_call_result_231803, pow_call_result_231811)
        
        # Applying the binary operator '*' (line 31)
        result_mul_231813 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 23), '*', result_mul_231799, result_sub_231812)
        
        # Getting the type of 'dif' (line 31)
        dif_231814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'dif')
        int_231815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'int')
        # Storing an element on a container (line 31)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 8), dif_231814, (int_231815, result_mul_231813))
        
        # Assigning a BinOp to a Subscript (line 32):
        
        # Assigning a BinOp to a Subscript (line 32):
        float_231816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 17), 'float')
        
        # Obtaining the type of the subscript
        int_231817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'int')
        # Getting the type of 'x' (line 32)
        x_231818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'x')
        # Obtaining the member '__getitem__' of a type (line 32)
        getitem___231819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), x_231818, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 32)
        subscript_call_result_231820 = invoke(stypy.reporting.localization.Localization(__file__, 32, 25), getitem___231819, int_231817)
        
        
        # Obtaining the type of the subscript
        int_231821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'int')
        # Getting the type of 'dif' (line 32)
        dif_231822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'dif')
        # Obtaining the member '__getitem__' of a type (line 32)
        getitem___231823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 33), dif_231822, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 32)
        subscript_call_result_231824 = invoke(stypy.reporting.localization.Localization(__file__, 32, 33), getitem___231823, int_231821)
        
        float_231825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 42), 'float')
        # Applying the binary operator '-' (line 32)
        result_sub_231826 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 33), '-', subscript_call_result_231824, float_231825)
        
        # Applying the binary operator '*' (line 32)
        result_mul_231827 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 25), '*', subscript_call_result_231820, result_sub_231826)
        
        float_231828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 49), 'float')
        # Applying the binary operator '+' (line 32)
        result_add_231829 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 25), '+', result_mul_231827, float_231828)
        
        # Applying the binary operator '*' (line 32)
        result_mul_231830 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 17), '*', float_231816, result_add_231829)
        
        # Getting the type of 'dif' (line 32)
        dif_231831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'dif')
        int_231832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 12), 'int')
        # Storing an element on a container (line 32)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 8), dif_231831, (int_231832, result_mul_231830))
        # Getting the type of 'dif' (line 33)
        dif_231833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'dif')
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', dif_231833)
        
        # ################# End of 'g1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'g1' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_231834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'g1'
        return stypy_return_type_231834


    @norecursion
    def fg1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_231835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'float')
        defaults = [float_231835]
        # Create a new context for function 'fg1'
        module_type_store = module_type_store.open_function_context('fg1', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.fg1.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.fg1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.fg1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.fg1.__dict__.__setitem__('stypy_function_name', 'TestTnc.fg1')
        TestTnc.fg1.__dict__.__setitem__('stypy_param_names_list', ['x', 'a'])
        TestTnc.fg1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.fg1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.fg1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.fg1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.fg1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.fg1.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.fg1', ['x', 'a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fg1', localization, ['x', 'a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fg1(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 36)
        tuple_231836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 36)
        # Adding element type (line 36)
        
        # Call to f1(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'x' (line 36)
        x_231839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'x', False)
        # Getting the type of 'a' (line 36)
        a_231840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'a', False)
        # Processing the call keyword arguments (line 36)
        kwargs_231841 = {}
        # Getting the type of 'self' (line 36)
        self_231837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'self', False)
        # Obtaining the member 'f1' of a type (line 36)
        f1_231838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), self_231837, 'f1')
        # Calling f1(args, kwargs) (line 36)
        f1_call_result_231842 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), f1_231838, *[x_231839, a_231840], **kwargs_231841)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 15), tuple_231836, f1_call_result_231842)
        # Adding element type (line 36)
        
        # Call to g1(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'x' (line 36)
        x_231845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 38), 'x', False)
        # Getting the type of 'a' (line 36)
        a_231846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), 'a', False)
        # Processing the call keyword arguments (line 36)
        kwargs_231847 = {}
        # Getting the type of 'self' (line 36)
        self_231843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'self', False)
        # Obtaining the member 'g1' of a type (line 36)
        g1_231844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 30), self_231843, 'g1')
        # Calling g1(args, kwargs) (line 36)
        g1_call_result_231848 = invoke(stypy.reporting.localization.Localization(__file__, 36, 30), g1_231844, *[x_231845, a_231846], **kwargs_231847)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 15), tuple_231836, g1_call_result_231848)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', tuple_231836)
        
        # ################# End of 'fg1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fg1' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_231849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fg1'
        return stypy_return_type_231849


    @norecursion
    def f3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f3'
        module_type_store = module_type_store.open_function_context('f3', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.f3.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.f3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.f3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.f3.__dict__.__setitem__('stypy_function_name', 'TestTnc.f3')
        TestTnc.f3.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.f3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.f3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.f3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.f3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.f3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.f3.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.f3', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f3', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f3(...)' code ##################

        
        # Obtaining the type of the subscript
        int_231850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'int')
        # Getting the type of 'x' (line 39)
        x_231851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'x')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___231852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), x_231851, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_231853 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), getitem___231852, int_231850)
        
        
        # Call to pow(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining the type of the subscript
        int_231855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'int')
        # Getting the type of 'x' (line 39)
        x_231856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___231857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 26), x_231856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_231858 = invoke(stypy.reporting.localization.Localization(__file__, 39, 26), getitem___231857, int_231855)
        
        
        # Obtaining the type of the subscript
        int_231859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'int')
        # Getting the type of 'x' (line 39)
        x_231860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___231861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 33), x_231860, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_231862 = invoke(stypy.reporting.localization.Localization(__file__, 39, 33), getitem___231861, int_231859)
        
        # Applying the binary operator '-' (line 39)
        result_sub_231863 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 26), '-', subscript_call_result_231858, subscript_call_result_231862)
        
        int_231864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 39), 'int')
        # Processing the call keyword arguments (line 39)
        kwargs_231865 = {}
        # Getting the type of 'pow' (line 39)
        pow_231854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'pow', False)
        # Calling pow(args, kwargs) (line 39)
        pow_call_result_231866 = invoke(stypy.reporting.localization.Localization(__file__, 39, 22), pow_231854, *[result_sub_231863, int_231864], **kwargs_231865)
        
        float_231867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 44), 'float')
        # Applying the binary operator '*' (line 39)
        result_mul_231868 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 22), '*', pow_call_result_231866, float_231867)
        
        # Applying the binary operator '+' (line 39)
        result_add_231869 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '+', subscript_call_result_231853, result_mul_231868)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', result_add_231869)
        
        # ################# End of 'f3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f3' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_231870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231870)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f3'
        return stypy_return_type_231870


    @norecursion
    def g3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'g3'
        module_type_store = module_type_store.open_function_context('g3', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.g3.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.g3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.g3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.g3.__dict__.__setitem__('stypy_function_name', 'TestTnc.g3')
        TestTnc.g3.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.g3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.g3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.g3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.g3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.g3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.g3.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.g3', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'g3', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'g3(...)' code ##################

        
        # Assigning a List to a Name (line 42):
        
        # Assigning a List to a Name (line 42):
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_231871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        int_231872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 14), list_231871, int_231872)
        # Adding element type (line 42)
        int_231873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 14), list_231871, int_231873)
        
        # Assigning a type to the variable 'dif' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'dif', list_231871)
        
        # Assigning a BinOp to a Subscript (line 43):
        
        # Assigning a BinOp to a Subscript (line 43):
        float_231874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'float')
        
        # Obtaining the type of the subscript
        int_231875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'int')
        # Getting the type of 'x' (line 43)
        x_231876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'x')
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___231877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 25), x_231876, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_231878 = invoke(stypy.reporting.localization.Localization(__file__, 43, 25), getitem___231877, int_231875)
        
        
        # Obtaining the type of the subscript
        int_231879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'int')
        # Getting the type of 'x' (line 43)
        x_231880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'x')
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___231881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), x_231880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_231882 = invoke(stypy.reporting.localization.Localization(__file__, 43, 32), getitem___231881, int_231879)
        
        # Applying the binary operator '-' (line 43)
        result_sub_231883 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 25), '-', subscript_call_result_231878, subscript_call_result_231882)
        
        # Applying the binary operator '*' (line 43)
        result_mul_231884 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 17), '*', float_231874, result_sub_231883)
        
        float_231885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 40), 'float')
        # Applying the binary operator '*' (line 43)
        result_mul_231886 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 38), '*', result_mul_231884, float_231885)
        
        # Getting the type of 'dif' (line 43)
        dif_231887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'dif')
        int_231888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 12), 'int')
        # Storing an element on a container (line 43)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 8), dif_231887, (int_231888, result_mul_231886))
        
        # Assigning a BinOp to a Subscript (line 44):
        
        # Assigning a BinOp to a Subscript (line 44):
        float_231889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'float')
        
        # Obtaining the type of the subscript
        int_231890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'int')
        # Getting the type of 'dif' (line 44)
        dif_231891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'dif')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___231892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), dif_231891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_231893 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), getitem___231892, int_231890)
        
        # Applying the binary operator '-' (line 44)
        result_sub_231894 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 17), '-', float_231889, subscript_call_result_231893)
        
        # Getting the type of 'dif' (line 44)
        dif_231895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'dif')
        int_231896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'int')
        # Storing an element on a container (line 44)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 8), dif_231895, (int_231896, result_sub_231894))
        # Getting the type of 'dif' (line 45)
        dif_231897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'dif')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', dif_231897)
        
        # ################# End of 'g3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'g3' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_231898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231898)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'g3'
        return stypy_return_type_231898


    @norecursion
    def fg3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fg3'
        module_type_store = module_type_store.open_function_context('fg3', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.fg3.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.fg3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.fg3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.fg3.__dict__.__setitem__('stypy_function_name', 'TestTnc.fg3')
        TestTnc.fg3.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.fg3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.fg3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.fg3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.fg3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.fg3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.fg3.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.fg3', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fg3', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fg3(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 48)
        tuple_231899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 48)
        # Adding element type (line 48)
        
        # Call to f3(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'x' (line 48)
        x_231902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'x', False)
        # Processing the call keyword arguments (line 48)
        kwargs_231903 = {}
        # Getting the type of 'self' (line 48)
        self_231900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'self', False)
        # Obtaining the member 'f3' of a type (line 48)
        f3_231901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), self_231900, 'f3')
        # Calling f3(args, kwargs) (line 48)
        f3_call_result_231904 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), f3_231901, *[x_231902], **kwargs_231903)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), tuple_231899, f3_call_result_231904)
        # Adding element type (line 48)
        
        # Call to g3(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'x' (line 48)
        x_231907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 35), 'x', False)
        # Processing the call keyword arguments (line 48)
        kwargs_231908 = {}
        # Getting the type of 'self' (line 48)
        self_231905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'self', False)
        # Obtaining the member 'g3' of a type (line 48)
        g3_231906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 27), self_231905, 'g3')
        # Calling g3(args, kwargs) (line 48)
        g3_call_result_231909 = invoke(stypy.reporting.localization.Localization(__file__, 48, 27), g3_231906, *[x_231907], **kwargs_231908)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 15), tuple_231899, g3_call_result_231909)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', tuple_231899)
        
        # ################# End of 'fg3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fg3' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_231910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fg3'
        return stypy_return_type_231910


    @norecursion
    def f4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f4'
        module_type_store = module_type_store.open_function_context('f4', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.f4.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.f4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.f4.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.f4.__dict__.__setitem__('stypy_function_name', 'TestTnc.f4')
        TestTnc.f4.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.f4.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.f4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.f4.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.f4.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.f4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.f4.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.f4', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f4', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f4(...)' code ##################

        
        # Call to pow(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Obtaining the type of the subscript
        int_231912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'int')
        # Getting the type of 'x' (line 51)
        x_231913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___231914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 19), x_231913, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_231915 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), getitem___231914, int_231912)
        
        float_231916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'float')
        # Applying the binary operator '+' (line 51)
        result_add_231917 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), '+', subscript_call_result_231915, float_231916)
        
        int_231918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'int')
        # Processing the call keyword arguments (line 51)
        kwargs_231919 = {}
        # Getting the type of 'pow' (line 51)
        pow_231911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'pow', False)
        # Calling pow(args, kwargs) (line 51)
        pow_call_result_231920 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), pow_231911, *[result_add_231917, int_231918], **kwargs_231919)
        
        float_231921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'float')
        # Applying the binary operator 'div' (line 51)
        result_div_231922 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), 'div', pow_call_result_231920, float_231921)
        
        
        # Obtaining the type of the subscript
        int_231923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 44), 'int')
        # Getting the type of 'x' (line 51)
        x_231924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'x')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___231925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 42), x_231924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_231926 = invoke(stypy.reporting.localization.Localization(__file__, 51, 42), getitem___231925, int_231923)
        
        # Applying the binary operator '+' (line 51)
        result_add_231927 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), '+', result_div_231922, subscript_call_result_231926)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', result_add_231927)
        
        # ################# End of 'f4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f4' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_231928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f4'
        return stypy_return_type_231928


    @norecursion
    def g4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'g4'
        module_type_store = module_type_store.open_function_context('g4', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.g4.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.g4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.g4.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.g4.__dict__.__setitem__('stypy_function_name', 'TestTnc.g4')
        TestTnc.g4.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.g4.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.g4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.g4.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.g4.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.g4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.g4.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.g4', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'g4', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'g4(...)' code ##################

        
        # Assigning a List to a Name (line 54):
        
        # Assigning a List to a Name (line 54):
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_231929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_231930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 14), list_231929, int_231930)
        # Adding element type (line 54)
        int_231931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 14), list_231929, int_231931)
        
        # Assigning a type to the variable 'dif' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'dif', list_231929)
        
        # Assigning a Call to a Subscript (line 55):
        
        # Assigning a Call to a Subscript (line 55):
        
        # Call to pow(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining the type of the subscript
        int_231933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'int')
        # Getting the type of 'x' (line 55)
        x_231934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___231935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 21), x_231934, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_231936 = invoke(stypy.reporting.localization.Localization(__file__, 55, 21), getitem___231935, int_231933)
        
        float_231937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'float')
        # Applying the binary operator '+' (line 55)
        result_add_231938 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 21), '+', subscript_call_result_231936, float_231937)
        
        int_231939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 33), 'int')
        # Processing the call keyword arguments (line 55)
        kwargs_231940 = {}
        # Getting the type of 'pow' (line 55)
        pow_231932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'pow', False)
        # Calling pow(args, kwargs) (line 55)
        pow_call_result_231941 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), pow_231932, *[result_add_231938, int_231939], **kwargs_231940)
        
        # Getting the type of 'dif' (line 55)
        dif_231942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'dif')
        int_231943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 12), 'int')
        # Storing an element on a container (line 55)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), dif_231942, (int_231943, pow_call_result_231941))
        
        # Assigning a Num to a Subscript (line 56):
        
        # Assigning a Num to a Subscript (line 56):
        float_231944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'float')
        # Getting the type of 'dif' (line 56)
        dif_231945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'dif')
        int_231946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'int')
        # Storing an element on a container (line 56)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), dif_231945, (int_231946, float_231944))
        # Getting the type of 'dif' (line 57)
        dif_231947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'dif')
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', dif_231947)
        
        # ################# End of 'g4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'g4' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_231948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'g4'
        return stypy_return_type_231948


    @norecursion
    def fg4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fg4'
        module_type_store = module_type_store.open_function_context('fg4', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.fg4.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.fg4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.fg4.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.fg4.__dict__.__setitem__('stypy_function_name', 'TestTnc.fg4')
        TestTnc.fg4.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.fg4.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.fg4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.fg4.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.fg4.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.fg4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.fg4.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.fg4', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fg4', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fg4(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 60)
        tuple_231949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 60)
        # Adding element type (line 60)
        
        # Call to f4(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'x' (line 60)
        x_231952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'x', False)
        # Processing the call keyword arguments (line 60)
        kwargs_231953 = {}
        # Getting the type of 'self' (line 60)
        self_231950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'self', False)
        # Obtaining the member 'f4' of a type (line 60)
        f4_231951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), self_231950, 'f4')
        # Calling f4(args, kwargs) (line 60)
        f4_call_result_231954 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), f4_231951, *[x_231952], **kwargs_231953)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 15), tuple_231949, f4_call_result_231954)
        # Adding element type (line 60)
        
        # Call to g4(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'x' (line 60)
        x_231957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'x', False)
        # Processing the call keyword arguments (line 60)
        kwargs_231958 = {}
        # Getting the type of 'self' (line 60)
        self_231955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'self', False)
        # Obtaining the member 'g4' of a type (line 60)
        g4_231956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 27), self_231955, 'g4')
        # Calling g4(args, kwargs) (line 60)
        g4_call_result_231959 = invoke(stypy.reporting.localization.Localization(__file__, 60, 27), g4_231956, *[x_231957], **kwargs_231958)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 15), tuple_231949, g4_call_result_231959)
        
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', tuple_231949)
        
        # ################# End of 'fg4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fg4' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_231960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_231960)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fg4'
        return stypy_return_type_231960


    @norecursion
    def f5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f5'
        module_type_store = module_type_store.open_function_context('f5', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.f5.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.f5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.f5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.f5.__dict__.__setitem__('stypy_function_name', 'TestTnc.f5')
        TestTnc.f5.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.f5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.f5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.f5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.f5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.f5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.f5.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.f5', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f5', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f5(...)' code ##################

        
        # Call to sin(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining the type of the subscript
        int_231963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'int')
        # Getting the type of 'x' (line 63)
        x_231964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___231965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), x_231964, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_231966 = invoke(stypy.reporting.localization.Localization(__file__, 63, 22), getitem___231965, int_231963)
        
        
        # Obtaining the type of the subscript
        int_231967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'int')
        # Getting the type of 'x' (line 63)
        x_231968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___231969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 29), x_231968, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_231970 = invoke(stypy.reporting.localization.Localization(__file__, 63, 29), getitem___231969, int_231967)
        
        # Applying the binary operator '+' (line 63)
        result_add_231971 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 22), '+', subscript_call_result_231966, subscript_call_result_231970)
        
        # Processing the call keyword arguments (line 63)
        kwargs_231972 = {}
        # Getting the type of 'np' (line 63)
        np_231961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'np', False)
        # Obtaining the member 'sin' of a type (line 63)
        sin_231962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 15), np_231961, 'sin')
        # Calling sin(args, kwargs) (line 63)
        sin_call_result_231973 = invoke(stypy.reporting.localization.Localization(__file__, 63, 15), sin_231962, *[result_add_231971], **kwargs_231972)
        
        
        # Call to pow(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining the type of the subscript
        int_231975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 43), 'int')
        # Getting the type of 'x' (line 63)
        x_231976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 41), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___231977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 41), x_231976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_231978 = invoke(stypy.reporting.localization.Localization(__file__, 63, 41), getitem___231977, int_231975)
        
        
        # Obtaining the type of the subscript
        int_231979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 50), 'int')
        # Getting the type of 'x' (line 63)
        x_231980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 48), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___231981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 48), x_231980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_231982 = invoke(stypy.reporting.localization.Localization(__file__, 63, 48), getitem___231981, int_231979)
        
        # Applying the binary operator '-' (line 63)
        result_sub_231983 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 41), '-', subscript_call_result_231978, subscript_call_result_231982)
        
        int_231984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 54), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_231985 = {}
        # Getting the type of 'pow' (line 63)
        pow_231974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'pow', False)
        # Calling pow(args, kwargs) (line 63)
        pow_call_result_231986 = invoke(stypy.reporting.localization.Localization(__file__, 63, 37), pow_231974, *[result_sub_231983, int_231984], **kwargs_231985)
        
        # Applying the binary operator '+' (line 63)
        result_add_231987 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 15), '+', sin_call_result_231973, pow_call_result_231986)
        
        float_231988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'float')
        
        # Obtaining the type of the subscript
        int_231989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 24), 'int')
        # Getting the type of 'x' (line 64)
        x_231990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'x')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___231991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 22), x_231990, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_231992 = invoke(stypy.reporting.localization.Localization(__file__, 64, 22), getitem___231991, int_231989)
        
        # Applying the binary operator '*' (line 64)
        result_mul_231993 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '*', float_231988, subscript_call_result_231992)
        
        # Applying the binary operator '-' (line 63)
        result_sub_231994 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 57), '-', result_add_231987, result_mul_231993)
        
        float_231995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'float')
        
        # Obtaining the type of the subscript
        int_231996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 37), 'int')
        # Getting the type of 'x' (line 64)
        x_231997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 'x')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___231998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 35), x_231997, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_231999 = invoke(stypy.reporting.localization.Localization(__file__, 64, 35), getitem___231998, int_231996)
        
        # Applying the binary operator '*' (line 64)
        result_mul_232000 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 29), '*', float_231995, subscript_call_result_231999)
        
        # Applying the binary operator '+' (line 64)
        result_add_232001 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 27), '+', result_sub_231994, result_mul_232000)
        
        float_232002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 42), 'float')
        # Applying the binary operator '+' (line 64)
        result_add_232003 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 40), '+', result_add_232001, float_232002)
        
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', result_add_232003)
        
        # ################# End of 'f5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f5' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_232004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232004)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f5'
        return stypy_return_type_232004


    @norecursion
    def g5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'g5'
        module_type_store = module_type_store.open_function_context('g5', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.g5.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.g5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.g5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.g5.__dict__.__setitem__('stypy_function_name', 'TestTnc.g5')
        TestTnc.g5.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.g5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.g5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.g5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.g5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.g5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.g5.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.g5', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'g5', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'g5(...)' code ##################

        
        # Assigning a List to a Name (line 67):
        
        # Assigning a List to a Name (line 67):
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_232005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        int_232006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 14), list_232005, int_232006)
        # Adding element type (line 67)
        int_232007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 14), list_232005, int_232007)
        
        # Assigning a type to the variable 'dif' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'dif', list_232005)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to cos(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining the type of the subscript
        int_232010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'int')
        # Getting the type of 'x' (line 68)
        x_232011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___232012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 20), x_232011, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_232013 = invoke(stypy.reporting.localization.Localization(__file__, 68, 20), getitem___232012, int_232010)
        
        
        # Obtaining the type of the subscript
        int_232014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'int')
        # Getting the type of 'x' (line 68)
        x_232015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___232016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 27), x_232015, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_232017 = invoke(stypy.reporting.localization.Localization(__file__, 68, 27), getitem___232016, int_232014)
        
        # Applying the binary operator '+' (line 68)
        result_add_232018 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 20), '+', subscript_call_result_232013, subscript_call_result_232017)
        
        # Processing the call keyword arguments (line 68)
        kwargs_232019 = {}
        # Getting the type of 'np' (line 68)
        np_232008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'np', False)
        # Obtaining the member 'cos' of a type (line 68)
        cos_232009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 13), np_232008, 'cos')
        # Calling cos(args, kwargs) (line 68)
        cos_call_result_232020 = invoke(stypy.reporting.localization.Localization(__file__, 68, 13), cos_232009, *[result_add_232018], **kwargs_232019)
        
        # Assigning a type to the variable 'v1' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'v1', cos_call_result_232020)
        
        # Assigning a BinOp to a Name (line 69):
        
        # Assigning a BinOp to a Name (line 69):
        float_232021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'float')
        
        # Obtaining the type of the subscript
        int_232022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'int')
        # Getting the type of 'x' (line 69)
        x_232023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'x')
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___232024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 18), x_232023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_232025 = invoke(stypy.reporting.localization.Localization(__file__, 69, 18), getitem___232024, int_232022)
        
        
        # Obtaining the type of the subscript
        int_232026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 27), 'int')
        # Getting the type of 'x' (line 69)
        x_232027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'x')
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___232028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), x_232027, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_232029 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), getitem___232028, int_232026)
        
        # Applying the binary operator '-' (line 69)
        result_sub_232030 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 18), '-', subscript_call_result_232025, subscript_call_result_232029)
        
        # Applying the binary operator '*' (line 69)
        result_mul_232031 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '*', float_232021, result_sub_232030)
        
        # Assigning a type to the variable 'v2' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'v2', result_mul_232031)
        
        # Assigning a BinOp to a Subscript (line 71):
        
        # Assigning a BinOp to a Subscript (line 71):
        # Getting the type of 'v1' (line 71)
        v1_232032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'v1')
        # Getting the type of 'v2' (line 71)
        v2_232033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'v2')
        # Applying the binary operator '+' (line 71)
        result_add_232034 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 17), '+', v1_232032, v2_232033)
        
        float_232035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'float')
        # Applying the binary operator '-' (line 71)
        result_sub_232036 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 25), '-', result_add_232034, float_232035)
        
        # Getting the type of 'dif' (line 71)
        dif_232037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'dif')
        int_232038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 12), 'int')
        # Storing an element on a container (line 71)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 8), dif_232037, (int_232038, result_sub_232036))
        
        # Assigning a BinOp to a Subscript (line 72):
        
        # Assigning a BinOp to a Subscript (line 72):
        # Getting the type of 'v1' (line 72)
        v1_232039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'v1')
        # Getting the type of 'v2' (line 72)
        v2_232040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'v2')
        # Applying the binary operator '-' (line 72)
        result_sub_232041 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 17), '-', v1_232039, v2_232040)
        
        float_232042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 27), 'float')
        # Applying the binary operator '+' (line 72)
        result_add_232043 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 25), '+', result_sub_232041, float_232042)
        
        # Getting the type of 'dif' (line 72)
        dif_232044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'dif')
        int_232045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'int')
        # Storing an element on a container (line 72)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 8), dif_232044, (int_232045, result_add_232043))
        # Getting the type of 'dif' (line 73)
        dif_232046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'dif')
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', dif_232046)
        
        # ################# End of 'g5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'g5' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_232047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232047)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'g5'
        return stypy_return_type_232047


    @norecursion
    def fg5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fg5'
        module_type_store = module_type_store.open_function_context('fg5', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.fg5.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.fg5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.fg5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.fg5.__dict__.__setitem__('stypy_function_name', 'TestTnc.fg5')
        TestTnc.fg5.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.fg5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.fg5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.fg5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.fg5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.fg5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.fg5.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.fg5', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fg5', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fg5(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_232048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        
        # Call to f5(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'x' (line 76)
        x_232051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'x', False)
        # Processing the call keyword arguments (line 76)
        kwargs_232052 = {}
        # Getting the type of 'self' (line 76)
        self_232049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'self', False)
        # Obtaining the member 'f5' of a type (line 76)
        f5_232050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 15), self_232049, 'f5')
        # Calling f5(args, kwargs) (line 76)
        f5_call_result_232053 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), f5_232050, *[x_232051], **kwargs_232052)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 15), tuple_232048, f5_call_result_232053)
        # Adding element type (line 76)
        
        # Call to g5(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'x' (line 76)
        x_232056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 35), 'x', False)
        # Processing the call keyword arguments (line 76)
        kwargs_232057 = {}
        # Getting the type of 'self' (line 76)
        self_232054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'self', False)
        # Obtaining the member 'g5' of a type (line 76)
        g5_232055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 27), self_232054, 'g5')
        # Calling g5(args, kwargs) (line 76)
        g5_call_result_232058 = invoke(stypy.reporting.localization.Localization(__file__, 76, 27), g5_232055, *[x_232056], **kwargs_232057)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 15), tuple_232048, g5_call_result_232058)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', tuple_232048)
        
        # ################# End of 'fg5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fg5' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_232059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fg5'
        return stypy_return_type_232059


    @norecursion
    def f38(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f38'
        module_type_store = module_type_store.open_function_context('f38', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.f38.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.f38.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.f38.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.f38.__dict__.__setitem__('stypy_function_name', 'TestTnc.f38')
        TestTnc.f38.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.f38.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.f38.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.f38.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.f38.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.f38.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.f38.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.f38', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f38', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f38(...)' code ##################

        float_232060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 16), 'float')
        
        # Call to pow(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining the type of the subscript
        int_232062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'int')
        # Getting the type of 'x' (line 79)
        x_232063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___232064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 28), x_232063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_232065 = invoke(stypy.reporting.localization.Localization(__file__, 79, 28), getitem___232064, int_232062)
        
        
        # Call to pow(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining the type of the subscript
        int_232067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 41), 'int')
        # Getting the type of 'x' (line 79)
        x_232068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 39), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___232069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 39), x_232068, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_232070 = invoke(stypy.reporting.localization.Localization(__file__, 79, 39), getitem___232069, int_232067)
        
        int_232071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 45), 'int')
        # Processing the call keyword arguments (line 79)
        kwargs_232072 = {}
        # Getting the type of 'pow' (line 79)
        pow_232066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 35), 'pow', False)
        # Calling pow(args, kwargs) (line 79)
        pow_call_result_232073 = invoke(stypy.reporting.localization.Localization(__file__, 79, 35), pow_232066, *[subscript_call_result_232070, int_232071], **kwargs_232072)
        
        # Applying the binary operator '-' (line 79)
        result_sub_232074 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 28), '-', subscript_call_result_232065, pow_call_result_232073)
        
        int_232075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 49), 'int')
        # Processing the call keyword arguments (line 79)
        kwargs_232076 = {}
        # Getting the type of 'pow' (line 79)
        pow_232061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'pow', False)
        # Calling pow(args, kwargs) (line 79)
        pow_call_result_232077 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), pow_232061, *[result_sub_232074, int_232075], **kwargs_232076)
        
        # Applying the binary operator '*' (line 79)
        result_mul_232078 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 16), '*', float_232060, pow_call_result_232077)
        
        
        # Call to pow(...): (line 80)
        # Processing the call arguments (line 80)
        float_232080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'float')
        
        # Obtaining the type of the subscript
        int_232081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 28), 'int')
        # Getting the type of 'x' (line 80)
        x_232082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___232083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 26), x_232082, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_232084 = invoke(stypy.reporting.localization.Localization(__file__, 80, 26), getitem___232083, int_232081)
        
        # Applying the binary operator '-' (line 80)
        result_sub_232085 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 20), '-', float_232080, subscript_call_result_232084)
        
        int_232086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 32), 'int')
        # Processing the call keyword arguments (line 80)
        kwargs_232087 = {}
        # Getting the type of 'pow' (line 80)
        pow_232079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'pow', False)
        # Calling pow(args, kwargs) (line 80)
        pow_call_result_232088 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), pow_232079, *[result_sub_232085, int_232086], **kwargs_232087)
        
        # Applying the binary operator '+' (line 79)
        result_add_232089 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 16), '+', result_mul_232078, pow_call_result_232088)
        
        float_232090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 37), 'float')
        
        # Call to pow(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Obtaining the type of the subscript
        int_232092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 50), 'int')
        # Getting the type of 'x' (line 80)
        x_232093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 48), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___232094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 48), x_232093, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_232095 = invoke(stypy.reporting.localization.Localization(__file__, 80, 48), getitem___232094, int_232092)
        
        
        # Call to pow(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Obtaining the type of the subscript
        int_232097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 61), 'int')
        # Getting the type of 'x' (line 80)
        x_232098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 59), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 80)
        getitem___232099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 59), x_232098, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 80)
        subscript_call_result_232100 = invoke(stypy.reporting.localization.Localization(__file__, 80, 59), getitem___232099, int_232097)
        
        int_232101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 65), 'int')
        # Processing the call keyword arguments (line 80)
        kwargs_232102 = {}
        # Getting the type of 'pow' (line 80)
        pow_232096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 55), 'pow', False)
        # Calling pow(args, kwargs) (line 80)
        pow_call_result_232103 = invoke(stypy.reporting.localization.Localization(__file__, 80, 55), pow_232096, *[subscript_call_result_232100, int_232101], **kwargs_232102)
        
        # Applying the binary operator '-' (line 80)
        result_sub_232104 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 48), '-', subscript_call_result_232095, pow_call_result_232103)
        
        int_232105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 69), 'int')
        # Processing the call keyword arguments (line 80)
        kwargs_232106 = {}
        # Getting the type of 'pow' (line 80)
        pow_232091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'pow', False)
        # Calling pow(args, kwargs) (line 80)
        pow_call_result_232107 = invoke(stypy.reporting.localization.Localization(__file__, 80, 44), pow_232091, *[result_sub_232104, int_232105], **kwargs_232106)
        
        # Applying the binary operator '*' (line 80)
        result_mul_232108 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 37), '*', float_232090, pow_call_result_232107)
        
        # Applying the binary operator '+' (line 80)
        result_add_232109 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 35), '+', result_add_232089, result_mul_232108)
        
        
        # Call to pow(...): (line 81)
        # Processing the call arguments (line 81)
        float_232111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'float')
        
        # Obtaining the type of the subscript
        int_232112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'int')
        # Getting the type of 'x' (line 81)
        x_232113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___232114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 26), x_232113, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_232115 = invoke(stypy.reporting.localization.Localization(__file__, 81, 26), getitem___232114, int_232112)
        
        # Applying the binary operator '-' (line 81)
        result_sub_232116 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 20), '-', float_232111, subscript_call_result_232115)
        
        int_232117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 32), 'int')
        # Processing the call keyword arguments (line 81)
        kwargs_232118 = {}
        # Getting the type of 'pow' (line 81)
        pow_232110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'pow', False)
        # Calling pow(args, kwargs) (line 81)
        pow_call_result_232119 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), pow_232110, *[result_sub_232116, int_232117], **kwargs_232118)
        
        # Applying the binary operator '+' (line 80)
        result_add_232120 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 72), '+', result_add_232109, pow_call_result_232119)
        
        float_232121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 37), 'float')
        
        # Call to pow(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining the type of the subscript
        int_232123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 51), 'int')
        # Getting the type of 'x' (line 81)
        x_232124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 49), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___232125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 49), x_232124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_232126 = invoke(stypy.reporting.localization.Localization(__file__, 81, 49), getitem___232125, int_232123)
        
        float_232127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 56), 'float')
        # Applying the binary operator '-' (line 81)
        result_sub_232128 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 49), '-', subscript_call_result_232126, float_232127)
        
        int_232129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 61), 'int')
        # Processing the call keyword arguments (line 81)
        kwargs_232130 = {}
        # Getting the type of 'pow' (line 81)
        pow_232122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 45), 'pow', False)
        # Calling pow(args, kwargs) (line 81)
        pow_call_result_232131 = invoke(stypy.reporting.localization.Localization(__file__, 81, 45), pow_232122, *[result_sub_232128, int_232129], **kwargs_232130)
        
        
        # Call to pow(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Obtaining the type of the subscript
        int_232133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 51), 'int')
        # Getting the type of 'x' (line 82)
        x_232134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 49), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___232135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 49), x_232134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_232136 = invoke(stypy.reporting.localization.Localization(__file__, 82, 49), getitem___232135, int_232133)
        
        float_232137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 56), 'float')
        # Applying the binary operator '-' (line 82)
        result_sub_232138 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 49), '-', subscript_call_result_232136, float_232137)
        
        int_232139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 61), 'int')
        # Processing the call keyword arguments (line 82)
        kwargs_232140 = {}
        # Getting the type of 'pow' (line 82)
        pow_232132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 45), 'pow', False)
        # Calling pow(args, kwargs) (line 82)
        pow_call_result_232141 = invoke(stypy.reporting.localization.Localization(__file__, 82, 45), pow_232132, *[result_sub_232138, int_232139], **kwargs_232140)
        
        # Applying the binary operator '+' (line 81)
        result_add_232142 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 45), '+', pow_call_result_232131, pow_call_result_232141)
        
        # Applying the binary operator '*' (line 81)
        result_mul_232143 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 37), '*', float_232121, result_add_232142)
        
        # Applying the binary operator '+' (line 81)
        result_add_232144 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 35), '+', result_add_232120, result_mul_232143)
        
        float_232145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 16), 'float')
        
        # Obtaining the type of the subscript
        int_232146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'int')
        # Getting the type of 'x' (line 83)
        x_232147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'x')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___232148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 24), x_232147, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_232149 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), getitem___232148, int_232146)
        
        float_232150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 31), 'float')
        # Applying the binary operator '-' (line 83)
        result_sub_232151 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 24), '-', subscript_call_result_232149, float_232150)
        
        # Applying the binary operator '*' (line 83)
        result_mul_232152 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 16), '*', float_232145, result_sub_232151)
        
        
        # Obtaining the type of the subscript
        int_232153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 41), 'int')
        # Getting the type of 'x' (line 83)
        x_232154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 39), 'x')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___232155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 39), x_232154, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_232156 = invoke(stypy.reporting.localization.Localization(__file__, 83, 39), getitem___232155, int_232153)
        
        float_232157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 46), 'float')
        # Applying the binary operator '-' (line 83)
        result_sub_232158 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 39), '-', subscript_call_result_232156, float_232157)
        
        # Applying the binary operator '*' (line 83)
        result_mul_232159 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 36), '*', result_mul_232152, result_sub_232158)
        
        # Applying the binary operator '+' (line 82)
        result_add_232160 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 65), '+', result_add_232144, result_mul_232159)
        
        float_232161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 54), 'float')
        # Applying the binary operator '*' (line 79)
        result_mul_232162 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), '*', result_add_232160, float_232161)
        
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', result_mul_232162)
        
        # ################# End of 'f38(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f38' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_232163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232163)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f38'
        return stypy_return_type_232163


    @norecursion
    def g38(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'g38'
        module_type_store = module_type_store.open_function_context('g38', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.g38.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.g38.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.g38.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.g38.__dict__.__setitem__('stypy_function_name', 'TestTnc.g38')
        TestTnc.g38.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.g38.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.g38.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.g38.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.g38.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.g38.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.g38.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.g38', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'g38', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'g38(...)' code ##################

        
        # Assigning a List to a Name (line 86):
        
        # Assigning a List to a Name (line 86):
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_232164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        int_232165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), list_232164, int_232165)
        # Adding element type (line 86)
        int_232166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), list_232164, int_232166)
        # Adding element type (line 86)
        int_232167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), list_232164, int_232167)
        # Adding element type (line 86)
        int_232168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 14), list_232164, int_232168)
        
        # Assigning a type to the variable 'dif' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'dif', list_232164)
        
        # Assigning a BinOp to a Subscript (line 87):
        
        # Assigning a BinOp to a Subscript (line 87):
        float_232169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'float')
        
        # Obtaining the type of the subscript
        int_232170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'int')
        # Getting the type of 'x' (line 87)
        x_232171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'x')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___232172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 27), x_232171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_232173 = invoke(stypy.reporting.localization.Localization(__file__, 87, 27), getitem___232172, int_232170)
        
        # Applying the binary operator '*' (line 87)
        result_mul_232174 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 18), '*', float_232169, subscript_call_result_232173)
        
        
        # Obtaining the type of the subscript
        int_232175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 37), 'int')
        # Getting the type of 'x' (line 87)
        x_232176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'x')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___232177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 35), x_232176, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_232178 = invoke(stypy.reporting.localization.Localization(__file__, 87, 35), getitem___232177, int_232175)
        
        
        # Call to pow(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Obtaining the type of the subscript
        int_232180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 48), 'int')
        # Getting the type of 'x' (line 87)
        x_232181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 46), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___232182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 46), x_232181, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_232183 = invoke(stypy.reporting.localization.Localization(__file__, 87, 46), getitem___232182, int_232180)
        
        int_232184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 52), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_232185 = {}
        # Getting the type of 'pow' (line 87)
        pow_232179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 42), 'pow', False)
        # Calling pow(args, kwargs) (line 87)
        pow_call_result_232186 = invoke(stypy.reporting.localization.Localization(__file__, 87, 42), pow_232179, *[subscript_call_result_232183, int_232184], **kwargs_232185)
        
        # Applying the binary operator '-' (line 87)
        result_sub_232187 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 35), '-', subscript_call_result_232178, pow_call_result_232186)
        
        # Applying the binary operator '*' (line 87)
        result_mul_232188 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 32), '*', result_mul_232174, result_sub_232187)
        
        float_232189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'float')
        float_232190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'float')
        
        # Obtaining the type of the subscript
        int_232191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 33), 'int')
        # Getting the type of 'x' (line 88)
        x_232192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'x')
        # Obtaining the member '__getitem__' of a type (line 88)
        getitem___232193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 31), x_232192, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 88)
        subscript_call_result_232194 = invoke(stypy.reporting.localization.Localization(__file__, 88, 31), getitem___232193, int_232191)
        
        # Applying the binary operator '-' (line 88)
        result_sub_232195 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 25), '-', float_232190, subscript_call_result_232194)
        
        # Applying the binary operator '*' (line 88)
        result_mul_232196 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 18), '*', float_232189, result_sub_232195)
        
        # Applying the binary operator '-' (line 87)
        result_sub_232197 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 18), '-', result_mul_232188, result_mul_232196)
        
        float_232198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 40), 'float')
        # Applying the binary operator '*' (line 87)
        result_mul_232199 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 17), '*', result_sub_232197, float_232198)
        
        # Getting the type of 'dif' (line 87)
        dif_232200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'dif')
        int_232201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 12), 'int')
        # Storing an element on a container (line 87)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), dif_232200, (int_232201, result_mul_232199))
        
        # Assigning a BinOp to a Subscript (line 89):
        
        # Assigning a BinOp to a Subscript (line 89):
        float_232202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'float')
        
        # Obtaining the type of the subscript
        int_232203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'int')
        # Getting the type of 'x' (line 89)
        x_232204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'x')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___232205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), x_232204, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_232206 = invoke(stypy.reporting.localization.Localization(__file__, 89, 27), getitem___232205, int_232203)
        
        
        # Call to pow(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining the type of the subscript
        int_232208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 40), 'int')
        # Getting the type of 'x' (line 89)
        x_232209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 38), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___232210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 38), x_232209, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_232211 = invoke(stypy.reporting.localization.Localization(__file__, 89, 38), getitem___232210, int_232208)
        
        int_232212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 44), 'int')
        # Processing the call keyword arguments (line 89)
        kwargs_232213 = {}
        # Getting the type of 'pow' (line 89)
        pow_232207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'pow', False)
        # Calling pow(args, kwargs) (line 89)
        pow_call_result_232214 = invoke(stypy.reporting.localization.Localization(__file__, 89, 34), pow_232207, *[subscript_call_result_232211, int_232212], **kwargs_232213)
        
        # Applying the binary operator '-' (line 89)
        result_sub_232215 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 27), '-', subscript_call_result_232206, pow_call_result_232214)
        
        # Applying the binary operator '*' (line 89)
        result_mul_232216 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 18), '*', float_232202, result_sub_232215)
        
        float_232217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 50), 'float')
        
        # Obtaining the type of the subscript
        int_232218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 60), 'int')
        # Getting the type of 'x' (line 89)
        x_232219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 58), 'x')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___232220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 58), x_232219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_232221 = invoke(stypy.reporting.localization.Localization(__file__, 89, 58), getitem___232220, int_232218)
        
        float_232222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 65), 'float')
        # Applying the binary operator '-' (line 89)
        result_sub_232223 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 58), '-', subscript_call_result_232221, float_232222)
        
        # Applying the binary operator '*' (line 89)
        result_mul_232224 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 50), '*', float_232217, result_sub_232223)
        
        # Applying the binary operator '+' (line 89)
        result_add_232225 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 18), '+', result_mul_232216, result_mul_232224)
        
        float_232226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'float')
        
        # Obtaining the type of the subscript
        int_232227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'int')
        # Getting the type of 'x' (line 90)
        x_232228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___232229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 26), x_232228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_232230 = invoke(stypy.reporting.localization.Localization(__file__, 90, 26), getitem___232229, int_232227)
        
        float_232231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 33), 'float')
        # Applying the binary operator '-' (line 90)
        result_sub_232232 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 26), '-', subscript_call_result_232230, float_232231)
        
        # Applying the binary operator '*' (line 90)
        result_mul_232233 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 18), '*', float_232226, result_sub_232232)
        
        # Applying the binary operator '+' (line 89)
        result_add_232234 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 70), '+', result_add_232225, result_mul_232233)
        
        float_232235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 41), 'float')
        # Applying the binary operator '*' (line 89)
        result_mul_232236 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 17), '*', result_add_232234, float_232235)
        
        # Getting the type of 'dif' (line 89)
        dif_232237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'dif')
        int_232238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'int')
        # Storing an element on a container (line 89)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), dif_232237, (int_232238, result_mul_232236))
        
        # Assigning a BinOp to a Subscript (line 91):
        
        # Assigning a BinOp to a Subscript (line 91):
        float_232239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'float')
        
        # Obtaining the type of the subscript
        int_232240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 30), 'int')
        # Getting the type of 'x' (line 91)
        x_232241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'x')
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___232242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 28), x_232241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_232243 = invoke(stypy.reporting.localization.Localization(__file__, 91, 28), getitem___232242, int_232240)
        
        # Applying the binary operator '*' (line 91)
        result_mul_232244 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 18), '*', float_232239, subscript_call_result_232243)
        
        
        # Obtaining the type of the subscript
        int_232245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 38), 'int')
        # Getting the type of 'x' (line 91)
        x_232246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 36), 'x')
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___232247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 36), x_232246, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_232248 = invoke(stypy.reporting.localization.Localization(__file__, 91, 36), getitem___232247, int_232245)
        
        
        # Call to pow(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining the type of the subscript
        int_232250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 49), 'int')
        # Getting the type of 'x' (line 91)
        x_232251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___232252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 47), x_232251, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_232253 = invoke(stypy.reporting.localization.Localization(__file__, 91, 47), getitem___232252, int_232250)
        
        int_232254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 53), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_232255 = {}
        # Getting the type of 'pow' (line 91)
        pow_232249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'pow', False)
        # Calling pow(args, kwargs) (line 91)
        pow_call_result_232256 = invoke(stypy.reporting.localization.Localization(__file__, 91, 43), pow_232249, *[subscript_call_result_232253, int_232254], **kwargs_232255)
        
        # Applying the binary operator '-' (line 91)
        result_sub_232257 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 36), '-', subscript_call_result_232248, pow_call_result_232256)
        
        # Applying the binary operator '*' (line 91)
        result_mul_232258 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 33), '*', result_mul_232244, result_sub_232257)
        
        float_232259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 18), 'float')
        float_232260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'float')
        
        # Obtaining the type of the subscript
        int_232261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 33), 'int')
        # Getting the type of 'x' (line 92)
        x_232262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'x')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___232263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 31), x_232262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_232264 = invoke(stypy.reporting.localization.Localization(__file__, 92, 31), getitem___232263, int_232261)
        
        # Applying the binary operator '-' (line 92)
        result_sub_232265 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 25), '-', float_232260, subscript_call_result_232264)
        
        # Applying the binary operator '*' (line 92)
        result_mul_232266 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 18), '*', float_232259, result_sub_232265)
        
        # Applying the binary operator '-' (line 91)
        result_sub_232267 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 18), '-', result_mul_232258, result_mul_232266)
        
        float_232268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 40), 'float')
        # Applying the binary operator '*' (line 91)
        result_mul_232269 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 17), '*', result_sub_232267, float_232268)
        
        # Getting the type of 'dif' (line 91)
        dif_232270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'dif')
        int_232271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 12), 'int')
        # Storing an element on a container (line 91)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 8), dif_232270, (int_232271, result_mul_232269))
        
        # Assigning a BinOp to a Subscript (line 93):
        
        # Assigning a BinOp to a Subscript (line 93):
        float_232272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 18), 'float')
        
        # Obtaining the type of the subscript
        int_232273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'int')
        # Getting the type of 'x' (line 93)
        x_232274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'x')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___232275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 27), x_232274, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_232276 = invoke(stypy.reporting.localization.Localization(__file__, 93, 27), getitem___232275, int_232273)
        
        
        # Call to pow(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining the type of the subscript
        int_232278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 40), 'int')
        # Getting the type of 'x' (line 93)
        x_232279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___232280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 38), x_232279, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_232281 = invoke(stypy.reporting.localization.Localization(__file__, 93, 38), getitem___232280, int_232278)
        
        int_232282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 44), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_232283 = {}
        # Getting the type of 'pow' (line 93)
        pow_232277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 'pow', False)
        # Calling pow(args, kwargs) (line 93)
        pow_call_result_232284 = invoke(stypy.reporting.localization.Localization(__file__, 93, 34), pow_232277, *[subscript_call_result_232281, int_232282], **kwargs_232283)
        
        # Applying the binary operator '-' (line 93)
        result_sub_232285 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 27), '-', subscript_call_result_232276, pow_call_result_232284)
        
        # Applying the binary operator '*' (line 93)
        result_mul_232286 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 18), '*', float_232272, result_sub_232285)
        
        float_232287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 50), 'float')
        
        # Obtaining the type of the subscript
        int_232288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 60), 'int')
        # Getting the type of 'x' (line 93)
        x_232289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 58), 'x')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___232290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 58), x_232289, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_232291 = invoke(stypy.reporting.localization.Localization(__file__, 93, 58), getitem___232290, int_232288)
        
        float_232292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 65), 'float')
        # Applying the binary operator '-' (line 93)
        result_sub_232293 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 58), '-', subscript_call_result_232291, float_232292)
        
        # Applying the binary operator '*' (line 93)
        result_mul_232294 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 50), '*', float_232287, result_sub_232293)
        
        # Applying the binary operator '+' (line 93)
        result_add_232295 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 18), '+', result_mul_232286, result_mul_232294)
        
        float_232296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 18), 'float')
        
        # Obtaining the type of the subscript
        int_232297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'int')
        # Getting the type of 'x' (line 94)
        x_232298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___232299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 26), x_232298, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_232300 = invoke(stypy.reporting.localization.Localization(__file__, 94, 26), getitem___232299, int_232297)
        
        float_232301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 33), 'float')
        # Applying the binary operator '-' (line 94)
        result_sub_232302 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 26), '-', subscript_call_result_232300, float_232301)
        
        # Applying the binary operator '*' (line 94)
        result_mul_232303 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 18), '*', float_232296, result_sub_232302)
        
        # Applying the binary operator '+' (line 93)
        result_add_232304 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 70), '+', result_add_232295, result_mul_232303)
        
        float_232305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 41), 'float')
        # Applying the binary operator '*' (line 93)
        result_mul_232306 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 17), '*', result_add_232304, float_232305)
        
        # Getting the type of 'dif' (line 93)
        dif_232307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'dif')
        int_232308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 12), 'int')
        # Storing an element on a container (line 93)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 8), dif_232307, (int_232308, result_mul_232306))
        # Getting the type of 'dif' (line 95)
        dif_232309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'dif')
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', dif_232309)
        
        # ################# End of 'g38(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'g38' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_232310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'g38'
        return stypy_return_type_232310


    @norecursion
    def fg38(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fg38'
        module_type_store = module_type_store.open_function_context('fg38', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.fg38.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.fg38.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.fg38.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.fg38.__dict__.__setitem__('stypy_function_name', 'TestTnc.fg38')
        TestTnc.fg38.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.fg38.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.fg38.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.fg38.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.fg38.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.fg38.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.fg38.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.fg38', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fg38', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fg38(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 98)
        tuple_232311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 98)
        # Adding element type (line 98)
        
        # Call to f38(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x' (line 98)
        x_232314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'x', False)
        # Processing the call keyword arguments (line 98)
        kwargs_232315 = {}
        # Getting the type of 'self' (line 98)
        self_232312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'self', False)
        # Obtaining the member 'f38' of a type (line 98)
        f38_232313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), self_232312, 'f38')
        # Calling f38(args, kwargs) (line 98)
        f38_call_result_232316 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), f38_232313, *[x_232314], **kwargs_232315)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 15), tuple_232311, f38_call_result_232316)
        # Adding element type (line 98)
        
        # Call to g38(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x' (line 98)
        x_232319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'x', False)
        # Processing the call keyword arguments (line 98)
        kwargs_232320 = {}
        # Getting the type of 'self' (line 98)
        self_232317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'self', False)
        # Obtaining the member 'g38' of a type (line 98)
        g38_232318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 28), self_232317, 'g38')
        # Calling g38(args, kwargs) (line 98)
        g38_call_result_232321 = invoke(stypy.reporting.localization.Localization(__file__, 98, 28), g38_232318, *[x_232319], **kwargs_232320)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 15), tuple_232311, g38_call_result_232321)
        
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', tuple_232311)
        
        # ################# End of 'fg38(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fg38' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_232322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fg38'
        return stypy_return_type_232322


    @norecursion
    def f45(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f45'
        module_type_store = module_type_store.open_function_context('f45', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.f45.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.f45.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.f45.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.f45.__dict__.__setitem__('stypy_function_name', 'TestTnc.f45')
        TestTnc.f45.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.f45.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.f45.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.f45.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.f45.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.f45.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.f45.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.f45', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f45', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f45(...)' code ##################

        float_232323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'float')
        
        # Obtaining the type of the subscript
        int_232324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'int')
        # Getting the type of 'x' (line 101)
        x_232325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___232326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 21), x_232325, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_232327 = invoke(stypy.reporting.localization.Localization(__file__, 101, 21), getitem___232326, int_232324)
        
        
        # Obtaining the type of the subscript
        int_232328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 30), 'int')
        # Getting the type of 'x' (line 101)
        x_232329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'x')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___232330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), x_232329, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_232331 = invoke(stypy.reporting.localization.Localization(__file__, 101, 28), getitem___232330, int_232328)
        
        # Applying the binary operator '*' (line 101)
        result_mul_232332 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 21), '*', subscript_call_result_232327, subscript_call_result_232331)
        
        
        # Obtaining the type of the subscript
        int_232333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 37), 'int')
        # Getting the type of 'x' (line 101)
        x_232334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'x')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___232335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), x_232334, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_232336 = invoke(stypy.reporting.localization.Localization(__file__, 101, 35), getitem___232335, int_232333)
        
        # Applying the binary operator '*' (line 101)
        result_mul_232337 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 33), '*', result_mul_232332, subscript_call_result_232336)
        
        
        # Obtaining the type of the subscript
        int_232338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 44), 'int')
        # Getting the type of 'x' (line 101)
        x_232339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'x')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___232340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 42), x_232339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_232341 = invoke(stypy.reporting.localization.Localization(__file__, 101, 42), getitem___232340, int_232338)
        
        # Applying the binary operator '*' (line 101)
        result_mul_232342 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 40), '*', result_mul_232337, subscript_call_result_232341)
        
        
        # Obtaining the type of the subscript
        int_232343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 51), 'int')
        # Getting the type of 'x' (line 101)
        x_232344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 49), 'x')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___232345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 49), x_232344, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_232346 = invoke(stypy.reporting.localization.Localization(__file__, 101, 49), getitem___232345, int_232343)
        
        # Applying the binary operator '*' (line 101)
        result_mul_232347 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 47), '*', result_mul_232342, subscript_call_result_232346)
        
        float_232348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 56), 'float')
        # Applying the binary operator 'div' (line 101)
        result_div_232349 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 54), 'div', result_mul_232347, float_232348)
        
        # Applying the binary operator '-' (line 101)
        result_sub_232350 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '-', float_232323, result_div_232349)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', result_sub_232350)
        
        # ################# End of 'f45(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f45' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_232351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f45'
        return stypy_return_type_232351


    @norecursion
    def g45(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'g45'
        module_type_store = module_type_store.open_function_context('g45', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.g45.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.g45.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.g45.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.g45.__dict__.__setitem__('stypy_function_name', 'TestTnc.g45')
        TestTnc.g45.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.g45.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.g45.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.g45.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.g45.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.g45.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.g45.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.g45', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'g45', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'g45(...)' code ##################

        
        # Assigning a BinOp to a Name (line 104):
        
        # Assigning a BinOp to a Name (line 104):
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_232352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        int_232353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 14), list_232352, int_232353)
        
        int_232354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'int')
        # Applying the binary operator '*' (line 104)
        result_mul_232355 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 14), '*', list_232352, int_232354)
        
        # Assigning a type to the variable 'dif' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'dif', result_mul_232355)
        
        # Assigning a BinOp to a Subscript (line 105):
        
        # Assigning a BinOp to a Subscript (line 105):
        
        
        # Obtaining the type of the subscript
        int_232356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'int')
        # Getting the type of 'x' (line 105)
        x_232357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'x')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___232358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), x_232357, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_232359 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), getitem___232358, int_232356)
        
        # Applying the 'usub' unary operator (line 105)
        result___neg___232360 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 17), 'usub', subscript_call_result_232359)
        
        
        # Obtaining the type of the subscript
        int_232361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'int')
        # Getting the type of 'x' (line 105)
        x_232362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___232363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 26), x_232362, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_232364 = invoke(stypy.reporting.localization.Localization(__file__, 105, 26), getitem___232363, int_232361)
        
        # Applying the binary operator '*' (line 105)
        result_mul_232365 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 17), '*', result___neg___232360, subscript_call_result_232364)
        
        
        # Obtaining the type of the subscript
        int_232366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 35), 'int')
        # Getting the type of 'x' (line 105)
        x_232367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'x')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___232368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 33), x_232367, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_232369 = invoke(stypy.reporting.localization.Localization(__file__, 105, 33), getitem___232368, int_232366)
        
        # Applying the binary operator '*' (line 105)
        result_mul_232370 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 31), '*', result_mul_232365, subscript_call_result_232369)
        
        
        # Obtaining the type of the subscript
        int_232371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 42), 'int')
        # Getting the type of 'x' (line 105)
        x_232372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'x')
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___232373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 40), x_232372, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_232374 = invoke(stypy.reporting.localization.Localization(__file__, 105, 40), getitem___232373, int_232371)
        
        # Applying the binary operator '*' (line 105)
        result_mul_232375 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 38), '*', result_mul_232370, subscript_call_result_232374)
        
        float_232376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 47), 'float')
        # Applying the binary operator 'div' (line 105)
        result_div_232377 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 45), 'div', result_mul_232375, float_232376)
        
        # Getting the type of 'dif' (line 105)
        dif_232378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'dif')
        int_232379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'int')
        # Storing an element on a container (line 105)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 8), dif_232378, (int_232379, result_div_232377))
        
        # Assigning a BinOp to a Subscript (line 106):
        
        # Assigning a BinOp to a Subscript (line 106):
        
        
        # Obtaining the type of the subscript
        int_232380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'int')
        # Getting the type of 'x' (line 106)
        x_232381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'x')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___232382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 19), x_232381, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_232383 = invoke(stypy.reporting.localization.Localization(__file__, 106, 19), getitem___232382, int_232380)
        
        # Applying the 'usub' unary operator (line 106)
        result___neg___232384 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 17), 'usub', subscript_call_result_232383)
        
        
        # Obtaining the type of the subscript
        int_232385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'int')
        # Getting the type of 'x' (line 106)
        x_232386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___232387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 26), x_232386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_232388 = invoke(stypy.reporting.localization.Localization(__file__, 106, 26), getitem___232387, int_232385)
        
        # Applying the binary operator '*' (line 106)
        result_mul_232389 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 17), '*', result___neg___232384, subscript_call_result_232388)
        
        
        # Obtaining the type of the subscript
        int_232390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'int')
        # Getting the type of 'x' (line 106)
        x_232391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'x')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___232392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 33), x_232391, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_232393 = invoke(stypy.reporting.localization.Localization(__file__, 106, 33), getitem___232392, int_232390)
        
        # Applying the binary operator '*' (line 106)
        result_mul_232394 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 31), '*', result_mul_232389, subscript_call_result_232393)
        
        
        # Obtaining the type of the subscript
        int_232395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 42), 'int')
        # Getting the type of 'x' (line 106)
        x_232396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'x')
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___232397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 40), x_232396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_232398 = invoke(stypy.reporting.localization.Localization(__file__, 106, 40), getitem___232397, int_232395)
        
        # Applying the binary operator '*' (line 106)
        result_mul_232399 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 38), '*', result_mul_232394, subscript_call_result_232398)
        
        float_232400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 47), 'float')
        # Applying the binary operator 'div' (line 106)
        result_div_232401 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 45), 'div', result_mul_232399, float_232400)
        
        # Getting the type of 'dif' (line 106)
        dif_232402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'dif')
        int_232403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 12), 'int')
        # Storing an element on a container (line 106)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 8), dif_232402, (int_232403, result_div_232401))
        
        # Assigning a BinOp to a Subscript (line 107):
        
        # Assigning a BinOp to a Subscript (line 107):
        
        
        # Obtaining the type of the subscript
        int_232404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'int')
        # Getting the type of 'x' (line 107)
        x_232405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'x')
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___232406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), x_232405, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_232407 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), getitem___232406, int_232404)
        
        # Applying the 'usub' unary operator (line 107)
        result___neg___232408 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 17), 'usub', subscript_call_result_232407)
        
        
        # Obtaining the type of the subscript
        int_232409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 28), 'int')
        # Getting the type of 'x' (line 107)
        x_232410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___232411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 26), x_232410, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_232412 = invoke(stypy.reporting.localization.Localization(__file__, 107, 26), getitem___232411, int_232409)
        
        # Applying the binary operator '*' (line 107)
        result_mul_232413 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 17), '*', result___neg___232408, subscript_call_result_232412)
        
        
        # Obtaining the type of the subscript
        int_232414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 35), 'int')
        # Getting the type of 'x' (line 107)
        x_232415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'x')
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___232416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 33), x_232415, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_232417 = invoke(stypy.reporting.localization.Localization(__file__, 107, 33), getitem___232416, int_232414)
        
        # Applying the binary operator '*' (line 107)
        result_mul_232418 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 31), '*', result_mul_232413, subscript_call_result_232417)
        
        
        # Obtaining the type of the subscript
        int_232419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 42), 'int')
        # Getting the type of 'x' (line 107)
        x_232420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 40), 'x')
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___232421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 40), x_232420, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_232422 = invoke(stypy.reporting.localization.Localization(__file__, 107, 40), getitem___232421, int_232419)
        
        # Applying the binary operator '*' (line 107)
        result_mul_232423 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 38), '*', result_mul_232418, subscript_call_result_232422)
        
        float_232424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 47), 'float')
        # Applying the binary operator 'div' (line 107)
        result_div_232425 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 45), 'div', result_mul_232423, float_232424)
        
        # Getting the type of 'dif' (line 107)
        dif_232426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'dif')
        int_232427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 12), 'int')
        # Storing an element on a container (line 107)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 8), dif_232426, (int_232427, result_div_232425))
        
        # Assigning a BinOp to a Subscript (line 108):
        
        # Assigning a BinOp to a Subscript (line 108):
        
        
        # Obtaining the type of the subscript
        int_232428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'int')
        # Getting the type of 'x' (line 108)
        x_232429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'x')
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___232430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 19), x_232429, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_232431 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), getitem___232430, int_232428)
        
        # Applying the 'usub' unary operator (line 108)
        result___neg___232432 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 17), 'usub', subscript_call_result_232431)
        
        
        # Obtaining the type of the subscript
        int_232433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 28), 'int')
        # Getting the type of 'x' (line 108)
        x_232434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___232435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 26), x_232434, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_232436 = invoke(stypy.reporting.localization.Localization(__file__, 108, 26), getitem___232435, int_232433)
        
        # Applying the binary operator '*' (line 108)
        result_mul_232437 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 17), '*', result___neg___232432, subscript_call_result_232436)
        
        
        # Obtaining the type of the subscript
        int_232438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 35), 'int')
        # Getting the type of 'x' (line 108)
        x_232439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'x')
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___232440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 33), x_232439, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_232441 = invoke(stypy.reporting.localization.Localization(__file__, 108, 33), getitem___232440, int_232438)
        
        # Applying the binary operator '*' (line 108)
        result_mul_232442 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 31), '*', result_mul_232437, subscript_call_result_232441)
        
        
        # Obtaining the type of the subscript
        int_232443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 42), 'int')
        # Getting the type of 'x' (line 108)
        x_232444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'x')
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___232445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 40), x_232444, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_232446 = invoke(stypy.reporting.localization.Localization(__file__, 108, 40), getitem___232445, int_232443)
        
        # Applying the binary operator '*' (line 108)
        result_mul_232447 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 38), '*', result_mul_232442, subscript_call_result_232446)
        
        float_232448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 47), 'float')
        # Applying the binary operator 'div' (line 108)
        result_div_232449 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 45), 'div', result_mul_232447, float_232448)
        
        # Getting the type of 'dif' (line 108)
        dif_232450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'dif')
        int_232451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'int')
        # Storing an element on a container (line 108)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 8), dif_232450, (int_232451, result_div_232449))
        
        # Assigning a BinOp to a Subscript (line 109):
        
        # Assigning a BinOp to a Subscript (line 109):
        
        
        # Obtaining the type of the subscript
        int_232452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'int')
        # Getting the type of 'x' (line 109)
        x_232453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'x')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___232454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), x_232453, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_232455 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), getitem___232454, int_232452)
        
        # Applying the 'usub' unary operator (line 109)
        result___neg___232456 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 17), 'usub', subscript_call_result_232455)
        
        
        # Obtaining the type of the subscript
        int_232457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'int')
        # Getting the type of 'x' (line 109)
        x_232458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___232459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 26), x_232458, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_232460 = invoke(stypy.reporting.localization.Localization(__file__, 109, 26), getitem___232459, int_232457)
        
        # Applying the binary operator '*' (line 109)
        result_mul_232461 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 17), '*', result___neg___232456, subscript_call_result_232460)
        
        
        # Obtaining the type of the subscript
        int_232462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'int')
        # Getting the type of 'x' (line 109)
        x_232463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'x')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___232464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 33), x_232463, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_232465 = invoke(stypy.reporting.localization.Localization(__file__, 109, 33), getitem___232464, int_232462)
        
        # Applying the binary operator '*' (line 109)
        result_mul_232466 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 31), '*', result_mul_232461, subscript_call_result_232465)
        
        
        # Obtaining the type of the subscript
        int_232467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 42), 'int')
        # Getting the type of 'x' (line 109)
        x_232468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'x')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___232469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 40), x_232468, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_232470 = invoke(stypy.reporting.localization.Localization(__file__, 109, 40), getitem___232469, int_232467)
        
        # Applying the binary operator '*' (line 109)
        result_mul_232471 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 38), '*', result_mul_232466, subscript_call_result_232470)
        
        float_232472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 47), 'float')
        # Applying the binary operator 'div' (line 109)
        result_div_232473 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 45), 'div', result_mul_232471, float_232472)
        
        # Getting the type of 'dif' (line 109)
        dif_232474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'dif')
        int_232475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'int')
        # Storing an element on a container (line 109)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 8), dif_232474, (int_232475, result_div_232473))
        # Getting the type of 'dif' (line 110)
        dif_232476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'dif')
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', dif_232476)
        
        # ################# End of 'g45(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'g45' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_232477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232477)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'g45'
        return stypy_return_type_232477


    @norecursion
    def fg45(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fg45'
        module_type_store = module_type_store.open_function_context('fg45', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.fg45.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.fg45.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.fg45.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.fg45.__dict__.__setitem__('stypy_function_name', 'TestTnc.fg45')
        TestTnc.fg45.__dict__.__setitem__('stypy_param_names_list', ['x'])
        TestTnc.fg45.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.fg45.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.fg45.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.fg45.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.fg45.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.fg45.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.fg45', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fg45', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fg45(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_232478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        
        # Call to f45(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'x' (line 113)
        x_232481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'x', False)
        # Processing the call keyword arguments (line 113)
        kwargs_232482 = {}
        # Getting the type of 'self' (line 113)
        self_232479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'self', False)
        # Obtaining the member 'f45' of a type (line 113)
        f45_232480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 15), self_232479, 'f45')
        # Calling f45(args, kwargs) (line 113)
        f45_call_result_232483 = invoke(stypy.reporting.localization.Localization(__file__, 113, 15), f45_232480, *[x_232481], **kwargs_232482)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), tuple_232478, f45_call_result_232483)
        # Adding element type (line 113)
        
        # Call to g45(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'x' (line 113)
        x_232486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 37), 'x', False)
        # Processing the call keyword arguments (line 113)
        kwargs_232487 = {}
        # Getting the type of 'self' (line 113)
        self_232484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'self', False)
        # Obtaining the member 'g45' of a type (line 113)
        g45_232485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 28), self_232484, 'g45')
        # Calling g45(args, kwargs) (line 113)
        g45_call_result_232488 = invoke(stypy.reporting.localization.Localization(__file__, 113, 28), g45_232485, *[x_232486], **kwargs_232487)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), tuple_232478, g45_call_result_232488)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', tuple_232478)
        
        # ################# End of 'fg45(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fg45' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_232489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fg45'
        return stypy_return_type_232489


    @norecursion
    def test_minimize_tnc1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc1'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc1', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc1')
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc1(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 118):
        
        # Assigning a List to a Name (line 118):
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_232490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        int_232491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 19), list_232490, int_232491)
        # Adding element type (line 118)
        int_232492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 19), list_232490, int_232492)
        
        # Assigning a type to the variable 'tuple_assignment_231676' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_assignment_231676', list_232490)
        
        # Assigning a Tuple to a Name (line 118):
        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_232493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_232494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        
        # Getting the type of 'np' (line 118)
        np_232495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'np')
        # Obtaining the member 'inf' of a type (line 118)
        inf_232496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 31), np_232495, 'inf')
        # Applying the 'usub' unary operator (line 118)
        result___neg___232497 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 30), 'usub', inf_232496)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 29), list_232494, result___neg___232497)
        # Adding element type (line 118)
        # Getting the type of 'None' (line 118)
        None_232498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 39), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 29), list_232494, None_232498)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 29), tuple_232493, list_232494)
        # Adding element type (line 118)
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_232499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        # Adding element type (line 118)
        float_232500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 46), list_232499, float_232500)
        # Adding element type (line 118)
        # Getting the type of 'None' (line 118)
        None_232501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 53), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 46), list_232499, None_232501)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 29), tuple_232493, list_232499)
        
        # Assigning a type to the variable 'tuple_assignment_231677' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_assignment_231677', tuple_232493)
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'tuple_assignment_231676' (line 118)
        tuple_assignment_231676_232502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_assignment_231676')
        # Assigning a type to the variable 'x0' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'x0', tuple_assignment_231676_232502)
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'tuple_assignment_231677' (line 118)
        tuple_assignment_231677_232503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_assignment_231677')
        # Assigning a type to the variable 'bnds' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'bnds', tuple_assignment_231677_232503)
        
        # Assigning a List to a Name (line 119):
        
        # Assigning a List to a Name (line 119):
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_232504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_232505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), list_232504, int_232505)
        # Adding element type (line 119)
        int_232506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 15), list_232504, int_232506)
        
        # Assigning a type to the variable 'xopt' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'xopt', list_232504)
        
        # Assigning a List to a Name (line 120):
        
        # Assigning a List to a Name (line 120):
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_232507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        
        # Assigning a type to the variable 'iterx' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'iterx', list_232507)
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to minimize(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'self' (line 122)
        self_232510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'self', False)
        # Obtaining the member 'f1' of a type (line 122)
        f1_232511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), self_232510, 'f1')
        # Getting the type of 'x0' (line 122)
        x0_232512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 41), 'x0', False)
        # Processing the call keyword arguments (line 122)
        str_232513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 52), 'str', 'TNC')
        keyword_232514 = str_232513
        # Getting the type of 'self' (line 122)
        self_232515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 63), 'self', False)
        # Obtaining the member 'g1' of a type (line 122)
        g1_232516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 63), self_232515, 'g1')
        keyword_232517 = g1_232516
        # Getting the type of 'bnds' (line 123)
        bnds_232518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'bnds', False)
        keyword_232519 = bnds_232518
        # Getting the type of 'self' (line 123)
        self_232520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 53), 'self', False)
        # Obtaining the member 'opts' of a type (line 123)
        opts_232521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 53), self_232520, 'opts')
        keyword_232522 = opts_232521
        # Getting the type of 'iterx' (line 124)
        iterx_232523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 41), 'iterx', False)
        # Obtaining the member 'append' of a type (line 124)
        append_232524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 41), iterx_232523, 'append')
        keyword_232525 = append_232524
        kwargs_232526 = {'callback': keyword_232525, 'options': keyword_232522, 'bounds': keyword_232519, 'method': keyword_232514, 'jac': keyword_232517}
        # Getting the type of 'optimize' (line 122)
        optimize_232508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 122)
        minimize_232509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 14), optimize_232508, 'minimize')
        # Calling minimize(args, kwargs) (line 122)
        minimize_call_result_232527 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), minimize_232509, *[f1_232511, x0_232512], **kwargs_232526)
        
        # Assigning a type to the variable 'res' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'res', minimize_call_result_232527)
        
        # Call to assert_allclose(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'res' (line 125)
        res_232529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'res', False)
        # Obtaining the member 'fun' of a type (line 125)
        fun_232530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 24), res_232529, 'fun')
        
        # Call to f1(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'xopt' (line 125)
        xopt_232533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'xopt', False)
        # Processing the call keyword arguments (line 125)
        kwargs_232534 = {}
        # Getting the type of 'self' (line 125)
        self_232531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 33), 'self', False)
        # Obtaining the member 'f1' of a type (line 125)
        f1_232532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 33), self_232531, 'f1')
        # Calling f1(args, kwargs) (line 125)
        f1_call_result_232535 = invoke(stypy.reporting.localization.Localization(__file__, 125, 33), f1_232532, *[xopt_232533], **kwargs_232534)
        
        # Processing the call keyword arguments (line 125)
        float_232536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 53), 'float')
        keyword_232537 = float_232536
        kwargs_232538 = {'atol': keyword_232537}
        # Getting the type of 'assert_allclose' (line 125)
        assert_allclose_232528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 125)
        assert_allclose_call_result_232539 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assert_allclose_232528, *[fun_232530, f1_call_result_232535], **kwargs_232538)
        
        
        # Call to assert_equal(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to len(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'iterx' (line 126)
        iterx_232542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'iterx', False)
        # Processing the call keyword arguments (line 126)
        kwargs_232543 = {}
        # Getting the type of 'len' (line 126)
        len_232541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 21), 'len', False)
        # Calling len(args, kwargs) (line 126)
        len_call_result_232544 = invoke(stypy.reporting.localization.Localization(__file__, 126, 21), len_232541, *[iterx_232542], **kwargs_232543)
        
        # Getting the type of 'res' (line 126)
        res_232545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'res', False)
        # Obtaining the member 'nit' of a type (line 126)
        nit_232546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 33), res_232545, 'nit')
        # Processing the call keyword arguments (line 126)
        kwargs_232547 = {}
        # Getting the type of 'assert_equal' (line 126)
        assert_equal_232540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 126)
        assert_equal_call_result_232548 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assert_equal_232540, *[len_call_result_232544, nit_232546], **kwargs_232547)
        
        
        # ################# End of 'test_minimize_tnc1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc1' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_232549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232549)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc1'
        return stypy_return_type_232549


    @norecursion
    def test_minimize_tnc1b(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc1b'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc1b', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc1b')
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc1b.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc1b', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc1b', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc1b(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to matrix(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_232552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        int_232553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 29), list_232552, int_232553)
        # Adding element type (line 129)
        int_232554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 29), list_232552, int_232554)
        
        # Processing the call keyword arguments (line 129)
        kwargs_232555 = {}
        # Getting the type of 'np' (line 129)
        np_232550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'np', False)
        # Obtaining the member 'matrix' of a type (line 129)
        matrix_232551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 19), np_232550, 'matrix')
        # Calling matrix(args, kwargs) (line 129)
        matrix_call_result_232556 = invoke(stypy.reporting.localization.Localization(__file__, 129, 19), matrix_232551, *[list_232552], **kwargs_232555)
        
        # Assigning a type to the variable 'tuple_assignment_231678' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'tuple_assignment_231678', matrix_call_result_232556)
        
        # Assigning a Tuple to a Name (line 129):
        
        # Obtaining an instance of the builtin type 'tuple' (line 129)
        tuple_232557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 129)
        # Adding element type (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_232558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        
        # Getting the type of 'np' (line 129)
        np_232559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 42), 'np')
        # Obtaining the member 'inf' of a type (line 129)
        inf_232560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 42), np_232559, 'inf')
        # Applying the 'usub' unary operator (line 129)
        result___neg___232561 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 41), 'usub', inf_232560)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 40), list_232558, result___neg___232561)
        # Adding element type (line 129)
        # Getting the type of 'None' (line 129)
        None_232562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 40), list_232558, None_232562)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 40), tuple_232557, list_232558)
        # Adding element type (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_232563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        float_232564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 56), list_232563, float_232564)
        # Adding element type (line 129)
        # Getting the type of 'None' (line 129)
        None_232565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 56), list_232563, None_232565)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 40), tuple_232557, list_232563)
        
        # Assigning a type to the variable 'tuple_assignment_231679' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'tuple_assignment_231679', tuple_232557)
        
        # Assigning a Name to a Name (line 129):
        # Getting the type of 'tuple_assignment_231678' (line 129)
        tuple_assignment_231678_232566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'tuple_assignment_231678')
        # Assigning a type to the variable 'x0' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'x0', tuple_assignment_231678_232566)
        
        # Assigning a Name to a Name (line 129):
        # Getting the type of 'tuple_assignment_231679' (line 129)
        tuple_assignment_231679_232567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'tuple_assignment_231679')
        # Assigning a type to the variable 'bnds' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'bnds', tuple_assignment_231679_232567)
        
        # Assigning a List to a Name (line 130):
        
        # Assigning a List to a Name (line 130):
        
        # Obtaining an instance of the builtin type 'list' (line 130)
        list_232568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 130)
        # Adding element type (line 130)
        int_232569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), list_232568, int_232569)
        # Adding element type (line 130)
        int_232570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), list_232568, int_232570)
        
        # Assigning a type to the variable 'xopt' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'xopt', list_232568)
        
        # Assigning a Attribute to a Name (line 131):
        
        # Assigning a Attribute to a Name (line 131):
        
        # Call to minimize(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'self' (line 131)
        self_232573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'self', False)
        # Obtaining the member 'f1' of a type (line 131)
        f1_232574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 30), self_232573, 'f1')
        # Getting the type of 'x0' (line 131)
        x0_232575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 39), 'x0', False)
        # Processing the call keyword arguments (line 131)
        str_232576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 50), 'str', 'TNC')
        keyword_232577 = str_232576
        # Getting the type of 'bnds' (line 132)
        bnds_232578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 37), 'bnds', False)
        keyword_232579 = bnds_232578
        # Getting the type of 'self' (line 132)
        self_232580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 51), 'self', False)
        # Obtaining the member 'opts' of a type (line 132)
        opts_232581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 51), self_232580, 'opts')
        keyword_232582 = opts_232581
        kwargs_232583 = {'options': keyword_232582, 'method': keyword_232577, 'bounds': keyword_232579}
        # Getting the type of 'optimize' (line 131)
        optimize_232571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 131)
        minimize_232572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), optimize_232571, 'minimize')
        # Calling minimize(args, kwargs) (line 131)
        minimize_call_result_232584 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), minimize_232572, *[f1_232574, x0_232575], **kwargs_232583)
        
        # Obtaining the member 'x' of a type (line 131)
        x_232585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), minimize_call_result_232584, 'x')
        # Assigning a type to the variable 'x' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'x', x_232585)
        
        # Call to assert_allclose(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Call to f1(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'x' (line 133)
        x_232589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 32), 'x', False)
        # Processing the call keyword arguments (line 133)
        kwargs_232590 = {}
        # Getting the type of 'self' (line 133)
        self_232587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'self', False)
        # Obtaining the member 'f1' of a type (line 133)
        f1_232588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), self_232587, 'f1')
        # Calling f1(args, kwargs) (line 133)
        f1_call_result_232591 = invoke(stypy.reporting.localization.Localization(__file__, 133, 24), f1_232588, *[x_232589], **kwargs_232590)
        
        
        # Call to f1(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'xopt' (line 133)
        xopt_232594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 44), 'xopt', False)
        # Processing the call keyword arguments (line 133)
        kwargs_232595 = {}
        # Getting the type of 'self' (line 133)
        self_232592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'self', False)
        # Obtaining the member 'f1' of a type (line 133)
        f1_232593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 36), self_232592, 'f1')
        # Calling f1(args, kwargs) (line 133)
        f1_call_result_232596 = invoke(stypy.reporting.localization.Localization(__file__, 133, 36), f1_232593, *[xopt_232594], **kwargs_232595)
        
        # Processing the call keyword arguments (line 133)
        float_232597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 56), 'float')
        keyword_232598 = float_232597
        kwargs_232599 = {'atol': keyword_232598}
        # Getting the type of 'assert_allclose' (line 133)
        assert_allclose_232586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 133)
        assert_allclose_call_result_232600 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assert_allclose_232586, *[f1_call_result_232591, f1_call_result_232596], **kwargs_232599)
        
        
        # ################# End of 'test_minimize_tnc1b(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc1b' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_232601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232601)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc1b'
        return stypy_return_type_232601


    @norecursion
    def test_minimize_tnc1c(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc1c'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc1c', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc1c')
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc1c.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc1c', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc1c', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc1c(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 136):
        
        # Assigning a List to a Name (line 136):
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_232602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        int_232603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_232602, int_232603)
        # Adding element type (line 136)
        int_232604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_232602, int_232604)
        
        # Assigning a type to the variable 'tuple_assignment_231680' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_assignment_231680', list_232602)
        
        # Assigning a Tuple to a Name (line 136):
        
        # Obtaining an instance of the builtin type 'tuple' (line 136)
        tuple_232605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 136)
        # Adding element type (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_232606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        
        # Getting the type of 'np' (line 136)
        np_232607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'np')
        # Obtaining the member 'inf' of a type (line 136)
        inf_232608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 31), np_232607, 'inf')
        # Applying the 'usub' unary operator (line 136)
        result___neg___232609 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 30), 'usub', inf_232608)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 29), list_232606, result___neg___232609)
        # Adding element type (line 136)
        # Getting the type of 'None' (line 136)
        None_232610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 39), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 29), list_232606, None_232610)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 29), tuple_232605, list_232606)
        # Adding element type (line 136)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_232611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        float_232612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 45), list_232611, float_232612)
        # Adding element type (line 136)
        # Getting the type of 'None' (line 136)
        None_232613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 52), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 45), list_232611, None_232613)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 29), tuple_232605, list_232611)
        
        # Assigning a type to the variable 'tuple_assignment_231681' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_assignment_231681', tuple_232605)
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'tuple_assignment_231680' (line 136)
        tuple_assignment_231680_232614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_assignment_231680')
        # Assigning a type to the variable 'x0' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'x0', tuple_assignment_231680_232614)
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'tuple_assignment_231681' (line 136)
        tuple_assignment_231681_232615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_assignment_231681')
        # Assigning a type to the variable 'bnds' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'bnds', tuple_assignment_231681_232615)
        
        # Assigning a List to a Name (line 137):
        
        # Assigning a List to a Name (line 137):
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_232616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        int_232617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 15), list_232616, int_232617)
        # Adding element type (line 137)
        int_232618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 15), list_232616, int_232618)
        
        # Assigning a type to the variable 'xopt' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'xopt', list_232616)
        
        # Assigning a Attribute to a Name (line 138):
        
        # Assigning a Attribute to a Name (line 138):
        
        # Call to minimize(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'self' (line 138)
        self_232621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 30), 'self', False)
        # Obtaining the member 'fg1' of a type (line 138)
        fg1_232622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 30), self_232621, 'fg1')
        # Getting the type of 'x0' (line 138)
        x0_232623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 40), 'x0', False)
        # Processing the call keyword arguments (line 138)
        str_232624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 51), 'str', 'TNC')
        keyword_232625 = str_232624
        # Getting the type of 'True' (line 139)
        True_232626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'True', False)
        keyword_232627 = True_232626
        # Getting the type of 'bnds' (line 139)
        bnds_232628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 47), 'bnds', False)
        keyword_232629 = bnds_232628
        # Getting the type of 'self' (line 140)
        self_232630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'self', False)
        # Obtaining the member 'opts' of a type (line 140)
        opts_232631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 38), self_232630, 'opts')
        keyword_232632 = opts_232631
        kwargs_232633 = {'options': keyword_232632, 'bounds': keyword_232629, 'method': keyword_232625, 'jac': keyword_232627}
        # Getting the type of 'optimize' (line 138)
        optimize_232619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 138)
        minimize_232620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), optimize_232619, 'minimize')
        # Calling minimize(args, kwargs) (line 138)
        minimize_call_result_232634 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), minimize_232620, *[fg1_232622, x0_232623], **kwargs_232633)
        
        # Obtaining the member 'x' of a type (line 138)
        x_232635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), minimize_call_result_232634, 'x')
        # Assigning a type to the variable 'x' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'x', x_232635)
        
        # Call to assert_allclose(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Call to f1(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'x' (line 141)
        x_232639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 32), 'x', False)
        # Processing the call keyword arguments (line 141)
        kwargs_232640 = {}
        # Getting the type of 'self' (line 141)
        self_232637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'self', False)
        # Obtaining the member 'f1' of a type (line 141)
        f1_232638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 24), self_232637, 'f1')
        # Calling f1(args, kwargs) (line 141)
        f1_call_result_232641 = invoke(stypy.reporting.localization.Localization(__file__, 141, 24), f1_232638, *[x_232639], **kwargs_232640)
        
        
        # Call to f1(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'xopt' (line 141)
        xopt_232644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 44), 'xopt', False)
        # Processing the call keyword arguments (line 141)
        kwargs_232645 = {}
        # Getting the type of 'self' (line 141)
        self_232642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 36), 'self', False)
        # Obtaining the member 'f1' of a type (line 141)
        f1_232643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 36), self_232642, 'f1')
        # Calling f1(args, kwargs) (line 141)
        f1_call_result_232646 = invoke(stypy.reporting.localization.Localization(__file__, 141, 36), f1_232643, *[xopt_232644], **kwargs_232645)
        
        # Processing the call keyword arguments (line 141)
        float_232647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 56), 'float')
        keyword_232648 = float_232647
        kwargs_232649 = {'atol': keyword_232648}
        # Getting the type of 'assert_allclose' (line 141)
        assert_allclose_232636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 141)
        assert_allclose_call_result_232650 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), assert_allclose_232636, *[f1_call_result_232641, f1_call_result_232646], **kwargs_232649)
        
        
        # ################# End of 'test_minimize_tnc1c(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc1c' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_232651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232651)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc1c'
        return stypy_return_type_232651


    @norecursion
    def test_minimize_tnc2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc2'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc2', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc2')
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc2(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 144):
        
        # Assigning a List to a Name (line 144):
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_232652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        int_232653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), list_232652, int_232653)
        # Adding element type (line 144)
        int_232654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 19), list_232652, int_232654)
        
        # Assigning a type to the variable 'tuple_assignment_231682' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'tuple_assignment_231682', list_232652)
        
        # Assigning a Tuple to a Name (line 144):
        
        # Obtaining an instance of the builtin type 'tuple' (line 144)
        tuple_232655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 144)
        # Adding element type (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_232656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        
        # Getting the type of 'np' (line 144)
        np_232657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 31), 'np')
        # Obtaining the member 'inf' of a type (line 144)
        inf_232658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 31), np_232657, 'inf')
        # Applying the 'usub' unary operator (line 144)
        result___neg___232659 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 30), 'usub', inf_232658)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 29), list_232656, result___neg___232659)
        # Adding element type (line 144)
        # Getting the type of 'None' (line 144)
        None_232660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 39), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 29), list_232656, None_232660)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 29), tuple_232655, list_232656)
        # Adding element type (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_232661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        float_232662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 46), list_232661, float_232662)
        # Adding element type (line 144)
        # Getting the type of 'None' (line 144)
        None_232663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 46), list_232661, None_232663)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 29), tuple_232655, list_232661)
        
        # Assigning a type to the variable 'tuple_assignment_231683' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'tuple_assignment_231683', tuple_232655)
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'tuple_assignment_231682' (line 144)
        tuple_assignment_231682_232664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'tuple_assignment_231682')
        # Assigning a type to the variable 'x0' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'x0', tuple_assignment_231682_232664)
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'tuple_assignment_231683' (line 144)
        tuple_assignment_231683_232665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'tuple_assignment_231683')
        # Assigning a type to the variable 'bnds' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'bnds', tuple_assignment_231683_232665)
        
        # Assigning a List to a Name (line 145):
        
        # Assigning a List to a Name (line 145):
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_232666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        # Adding element type (line 145)
        float_232667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_232666, float_232667)
        # Adding element type (line 145)
        float_232668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 15), list_232666, float_232668)
        
        # Assigning a type to the variable 'xopt' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'xopt', list_232666)
        
        # Assigning a Attribute to a Name (line 146):
        
        # Assigning a Attribute to a Name (line 146):
        
        # Call to minimize(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_232671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'self', False)
        # Obtaining the member 'f1' of a type (line 146)
        f1_232672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 30), self_232671, 'f1')
        # Getting the type of 'x0' (line 146)
        x0_232673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'x0', False)
        # Processing the call keyword arguments (line 146)
        str_232674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 50), 'str', 'TNC')
        keyword_232675 = str_232674
        # Getting the type of 'self' (line 147)
        self_232676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 34), 'self', False)
        # Obtaining the member 'g1' of a type (line 147)
        g1_232677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 34), self_232676, 'g1')
        keyword_232678 = g1_232677
        # Getting the type of 'bnds' (line 147)
        bnds_232679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 50), 'bnds', False)
        keyword_232680 = bnds_232679
        # Getting the type of 'self' (line 148)
        self_232681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 38), 'self', False)
        # Obtaining the member 'opts' of a type (line 148)
        opts_232682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 38), self_232681, 'opts')
        keyword_232683 = opts_232682
        kwargs_232684 = {'options': keyword_232683, 'bounds': keyword_232680, 'method': keyword_232675, 'jac': keyword_232678}
        # Getting the type of 'optimize' (line 146)
        optimize_232669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 146)
        minimize_232670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), optimize_232669, 'minimize')
        # Calling minimize(args, kwargs) (line 146)
        minimize_call_result_232685 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), minimize_232670, *[f1_232672, x0_232673], **kwargs_232684)
        
        # Obtaining the member 'x' of a type (line 146)
        x_232686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), minimize_call_result_232685, 'x')
        # Assigning a type to the variable 'x' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'x', x_232686)
        
        # Call to assert_allclose(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to f1(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'x' (line 149)
        x_232690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'x', False)
        # Processing the call keyword arguments (line 149)
        kwargs_232691 = {}
        # Getting the type of 'self' (line 149)
        self_232688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'self', False)
        # Obtaining the member 'f1' of a type (line 149)
        f1_232689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), self_232688, 'f1')
        # Calling f1(args, kwargs) (line 149)
        f1_call_result_232692 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), f1_232689, *[x_232690], **kwargs_232691)
        
        
        # Call to f1(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'xopt' (line 149)
        xopt_232695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 44), 'xopt', False)
        # Processing the call keyword arguments (line 149)
        kwargs_232696 = {}
        # Getting the type of 'self' (line 149)
        self_232693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 36), 'self', False)
        # Obtaining the member 'f1' of a type (line 149)
        f1_232694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 36), self_232693, 'f1')
        # Calling f1(args, kwargs) (line 149)
        f1_call_result_232697 = invoke(stypy.reporting.localization.Localization(__file__, 149, 36), f1_232694, *[xopt_232695], **kwargs_232696)
        
        # Processing the call keyword arguments (line 149)
        float_232698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 56), 'float')
        keyword_232699 = float_232698
        kwargs_232700 = {'atol': keyword_232699}
        # Getting the type of 'assert_allclose' (line 149)
        assert_allclose_232687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 149)
        assert_allclose_call_result_232701 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assert_allclose_232687, *[f1_call_result_232692, f1_call_result_232697], **kwargs_232700)
        
        
        # ################# End of 'test_minimize_tnc2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc2' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_232702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232702)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc2'
        return stypy_return_type_232702


    @norecursion
    def test_minimize_tnc3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc3'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc3', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc3')
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc3(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 152):
        
        # Assigning a List to a Name (line 152):
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_232703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        int_232704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_232703, int_232704)
        # Adding element type (line 152)
        int_232705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 19), list_232703, int_232705)
        
        # Assigning a type to the variable 'tuple_assignment_231684' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_assignment_231684', list_232703)
        
        # Assigning a Tuple to a Name (line 152):
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_232706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_232707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        
        # Getting the type of 'np' (line 152)
        np_232708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'np')
        # Obtaining the member 'inf' of a type (line 152)
        inf_232709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 31), np_232708, 'inf')
        # Applying the 'usub' unary operator (line 152)
        result___neg___232710 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 30), 'usub', inf_232709)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 29), list_232707, result___neg___232710)
        # Adding element type (line 152)
        # Getting the type of 'None' (line 152)
        None_232711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 29), list_232707, None_232711)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 29), tuple_232706, list_232707)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_232712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        float_232713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 46), list_232712, float_232713)
        # Adding element type (line 152)
        # Getting the type of 'None' (line 152)
        None_232714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 52), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 46), list_232712, None_232714)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 29), tuple_232706, list_232712)
        
        # Assigning a type to the variable 'tuple_assignment_231685' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_assignment_231685', tuple_232706)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'tuple_assignment_231684' (line 152)
        tuple_assignment_231684_232715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_assignment_231684')
        # Assigning a type to the variable 'x0' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'x0', tuple_assignment_231684_232715)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'tuple_assignment_231685' (line 152)
        tuple_assignment_231685_232716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_assignment_231685')
        # Assigning a type to the variable 'bnds' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'bnds', tuple_assignment_231685_232716)
        
        # Assigning a List to a Name (line 153):
        
        # Assigning a List to a Name (line 153):
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_232717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        # Adding element type (line 153)
        int_232718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 15), list_232717, int_232718)
        # Adding element type (line 153)
        int_232719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 15), list_232717, int_232719)
        
        # Assigning a type to the variable 'xopt' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'xopt', list_232717)
        
        # Assigning a Attribute to a Name (line 154):
        
        # Assigning a Attribute to a Name (line 154):
        
        # Call to minimize(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'self' (line 154)
        self_232722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'self', False)
        # Obtaining the member 'f3' of a type (line 154)
        f3_232723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 30), self_232722, 'f3')
        # Getting the type of 'x0' (line 154)
        x0_232724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'x0', False)
        # Processing the call keyword arguments (line 154)
        str_232725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 50), 'str', 'TNC')
        keyword_232726 = str_232725
        # Getting the type of 'self' (line 155)
        self_232727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'self', False)
        # Obtaining the member 'g3' of a type (line 155)
        g3_232728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 34), self_232727, 'g3')
        keyword_232729 = g3_232728
        # Getting the type of 'bnds' (line 155)
        bnds_232730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 50), 'bnds', False)
        keyword_232731 = bnds_232730
        # Getting the type of 'self' (line 156)
        self_232732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 38), 'self', False)
        # Obtaining the member 'opts' of a type (line 156)
        opts_232733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 38), self_232732, 'opts')
        keyword_232734 = opts_232733
        kwargs_232735 = {'options': keyword_232734, 'bounds': keyword_232731, 'method': keyword_232726, 'jac': keyword_232729}
        # Getting the type of 'optimize' (line 154)
        optimize_232720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 154)
        minimize_232721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), optimize_232720, 'minimize')
        # Calling minimize(args, kwargs) (line 154)
        minimize_call_result_232736 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), minimize_232721, *[f3_232723, x0_232724], **kwargs_232735)
        
        # Obtaining the member 'x' of a type (line 154)
        x_232737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), minimize_call_result_232736, 'x')
        # Assigning a type to the variable 'x' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'x', x_232737)
        
        # Call to assert_allclose(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Call to f3(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'x' (line 157)
        x_232741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'x', False)
        # Processing the call keyword arguments (line 157)
        kwargs_232742 = {}
        # Getting the type of 'self' (line 157)
        self_232739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'self', False)
        # Obtaining the member 'f3' of a type (line 157)
        f3_232740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), self_232739, 'f3')
        # Calling f3(args, kwargs) (line 157)
        f3_call_result_232743 = invoke(stypy.reporting.localization.Localization(__file__, 157, 24), f3_232740, *[x_232741], **kwargs_232742)
        
        
        # Call to f3(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'xopt' (line 157)
        xopt_232746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 44), 'xopt', False)
        # Processing the call keyword arguments (line 157)
        kwargs_232747 = {}
        # Getting the type of 'self' (line 157)
        self_232744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'self', False)
        # Obtaining the member 'f3' of a type (line 157)
        f3_232745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 36), self_232744, 'f3')
        # Calling f3(args, kwargs) (line 157)
        f3_call_result_232748 = invoke(stypy.reporting.localization.Localization(__file__, 157, 36), f3_232745, *[xopt_232746], **kwargs_232747)
        
        # Processing the call keyword arguments (line 157)
        float_232749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 56), 'float')
        keyword_232750 = float_232749
        kwargs_232751 = {'atol': keyword_232750}
        # Getting the type of 'assert_allclose' (line 157)
        assert_allclose_232738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 157)
        assert_allclose_call_result_232752 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), assert_allclose_232738, *[f3_call_result_232743, f3_call_result_232748], **kwargs_232751)
        
        
        # ################# End of 'test_minimize_tnc3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc3' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_232753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc3'
        return stypy_return_type_232753


    @norecursion
    def test_minimize_tnc4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc4'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc4', 159, 4, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc4')
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc4.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc4', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc4', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc4(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 160):
        
        # Assigning a List to a Name (line 160):
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_232754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        float_232755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 18), list_232754, float_232755)
        # Adding element type (line 160)
        float_232756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 18), list_232754, float_232756)
        
        # Assigning a type to the variable 'tuple_assignment_231686' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_assignment_231686', list_232754)
        
        # Assigning a List to a Name (line 160):
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_232757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_232758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        int_232759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), tuple_232758, int_232759)
        # Adding element type (line 160)
        # Getting the type of 'None' (line 160)
        None_232760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 39), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 36), tuple_232758, None_232760)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 34), list_232757, tuple_232758)
        # Adding element type (line 160)
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_232761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        int_232762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 47), tuple_232761, int_232762)
        # Adding element type (line 160)
        # Getting the type of 'None' (line 160)
        None_232763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 50), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 47), tuple_232761, None_232763)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 34), list_232757, tuple_232761)
        
        # Assigning a type to the variable 'tuple_assignment_231687' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_assignment_231687', list_232757)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_assignment_231686' (line 160)
        tuple_assignment_231686_232764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_assignment_231686')
        # Assigning a type to the variable 'x0' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'x0', tuple_assignment_231686_232764)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_assignment_231687' (line 160)
        tuple_assignment_231687_232765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_assignment_231687')
        # Assigning a type to the variable 'bnds' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'bnds', tuple_assignment_231687_232765)
        
        # Assigning a List to a Name (line 161):
        
        # Assigning a List to a Name (line 161):
        
        # Obtaining an instance of the builtin type 'list' (line 161)
        list_232766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 161)
        # Adding element type (line 161)
        int_232767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 15), list_232766, int_232767)
        # Adding element type (line 161)
        int_232768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 15), list_232766, int_232768)
        
        # Assigning a type to the variable 'xopt' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'xopt', list_232766)
        
        # Assigning a Attribute to a Name (line 162):
        
        # Assigning a Attribute to a Name (line 162):
        
        # Call to minimize(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'self' (line 162)
        self_232771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'self', False)
        # Obtaining the member 'f4' of a type (line 162)
        f4_232772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 30), self_232771, 'f4')
        # Getting the type of 'x0' (line 162)
        x0_232773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 39), 'x0', False)
        # Processing the call keyword arguments (line 162)
        str_232774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 50), 'str', 'TNC')
        keyword_232775 = str_232774
        # Getting the type of 'self' (line 163)
        self_232776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'self', False)
        # Obtaining the member 'g4' of a type (line 163)
        g4_232777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 34), self_232776, 'g4')
        keyword_232778 = g4_232777
        # Getting the type of 'bnds' (line 163)
        bnds_232779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 50), 'bnds', False)
        keyword_232780 = bnds_232779
        # Getting the type of 'self' (line 164)
        self_232781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 38), 'self', False)
        # Obtaining the member 'opts' of a type (line 164)
        opts_232782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 38), self_232781, 'opts')
        keyword_232783 = opts_232782
        kwargs_232784 = {'options': keyword_232783, 'bounds': keyword_232780, 'method': keyword_232775, 'jac': keyword_232778}
        # Getting the type of 'optimize' (line 162)
        optimize_232769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 162)
        minimize_232770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), optimize_232769, 'minimize')
        # Calling minimize(args, kwargs) (line 162)
        minimize_call_result_232785 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), minimize_232770, *[f4_232772, x0_232773], **kwargs_232784)
        
        # Obtaining the member 'x' of a type (line 162)
        x_232786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), minimize_call_result_232785, 'x')
        # Assigning a type to the variable 'x' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'x', x_232786)
        
        # Call to assert_allclose(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to f4(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'x' (line 165)
        x_232790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'x', False)
        # Processing the call keyword arguments (line 165)
        kwargs_232791 = {}
        # Getting the type of 'self' (line 165)
        self_232788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'self', False)
        # Obtaining the member 'f4' of a type (line 165)
        f4_232789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 24), self_232788, 'f4')
        # Calling f4(args, kwargs) (line 165)
        f4_call_result_232792 = invoke(stypy.reporting.localization.Localization(__file__, 165, 24), f4_232789, *[x_232790], **kwargs_232791)
        
        
        # Call to f4(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'xopt' (line 165)
        xopt_232795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 44), 'xopt', False)
        # Processing the call keyword arguments (line 165)
        kwargs_232796 = {}
        # Getting the type of 'self' (line 165)
        self_232793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 36), 'self', False)
        # Obtaining the member 'f4' of a type (line 165)
        f4_232794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 36), self_232793, 'f4')
        # Calling f4(args, kwargs) (line 165)
        f4_call_result_232797 = invoke(stypy.reporting.localization.Localization(__file__, 165, 36), f4_232794, *[xopt_232795], **kwargs_232796)
        
        # Processing the call keyword arguments (line 165)
        float_232798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 56), 'float')
        keyword_232799 = float_232798
        kwargs_232800 = {'atol': keyword_232799}
        # Getting the type of 'assert_allclose' (line 165)
        assert_allclose_232787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 165)
        assert_allclose_call_result_232801 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert_allclose_232787, *[f4_call_result_232792, f4_call_result_232797], **kwargs_232800)
        
        
        # ################# End of 'test_minimize_tnc4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc4' in the type store
        # Getting the type of 'stypy_return_type' (line 159)
        stypy_return_type_232802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc4'
        return stypy_return_type_232802


    @norecursion
    def test_minimize_tnc5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc5'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc5', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc5')
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc5(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 168):
        
        # Assigning a List to a Name (line 168):
        
        # Obtaining an instance of the builtin type 'list' (line 168)
        list_232803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 168)
        # Adding element type (line 168)
        int_232804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 19), list_232803, int_232804)
        # Adding element type (line 168)
        int_232805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 19), list_232803, int_232805)
        
        # Assigning a type to the variable 'tuple_assignment_231688' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_assignment_231688', list_232803)
        
        # Assigning a List to a Name (line 168):
        
        # Obtaining an instance of the builtin type 'list' (line 168)
        list_232806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 168)
        # Adding element type (line 168)
        
        # Obtaining an instance of the builtin type 'tuple' (line 168)
        tuple_232807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 168)
        # Adding element type (line 168)
        float_232808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 29), tuple_232807, float_232808)
        # Adding element type (line 168)
        int_232809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 29), tuple_232807, int_232809)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 27), list_232806, tuple_232807)
        # Adding element type (line 168)
        
        # Obtaining an instance of the builtin type 'tuple' (line 168)
        tuple_232810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 168)
        # Adding element type (line 168)
        int_232811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 39), tuple_232810, int_232811)
        # Adding element type (line 168)
        int_232812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 39), tuple_232810, int_232812)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 27), list_232806, tuple_232810)
        
        # Assigning a type to the variable 'tuple_assignment_231689' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_assignment_231689', list_232806)
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'tuple_assignment_231688' (line 168)
        tuple_assignment_231688_232813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_assignment_231688')
        # Assigning a type to the variable 'x0' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'x0', tuple_assignment_231688_232813)
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'tuple_assignment_231689' (line 168)
        tuple_assignment_231689_232814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_assignment_231689')
        # Assigning a type to the variable 'bnds' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'bnds', tuple_assignment_231689_232814)
        
        # Assigning a List to a Name (line 169):
        
        # Assigning a List to a Name (line 169):
        
        # Obtaining an instance of the builtin type 'list' (line 169)
        list_232815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 169)
        # Adding element type (line 169)
        float_232816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 15), list_232815, float_232816)
        # Adding element type (line 169)
        float_232817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 15), list_232815, float_232817)
        
        # Assigning a type to the variable 'xopt' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'xopt', list_232815)
        
        # Assigning a Attribute to a Name (line 170):
        
        # Assigning a Attribute to a Name (line 170):
        
        # Call to minimize(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'self' (line 170)
        self_232820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'self', False)
        # Obtaining the member 'f5' of a type (line 170)
        f5_232821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 30), self_232820, 'f5')
        # Getting the type of 'x0' (line 170)
        x0_232822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 39), 'x0', False)
        # Processing the call keyword arguments (line 170)
        str_232823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 50), 'str', 'TNC')
        keyword_232824 = str_232823
        # Getting the type of 'self' (line 171)
        self_232825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 34), 'self', False)
        # Obtaining the member 'g5' of a type (line 171)
        g5_232826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 34), self_232825, 'g5')
        keyword_232827 = g5_232826
        # Getting the type of 'bnds' (line 171)
        bnds_232828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 50), 'bnds', False)
        keyword_232829 = bnds_232828
        # Getting the type of 'self' (line 172)
        self_232830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 38), 'self', False)
        # Obtaining the member 'opts' of a type (line 172)
        opts_232831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 38), self_232830, 'opts')
        keyword_232832 = opts_232831
        kwargs_232833 = {'options': keyword_232832, 'bounds': keyword_232829, 'method': keyword_232824, 'jac': keyword_232827}
        # Getting the type of 'optimize' (line 170)
        optimize_232818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 170)
        minimize_232819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), optimize_232818, 'minimize')
        # Calling minimize(args, kwargs) (line 170)
        minimize_call_result_232834 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), minimize_232819, *[f5_232821, x0_232822], **kwargs_232833)
        
        # Obtaining the member 'x' of a type (line 170)
        x_232835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 12), minimize_call_result_232834, 'x')
        # Assigning a type to the variable 'x' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'x', x_232835)
        
        # Call to assert_allclose(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Call to f5(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'x' (line 173)
        x_232839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'x', False)
        # Processing the call keyword arguments (line 173)
        kwargs_232840 = {}
        # Getting the type of 'self' (line 173)
        self_232837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'self', False)
        # Obtaining the member 'f5' of a type (line 173)
        f5_232838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 24), self_232837, 'f5')
        # Calling f5(args, kwargs) (line 173)
        f5_call_result_232841 = invoke(stypy.reporting.localization.Localization(__file__, 173, 24), f5_232838, *[x_232839], **kwargs_232840)
        
        
        # Call to f5(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'xopt' (line 173)
        xopt_232844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 44), 'xopt', False)
        # Processing the call keyword arguments (line 173)
        kwargs_232845 = {}
        # Getting the type of 'self' (line 173)
        self_232842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 36), 'self', False)
        # Obtaining the member 'f5' of a type (line 173)
        f5_232843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 36), self_232842, 'f5')
        # Calling f5(args, kwargs) (line 173)
        f5_call_result_232846 = invoke(stypy.reporting.localization.Localization(__file__, 173, 36), f5_232843, *[xopt_232844], **kwargs_232845)
        
        # Processing the call keyword arguments (line 173)
        float_232847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 56), 'float')
        keyword_232848 = float_232847
        kwargs_232849 = {'atol': keyword_232848}
        # Getting the type of 'assert_allclose' (line 173)
        assert_allclose_232836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 173)
        assert_allclose_call_result_232850 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), assert_allclose_232836, *[f5_call_result_232841, f5_call_result_232846], **kwargs_232849)
        
        
        # ################# End of 'test_minimize_tnc5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc5' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_232851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc5'
        return stypy_return_type_232851


    @norecursion
    def test_minimize_tnc38(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc38'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc38', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc38')
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc38.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc38', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc38', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc38(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to array(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_232854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        int_232855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 28), list_232854, int_232855)
        # Adding element type (line 176)
        int_232856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 28), list_232854, int_232856)
        # Adding element type (line 176)
        int_232857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 28), list_232854, int_232857)
        # Adding element type (line 176)
        int_232858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 28), list_232854, int_232858)
        
        # Processing the call keyword arguments (line 176)
        kwargs_232859 = {}
        # Getting the type of 'np' (line 176)
        np_232852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 176)
        array_232853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), np_232852, 'array')
        # Calling array(args, kwargs) (line 176)
        array_call_result_232860 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), array_232853, *[list_232854], **kwargs_232859)
        
        # Assigning a type to the variable 'tuple_assignment_231690' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_assignment_231690', array_call_result_232860)
        
        # Assigning a BinOp to a Name (line 176):
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_232861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        
        # Obtaining an instance of the builtin type 'tuple' (line 176)
        tuple_232862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 176)
        # Adding element type (line 176)
        int_232863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 49), tuple_232862, int_232863)
        # Adding element type (line 176)
        int_232864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 49), tuple_232862, int_232864)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 47), list_232861, tuple_232862)
        
        int_232865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 59), 'int')
        # Applying the binary operator '*' (line 176)
        result_mul_232866 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 47), '*', list_232861, int_232865)
        
        # Assigning a type to the variable 'tuple_assignment_231691' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_assignment_231691', result_mul_232866)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_assignment_231690' (line 176)
        tuple_assignment_231690_232867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_assignment_231690')
        # Assigning a type to the variable 'x0' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'x0', tuple_assignment_231690_232867)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_assignment_231691' (line 176)
        tuple_assignment_231691_232868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_assignment_231691')
        # Assigning a type to the variable 'bnds' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'bnds', tuple_assignment_231691_232868)
        
        # Assigning a BinOp to a Name (line 177):
        
        # Assigning a BinOp to a Name (line 177):
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_232869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        int_232870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 15), list_232869, int_232870)
        
        int_232871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 19), 'int')
        # Applying the binary operator '*' (line 177)
        result_mul_232872 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), '*', list_232869, int_232871)
        
        # Assigning a type to the variable 'xopt' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'xopt', result_mul_232872)
        
        # Assigning a Attribute to a Name (line 178):
        
        # Assigning a Attribute to a Name (line 178):
        
        # Call to minimize(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'self' (line 178)
        self_232875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'self', False)
        # Obtaining the member 'f38' of a type (line 178)
        f38_232876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 30), self_232875, 'f38')
        # Getting the type of 'x0' (line 178)
        x0_232877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 40), 'x0', False)
        # Processing the call keyword arguments (line 178)
        str_232878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 51), 'str', 'TNC')
        keyword_232879 = str_232878
        # Getting the type of 'self' (line 179)
        self_232880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'self', False)
        # Obtaining the member 'g38' of a type (line 179)
        g38_232881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 34), self_232880, 'g38')
        keyword_232882 = g38_232881
        # Getting the type of 'bnds' (line 179)
        bnds_232883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 51), 'bnds', False)
        keyword_232884 = bnds_232883
        # Getting the type of 'self' (line 180)
        self_232885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 38), 'self', False)
        # Obtaining the member 'opts' of a type (line 180)
        opts_232886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 38), self_232885, 'opts')
        keyword_232887 = opts_232886
        kwargs_232888 = {'options': keyword_232887, 'bounds': keyword_232884, 'method': keyword_232879, 'jac': keyword_232882}
        # Getting the type of 'optimize' (line 178)
        optimize_232873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 178)
        minimize_232874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), optimize_232873, 'minimize')
        # Calling minimize(args, kwargs) (line 178)
        minimize_call_result_232889 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), minimize_232874, *[f38_232876, x0_232877], **kwargs_232888)
        
        # Obtaining the member 'x' of a type (line 178)
        x_232890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), minimize_call_result_232889, 'x')
        # Assigning a type to the variable 'x' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'x', x_232890)
        
        # Call to assert_allclose(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to f38(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'x' (line 181)
        x_232894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'x', False)
        # Processing the call keyword arguments (line 181)
        kwargs_232895 = {}
        # Getting the type of 'self' (line 181)
        self_232892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'self', False)
        # Obtaining the member 'f38' of a type (line 181)
        f38_232893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 24), self_232892, 'f38')
        # Calling f38(args, kwargs) (line 181)
        f38_call_result_232896 = invoke(stypy.reporting.localization.Localization(__file__, 181, 24), f38_232893, *[x_232894], **kwargs_232895)
        
        
        # Call to f38(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'xopt' (line 181)
        xopt_232899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'xopt', False)
        # Processing the call keyword arguments (line 181)
        kwargs_232900 = {}
        # Getting the type of 'self' (line 181)
        self_232897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 37), 'self', False)
        # Obtaining the member 'f38' of a type (line 181)
        f38_232898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 37), self_232897, 'f38')
        # Calling f38(args, kwargs) (line 181)
        f38_call_result_232901 = invoke(stypy.reporting.localization.Localization(__file__, 181, 37), f38_232898, *[xopt_232899], **kwargs_232900)
        
        # Processing the call keyword arguments (line 181)
        float_232902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 58), 'float')
        keyword_232903 = float_232902
        kwargs_232904 = {'atol': keyword_232903}
        # Getting the type of 'assert_allclose' (line 181)
        assert_allclose_232891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 181)
        assert_allclose_call_result_232905 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assert_allclose_232891, *[f38_call_result_232896, f38_call_result_232901], **kwargs_232904)
        
        
        # ################# End of 'test_minimize_tnc38(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc38' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_232906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232906)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc38'
        return stypy_return_type_232906


    @norecursion
    def test_minimize_tnc45(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_tnc45'
        module_type_store = module_type_store.open_function_context('test_minimize_tnc45', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_minimize_tnc45')
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_minimize_tnc45.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_minimize_tnc45', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_tnc45', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_tnc45(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 184):
        
        # Assigning a BinOp to a Name (line 184):
        
        # Obtaining an instance of the builtin type 'list' (line 184)
        list_232907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 184)
        # Adding element type (line 184)
        int_232908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 19), list_232907, int_232908)
        
        int_232909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 25), 'int')
        # Applying the binary operator '*' (line 184)
        result_mul_232910 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 19), '*', list_232907, int_232909)
        
        # Assigning a type to the variable 'tuple_assignment_231692' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_assignment_231692', result_mul_232910)
        
        # Assigning a List to a Name (line 184):
        
        # Obtaining an instance of the builtin type 'list' (line 184)
        list_232911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 184)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_232912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        int_232913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 30), tuple_232912, int_232913)
        # Adding element type (line 184)
        int_232914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 30), tuple_232912, int_232914)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), list_232911, tuple_232912)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_232915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        int_232916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 38), tuple_232915, int_232916)
        # Adding element type (line 184)
        int_232917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 38), tuple_232915, int_232917)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), list_232911, tuple_232915)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_232918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        int_232919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 46), tuple_232918, int_232919)
        # Adding element type (line 184)
        int_232920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 46), tuple_232918, int_232920)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), list_232911, tuple_232918)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_232921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        int_232922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 54), tuple_232921, int_232922)
        # Adding element type (line 184)
        int_232923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 54), tuple_232921, int_232923)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), list_232911, tuple_232921)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_232924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        int_232925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 62), tuple_232924, int_232925)
        # Adding element type (line 184)
        int_232926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 62), tuple_232924, int_232926)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 28), list_232911, tuple_232924)
        
        # Assigning a type to the variable 'tuple_assignment_231693' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_assignment_231693', list_232911)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'tuple_assignment_231692' (line 184)
        tuple_assignment_231692_232927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_assignment_231692')
        # Assigning a type to the variable 'x0' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'x0', tuple_assignment_231692_232927)
        
        # Assigning a Name to a Name (line 184):
        # Getting the type of 'tuple_assignment_231693' (line 184)
        tuple_assignment_231693_232928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'tuple_assignment_231693')
        # Assigning a type to the variable 'bnds' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'bnds', tuple_assignment_231693_232928)
        
        # Assigning a List to a Name (line 185):
        
        # Assigning a List to a Name (line 185):
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_232929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        int_232930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 15), list_232929, int_232930)
        # Adding element type (line 185)
        int_232931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 15), list_232929, int_232931)
        # Adding element type (line 185)
        int_232932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 15), list_232929, int_232932)
        # Adding element type (line 185)
        int_232933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 15), list_232929, int_232933)
        # Adding element type (line 185)
        int_232934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 15), list_232929, int_232934)
        
        # Assigning a type to the variable 'xopt' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'xopt', list_232929)
        
        # Assigning a Attribute to a Name (line 186):
        
        # Assigning a Attribute to a Name (line 186):
        
        # Call to minimize(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'self' (line 186)
        self_232937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 30), 'self', False)
        # Obtaining the member 'f45' of a type (line 186)
        f45_232938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 30), self_232937, 'f45')
        # Getting the type of 'x0' (line 186)
        x0_232939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'x0', False)
        # Processing the call keyword arguments (line 186)
        str_232940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 51), 'str', 'TNC')
        keyword_232941 = str_232940
        # Getting the type of 'self' (line 187)
        self_232942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'self', False)
        # Obtaining the member 'g45' of a type (line 187)
        g45_232943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 34), self_232942, 'g45')
        keyword_232944 = g45_232943
        # Getting the type of 'bnds' (line 187)
        bnds_232945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 51), 'bnds', False)
        keyword_232946 = bnds_232945
        # Getting the type of 'self' (line 188)
        self_232947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 38), 'self', False)
        # Obtaining the member 'opts' of a type (line 188)
        opts_232948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 38), self_232947, 'opts')
        keyword_232949 = opts_232948
        kwargs_232950 = {'options': keyword_232949, 'bounds': keyword_232946, 'method': keyword_232941, 'jac': keyword_232944}
        # Getting the type of 'optimize' (line 186)
        optimize_232935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'optimize', False)
        # Obtaining the member 'minimize' of a type (line 186)
        minimize_232936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), optimize_232935, 'minimize')
        # Calling minimize(args, kwargs) (line 186)
        minimize_call_result_232951 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), minimize_232936, *[f45_232938, x0_232939], **kwargs_232950)
        
        # Obtaining the member 'x' of a type (line 186)
        x_232952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), minimize_call_result_232951, 'x')
        # Assigning a type to the variable 'x' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'x', x_232952)
        
        # Call to assert_allclose(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Call to f45(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'x' (line 189)
        x_232956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'x', False)
        # Processing the call keyword arguments (line 189)
        kwargs_232957 = {}
        # Getting the type of 'self' (line 189)
        self_232954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'self', False)
        # Obtaining the member 'f45' of a type (line 189)
        f45_232955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 24), self_232954, 'f45')
        # Calling f45(args, kwargs) (line 189)
        f45_call_result_232958 = invoke(stypy.reporting.localization.Localization(__file__, 189, 24), f45_232955, *[x_232956], **kwargs_232957)
        
        
        # Call to f45(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'xopt' (line 189)
        xopt_232961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 46), 'xopt', False)
        # Processing the call keyword arguments (line 189)
        kwargs_232962 = {}
        # Getting the type of 'self' (line 189)
        self_232959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'self', False)
        # Obtaining the member 'f45' of a type (line 189)
        f45_232960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 37), self_232959, 'f45')
        # Calling f45(args, kwargs) (line 189)
        f45_call_result_232963 = invoke(stypy.reporting.localization.Localization(__file__, 189, 37), f45_232960, *[xopt_232961], **kwargs_232962)
        
        # Processing the call keyword arguments (line 189)
        float_232964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 58), 'float')
        keyword_232965 = float_232964
        kwargs_232966 = {'atol': keyword_232965}
        # Getting the type of 'assert_allclose' (line 189)
        assert_allclose_232953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 189)
        assert_allclose_call_result_232967 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assert_allclose_232953, *[f45_call_result_232958, f45_call_result_232963], **kwargs_232966)
        
        
        # ################# End of 'test_minimize_tnc45(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_tnc45' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_232968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232968)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_tnc45'
        return stypy_return_type_232968


    @norecursion
    def test_tnc1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc1'
        module_type_store = module_type_store.open_function_context('test_tnc1', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc1')
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc1(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 193):
        
        # Assigning a Attribute to a Name (line 193):
        # Getting the type of 'self' (line 193)
        self_232969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'self')
        # Obtaining the member 'fg1' of a type (line 193)
        fg1_232970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 24), self_232969, 'fg1')
        # Assigning a type to the variable 'tuple_assignment_231694' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_assignment_231694', fg1_232970)
        
        # Assigning a List to a Name (line 193):
        
        # Obtaining an instance of the builtin type 'list' (line 193)
        list_232971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 193)
        # Adding element type (line 193)
        int_232972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 34), list_232971, int_232972)
        # Adding element type (line 193)
        int_232973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 34), list_232971, int_232973)
        
        # Assigning a type to the variable 'tuple_assignment_231695' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_assignment_231695', list_232971)
        
        # Assigning a Tuple to a Name (line 193):
        
        # Obtaining an instance of the builtin type 'tuple' (line 193)
        tuple_232974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 193)
        # Adding element type (line 193)
        
        # Obtaining an instance of the builtin type 'list' (line 193)
        list_232975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 193)
        # Adding element type (line 193)
        
        # Getting the type of 'np' (line 193)
        np_232976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 46), 'np')
        # Obtaining the member 'inf' of a type (line 193)
        inf_232977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 46), np_232976, 'inf')
        # Applying the 'usub' unary operator (line 193)
        result___neg___232978 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 45), 'usub', inf_232977)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 44), list_232975, result___neg___232978)
        # Adding element type (line 193)
        # Getting the type of 'None' (line 193)
        None_232979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 54), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 44), list_232975, None_232979)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 44), tuple_232974, list_232975)
        # Adding element type (line 193)
        
        # Obtaining an instance of the builtin type 'list' (line 193)
        list_232980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 193)
        # Adding element type (line 193)
        float_232981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 61), list_232980, float_232981)
        # Adding element type (line 193)
        # Getting the type of 'None' (line 193)
        None_232982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 68), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 61), list_232980, None_232982)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 44), tuple_232974, list_232980)
        
        # Assigning a type to the variable 'tuple_assignment_231696' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_assignment_231696', tuple_232974)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_assignment_231694' (line 193)
        tuple_assignment_231694_232983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_assignment_231694')
        # Assigning a type to the variable 'fg' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'fg', tuple_assignment_231694_232983)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_assignment_231695' (line 193)
        tuple_assignment_231695_232984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_assignment_231695')
        # Assigning a type to the variable 'x' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'x', tuple_assignment_231695_232984)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_assignment_231696' (line 193)
        tuple_assignment_231696_232985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_assignment_231696')
        # Assigning a type to the variable 'bounds' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'bounds', tuple_assignment_231696_232985)
        
        # Assigning a List to a Name (line 194):
        
        # Assigning a List to a Name (line 194):
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_232986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        int_232987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 15), list_232986, int_232987)
        # Adding element type (line 194)
        int_232988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 15), list_232986, int_232988)
        
        # Assigning a type to the variable 'xopt' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'xopt', list_232986)
        
        # Assigning a Call to a Tuple (line 196):
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        int_232989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'int')
        
        # Call to fmin_tnc(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'fg' (line 196)
        fg_232992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 38), 'fg', False)
        # Getting the type of 'x' (line 196)
        x_232993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'x', False)
        # Processing the call keyword arguments (line 196)
        # Getting the type of 'bounds' (line 196)
        bounds_232994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 52), 'bounds', False)
        keyword_232995 = bounds_232994
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_232996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        float_232997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 66), tuple_232996, float_232997)
        
        keyword_232998 = tuple_232996
        # Getting the type of 'optimize' (line 197)
        optimize_232999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 197)
        tnc_233000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 47), optimize_232999, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 197)
        MSG_NONE_233001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 47), tnc_233000, 'MSG_NONE')
        keyword_233002 = MSG_NONE_233001
        int_233003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 45), 'int')
        keyword_233004 = int_233003
        kwargs_233005 = {'args': keyword_232998, 'messages': keyword_233002, 'bounds': keyword_232995, 'maxfun': keyword_233004}
        # Getting the type of 'optimize' (line 196)
        optimize_232990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 196)
        fmin_tnc_232991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 20), optimize_232990, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 196)
        fmin_tnc_call_result_233006 = invoke(stypy.reporting.localization.Localization(__file__, 196, 20), fmin_tnc_232991, *[fg_232992, x_232993], **kwargs_233005)
        
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___233007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), fmin_tnc_call_result_233006, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_233008 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), getitem___233007, int_232989)
        
        # Assigning a type to the variable 'tuple_var_assignment_231697' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_231697', subscript_call_result_233008)
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        int_233009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'int')
        
        # Call to fmin_tnc(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'fg' (line 196)
        fg_233012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 38), 'fg', False)
        # Getting the type of 'x' (line 196)
        x_233013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'x', False)
        # Processing the call keyword arguments (line 196)
        # Getting the type of 'bounds' (line 196)
        bounds_233014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 52), 'bounds', False)
        keyword_233015 = bounds_233014
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_233016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        float_233017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 66), tuple_233016, float_233017)
        
        keyword_233018 = tuple_233016
        # Getting the type of 'optimize' (line 197)
        optimize_233019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 197)
        tnc_233020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 47), optimize_233019, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 197)
        MSG_NONE_233021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 47), tnc_233020, 'MSG_NONE')
        keyword_233022 = MSG_NONE_233021
        int_233023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 45), 'int')
        keyword_233024 = int_233023
        kwargs_233025 = {'args': keyword_233018, 'messages': keyword_233022, 'bounds': keyword_233015, 'maxfun': keyword_233024}
        # Getting the type of 'optimize' (line 196)
        optimize_233010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 196)
        fmin_tnc_233011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 20), optimize_233010, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 196)
        fmin_tnc_call_result_233026 = invoke(stypy.reporting.localization.Localization(__file__, 196, 20), fmin_tnc_233011, *[fg_233012, x_233013], **kwargs_233025)
        
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___233027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), fmin_tnc_call_result_233026, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_233028 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), getitem___233027, int_233009)
        
        # Assigning a type to the variable 'tuple_var_assignment_231698' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_231698', subscript_call_result_233028)
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        int_233029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'int')
        
        # Call to fmin_tnc(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'fg' (line 196)
        fg_233032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 38), 'fg', False)
        # Getting the type of 'x' (line 196)
        x_233033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'x', False)
        # Processing the call keyword arguments (line 196)
        # Getting the type of 'bounds' (line 196)
        bounds_233034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 52), 'bounds', False)
        keyword_233035 = bounds_233034
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_233036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        float_233037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 66), tuple_233036, float_233037)
        
        keyword_233038 = tuple_233036
        # Getting the type of 'optimize' (line 197)
        optimize_233039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 197)
        tnc_233040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 47), optimize_233039, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 197)
        MSG_NONE_233041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 47), tnc_233040, 'MSG_NONE')
        keyword_233042 = MSG_NONE_233041
        int_233043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 45), 'int')
        keyword_233044 = int_233043
        kwargs_233045 = {'args': keyword_233038, 'messages': keyword_233042, 'bounds': keyword_233035, 'maxfun': keyword_233044}
        # Getting the type of 'optimize' (line 196)
        optimize_233030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 196)
        fmin_tnc_233031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 20), optimize_233030, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 196)
        fmin_tnc_call_result_233046 = invoke(stypy.reporting.localization.Localization(__file__, 196, 20), fmin_tnc_233031, *[fg_233032, x_233033], **kwargs_233045)
        
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___233047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), fmin_tnc_call_result_233046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_233048 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), getitem___233047, int_233029)
        
        # Assigning a type to the variable 'tuple_var_assignment_231699' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_231699', subscript_call_result_233048)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'tuple_var_assignment_231697' (line 196)
        tuple_var_assignment_231697_233049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_231697')
        # Assigning a type to the variable 'x' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'x', tuple_var_assignment_231697_233049)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'tuple_var_assignment_231698' (line 196)
        tuple_var_assignment_231698_233050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_231698')
        # Assigning a type to the variable 'nf' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'nf', tuple_var_assignment_231698_233050)
        
        # Assigning a Name to a Name (line 196):
        # Getting the type of 'tuple_var_assignment_231699' (line 196)
        tuple_var_assignment_231699_233051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_231699')
        # Assigning a type to the variable 'rc' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'rc', tuple_var_assignment_231699_233051)
        
        # Call to assert_allclose(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Call to f1(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'x' (line 200)
        x_233055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'x', False)
        # Processing the call keyword arguments (line 200)
        kwargs_233056 = {}
        # Getting the type of 'self' (line 200)
        self_233053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 24), 'self', False)
        # Obtaining the member 'f1' of a type (line 200)
        f1_233054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 24), self_233053, 'f1')
        # Calling f1(args, kwargs) (line 200)
        f1_call_result_233057 = invoke(stypy.reporting.localization.Localization(__file__, 200, 24), f1_233054, *[x_233055], **kwargs_233056)
        
        
        # Call to f1(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'xopt' (line 200)
        xopt_233060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 44), 'xopt', False)
        # Processing the call keyword arguments (line 200)
        kwargs_233061 = {}
        # Getting the type of 'self' (line 200)
        self_233058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'self', False)
        # Obtaining the member 'f1' of a type (line 200)
        f1_233059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 36), self_233058, 'f1')
        # Calling f1(args, kwargs) (line 200)
        f1_call_result_233062 = invoke(stypy.reporting.localization.Localization(__file__, 200, 36), f1_233059, *[xopt_233060], **kwargs_233061)
        
        # Processing the call keyword arguments (line 200)
        float_233063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 56), 'float')
        keyword_233064 = float_233063
        str_233065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 202)
        rc_233066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 55), 'rc', False)
        # Getting the type of 'optimize' (line 202)
        optimize_233067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 202)
        tnc_233068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 32), optimize_233067, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 202)
        RCSTRINGS_233069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 32), tnc_233068, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___233070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 32), RCSTRINGS_233069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_233071 = invoke(stypy.reporting.localization.Localization(__file__, 202, 32), getitem___233070, rc_233066)
        
        # Applying the binary operator '+' (line 201)
        result_add_233072 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 32), '+', str_233065, subscript_call_result_233071)
        
        keyword_233073 = result_add_233072
        kwargs_233074 = {'err_msg': keyword_233073, 'atol': keyword_233064}
        # Getting the type of 'assert_allclose' (line 200)
        assert_allclose_233052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 200)
        assert_allclose_call_result_233075 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), assert_allclose_233052, *[f1_call_result_233057, f1_call_result_233062], **kwargs_233074)
        
        
        # ################# End of 'test_tnc1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc1' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_233076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc1'
        return stypy_return_type_233076


    @norecursion
    def test_tnc1b(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc1b'
        module_type_store = module_type_store.open_function_context('test_tnc1b', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc1b')
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc1b.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc1b', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc1b', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc1b(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 205):
        
        # Assigning a List to a Name (line 205):
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_233077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        int_233078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 20), list_233077, int_233078)
        # Adding element type (line 205)
        int_233079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 20), list_233077, int_233079)
        
        # Assigning a type to the variable 'tuple_assignment_231700' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_assignment_231700', list_233077)
        
        # Assigning a Tuple to a Name (line 205):
        
        # Obtaining an instance of the builtin type 'tuple' (line 205)
        tuple_233080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 205)
        # Adding element type (line 205)
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_233081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        
        # Getting the type of 'np' (line 205)
        np_233082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 32), 'np')
        # Obtaining the member 'inf' of a type (line 205)
        inf_233083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 32), np_233082, 'inf')
        # Applying the 'usub' unary operator (line 205)
        result___neg___233084 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 31), 'usub', inf_233083)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 30), list_233081, result___neg___233084)
        # Adding element type (line 205)
        # Getting the type of 'None' (line 205)
        None_233085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 40), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 30), list_233081, None_233085)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 30), tuple_233080, list_233081)
        # Adding element type (line 205)
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_233086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        float_233087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_233086, float_233087)
        # Adding element type (line 205)
        # Getting the type of 'None' (line 205)
        None_233088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 54), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_233086, None_233088)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 30), tuple_233080, list_233086)
        
        # Assigning a type to the variable 'tuple_assignment_231701' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_assignment_231701', tuple_233080)
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'tuple_assignment_231700' (line 205)
        tuple_assignment_231700_233089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_assignment_231700')
        # Assigning a type to the variable 'x' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'x', tuple_assignment_231700_233089)
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'tuple_assignment_231701' (line 205)
        tuple_assignment_231701_233090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_assignment_231701')
        # Assigning a type to the variable 'bounds' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'bounds', tuple_assignment_231701_233090)
        
        # Assigning a List to a Name (line 206):
        
        # Assigning a List to a Name (line 206):
        
        # Obtaining an instance of the builtin type 'list' (line 206)
        list_233091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 206)
        # Adding element type (line 206)
        int_233092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 15), list_233091, int_233092)
        # Adding element type (line 206)
        int_233093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 15), list_233091, int_233093)
        
        # Assigning a type to the variable 'xopt' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'xopt', list_233091)
        
        # Assigning a Call to a Tuple (line 208):
        
        # Assigning a Subscript to a Name (line 208):
        
        # Obtaining the type of the subscript
        int_233094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 8), 'int')
        
        # Call to fmin_tnc(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'self' (line 208)
        self_233097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 38), 'self', False)
        # Obtaining the member 'f1' of a type (line 208)
        f1_233098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 38), self_233097, 'f1')
        # Getting the type of 'x' (line 208)
        x_233099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'x', False)
        # Processing the call keyword arguments (line 208)
        # Getting the type of 'True' (line 208)
        True_233100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 62), 'True', False)
        keyword_233101 = True_233100
        # Getting the type of 'bounds' (line 209)
        bounds_233102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 45), 'bounds', False)
        keyword_233103 = bounds_233102
        # Getting the type of 'optimize' (line 210)
        optimize_233104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 210)
        tnc_233105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 47), optimize_233104, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 210)
        MSG_NONE_233106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 47), tnc_233105, 'MSG_NONE')
        keyword_233107 = MSG_NONE_233106
        int_233108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 45), 'int')
        keyword_233109 = int_233108
        kwargs_233110 = {'messages': keyword_233107, 'approx_grad': keyword_233101, 'bounds': keyword_233103, 'maxfun': keyword_233109}
        # Getting the type of 'optimize' (line 208)
        optimize_233095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 208)
        fmin_tnc_233096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), optimize_233095, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 208)
        fmin_tnc_call_result_233111 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), fmin_tnc_233096, *[f1_233098, x_233099], **kwargs_233110)
        
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___233112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), fmin_tnc_call_result_233111, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_233113 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), getitem___233112, int_233094)
        
        # Assigning a type to the variable 'tuple_var_assignment_231702' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'tuple_var_assignment_231702', subscript_call_result_233113)
        
        # Assigning a Subscript to a Name (line 208):
        
        # Obtaining the type of the subscript
        int_233114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 8), 'int')
        
        # Call to fmin_tnc(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'self' (line 208)
        self_233117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 38), 'self', False)
        # Obtaining the member 'f1' of a type (line 208)
        f1_233118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 38), self_233117, 'f1')
        # Getting the type of 'x' (line 208)
        x_233119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'x', False)
        # Processing the call keyword arguments (line 208)
        # Getting the type of 'True' (line 208)
        True_233120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 62), 'True', False)
        keyword_233121 = True_233120
        # Getting the type of 'bounds' (line 209)
        bounds_233122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 45), 'bounds', False)
        keyword_233123 = bounds_233122
        # Getting the type of 'optimize' (line 210)
        optimize_233124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 210)
        tnc_233125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 47), optimize_233124, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 210)
        MSG_NONE_233126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 47), tnc_233125, 'MSG_NONE')
        keyword_233127 = MSG_NONE_233126
        int_233128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 45), 'int')
        keyword_233129 = int_233128
        kwargs_233130 = {'messages': keyword_233127, 'approx_grad': keyword_233121, 'bounds': keyword_233123, 'maxfun': keyword_233129}
        # Getting the type of 'optimize' (line 208)
        optimize_233115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 208)
        fmin_tnc_233116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), optimize_233115, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 208)
        fmin_tnc_call_result_233131 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), fmin_tnc_233116, *[f1_233118, x_233119], **kwargs_233130)
        
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___233132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), fmin_tnc_call_result_233131, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_233133 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), getitem___233132, int_233114)
        
        # Assigning a type to the variable 'tuple_var_assignment_231703' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'tuple_var_assignment_231703', subscript_call_result_233133)
        
        # Assigning a Subscript to a Name (line 208):
        
        # Obtaining the type of the subscript
        int_233134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 8), 'int')
        
        # Call to fmin_tnc(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'self' (line 208)
        self_233137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 38), 'self', False)
        # Obtaining the member 'f1' of a type (line 208)
        f1_233138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 38), self_233137, 'f1')
        # Getting the type of 'x' (line 208)
        x_233139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'x', False)
        # Processing the call keyword arguments (line 208)
        # Getting the type of 'True' (line 208)
        True_233140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 62), 'True', False)
        keyword_233141 = True_233140
        # Getting the type of 'bounds' (line 209)
        bounds_233142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 45), 'bounds', False)
        keyword_233143 = bounds_233142
        # Getting the type of 'optimize' (line 210)
        optimize_233144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 210)
        tnc_233145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 47), optimize_233144, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 210)
        MSG_NONE_233146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 47), tnc_233145, 'MSG_NONE')
        keyword_233147 = MSG_NONE_233146
        int_233148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 45), 'int')
        keyword_233149 = int_233148
        kwargs_233150 = {'messages': keyword_233147, 'approx_grad': keyword_233141, 'bounds': keyword_233143, 'maxfun': keyword_233149}
        # Getting the type of 'optimize' (line 208)
        optimize_233135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 208)
        fmin_tnc_233136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), optimize_233135, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 208)
        fmin_tnc_call_result_233151 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), fmin_tnc_233136, *[f1_233138, x_233139], **kwargs_233150)
        
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___233152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), fmin_tnc_call_result_233151, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_233153 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), getitem___233152, int_233134)
        
        # Assigning a type to the variable 'tuple_var_assignment_231704' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'tuple_var_assignment_231704', subscript_call_result_233153)
        
        # Assigning a Name to a Name (line 208):
        # Getting the type of 'tuple_var_assignment_231702' (line 208)
        tuple_var_assignment_231702_233154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'tuple_var_assignment_231702')
        # Assigning a type to the variable 'x' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'x', tuple_var_assignment_231702_233154)
        
        # Assigning a Name to a Name (line 208):
        # Getting the type of 'tuple_var_assignment_231703' (line 208)
        tuple_var_assignment_231703_233155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'tuple_var_assignment_231703')
        # Assigning a type to the variable 'nf' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'nf', tuple_var_assignment_231703_233155)
        
        # Assigning a Name to a Name (line 208):
        # Getting the type of 'tuple_var_assignment_231704' (line 208)
        tuple_var_assignment_231704_233156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'tuple_var_assignment_231704')
        # Assigning a type to the variable 'rc' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'rc', tuple_var_assignment_231704_233156)
        
        # Call to assert_allclose(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to f1(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'x' (line 213)
        x_233160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'x', False)
        # Processing the call keyword arguments (line 213)
        kwargs_233161 = {}
        # Getting the type of 'self' (line 213)
        self_233158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'self', False)
        # Obtaining the member 'f1' of a type (line 213)
        f1_233159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 24), self_233158, 'f1')
        # Calling f1(args, kwargs) (line 213)
        f1_call_result_233162 = invoke(stypy.reporting.localization.Localization(__file__, 213, 24), f1_233159, *[x_233160], **kwargs_233161)
        
        
        # Call to f1(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'xopt' (line 213)
        xopt_233165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 44), 'xopt', False)
        # Processing the call keyword arguments (line 213)
        kwargs_233166 = {}
        # Getting the type of 'self' (line 213)
        self_233163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'self', False)
        # Obtaining the member 'f1' of a type (line 213)
        f1_233164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 36), self_233163, 'f1')
        # Calling f1(args, kwargs) (line 213)
        f1_call_result_233167 = invoke(stypy.reporting.localization.Localization(__file__, 213, 36), f1_233164, *[xopt_233165], **kwargs_233166)
        
        # Processing the call keyword arguments (line 213)
        float_233168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 56), 'float')
        keyword_233169 = float_233168
        str_233170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 215)
        rc_233171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 55), 'rc', False)
        # Getting the type of 'optimize' (line 215)
        optimize_233172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 215)
        tnc_233173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 32), optimize_233172, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 215)
        RCSTRINGS_233174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 32), tnc_233173, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___233175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 32), RCSTRINGS_233174, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_233176 = invoke(stypy.reporting.localization.Localization(__file__, 215, 32), getitem___233175, rc_233171)
        
        # Applying the binary operator '+' (line 214)
        result_add_233177 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 32), '+', str_233170, subscript_call_result_233176)
        
        keyword_233178 = result_add_233177
        kwargs_233179 = {'err_msg': keyword_233178, 'atol': keyword_233169}
        # Getting the type of 'assert_allclose' (line 213)
        assert_allclose_233157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 213)
        assert_allclose_call_result_233180 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assert_allclose_233157, *[f1_call_result_233162, f1_call_result_233167], **kwargs_233179)
        
        
        # ################# End of 'test_tnc1b(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc1b' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_233181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc1b'
        return stypy_return_type_233181


    @norecursion
    def test_tnc1c(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc1c'
        module_type_store = module_type_store.open_function_context('test_tnc1c', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc1c')
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc1c.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc1c', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc1c', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc1c(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 218):
        
        # Assigning a List to a Name (line 218):
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_233182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        int_233183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 20), list_233182, int_233183)
        # Adding element type (line 218)
        int_233184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 20), list_233182, int_233184)
        
        # Assigning a type to the variable 'tuple_assignment_231705' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tuple_assignment_231705', list_233182)
        
        # Assigning a Tuple to a Name (line 218):
        
        # Obtaining an instance of the builtin type 'tuple' (line 218)
        tuple_233185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 218)
        # Adding element type (line 218)
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_233186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        
        # Getting the type of 'np' (line 218)
        np_233187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 32), 'np')
        # Obtaining the member 'inf' of a type (line 218)
        inf_233188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 32), np_233187, 'inf')
        # Applying the 'usub' unary operator (line 218)
        result___neg___233189 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 31), 'usub', inf_233188)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 30), list_233186, result___neg___233189)
        # Adding element type (line 218)
        # Getting the type of 'None' (line 218)
        None_233190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 40), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 30), list_233186, None_233190)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 30), tuple_233185, list_233186)
        # Adding element type (line 218)
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_233191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        float_233192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 47), list_233191, float_233192)
        # Adding element type (line 218)
        # Getting the type of 'None' (line 218)
        None_233193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 54), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 47), list_233191, None_233193)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 30), tuple_233185, list_233191)
        
        # Assigning a type to the variable 'tuple_assignment_231706' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tuple_assignment_231706', tuple_233185)
        
        # Assigning a Name to a Name (line 218):
        # Getting the type of 'tuple_assignment_231705' (line 218)
        tuple_assignment_231705_233194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tuple_assignment_231705')
        # Assigning a type to the variable 'x' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'x', tuple_assignment_231705_233194)
        
        # Assigning a Name to a Name (line 218):
        # Getting the type of 'tuple_assignment_231706' (line 218)
        tuple_assignment_231706_233195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tuple_assignment_231706')
        # Assigning a type to the variable 'bounds' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'bounds', tuple_assignment_231706_233195)
        
        # Assigning a List to a Name (line 219):
        
        # Assigning a List to a Name (line 219):
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_233196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        int_233197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 15), list_233196, int_233197)
        # Adding element type (line 219)
        int_233198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 15), list_233196, int_233198)
        
        # Assigning a type to the variable 'xopt' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'xopt', list_233196)
        
        # Assigning a Call to a Tuple (line 221):
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_233199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        
        # Call to fmin_tnc(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_233202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 38), 'self', False)
        # Obtaining the member 'f1' of a type (line 221)
        f1_233203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 38), self_233202, 'f1')
        # Getting the type of 'x' (line 221)
        x_233204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 47), 'x', False)
        # Processing the call keyword arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_233205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 57), 'self', False)
        # Obtaining the member 'g1' of a type (line 221)
        g1_233206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 57), self_233205, 'g1')
        keyword_233207 = g1_233206
        # Getting the type of 'bounds' (line 222)
        bounds_233208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 45), 'bounds', False)
        keyword_233209 = bounds_233208
        # Getting the type of 'optimize' (line 223)
        optimize_233210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 223)
        tnc_233211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 47), optimize_233210, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 223)
        MSG_NONE_233212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 47), tnc_233211, 'MSG_NONE')
        keyword_233213 = MSG_NONE_233212
        int_233214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 45), 'int')
        keyword_233215 = int_233214
        kwargs_233216 = {'maxfun': keyword_233215, 'messages': keyword_233213, 'fprime': keyword_233207, 'bounds': keyword_233209}
        # Getting the type of 'optimize' (line 221)
        optimize_233200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 221)
        fmin_tnc_233201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), optimize_233200, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 221)
        fmin_tnc_call_result_233217 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), fmin_tnc_233201, *[f1_233203, x_233204], **kwargs_233216)
        
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___233218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), fmin_tnc_call_result_233217, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_233219 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___233218, int_233199)
        
        # Assigning a type to the variable 'tuple_var_assignment_231707' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_231707', subscript_call_result_233219)
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_233220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        
        # Call to fmin_tnc(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_233223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 38), 'self', False)
        # Obtaining the member 'f1' of a type (line 221)
        f1_233224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 38), self_233223, 'f1')
        # Getting the type of 'x' (line 221)
        x_233225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 47), 'x', False)
        # Processing the call keyword arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_233226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 57), 'self', False)
        # Obtaining the member 'g1' of a type (line 221)
        g1_233227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 57), self_233226, 'g1')
        keyword_233228 = g1_233227
        # Getting the type of 'bounds' (line 222)
        bounds_233229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 45), 'bounds', False)
        keyword_233230 = bounds_233229
        # Getting the type of 'optimize' (line 223)
        optimize_233231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 223)
        tnc_233232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 47), optimize_233231, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 223)
        MSG_NONE_233233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 47), tnc_233232, 'MSG_NONE')
        keyword_233234 = MSG_NONE_233233
        int_233235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 45), 'int')
        keyword_233236 = int_233235
        kwargs_233237 = {'maxfun': keyword_233236, 'messages': keyword_233234, 'fprime': keyword_233228, 'bounds': keyword_233230}
        # Getting the type of 'optimize' (line 221)
        optimize_233221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 221)
        fmin_tnc_233222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), optimize_233221, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 221)
        fmin_tnc_call_result_233238 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), fmin_tnc_233222, *[f1_233224, x_233225], **kwargs_233237)
        
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___233239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), fmin_tnc_call_result_233238, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_233240 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___233239, int_233220)
        
        # Assigning a type to the variable 'tuple_var_assignment_231708' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_231708', subscript_call_result_233240)
        
        # Assigning a Subscript to a Name (line 221):
        
        # Obtaining the type of the subscript
        int_233241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        
        # Call to fmin_tnc(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_233244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 38), 'self', False)
        # Obtaining the member 'f1' of a type (line 221)
        f1_233245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 38), self_233244, 'f1')
        # Getting the type of 'x' (line 221)
        x_233246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 47), 'x', False)
        # Processing the call keyword arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_233247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 57), 'self', False)
        # Obtaining the member 'g1' of a type (line 221)
        g1_233248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 57), self_233247, 'g1')
        keyword_233249 = g1_233248
        # Getting the type of 'bounds' (line 222)
        bounds_233250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 45), 'bounds', False)
        keyword_233251 = bounds_233250
        # Getting the type of 'optimize' (line 223)
        optimize_233252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 223)
        tnc_233253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 47), optimize_233252, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 223)
        MSG_NONE_233254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 47), tnc_233253, 'MSG_NONE')
        keyword_233255 = MSG_NONE_233254
        int_233256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 45), 'int')
        keyword_233257 = int_233256
        kwargs_233258 = {'maxfun': keyword_233257, 'messages': keyword_233255, 'fprime': keyword_233249, 'bounds': keyword_233251}
        # Getting the type of 'optimize' (line 221)
        optimize_233242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 221)
        fmin_tnc_233243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), optimize_233242, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 221)
        fmin_tnc_call_result_233259 = invoke(stypy.reporting.localization.Localization(__file__, 221, 20), fmin_tnc_233243, *[f1_233245, x_233246], **kwargs_233258)
        
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___233260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), fmin_tnc_call_result_233259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_233261 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___233260, int_233241)
        
        # Assigning a type to the variable 'tuple_var_assignment_231709' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_231709', subscript_call_result_233261)
        
        # Assigning a Name to a Name (line 221):
        # Getting the type of 'tuple_var_assignment_231707' (line 221)
        tuple_var_assignment_231707_233262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_231707')
        # Assigning a type to the variable 'x' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'x', tuple_var_assignment_231707_233262)
        
        # Assigning a Name to a Name (line 221):
        # Getting the type of 'tuple_var_assignment_231708' (line 221)
        tuple_var_assignment_231708_233263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_231708')
        # Assigning a type to the variable 'nf' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'nf', tuple_var_assignment_231708_233263)
        
        # Assigning a Name to a Name (line 221):
        # Getting the type of 'tuple_var_assignment_231709' (line 221)
        tuple_var_assignment_231709_233264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_231709')
        # Assigning a type to the variable 'rc' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'rc', tuple_var_assignment_231709_233264)
        
        # Call to assert_allclose(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Call to f1(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'x' (line 226)
        x_233268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 32), 'x', False)
        # Processing the call keyword arguments (line 226)
        kwargs_233269 = {}
        # Getting the type of 'self' (line 226)
        self_233266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'self', False)
        # Obtaining the member 'f1' of a type (line 226)
        f1_233267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 24), self_233266, 'f1')
        # Calling f1(args, kwargs) (line 226)
        f1_call_result_233270 = invoke(stypy.reporting.localization.Localization(__file__, 226, 24), f1_233267, *[x_233268], **kwargs_233269)
        
        
        # Call to f1(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'xopt' (line 226)
        xopt_233273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'xopt', False)
        # Processing the call keyword arguments (line 226)
        kwargs_233274 = {}
        # Getting the type of 'self' (line 226)
        self_233271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'self', False)
        # Obtaining the member 'f1' of a type (line 226)
        f1_233272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 36), self_233271, 'f1')
        # Calling f1(args, kwargs) (line 226)
        f1_call_result_233275 = invoke(stypy.reporting.localization.Localization(__file__, 226, 36), f1_233272, *[xopt_233273], **kwargs_233274)
        
        # Processing the call keyword arguments (line 226)
        float_233276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 56), 'float')
        keyword_233277 = float_233276
        str_233278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 228)
        rc_233279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 55), 'rc', False)
        # Getting the type of 'optimize' (line 228)
        optimize_233280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 228)
        tnc_233281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 32), optimize_233280, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 228)
        RCSTRINGS_233282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 32), tnc_233281, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___233283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 32), RCSTRINGS_233282, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_233284 = invoke(stypy.reporting.localization.Localization(__file__, 228, 32), getitem___233283, rc_233279)
        
        # Applying the binary operator '+' (line 227)
        result_add_233285 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 32), '+', str_233278, subscript_call_result_233284)
        
        keyword_233286 = result_add_233285
        kwargs_233287 = {'err_msg': keyword_233286, 'atol': keyword_233277}
        # Getting the type of 'assert_allclose' (line 226)
        assert_allclose_233265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 226)
        assert_allclose_call_result_233288 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), assert_allclose_233265, *[f1_call_result_233270, f1_call_result_233275], **kwargs_233287)
        
        
        # ################# End of 'test_tnc1c(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc1c' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_233289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc1c'
        return stypy_return_type_233289


    @norecursion
    def test_tnc2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc2'
        module_type_store = module_type_store.open_function_context('test_tnc2', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc2')
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc2(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 231):
        
        # Assigning a Attribute to a Name (line 231):
        # Getting the type of 'self' (line 231)
        self_233290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'self')
        # Obtaining the member 'fg1' of a type (line 231)
        fg1_233291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 24), self_233290, 'fg1')
        # Assigning a type to the variable 'tuple_assignment_231710' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_assignment_231710', fg1_233291)
        
        # Assigning a List to a Name (line 231):
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_233292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        int_233293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), list_233292, int_233293)
        # Adding element type (line 231)
        int_233294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), list_233292, int_233294)
        
        # Assigning a type to the variable 'tuple_assignment_231711' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_assignment_231711', list_233292)
        
        # Assigning a Tuple to a Name (line 231):
        
        # Obtaining an instance of the builtin type 'tuple' (line 231)
        tuple_233295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 231)
        # Adding element type (line 231)
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_233296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        
        # Getting the type of 'np' (line 231)
        np_233297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 46), 'np')
        # Obtaining the member 'inf' of a type (line 231)
        inf_233298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 46), np_233297, 'inf')
        # Applying the 'usub' unary operator (line 231)
        result___neg___233299 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 45), 'usub', inf_233298)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 44), list_233296, result___neg___233299)
        # Adding element type (line 231)
        # Getting the type of 'None' (line 231)
        None_233300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 54), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 44), list_233296, None_233300)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 44), tuple_233295, list_233296)
        # Adding element type (line 231)
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_233301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        float_233302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 61), list_233301, float_233302)
        # Adding element type (line 231)
        # Getting the type of 'None' (line 231)
        None_233303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 67), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 61), list_233301, None_233303)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 44), tuple_233295, list_233301)
        
        # Assigning a type to the variable 'tuple_assignment_231712' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_assignment_231712', tuple_233295)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_assignment_231710' (line 231)
        tuple_assignment_231710_233304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_assignment_231710')
        # Assigning a type to the variable 'fg' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'fg', tuple_assignment_231710_233304)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_assignment_231711' (line 231)
        tuple_assignment_231711_233305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_assignment_231711')
        # Assigning a type to the variable 'x' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'x', tuple_assignment_231711_233305)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_assignment_231712' (line 231)
        tuple_assignment_231712_233306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_assignment_231712')
        # Assigning a type to the variable 'bounds' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'bounds', tuple_assignment_231712_233306)
        
        # Assigning a List to a Name (line 232):
        
        # Assigning a List to a Name (line 232):
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_233307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        float_233308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 15), list_233307, float_233308)
        # Adding element type (line 232)
        float_233309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 15), list_233307, float_233309)
        
        # Assigning a type to the variable 'xopt' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'xopt', list_233307)
        
        # Assigning a Call to a Tuple (line 234):
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_233310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
        
        # Call to fmin_tnc(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'fg' (line 234)
        fg_233313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 38), 'fg', False)
        # Getting the type of 'x' (line 234)
        x_233314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 42), 'x', False)
        # Processing the call keyword arguments (line 234)
        # Getting the type of 'bounds' (line 234)
        bounds_233315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 52), 'bounds', False)
        keyword_233316 = bounds_233315
        # Getting the type of 'optimize' (line 235)
        optimize_233317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 235)
        tnc_233318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 47), optimize_233317, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 235)
        MSG_NONE_233319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 47), tnc_233318, 'MSG_NONE')
        keyword_233320 = MSG_NONE_233319
        int_233321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 45), 'int')
        keyword_233322 = int_233321
        kwargs_233323 = {'messages': keyword_233320, 'bounds': keyword_233316, 'maxfun': keyword_233322}
        # Getting the type of 'optimize' (line 234)
        optimize_233311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 234)
        fmin_tnc_233312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 20), optimize_233311, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 234)
        fmin_tnc_call_result_233324 = invoke(stypy.reporting.localization.Localization(__file__, 234, 20), fmin_tnc_233312, *[fg_233313, x_233314], **kwargs_233323)
        
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___233325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), fmin_tnc_call_result_233324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_233326 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), getitem___233325, int_233310)
        
        # Assigning a type to the variable 'tuple_var_assignment_231713' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_231713', subscript_call_result_233326)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_233327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
        
        # Call to fmin_tnc(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'fg' (line 234)
        fg_233330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 38), 'fg', False)
        # Getting the type of 'x' (line 234)
        x_233331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 42), 'x', False)
        # Processing the call keyword arguments (line 234)
        # Getting the type of 'bounds' (line 234)
        bounds_233332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 52), 'bounds', False)
        keyword_233333 = bounds_233332
        # Getting the type of 'optimize' (line 235)
        optimize_233334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 235)
        tnc_233335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 47), optimize_233334, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 235)
        MSG_NONE_233336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 47), tnc_233335, 'MSG_NONE')
        keyword_233337 = MSG_NONE_233336
        int_233338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 45), 'int')
        keyword_233339 = int_233338
        kwargs_233340 = {'messages': keyword_233337, 'bounds': keyword_233333, 'maxfun': keyword_233339}
        # Getting the type of 'optimize' (line 234)
        optimize_233328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 234)
        fmin_tnc_233329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 20), optimize_233328, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 234)
        fmin_tnc_call_result_233341 = invoke(stypy.reporting.localization.Localization(__file__, 234, 20), fmin_tnc_233329, *[fg_233330, x_233331], **kwargs_233340)
        
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___233342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), fmin_tnc_call_result_233341, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_233343 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), getitem___233342, int_233327)
        
        # Assigning a type to the variable 'tuple_var_assignment_231714' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_231714', subscript_call_result_233343)
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        int_233344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 8), 'int')
        
        # Call to fmin_tnc(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'fg' (line 234)
        fg_233347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 38), 'fg', False)
        # Getting the type of 'x' (line 234)
        x_233348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 42), 'x', False)
        # Processing the call keyword arguments (line 234)
        # Getting the type of 'bounds' (line 234)
        bounds_233349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 52), 'bounds', False)
        keyword_233350 = bounds_233349
        # Getting the type of 'optimize' (line 235)
        optimize_233351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 235)
        tnc_233352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 47), optimize_233351, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 235)
        MSG_NONE_233353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 47), tnc_233352, 'MSG_NONE')
        keyword_233354 = MSG_NONE_233353
        int_233355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 45), 'int')
        keyword_233356 = int_233355
        kwargs_233357 = {'messages': keyword_233354, 'bounds': keyword_233350, 'maxfun': keyword_233356}
        # Getting the type of 'optimize' (line 234)
        optimize_233345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 234)
        fmin_tnc_233346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 20), optimize_233345, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 234)
        fmin_tnc_call_result_233358 = invoke(stypy.reporting.localization.Localization(__file__, 234, 20), fmin_tnc_233346, *[fg_233347, x_233348], **kwargs_233357)
        
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___233359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), fmin_tnc_call_result_233358, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_233360 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), getitem___233359, int_233344)
        
        # Assigning a type to the variable 'tuple_var_assignment_231715' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_231715', subscript_call_result_233360)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_231713' (line 234)
        tuple_var_assignment_231713_233361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_231713')
        # Assigning a type to the variable 'x' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'x', tuple_var_assignment_231713_233361)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_231714' (line 234)
        tuple_var_assignment_231714_233362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_231714')
        # Assigning a type to the variable 'nf' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'nf', tuple_var_assignment_231714_233362)
        
        # Assigning a Name to a Name (line 234):
        # Getting the type of 'tuple_var_assignment_231715' (line 234)
        tuple_var_assignment_231715_233363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'tuple_var_assignment_231715')
        # Assigning a type to the variable 'rc' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'rc', tuple_var_assignment_231715_233363)
        
        # Call to assert_allclose(...): (line 238)
        # Processing the call arguments (line 238)
        
        # Call to f1(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'x' (line 238)
        x_233367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 32), 'x', False)
        # Processing the call keyword arguments (line 238)
        kwargs_233368 = {}
        # Getting the type of 'self' (line 238)
        self_233365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'self', False)
        # Obtaining the member 'f1' of a type (line 238)
        f1_233366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 24), self_233365, 'f1')
        # Calling f1(args, kwargs) (line 238)
        f1_call_result_233369 = invoke(stypy.reporting.localization.Localization(__file__, 238, 24), f1_233366, *[x_233367], **kwargs_233368)
        
        
        # Call to f1(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'xopt' (line 238)
        xopt_233372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 44), 'xopt', False)
        # Processing the call keyword arguments (line 238)
        kwargs_233373 = {}
        # Getting the type of 'self' (line 238)
        self_233370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'self', False)
        # Obtaining the member 'f1' of a type (line 238)
        f1_233371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 36), self_233370, 'f1')
        # Calling f1(args, kwargs) (line 238)
        f1_call_result_233374 = invoke(stypy.reporting.localization.Localization(__file__, 238, 36), f1_233371, *[xopt_233372], **kwargs_233373)
        
        # Processing the call keyword arguments (line 238)
        float_233375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 56), 'float')
        keyword_233376 = float_233375
        str_233377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 240)
        rc_233378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 55), 'rc', False)
        # Getting the type of 'optimize' (line 240)
        optimize_233379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 240)
        tnc_233380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 32), optimize_233379, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 240)
        RCSTRINGS_233381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 32), tnc_233380, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___233382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 32), RCSTRINGS_233381, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_233383 = invoke(stypy.reporting.localization.Localization(__file__, 240, 32), getitem___233382, rc_233378)
        
        # Applying the binary operator '+' (line 239)
        result_add_233384 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 32), '+', str_233377, subscript_call_result_233383)
        
        keyword_233385 = result_add_233384
        kwargs_233386 = {'err_msg': keyword_233385, 'atol': keyword_233376}
        # Getting the type of 'assert_allclose' (line 238)
        assert_allclose_233364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 238)
        assert_allclose_call_result_233387 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), assert_allclose_233364, *[f1_call_result_233369, f1_call_result_233374], **kwargs_233386)
        
        
        # ################# End of 'test_tnc2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc2' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_233388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233388)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc2'
        return stypy_return_type_233388


    @norecursion
    def test_tnc3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc3'
        module_type_store = module_type_store.open_function_context('test_tnc3', 242, 4, False)
        # Assigning a type to the variable 'self' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc3')
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc3(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 243):
        
        # Assigning a Attribute to a Name (line 243):
        # Getting the type of 'self' (line 243)
        self_233389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'self')
        # Obtaining the member 'fg3' of a type (line 243)
        fg3_233390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 24), self_233389, 'fg3')
        # Assigning a type to the variable 'tuple_assignment_231716' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_assignment_231716', fg3_233390)
        
        # Assigning a List to a Name (line 243):
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_233391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        int_233392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 34), list_233391, int_233392)
        # Adding element type (line 243)
        int_233393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 34), list_233391, int_233393)
        
        # Assigning a type to the variable 'tuple_assignment_231717' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_assignment_231717', list_233391)
        
        # Assigning a Tuple to a Name (line 243):
        
        # Obtaining an instance of the builtin type 'tuple' (line 243)
        tuple_233394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 243)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_233395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        
        # Getting the type of 'np' (line 243)
        np_233396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 46), 'np')
        # Obtaining the member 'inf' of a type (line 243)
        inf_233397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 46), np_233396, 'inf')
        # Applying the 'usub' unary operator (line 243)
        result___neg___233398 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 45), 'usub', inf_233397)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), list_233395, result___neg___233398)
        # Adding element type (line 243)
        # Getting the type of 'None' (line 243)
        None_233399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 54), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), list_233395, None_233399)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), tuple_233394, list_233395)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_233400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        float_233401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 61), list_233400, float_233401)
        # Adding element type (line 243)
        # Getting the type of 'None' (line 243)
        None_233402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 67), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 61), list_233400, None_233402)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), tuple_233394, list_233400)
        
        # Assigning a type to the variable 'tuple_assignment_231718' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_assignment_231718', tuple_233394)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'tuple_assignment_231716' (line 243)
        tuple_assignment_231716_233403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_assignment_231716')
        # Assigning a type to the variable 'fg' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'fg', tuple_assignment_231716_233403)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'tuple_assignment_231717' (line 243)
        tuple_assignment_231717_233404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_assignment_231717')
        # Assigning a type to the variable 'x' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'x', tuple_assignment_231717_233404)
        
        # Assigning a Name to a Name (line 243):
        # Getting the type of 'tuple_assignment_231718' (line 243)
        tuple_assignment_231718_233405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_assignment_231718')
        # Assigning a type to the variable 'bounds' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'bounds', tuple_assignment_231718_233405)
        
        # Assigning a List to a Name (line 244):
        
        # Assigning a List to a Name (line 244):
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_233406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        int_233407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 15), list_233406, int_233407)
        # Adding element type (line 244)
        int_233408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 15), list_233406, int_233408)
        
        # Assigning a type to the variable 'xopt' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'xopt', list_233406)
        
        # Assigning a Call to a Tuple (line 246):
        
        # Assigning a Subscript to a Name (line 246):
        
        # Obtaining the type of the subscript
        int_233409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 8), 'int')
        
        # Call to fmin_tnc(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'fg' (line 246)
        fg_233412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'fg', False)
        # Getting the type of 'x' (line 246)
        x_233413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'x', False)
        # Processing the call keyword arguments (line 246)
        # Getting the type of 'bounds' (line 246)
        bounds_233414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 52), 'bounds', False)
        keyword_233415 = bounds_233414
        # Getting the type of 'optimize' (line 247)
        optimize_233416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 247)
        tnc_233417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 47), optimize_233416, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 247)
        MSG_NONE_233418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 47), tnc_233417, 'MSG_NONE')
        keyword_233419 = MSG_NONE_233418
        int_233420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 45), 'int')
        keyword_233421 = int_233420
        kwargs_233422 = {'messages': keyword_233419, 'bounds': keyword_233415, 'maxfun': keyword_233421}
        # Getting the type of 'optimize' (line 246)
        optimize_233410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 246)
        fmin_tnc_233411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 20), optimize_233410, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 246)
        fmin_tnc_call_result_233423 = invoke(stypy.reporting.localization.Localization(__file__, 246, 20), fmin_tnc_233411, *[fg_233412, x_233413], **kwargs_233422)
        
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___233424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), fmin_tnc_call_result_233423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_233425 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), getitem___233424, int_233409)
        
        # Assigning a type to the variable 'tuple_var_assignment_231719' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_231719', subscript_call_result_233425)
        
        # Assigning a Subscript to a Name (line 246):
        
        # Obtaining the type of the subscript
        int_233426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 8), 'int')
        
        # Call to fmin_tnc(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'fg' (line 246)
        fg_233429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'fg', False)
        # Getting the type of 'x' (line 246)
        x_233430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'x', False)
        # Processing the call keyword arguments (line 246)
        # Getting the type of 'bounds' (line 246)
        bounds_233431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 52), 'bounds', False)
        keyword_233432 = bounds_233431
        # Getting the type of 'optimize' (line 247)
        optimize_233433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 247)
        tnc_233434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 47), optimize_233433, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 247)
        MSG_NONE_233435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 47), tnc_233434, 'MSG_NONE')
        keyword_233436 = MSG_NONE_233435
        int_233437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 45), 'int')
        keyword_233438 = int_233437
        kwargs_233439 = {'messages': keyword_233436, 'bounds': keyword_233432, 'maxfun': keyword_233438}
        # Getting the type of 'optimize' (line 246)
        optimize_233427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 246)
        fmin_tnc_233428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 20), optimize_233427, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 246)
        fmin_tnc_call_result_233440 = invoke(stypy.reporting.localization.Localization(__file__, 246, 20), fmin_tnc_233428, *[fg_233429, x_233430], **kwargs_233439)
        
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___233441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), fmin_tnc_call_result_233440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_233442 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), getitem___233441, int_233426)
        
        # Assigning a type to the variable 'tuple_var_assignment_231720' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_231720', subscript_call_result_233442)
        
        # Assigning a Subscript to a Name (line 246):
        
        # Obtaining the type of the subscript
        int_233443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 8), 'int')
        
        # Call to fmin_tnc(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'fg' (line 246)
        fg_233446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'fg', False)
        # Getting the type of 'x' (line 246)
        x_233447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'x', False)
        # Processing the call keyword arguments (line 246)
        # Getting the type of 'bounds' (line 246)
        bounds_233448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 52), 'bounds', False)
        keyword_233449 = bounds_233448
        # Getting the type of 'optimize' (line 247)
        optimize_233450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 247)
        tnc_233451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 47), optimize_233450, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 247)
        MSG_NONE_233452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 47), tnc_233451, 'MSG_NONE')
        keyword_233453 = MSG_NONE_233452
        int_233454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 45), 'int')
        keyword_233455 = int_233454
        kwargs_233456 = {'messages': keyword_233453, 'bounds': keyword_233449, 'maxfun': keyword_233455}
        # Getting the type of 'optimize' (line 246)
        optimize_233444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 246)
        fmin_tnc_233445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 20), optimize_233444, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 246)
        fmin_tnc_call_result_233457 = invoke(stypy.reporting.localization.Localization(__file__, 246, 20), fmin_tnc_233445, *[fg_233446, x_233447], **kwargs_233456)
        
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___233458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), fmin_tnc_call_result_233457, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 246)
        subscript_call_result_233459 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), getitem___233458, int_233443)
        
        # Assigning a type to the variable 'tuple_var_assignment_231721' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_231721', subscript_call_result_233459)
        
        # Assigning a Name to a Name (line 246):
        # Getting the type of 'tuple_var_assignment_231719' (line 246)
        tuple_var_assignment_231719_233460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_231719')
        # Assigning a type to the variable 'x' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'x', tuple_var_assignment_231719_233460)
        
        # Assigning a Name to a Name (line 246):
        # Getting the type of 'tuple_var_assignment_231720' (line 246)
        tuple_var_assignment_231720_233461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_231720')
        # Assigning a type to the variable 'nf' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'nf', tuple_var_assignment_231720_233461)
        
        # Assigning a Name to a Name (line 246):
        # Getting the type of 'tuple_var_assignment_231721' (line 246)
        tuple_var_assignment_231721_233462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_231721')
        # Assigning a type to the variable 'rc' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'rc', tuple_var_assignment_231721_233462)
        
        # Call to assert_allclose(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to f3(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'x' (line 250)
        x_233466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 32), 'x', False)
        # Processing the call keyword arguments (line 250)
        kwargs_233467 = {}
        # Getting the type of 'self' (line 250)
        self_233464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'self', False)
        # Obtaining the member 'f3' of a type (line 250)
        f3_233465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 24), self_233464, 'f3')
        # Calling f3(args, kwargs) (line 250)
        f3_call_result_233468 = invoke(stypy.reporting.localization.Localization(__file__, 250, 24), f3_233465, *[x_233466], **kwargs_233467)
        
        
        # Call to f3(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'xopt' (line 250)
        xopt_233471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 44), 'xopt', False)
        # Processing the call keyword arguments (line 250)
        kwargs_233472 = {}
        # Getting the type of 'self' (line 250)
        self_233469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 36), 'self', False)
        # Obtaining the member 'f3' of a type (line 250)
        f3_233470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 36), self_233469, 'f3')
        # Calling f3(args, kwargs) (line 250)
        f3_call_result_233473 = invoke(stypy.reporting.localization.Localization(__file__, 250, 36), f3_233470, *[xopt_233471], **kwargs_233472)
        
        # Processing the call keyword arguments (line 250)
        float_233474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 56), 'float')
        keyword_233475 = float_233474
        str_233476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 252)
        rc_233477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 55), 'rc', False)
        # Getting the type of 'optimize' (line 252)
        optimize_233478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 252)
        tnc_233479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 32), optimize_233478, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 252)
        RCSTRINGS_233480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 32), tnc_233479, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___233481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 32), RCSTRINGS_233480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_233482 = invoke(stypy.reporting.localization.Localization(__file__, 252, 32), getitem___233481, rc_233477)
        
        # Applying the binary operator '+' (line 251)
        result_add_233483 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 32), '+', str_233476, subscript_call_result_233482)
        
        keyword_233484 = result_add_233483
        kwargs_233485 = {'err_msg': keyword_233484, 'atol': keyword_233475}
        # Getting the type of 'assert_allclose' (line 250)
        assert_allclose_233463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 250)
        assert_allclose_call_result_233486 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), assert_allclose_233463, *[f3_call_result_233468, f3_call_result_233473], **kwargs_233485)
        
        
        # ################# End of 'test_tnc3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc3' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_233487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233487)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc3'
        return stypy_return_type_233487


    @norecursion
    def test_tnc4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc4'
        module_type_store = module_type_store.open_function_context('test_tnc4', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc4')
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc4.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc4', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc4', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc4(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 255):
        
        # Assigning a Attribute to a Name (line 255):
        # Getting the type of 'self' (line 255)
        self_233488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'self')
        # Obtaining the member 'fg4' of a type (line 255)
        fg4_233489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 24), self_233488, 'fg4')
        # Assigning a type to the variable 'tuple_assignment_231722' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'tuple_assignment_231722', fg4_233489)
        
        # Assigning a List to a Name (line 255):
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_233490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        float_233491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 34), list_233490, float_233491)
        # Adding element type (line 255)
        float_233492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 34), list_233490, float_233492)
        
        # Assigning a type to the variable 'tuple_assignment_231723' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'tuple_assignment_231723', list_233490)
        
        # Assigning a List to a Name (line 255):
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_233493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        
        # Obtaining an instance of the builtin type 'tuple' (line 255)
        tuple_233494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 255)
        # Adding element type (line 255)
        int_233495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 52), tuple_233494, int_233495)
        # Adding element type (line 255)
        # Getting the type of 'None' (line 255)
        None_233496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 55), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 52), tuple_233494, None_233496)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 50), list_233493, tuple_233494)
        # Adding element type (line 255)
        
        # Obtaining an instance of the builtin type 'tuple' (line 255)
        tuple_233497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 255)
        # Adding element type (line 255)
        int_233498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 63), tuple_233497, int_233498)
        # Adding element type (line 255)
        # Getting the type of 'None' (line 255)
        None_233499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 66), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 63), tuple_233497, None_233499)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 50), list_233493, tuple_233497)
        
        # Assigning a type to the variable 'tuple_assignment_231724' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'tuple_assignment_231724', list_233493)
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'tuple_assignment_231722' (line 255)
        tuple_assignment_231722_233500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'tuple_assignment_231722')
        # Assigning a type to the variable 'fg' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'fg', tuple_assignment_231722_233500)
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'tuple_assignment_231723' (line 255)
        tuple_assignment_231723_233501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'tuple_assignment_231723')
        # Assigning a type to the variable 'x' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'x', tuple_assignment_231723_233501)
        
        # Assigning a Name to a Name (line 255):
        # Getting the type of 'tuple_assignment_231724' (line 255)
        tuple_assignment_231724_233502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'tuple_assignment_231724')
        # Assigning a type to the variable 'bounds' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'bounds', tuple_assignment_231724_233502)
        
        # Assigning a List to a Name (line 256):
        
        # Assigning a List to a Name (line 256):
        
        # Obtaining an instance of the builtin type 'list' (line 256)
        list_233503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 256)
        # Adding element type (line 256)
        int_233504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 15), list_233503, int_233504)
        # Adding element type (line 256)
        int_233505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 15), list_233503, int_233505)
        
        # Assigning a type to the variable 'xopt' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'xopt', list_233503)
        
        # Assigning a Call to a Tuple (line 258):
        
        # Assigning a Subscript to a Name (line 258):
        
        # Obtaining the type of the subscript
        int_233506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 8), 'int')
        
        # Call to fmin_tnc(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'fg' (line 258)
        fg_233509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 38), 'fg', False)
        # Getting the type of 'x' (line 258)
        x_233510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 42), 'x', False)
        # Processing the call keyword arguments (line 258)
        # Getting the type of 'bounds' (line 258)
        bounds_233511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 52), 'bounds', False)
        keyword_233512 = bounds_233511
        # Getting the type of 'optimize' (line 259)
        optimize_233513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 259)
        tnc_233514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 47), optimize_233513, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 259)
        MSG_NONE_233515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 47), tnc_233514, 'MSG_NONE')
        keyword_233516 = MSG_NONE_233515
        int_233517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 45), 'int')
        keyword_233518 = int_233517
        kwargs_233519 = {'messages': keyword_233516, 'bounds': keyword_233512, 'maxfun': keyword_233518}
        # Getting the type of 'optimize' (line 258)
        optimize_233507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 258)
        fmin_tnc_233508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), optimize_233507, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 258)
        fmin_tnc_call_result_233520 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), fmin_tnc_233508, *[fg_233509, x_233510], **kwargs_233519)
        
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___233521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), fmin_tnc_call_result_233520, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_233522 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), getitem___233521, int_233506)
        
        # Assigning a type to the variable 'tuple_var_assignment_231725' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'tuple_var_assignment_231725', subscript_call_result_233522)
        
        # Assigning a Subscript to a Name (line 258):
        
        # Obtaining the type of the subscript
        int_233523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 8), 'int')
        
        # Call to fmin_tnc(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'fg' (line 258)
        fg_233526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 38), 'fg', False)
        # Getting the type of 'x' (line 258)
        x_233527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 42), 'x', False)
        # Processing the call keyword arguments (line 258)
        # Getting the type of 'bounds' (line 258)
        bounds_233528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 52), 'bounds', False)
        keyword_233529 = bounds_233528
        # Getting the type of 'optimize' (line 259)
        optimize_233530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 259)
        tnc_233531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 47), optimize_233530, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 259)
        MSG_NONE_233532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 47), tnc_233531, 'MSG_NONE')
        keyword_233533 = MSG_NONE_233532
        int_233534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 45), 'int')
        keyword_233535 = int_233534
        kwargs_233536 = {'messages': keyword_233533, 'bounds': keyword_233529, 'maxfun': keyword_233535}
        # Getting the type of 'optimize' (line 258)
        optimize_233524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 258)
        fmin_tnc_233525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), optimize_233524, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 258)
        fmin_tnc_call_result_233537 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), fmin_tnc_233525, *[fg_233526, x_233527], **kwargs_233536)
        
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___233538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), fmin_tnc_call_result_233537, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_233539 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), getitem___233538, int_233523)
        
        # Assigning a type to the variable 'tuple_var_assignment_231726' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'tuple_var_assignment_231726', subscript_call_result_233539)
        
        # Assigning a Subscript to a Name (line 258):
        
        # Obtaining the type of the subscript
        int_233540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 8), 'int')
        
        # Call to fmin_tnc(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'fg' (line 258)
        fg_233543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 38), 'fg', False)
        # Getting the type of 'x' (line 258)
        x_233544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 42), 'x', False)
        # Processing the call keyword arguments (line 258)
        # Getting the type of 'bounds' (line 258)
        bounds_233545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 52), 'bounds', False)
        keyword_233546 = bounds_233545
        # Getting the type of 'optimize' (line 259)
        optimize_233547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 259)
        tnc_233548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 47), optimize_233547, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 259)
        MSG_NONE_233549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 47), tnc_233548, 'MSG_NONE')
        keyword_233550 = MSG_NONE_233549
        int_233551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 45), 'int')
        keyword_233552 = int_233551
        kwargs_233553 = {'messages': keyword_233550, 'bounds': keyword_233546, 'maxfun': keyword_233552}
        # Getting the type of 'optimize' (line 258)
        optimize_233541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 258)
        fmin_tnc_233542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), optimize_233541, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 258)
        fmin_tnc_call_result_233554 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), fmin_tnc_233542, *[fg_233543, x_233544], **kwargs_233553)
        
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___233555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), fmin_tnc_call_result_233554, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_233556 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), getitem___233555, int_233540)
        
        # Assigning a type to the variable 'tuple_var_assignment_231727' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'tuple_var_assignment_231727', subscript_call_result_233556)
        
        # Assigning a Name to a Name (line 258):
        # Getting the type of 'tuple_var_assignment_231725' (line 258)
        tuple_var_assignment_231725_233557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'tuple_var_assignment_231725')
        # Assigning a type to the variable 'x' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'x', tuple_var_assignment_231725_233557)
        
        # Assigning a Name to a Name (line 258):
        # Getting the type of 'tuple_var_assignment_231726' (line 258)
        tuple_var_assignment_231726_233558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'tuple_var_assignment_231726')
        # Assigning a type to the variable 'nf' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'nf', tuple_var_assignment_231726_233558)
        
        # Assigning a Name to a Name (line 258):
        # Getting the type of 'tuple_var_assignment_231727' (line 258)
        tuple_var_assignment_231727_233559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'tuple_var_assignment_231727')
        # Assigning a type to the variable 'rc' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'rc', tuple_var_assignment_231727_233559)
        
        # Call to assert_allclose(...): (line 262)
        # Processing the call arguments (line 262)
        
        # Call to f4(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'x' (line 262)
        x_233563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 32), 'x', False)
        # Processing the call keyword arguments (line 262)
        kwargs_233564 = {}
        # Getting the type of 'self' (line 262)
        self_233561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'self', False)
        # Obtaining the member 'f4' of a type (line 262)
        f4_233562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 24), self_233561, 'f4')
        # Calling f4(args, kwargs) (line 262)
        f4_call_result_233565 = invoke(stypy.reporting.localization.Localization(__file__, 262, 24), f4_233562, *[x_233563], **kwargs_233564)
        
        
        # Call to f4(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'xopt' (line 262)
        xopt_233568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 44), 'xopt', False)
        # Processing the call keyword arguments (line 262)
        kwargs_233569 = {}
        # Getting the type of 'self' (line 262)
        self_233566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 36), 'self', False)
        # Obtaining the member 'f4' of a type (line 262)
        f4_233567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 36), self_233566, 'f4')
        # Calling f4(args, kwargs) (line 262)
        f4_call_result_233570 = invoke(stypy.reporting.localization.Localization(__file__, 262, 36), f4_233567, *[xopt_233568], **kwargs_233569)
        
        # Processing the call keyword arguments (line 262)
        float_233571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 56), 'float')
        keyword_233572 = float_233571
        str_233573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 264)
        rc_233574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 55), 'rc', False)
        # Getting the type of 'optimize' (line 264)
        optimize_233575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 264)
        tnc_233576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 32), optimize_233575, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 264)
        RCSTRINGS_233577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 32), tnc_233576, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___233578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 32), RCSTRINGS_233577, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_233579 = invoke(stypy.reporting.localization.Localization(__file__, 264, 32), getitem___233578, rc_233574)
        
        # Applying the binary operator '+' (line 263)
        result_add_233580 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 32), '+', str_233573, subscript_call_result_233579)
        
        keyword_233581 = result_add_233580
        kwargs_233582 = {'err_msg': keyword_233581, 'atol': keyword_233572}
        # Getting the type of 'assert_allclose' (line 262)
        assert_allclose_233560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 262)
        assert_allclose_call_result_233583 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assert_allclose_233560, *[f4_call_result_233565, f4_call_result_233570], **kwargs_233582)
        
        
        # ################# End of 'test_tnc4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc4' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_233584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc4'
        return stypy_return_type_233584


    @norecursion
    def test_tnc5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc5'
        module_type_store = module_type_store.open_function_context('test_tnc5', 266, 4, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc5')
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc5(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 267):
        
        # Assigning a Attribute to a Name (line 267):
        # Getting the type of 'self' (line 267)
        self_233585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'self')
        # Obtaining the member 'fg5' of a type (line 267)
        fg5_233586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 24), self_233585, 'fg5')
        # Assigning a type to the variable 'tuple_assignment_231728' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_231728', fg5_233586)
        
        # Assigning a List to a Name (line 267):
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_233587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_233588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 34), list_233587, int_233588)
        # Adding element type (line 267)
        int_233589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 34), list_233587, int_233589)
        
        # Assigning a type to the variable 'tuple_assignment_231729' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_231729', list_233587)
        
        # Assigning a List to a Name (line 267):
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_233590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'tuple' (line 267)
        tuple_233591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 267)
        # Adding element type (line 267)
        float_233592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 44), tuple_233591, float_233592)
        # Adding element type (line 267)
        int_233593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 44), tuple_233591, int_233593)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 42), list_233590, tuple_233591)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'tuple' (line 267)
        tuple_233594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 267)
        # Adding element type (line 267)
        int_233595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 54), tuple_233594, int_233595)
        # Adding element type (line 267)
        int_233596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 54), tuple_233594, int_233596)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 42), list_233590, tuple_233594)
        
        # Assigning a type to the variable 'tuple_assignment_231730' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_231730', list_233590)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'tuple_assignment_231728' (line 267)
        tuple_assignment_231728_233597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_231728')
        # Assigning a type to the variable 'fg' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'fg', tuple_assignment_231728_233597)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'tuple_assignment_231729' (line 267)
        tuple_assignment_231729_233598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_231729')
        # Assigning a type to the variable 'x' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'x', tuple_assignment_231729_233598)
        
        # Assigning a Name to a Name (line 267):
        # Getting the type of 'tuple_assignment_231730' (line 267)
        tuple_assignment_231730_233599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'tuple_assignment_231730')
        # Assigning a type to the variable 'bounds' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'bounds', tuple_assignment_231730_233599)
        
        # Assigning a List to a Name (line 268):
        
        # Assigning a List to a Name (line 268):
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_233600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        # Adding element type (line 268)
        float_233601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 15), list_233600, float_233601)
        # Adding element type (line 268)
        float_233602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 15), list_233600, float_233602)
        
        # Assigning a type to the variable 'xopt' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'xopt', list_233600)
        
        # Assigning a Call to a Tuple (line 270):
        
        # Assigning a Subscript to a Name (line 270):
        
        # Obtaining the type of the subscript
        int_233603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 8), 'int')
        
        # Call to fmin_tnc(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'fg' (line 270)
        fg_233606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'fg', False)
        # Getting the type of 'x' (line 270)
        x_233607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), 'x', False)
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'bounds' (line 270)
        bounds_233608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 52), 'bounds', False)
        keyword_233609 = bounds_233608
        # Getting the type of 'optimize' (line 271)
        optimize_233610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 271)
        tnc_233611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 47), optimize_233610, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 271)
        MSG_NONE_233612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 47), tnc_233611, 'MSG_NONE')
        keyword_233613 = MSG_NONE_233612
        int_233614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 45), 'int')
        keyword_233615 = int_233614
        kwargs_233616 = {'messages': keyword_233613, 'bounds': keyword_233609, 'maxfun': keyword_233615}
        # Getting the type of 'optimize' (line 270)
        optimize_233604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 270)
        fmin_tnc_233605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), optimize_233604, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 270)
        fmin_tnc_call_result_233617 = invoke(stypy.reporting.localization.Localization(__file__, 270, 20), fmin_tnc_233605, *[fg_233606, x_233607], **kwargs_233616)
        
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___233618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), fmin_tnc_call_result_233617, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_233619 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), getitem___233618, int_233603)
        
        # Assigning a type to the variable 'tuple_var_assignment_231731' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_231731', subscript_call_result_233619)
        
        # Assigning a Subscript to a Name (line 270):
        
        # Obtaining the type of the subscript
        int_233620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 8), 'int')
        
        # Call to fmin_tnc(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'fg' (line 270)
        fg_233623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'fg', False)
        # Getting the type of 'x' (line 270)
        x_233624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), 'x', False)
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'bounds' (line 270)
        bounds_233625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 52), 'bounds', False)
        keyword_233626 = bounds_233625
        # Getting the type of 'optimize' (line 271)
        optimize_233627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 271)
        tnc_233628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 47), optimize_233627, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 271)
        MSG_NONE_233629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 47), tnc_233628, 'MSG_NONE')
        keyword_233630 = MSG_NONE_233629
        int_233631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 45), 'int')
        keyword_233632 = int_233631
        kwargs_233633 = {'messages': keyword_233630, 'bounds': keyword_233626, 'maxfun': keyword_233632}
        # Getting the type of 'optimize' (line 270)
        optimize_233621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 270)
        fmin_tnc_233622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), optimize_233621, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 270)
        fmin_tnc_call_result_233634 = invoke(stypy.reporting.localization.Localization(__file__, 270, 20), fmin_tnc_233622, *[fg_233623, x_233624], **kwargs_233633)
        
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___233635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), fmin_tnc_call_result_233634, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_233636 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), getitem___233635, int_233620)
        
        # Assigning a type to the variable 'tuple_var_assignment_231732' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_231732', subscript_call_result_233636)
        
        # Assigning a Subscript to a Name (line 270):
        
        # Obtaining the type of the subscript
        int_233637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 8), 'int')
        
        # Call to fmin_tnc(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'fg' (line 270)
        fg_233640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'fg', False)
        # Getting the type of 'x' (line 270)
        x_233641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 42), 'x', False)
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'bounds' (line 270)
        bounds_233642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 52), 'bounds', False)
        keyword_233643 = bounds_233642
        # Getting the type of 'optimize' (line 271)
        optimize_233644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 271)
        tnc_233645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 47), optimize_233644, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 271)
        MSG_NONE_233646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 47), tnc_233645, 'MSG_NONE')
        keyword_233647 = MSG_NONE_233646
        int_233648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 45), 'int')
        keyword_233649 = int_233648
        kwargs_233650 = {'messages': keyword_233647, 'bounds': keyword_233643, 'maxfun': keyword_233649}
        # Getting the type of 'optimize' (line 270)
        optimize_233638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 270)
        fmin_tnc_233639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), optimize_233638, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 270)
        fmin_tnc_call_result_233651 = invoke(stypy.reporting.localization.Localization(__file__, 270, 20), fmin_tnc_233639, *[fg_233640, x_233641], **kwargs_233650)
        
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___233652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), fmin_tnc_call_result_233651, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_233653 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), getitem___233652, int_233637)
        
        # Assigning a type to the variable 'tuple_var_assignment_231733' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_231733', subscript_call_result_233653)
        
        # Assigning a Name to a Name (line 270):
        # Getting the type of 'tuple_var_assignment_231731' (line 270)
        tuple_var_assignment_231731_233654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_231731')
        # Assigning a type to the variable 'x' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'x', tuple_var_assignment_231731_233654)
        
        # Assigning a Name to a Name (line 270):
        # Getting the type of 'tuple_var_assignment_231732' (line 270)
        tuple_var_assignment_231732_233655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_231732')
        # Assigning a type to the variable 'nf' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'nf', tuple_var_assignment_231732_233655)
        
        # Assigning a Name to a Name (line 270):
        # Getting the type of 'tuple_var_assignment_231733' (line 270)
        tuple_var_assignment_231733_233656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_231733')
        # Assigning a type to the variable 'rc' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'rc', tuple_var_assignment_231733_233656)
        
        # Call to assert_allclose(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Call to f5(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'x' (line 274)
        x_233660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 32), 'x', False)
        # Processing the call keyword arguments (line 274)
        kwargs_233661 = {}
        # Getting the type of 'self' (line 274)
        self_233658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'self', False)
        # Obtaining the member 'f5' of a type (line 274)
        f5_233659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), self_233658, 'f5')
        # Calling f5(args, kwargs) (line 274)
        f5_call_result_233662 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), f5_233659, *[x_233660], **kwargs_233661)
        
        
        # Call to f5(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'xopt' (line 274)
        xopt_233665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 44), 'xopt', False)
        # Processing the call keyword arguments (line 274)
        kwargs_233666 = {}
        # Getting the type of 'self' (line 274)
        self_233663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'self', False)
        # Obtaining the member 'f5' of a type (line 274)
        f5_233664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 36), self_233663, 'f5')
        # Calling f5(args, kwargs) (line 274)
        f5_call_result_233667 = invoke(stypy.reporting.localization.Localization(__file__, 274, 36), f5_233664, *[xopt_233665], **kwargs_233666)
        
        # Processing the call keyword arguments (line 274)
        float_233668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 56), 'float')
        keyword_233669 = float_233668
        str_233670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 276)
        rc_233671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 55), 'rc', False)
        # Getting the type of 'optimize' (line 276)
        optimize_233672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 276)
        tnc_233673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 32), optimize_233672, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 276)
        RCSTRINGS_233674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 32), tnc_233673, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___233675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 32), RCSTRINGS_233674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_233676 = invoke(stypy.reporting.localization.Localization(__file__, 276, 32), getitem___233675, rc_233671)
        
        # Applying the binary operator '+' (line 275)
        result_add_233677 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 32), '+', str_233670, subscript_call_result_233676)
        
        keyword_233678 = result_add_233677
        kwargs_233679 = {'err_msg': keyword_233678, 'atol': keyword_233669}
        # Getting the type of 'assert_allclose' (line 274)
        assert_allclose_233657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 274)
        assert_allclose_call_result_233680 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), assert_allclose_233657, *[f5_call_result_233662, f5_call_result_233667], **kwargs_233679)
        
        
        # ################# End of 'test_tnc5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc5' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_233681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233681)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc5'
        return stypy_return_type_233681


    @norecursion
    def test_tnc38(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc38'
        module_type_store = module_type_store.open_function_context('test_tnc38', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc38')
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc38.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc38', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc38', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc38(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 279):
        
        # Assigning a Attribute to a Name (line 279):
        # Getting the type of 'self' (line 279)
        self_233682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'self')
        # Obtaining the member 'fg38' of a type (line 279)
        fg38_233683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), self_233682, 'fg38')
        # Assigning a type to the variable 'tuple_assignment_231734' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_assignment_231734', fg38_233683)
        
        # Assigning a Call to a Name (line 279):
        
        # Call to array(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_233686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        int_233687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 44), list_233686, int_233687)
        # Adding element type (line 279)
        int_233688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 44), list_233686, int_233688)
        # Adding element type (line 279)
        int_233689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 44), list_233686, int_233689)
        # Adding element type (line 279)
        int_233690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 44), list_233686, int_233690)
        
        # Processing the call keyword arguments (line 279)
        kwargs_233691 = {}
        # Getting the type of 'np' (line 279)
        np_233684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 279)
        array_233685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 35), np_233684, 'array')
        # Calling array(args, kwargs) (line 279)
        array_call_result_233692 = invoke(stypy.reporting.localization.Localization(__file__, 279, 35), array_233685, *[list_233686], **kwargs_233691)
        
        # Assigning a type to the variable 'tuple_assignment_231735' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_assignment_231735', array_call_result_233692)
        
        # Assigning a BinOp to a Name (line 279):
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_233693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        
        # Obtaining an instance of the builtin type 'tuple' (line 279)
        tuple_233694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 65), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 279)
        # Adding element type (line 279)
        int_233695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 65), tuple_233694, int_233695)
        # Adding element type (line 279)
        int_233696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 65), tuple_233694, int_233696)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 63), list_233693, tuple_233694)
        
        int_233697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 75), 'int')
        # Applying the binary operator '*' (line 279)
        result_mul_233698 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 63), '*', list_233693, int_233697)
        
        # Assigning a type to the variable 'tuple_assignment_231736' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_assignment_231736', result_mul_233698)
        
        # Assigning a Name to a Name (line 279):
        # Getting the type of 'tuple_assignment_231734' (line 279)
        tuple_assignment_231734_233699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_assignment_231734')
        # Assigning a type to the variable 'fg' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'fg', tuple_assignment_231734_233699)
        
        # Assigning a Name to a Name (line 279):
        # Getting the type of 'tuple_assignment_231735' (line 279)
        tuple_assignment_231735_233700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_assignment_231735')
        # Assigning a type to the variable 'x' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'x', tuple_assignment_231735_233700)
        
        # Assigning a Name to a Name (line 279):
        # Getting the type of 'tuple_assignment_231736' (line 279)
        tuple_assignment_231736_233701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_assignment_231736')
        # Assigning a type to the variable 'bounds' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'bounds', tuple_assignment_231736_233701)
        
        # Assigning a BinOp to a Name (line 280):
        
        # Assigning a BinOp to a Name (line 280):
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_233702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        int_233703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 15), list_233702, int_233703)
        
        int_233704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 19), 'int')
        # Applying the binary operator '*' (line 280)
        result_mul_233705 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), '*', list_233702, int_233704)
        
        # Assigning a type to the variable 'xopt' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'xopt', result_mul_233705)
        
        # Assigning a Call to a Tuple (line 282):
        
        # Assigning a Subscript to a Name (line 282):
        
        # Obtaining the type of the subscript
        int_233706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 8), 'int')
        
        # Call to fmin_tnc(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'fg' (line 282)
        fg_233709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 38), 'fg', False)
        # Getting the type of 'x' (line 282)
        x_233710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 42), 'x', False)
        # Processing the call keyword arguments (line 282)
        # Getting the type of 'bounds' (line 282)
        bounds_233711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 52), 'bounds', False)
        keyword_233712 = bounds_233711
        # Getting the type of 'optimize' (line 283)
        optimize_233713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 283)
        tnc_233714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 47), optimize_233713, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 283)
        MSG_NONE_233715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 47), tnc_233714, 'MSG_NONE')
        keyword_233716 = MSG_NONE_233715
        int_233717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 45), 'int')
        keyword_233718 = int_233717
        kwargs_233719 = {'messages': keyword_233716, 'bounds': keyword_233712, 'maxfun': keyword_233718}
        # Getting the type of 'optimize' (line 282)
        optimize_233707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 282)
        fmin_tnc_233708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 20), optimize_233707, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 282)
        fmin_tnc_call_result_233720 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), fmin_tnc_233708, *[fg_233709, x_233710], **kwargs_233719)
        
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___233721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), fmin_tnc_call_result_233720, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_233722 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), getitem___233721, int_233706)
        
        # Assigning a type to the variable 'tuple_var_assignment_231737' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_231737', subscript_call_result_233722)
        
        # Assigning a Subscript to a Name (line 282):
        
        # Obtaining the type of the subscript
        int_233723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 8), 'int')
        
        # Call to fmin_tnc(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'fg' (line 282)
        fg_233726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 38), 'fg', False)
        # Getting the type of 'x' (line 282)
        x_233727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 42), 'x', False)
        # Processing the call keyword arguments (line 282)
        # Getting the type of 'bounds' (line 282)
        bounds_233728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 52), 'bounds', False)
        keyword_233729 = bounds_233728
        # Getting the type of 'optimize' (line 283)
        optimize_233730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 283)
        tnc_233731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 47), optimize_233730, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 283)
        MSG_NONE_233732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 47), tnc_233731, 'MSG_NONE')
        keyword_233733 = MSG_NONE_233732
        int_233734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 45), 'int')
        keyword_233735 = int_233734
        kwargs_233736 = {'messages': keyword_233733, 'bounds': keyword_233729, 'maxfun': keyword_233735}
        # Getting the type of 'optimize' (line 282)
        optimize_233724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 282)
        fmin_tnc_233725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 20), optimize_233724, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 282)
        fmin_tnc_call_result_233737 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), fmin_tnc_233725, *[fg_233726, x_233727], **kwargs_233736)
        
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___233738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), fmin_tnc_call_result_233737, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_233739 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), getitem___233738, int_233723)
        
        # Assigning a type to the variable 'tuple_var_assignment_231738' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_231738', subscript_call_result_233739)
        
        # Assigning a Subscript to a Name (line 282):
        
        # Obtaining the type of the subscript
        int_233740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 8), 'int')
        
        # Call to fmin_tnc(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'fg' (line 282)
        fg_233743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 38), 'fg', False)
        # Getting the type of 'x' (line 282)
        x_233744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 42), 'x', False)
        # Processing the call keyword arguments (line 282)
        # Getting the type of 'bounds' (line 282)
        bounds_233745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 52), 'bounds', False)
        keyword_233746 = bounds_233745
        # Getting the type of 'optimize' (line 283)
        optimize_233747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 283)
        tnc_233748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 47), optimize_233747, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 283)
        MSG_NONE_233749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 47), tnc_233748, 'MSG_NONE')
        keyword_233750 = MSG_NONE_233749
        int_233751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 45), 'int')
        keyword_233752 = int_233751
        kwargs_233753 = {'messages': keyword_233750, 'bounds': keyword_233746, 'maxfun': keyword_233752}
        # Getting the type of 'optimize' (line 282)
        optimize_233741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 282)
        fmin_tnc_233742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 20), optimize_233741, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 282)
        fmin_tnc_call_result_233754 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), fmin_tnc_233742, *[fg_233743, x_233744], **kwargs_233753)
        
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___233755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), fmin_tnc_call_result_233754, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_233756 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), getitem___233755, int_233740)
        
        # Assigning a type to the variable 'tuple_var_assignment_231739' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_231739', subscript_call_result_233756)
        
        # Assigning a Name to a Name (line 282):
        # Getting the type of 'tuple_var_assignment_231737' (line 282)
        tuple_var_assignment_231737_233757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_231737')
        # Assigning a type to the variable 'x' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'x', tuple_var_assignment_231737_233757)
        
        # Assigning a Name to a Name (line 282):
        # Getting the type of 'tuple_var_assignment_231738' (line 282)
        tuple_var_assignment_231738_233758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_231738')
        # Assigning a type to the variable 'nf' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'nf', tuple_var_assignment_231738_233758)
        
        # Assigning a Name to a Name (line 282):
        # Getting the type of 'tuple_var_assignment_231739' (line 282)
        tuple_var_assignment_231739_233759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_231739')
        # Assigning a type to the variable 'rc' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'rc', tuple_var_assignment_231739_233759)
        
        # Call to assert_allclose(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Call to f38(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'x' (line 286)
        x_233763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 33), 'x', False)
        # Processing the call keyword arguments (line 286)
        kwargs_233764 = {}
        # Getting the type of 'self' (line 286)
        self_233761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 24), 'self', False)
        # Obtaining the member 'f38' of a type (line 286)
        f38_233762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 24), self_233761, 'f38')
        # Calling f38(args, kwargs) (line 286)
        f38_call_result_233765 = invoke(stypy.reporting.localization.Localization(__file__, 286, 24), f38_233762, *[x_233763], **kwargs_233764)
        
        
        # Call to f38(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'xopt' (line 286)
        xopt_233768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 46), 'xopt', False)
        # Processing the call keyword arguments (line 286)
        kwargs_233769 = {}
        # Getting the type of 'self' (line 286)
        self_233766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), 'self', False)
        # Obtaining the member 'f38' of a type (line 286)
        f38_233767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 37), self_233766, 'f38')
        # Calling f38(args, kwargs) (line 286)
        f38_call_result_233770 = invoke(stypy.reporting.localization.Localization(__file__, 286, 37), f38_233767, *[xopt_233768], **kwargs_233769)
        
        # Processing the call keyword arguments (line 286)
        float_233771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 58), 'float')
        keyword_233772 = float_233771
        str_233773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 288)
        rc_233774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 55), 'rc', False)
        # Getting the type of 'optimize' (line 288)
        optimize_233775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 288)
        tnc_233776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 32), optimize_233775, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 288)
        RCSTRINGS_233777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 32), tnc_233776, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___233778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 32), RCSTRINGS_233777, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 288)
        subscript_call_result_233779 = invoke(stypy.reporting.localization.Localization(__file__, 288, 32), getitem___233778, rc_233774)
        
        # Applying the binary operator '+' (line 287)
        result_add_233780 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 32), '+', str_233773, subscript_call_result_233779)
        
        keyword_233781 = result_add_233780
        kwargs_233782 = {'err_msg': keyword_233781, 'atol': keyword_233772}
        # Getting the type of 'assert_allclose' (line 286)
        assert_allclose_233760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 286)
        assert_allclose_call_result_233783 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), assert_allclose_233760, *[f38_call_result_233765, f38_call_result_233770], **kwargs_233782)
        
        
        # ################# End of 'test_tnc38(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc38' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_233784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233784)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc38'
        return stypy_return_type_233784


    @norecursion
    def test_tnc45(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tnc45'
        module_type_store = module_type_store.open_function_context('test_tnc45', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_localization', localization)
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_function_name', 'TestTnc.test_tnc45')
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_param_names_list', [])
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTnc.test_tnc45.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.test_tnc45', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tnc45', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tnc45(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 291):
        
        # Assigning a Attribute to a Name (line 291):
        # Getting the type of 'self' (line 291)
        self_233785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'self')
        # Obtaining the member 'fg45' of a type (line 291)
        fg45_233786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 24), self_233785, 'fg45')
        # Assigning a type to the variable 'tuple_assignment_231740' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_assignment_231740', fg45_233786)
        
        # Assigning a BinOp to a Name (line 291):
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_233787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        int_233788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 35), list_233787, int_233788)
        
        int_233789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 41), 'int')
        # Applying the binary operator '*' (line 291)
        result_mul_233790 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 35), '*', list_233787, int_233789)
        
        # Assigning a type to the variable 'tuple_assignment_231741' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_assignment_231741', result_mul_233790)
        
        # Assigning a List to a Name (line 291):
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_233791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_233792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        int_233793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 46), tuple_233792, int_233793)
        # Adding element type (line 291)
        int_233794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 46), tuple_233792, int_233794)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 44), list_233791, tuple_233792)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_233795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        int_233796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 54), tuple_233795, int_233796)
        # Adding element type (line 291)
        int_233797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 54), tuple_233795, int_233797)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 44), list_233791, tuple_233795)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_233798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 62), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        int_233799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 62), tuple_233798, int_233799)
        # Adding element type (line 291)
        int_233800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 62), tuple_233798, int_233800)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 44), list_233791, tuple_233798)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 292)
        tuple_233801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 292)
        # Adding element type (line 292)
        int_233802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 46), tuple_233801, int_233802)
        # Adding element type (line 292)
        int_233803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 46), tuple_233801, int_233803)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 44), list_233791, tuple_233801)
        # Adding element type (line 291)
        
        # Obtaining an instance of the builtin type 'tuple' (line 292)
        tuple_233804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 292)
        # Adding element type (line 292)
        int_233805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 54), tuple_233804, int_233805)
        # Adding element type (line 292)
        int_233806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 54), tuple_233804, int_233806)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 44), list_233791, tuple_233804)
        
        # Assigning a type to the variable 'tuple_assignment_231742' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_assignment_231742', list_233791)
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'tuple_assignment_231740' (line 291)
        tuple_assignment_231740_233807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_assignment_231740')
        # Assigning a type to the variable 'fg' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'fg', tuple_assignment_231740_233807)
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'tuple_assignment_231741' (line 291)
        tuple_assignment_231741_233808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_assignment_231741')
        # Assigning a type to the variable 'x' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'x', tuple_assignment_231741_233808)
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'tuple_assignment_231742' (line 291)
        tuple_assignment_231742_233809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_assignment_231742')
        # Assigning a type to the variable 'bounds' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'bounds', tuple_assignment_231742_233809)
        
        # Assigning a List to a Name (line 293):
        
        # Assigning a List to a Name (line 293):
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_233810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        int_233811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_233810, int_233811)
        # Adding element type (line 293)
        int_233812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_233810, int_233812)
        # Adding element type (line 293)
        int_233813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_233810, int_233813)
        # Adding element type (line 293)
        int_233814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_233810, int_233814)
        # Adding element type (line 293)
        int_233815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_233810, int_233815)
        
        # Assigning a type to the variable 'xopt' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'xopt', list_233810)
        
        # Assigning a Call to a Tuple (line 295):
        
        # Assigning a Subscript to a Name (line 295):
        
        # Obtaining the type of the subscript
        int_233816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 8), 'int')
        
        # Call to fmin_tnc(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'fg' (line 295)
        fg_233819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'fg', False)
        # Getting the type of 'x' (line 295)
        x_233820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 42), 'x', False)
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'bounds' (line 295)
        bounds_233821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'bounds', False)
        keyword_233822 = bounds_233821
        # Getting the type of 'optimize' (line 296)
        optimize_233823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 296)
        tnc_233824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 47), optimize_233823, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 296)
        MSG_NONE_233825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 47), tnc_233824, 'MSG_NONE')
        keyword_233826 = MSG_NONE_233825
        int_233827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 45), 'int')
        keyword_233828 = int_233827
        kwargs_233829 = {'messages': keyword_233826, 'bounds': keyword_233822, 'maxfun': keyword_233828}
        # Getting the type of 'optimize' (line 295)
        optimize_233817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 295)
        fmin_tnc_233818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 20), optimize_233817, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 295)
        fmin_tnc_call_result_233830 = invoke(stypy.reporting.localization.Localization(__file__, 295, 20), fmin_tnc_233818, *[fg_233819, x_233820], **kwargs_233829)
        
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___233831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), fmin_tnc_call_result_233830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_233832 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), getitem___233831, int_233816)
        
        # Assigning a type to the variable 'tuple_var_assignment_231743' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'tuple_var_assignment_231743', subscript_call_result_233832)
        
        # Assigning a Subscript to a Name (line 295):
        
        # Obtaining the type of the subscript
        int_233833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 8), 'int')
        
        # Call to fmin_tnc(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'fg' (line 295)
        fg_233836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'fg', False)
        # Getting the type of 'x' (line 295)
        x_233837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 42), 'x', False)
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'bounds' (line 295)
        bounds_233838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'bounds', False)
        keyword_233839 = bounds_233838
        # Getting the type of 'optimize' (line 296)
        optimize_233840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 296)
        tnc_233841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 47), optimize_233840, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 296)
        MSG_NONE_233842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 47), tnc_233841, 'MSG_NONE')
        keyword_233843 = MSG_NONE_233842
        int_233844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 45), 'int')
        keyword_233845 = int_233844
        kwargs_233846 = {'messages': keyword_233843, 'bounds': keyword_233839, 'maxfun': keyword_233845}
        # Getting the type of 'optimize' (line 295)
        optimize_233834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 295)
        fmin_tnc_233835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 20), optimize_233834, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 295)
        fmin_tnc_call_result_233847 = invoke(stypy.reporting.localization.Localization(__file__, 295, 20), fmin_tnc_233835, *[fg_233836, x_233837], **kwargs_233846)
        
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___233848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), fmin_tnc_call_result_233847, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_233849 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), getitem___233848, int_233833)
        
        # Assigning a type to the variable 'tuple_var_assignment_231744' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'tuple_var_assignment_231744', subscript_call_result_233849)
        
        # Assigning a Subscript to a Name (line 295):
        
        # Obtaining the type of the subscript
        int_233850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 8), 'int')
        
        # Call to fmin_tnc(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'fg' (line 295)
        fg_233853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'fg', False)
        # Getting the type of 'x' (line 295)
        x_233854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 42), 'x', False)
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'bounds' (line 295)
        bounds_233855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'bounds', False)
        keyword_233856 = bounds_233855
        # Getting the type of 'optimize' (line 296)
        optimize_233857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 47), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 296)
        tnc_233858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 47), optimize_233857, 'tnc')
        # Obtaining the member 'MSG_NONE' of a type (line 296)
        MSG_NONE_233859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 47), tnc_233858, 'MSG_NONE')
        keyword_233860 = MSG_NONE_233859
        int_233861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 45), 'int')
        keyword_233862 = int_233861
        kwargs_233863 = {'messages': keyword_233860, 'bounds': keyword_233856, 'maxfun': keyword_233862}
        # Getting the type of 'optimize' (line 295)
        optimize_233851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'optimize', False)
        # Obtaining the member 'fmin_tnc' of a type (line 295)
        fmin_tnc_233852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 20), optimize_233851, 'fmin_tnc')
        # Calling fmin_tnc(args, kwargs) (line 295)
        fmin_tnc_call_result_233864 = invoke(stypy.reporting.localization.Localization(__file__, 295, 20), fmin_tnc_233852, *[fg_233853, x_233854], **kwargs_233863)
        
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___233865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), fmin_tnc_call_result_233864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_233866 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), getitem___233865, int_233850)
        
        # Assigning a type to the variable 'tuple_var_assignment_231745' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'tuple_var_assignment_231745', subscript_call_result_233866)
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'tuple_var_assignment_231743' (line 295)
        tuple_var_assignment_231743_233867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'tuple_var_assignment_231743')
        # Assigning a type to the variable 'x' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'x', tuple_var_assignment_231743_233867)
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'tuple_var_assignment_231744' (line 295)
        tuple_var_assignment_231744_233868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'tuple_var_assignment_231744')
        # Assigning a type to the variable 'nf' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'nf', tuple_var_assignment_231744_233868)
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'tuple_var_assignment_231745' (line 295)
        tuple_var_assignment_231745_233869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'tuple_var_assignment_231745')
        # Assigning a type to the variable 'rc' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'rc', tuple_var_assignment_231745_233869)
        
        # Call to assert_allclose(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Call to f45(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'x' (line 299)
        x_233873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 33), 'x', False)
        # Processing the call keyword arguments (line 299)
        kwargs_233874 = {}
        # Getting the type of 'self' (line 299)
        self_233871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'self', False)
        # Obtaining the member 'f45' of a type (line 299)
        f45_233872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 24), self_233871, 'f45')
        # Calling f45(args, kwargs) (line 299)
        f45_call_result_233875 = invoke(stypy.reporting.localization.Localization(__file__, 299, 24), f45_233872, *[x_233873], **kwargs_233874)
        
        
        # Call to f45(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'xopt' (line 299)
        xopt_233878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'xopt', False)
        # Processing the call keyword arguments (line 299)
        kwargs_233879 = {}
        # Getting the type of 'self' (line 299)
        self_233876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 37), 'self', False)
        # Obtaining the member 'f45' of a type (line 299)
        f45_233877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 37), self_233876, 'f45')
        # Calling f45(args, kwargs) (line 299)
        f45_call_result_233880 = invoke(stypy.reporting.localization.Localization(__file__, 299, 37), f45_233877, *[xopt_233878], **kwargs_233879)
        
        # Processing the call keyword arguments (line 299)
        float_233881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 58), 'float')
        keyword_233882 = float_233881
        str_233883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 32), 'str', 'TNC failed with status: ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rc' (line 301)
        rc_233884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 55), 'rc', False)
        # Getting the type of 'optimize' (line 301)
        optimize_233885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 32), 'optimize', False)
        # Obtaining the member 'tnc' of a type (line 301)
        tnc_233886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 32), optimize_233885, 'tnc')
        # Obtaining the member 'RCSTRINGS' of a type (line 301)
        RCSTRINGS_233887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 32), tnc_233886, 'RCSTRINGS')
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___233888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 32), RCSTRINGS_233887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_233889 = invoke(stypy.reporting.localization.Localization(__file__, 301, 32), getitem___233888, rc_233884)
        
        # Applying the binary operator '+' (line 300)
        result_add_233890 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 32), '+', str_233883, subscript_call_result_233889)
        
        keyword_233891 = result_add_233890
        kwargs_233892 = {'err_msg': keyword_233891, 'atol': keyword_233882}
        # Getting the type of 'assert_allclose' (line 299)
        assert_allclose_233870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 299)
        assert_allclose_call_result_233893 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), assert_allclose_233870, *[f45_call_result_233875, f45_call_result_233880], **kwargs_233892)
        
        
        # ################# End of 'test_tnc45(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tnc45' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_233894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233894)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tnc45'
        return stypy_return_type_233894


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTnc.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTnc' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestTnc', TestTnc)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

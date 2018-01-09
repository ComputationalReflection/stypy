
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_, assert_allclose
5: import scipy.special.orthogonal as orth
6: 
7: from scipy.special._testutils import FuncData
8: 
9: 
10: def test_eval_chebyt():
11:     n = np.arange(0, 10000, 7)
12:     x = 2*np.random.rand() - 1
13:     v1 = np.cos(n*np.arccos(x))
14:     v2 = orth.eval_chebyt(n, x)
15:     assert_(np.allclose(v1, v2, rtol=1e-15))
16: 
17: 
18: def test_eval_genlaguerre_restriction():
19:     # check it returns nan for alpha <= -1
20:     assert_(np.isnan(orth.eval_genlaguerre(0, -1, 0)))
21:     assert_(np.isnan(orth.eval_genlaguerre(0.1, -1, 0)))
22: 
23: 
24: def test_warnings():
25:     # ticket 1334
26:     olderr = np.seterr(all='raise')
27:     try:
28:         # these should raise no fp warnings
29:         orth.eval_legendre(1, 0)
30:         orth.eval_laguerre(1, 1)
31:         orth.eval_gegenbauer(1, 1, 0)
32:     finally:
33:         np.seterr(**olderr)
34: 
35: 
36: class TestPolys(object):
37:     '''
38:     Check that the eval_* functions agree with the constructed polynomials
39: 
40:     '''
41: 
42:     def check_poly(self, func, cls, param_ranges=[], x_range=[], nn=10,
43:                    nparam=10, nx=10, rtol=1e-8):
44:         np.random.seed(1234)
45: 
46:         dataset = []
47:         for n in np.arange(nn):
48:             params = [a + (b-a)*np.random.rand(nparam) for a,b in param_ranges]
49:             params = np.asarray(params).T
50:             if not param_ranges:
51:                 params = [0]
52:             for p in params:
53:                 if param_ranges:
54:                     p = (n,) + tuple(p)
55:                 else:
56:                     p = (n,)
57:                 x = x_range[0] + (x_range[1] - x_range[0])*np.random.rand(nx)
58:                 x[0] = x_range[0]  # always include domain start point
59:                 x[1] = x_range[1]  # always include domain end point
60:                 poly = np.poly1d(cls(*p).coef)
61:                 z = np.c_[np.tile(p, (nx,1)), x, poly(x)]
62:                 dataset.append(z)
63: 
64:         dataset = np.concatenate(dataset, axis=0)
65: 
66:         def polyfunc(*p):
67:             p = (p[0].astype(int),) + p[1:]
68:             return func(*p)
69: 
70:         olderr = np.seterr(all='raise')
71:         try:
72:             ds = FuncData(polyfunc, dataset, list(range(len(param_ranges)+2)), -1,
73:                           rtol=rtol)
74:             ds.check()
75:         finally:
76:             np.seterr(**olderr)
77: 
78:     def test_jacobi(self):
79:         self.check_poly(orth.eval_jacobi, orth.jacobi,
80:                    param_ranges=[(-0.99, 10), (-0.99, 10)], x_range=[-1, 1],
81:                    rtol=1e-5)
82: 
83:     def test_sh_jacobi(self):
84:         self.check_poly(orth.eval_sh_jacobi, orth.sh_jacobi,
85:                    param_ranges=[(1, 10), (0, 1)], x_range=[0, 1],
86:                    rtol=1e-5)
87: 
88:     def test_gegenbauer(self):
89:         self.check_poly(orth.eval_gegenbauer, orth.gegenbauer,
90:                    param_ranges=[(-0.499, 10)], x_range=[-1, 1],
91:                    rtol=1e-7)
92: 
93:     def test_chebyt(self):
94:         self.check_poly(orth.eval_chebyt, orth.chebyt,
95:                    param_ranges=[], x_range=[-1, 1])
96: 
97:     def test_chebyu(self):
98:         self.check_poly(orth.eval_chebyu, orth.chebyu,
99:                    param_ranges=[], x_range=[-1, 1])
100: 
101:     def test_chebys(self):
102:         self.check_poly(orth.eval_chebys, orth.chebys,
103:                    param_ranges=[], x_range=[-2, 2])
104: 
105:     def test_chebyc(self):
106:         self.check_poly(orth.eval_chebyc, orth.chebyc,
107:                    param_ranges=[], x_range=[-2, 2])
108: 
109:     def test_sh_chebyt(self):
110:         olderr = np.seterr(all='ignore')
111:         try:
112:             self.check_poly(orth.eval_sh_chebyt, orth.sh_chebyt,
113:                             param_ranges=[], x_range=[0, 1])
114:         finally:
115:             np.seterr(**olderr)
116: 
117:     def test_sh_chebyu(self):
118:         self.check_poly(orth.eval_sh_chebyu, orth.sh_chebyu,
119:                    param_ranges=[], x_range=[0, 1])
120: 
121:     def test_legendre(self):
122:         self.check_poly(orth.eval_legendre, orth.legendre,
123:                    param_ranges=[], x_range=[-1, 1])
124: 
125:     def test_sh_legendre(self):
126:         olderr = np.seterr(all='ignore')
127:         try:
128:             self.check_poly(orth.eval_sh_legendre, orth.sh_legendre,
129:                             param_ranges=[], x_range=[0, 1])
130:         finally:
131:             np.seterr(**olderr)
132: 
133:     def test_genlaguerre(self):
134:         self.check_poly(orth.eval_genlaguerre, orth.genlaguerre,
135:                    param_ranges=[(-0.99, 10)], x_range=[0, 100])
136: 
137:     def test_laguerre(self):
138:         self.check_poly(orth.eval_laguerre, orth.laguerre,
139:                    param_ranges=[], x_range=[0, 100])
140: 
141:     def test_hermite(self):
142:         self.check_poly(orth.eval_hermite, orth.hermite,
143:                    param_ranges=[], x_range=[-100, 100])
144: 
145:     def test_hermitenorm(self):
146:         self.check_poly(orth.eval_hermitenorm, orth.hermitenorm,
147:                         param_ranges=[], x_range=[-100, 100])
148: 
149: 
150: class TestRecurrence(object):
151:     '''
152:     Check that the eval_* functions sig='ld->d' and 'dd->d' agree.
153: 
154:     '''
155: 
156:     def check_poly(self, func, param_ranges=[], x_range=[], nn=10,
157:                    nparam=10, nx=10, rtol=1e-8):
158:         np.random.seed(1234)
159: 
160:         dataset = []
161:         for n in np.arange(nn):
162:             params = [a + (b-a)*np.random.rand(nparam) for a,b in param_ranges]
163:             params = np.asarray(params).T
164:             if not param_ranges:
165:                 params = [0]
166:             for p in params:
167:                 if param_ranges:
168:                     p = (n,) + tuple(p)
169:                 else:
170:                     p = (n,)
171:                 x = x_range[0] + (x_range[1] - x_range[0])*np.random.rand(nx)
172:                 x[0] = x_range[0]  # always include domain start point
173:                 x[1] = x_range[1]  # always include domain end point
174:                 kw = dict(sig=(len(p)+1)*'d'+'->d')
175:                 z = np.c_[np.tile(p, (nx,1)), x, func(*(p + (x,)), **kw)]
176:                 dataset.append(z)
177: 
178:         dataset = np.concatenate(dataset, axis=0)
179: 
180:         def polyfunc(*p):
181:             p = (p[0].astype(int),) + p[1:]
182:             kw = dict(sig='l'+(len(p)-1)*'d'+'->d')
183:             return func(*p, **kw)
184: 
185:         olderr = np.seterr(all='raise')
186:         try:
187:             ds = FuncData(polyfunc, dataset, list(range(len(param_ranges)+2)), -1,
188:                           rtol=rtol)
189:             ds.check()
190:         finally:
191:             np.seterr(**olderr)
192: 
193:     def test_jacobi(self):
194:         self.check_poly(orth.eval_jacobi,
195:                    param_ranges=[(-0.99, 10), (-0.99, 10)], x_range=[-1, 1])
196: 
197:     def test_sh_jacobi(self):
198:         self.check_poly(orth.eval_sh_jacobi,
199:                    param_ranges=[(1, 10), (0, 1)], x_range=[0, 1])
200: 
201:     def test_gegenbauer(self):
202:         self.check_poly(orth.eval_gegenbauer,
203:                    param_ranges=[(-0.499, 10)], x_range=[-1, 1])
204: 
205:     def test_chebyt(self):
206:         self.check_poly(orth.eval_chebyt,
207:                    param_ranges=[], x_range=[-1, 1])
208: 
209:     def test_chebyu(self):
210:         self.check_poly(orth.eval_chebyu,
211:                    param_ranges=[], x_range=[-1, 1])
212: 
213:     def test_chebys(self):
214:         self.check_poly(orth.eval_chebys,
215:                    param_ranges=[], x_range=[-2, 2])
216: 
217:     def test_chebyc(self):
218:         self.check_poly(orth.eval_chebyc,
219:                    param_ranges=[], x_range=[-2, 2])
220: 
221:     def test_sh_chebyt(self):
222:         self.check_poly(orth.eval_sh_chebyt,
223:                    param_ranges=[], x_range=[0, 1])
224: 
225:     def test_sh_chebyu(self):
226:         self.check_poly(orth.eval_sh_chebyu,
227:                    param_ranges=[], x_range=[0, 1])
228: 
229:     def test_legendre(self):
230:         self.check_poly(orth.eval_legendre,
231:                    param_ranges=[], x_range=[-1, 1])
232: 
233:     def test_sh_legendre(self):
234:         self.check_poly(orth.eval_sh_legendre,
235:                    param_ranges=[], x_range=[0, 1])
236: 
237:     def test_genlaguerre(self):
238:         self.check_poly(orth.eval_genlaguerre,
239:                    param_ranges=[(-0.99, 10)], x_range=[0, 100])
240: 
241:     def test_laguerre(self):
242:         self.check_poly(orth.eval_laguerre,
243:                    param_ranges=[], x_range=[0, 100])
244: 
245:     def test_hermite(self):
246:         v = orth.eval_hermite(70, 1.0)
247:         a = -1.457076485701412e60
248:         assert_allclose(v,a)
249: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_558054 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_558054) is not StypyTypeError):

    if (import_558054 != 'pyd_module'):
        __import__(import_558054)
        sys_modules_558055 = sys.modules[import_558054]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_558055.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_558054)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_, assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_558056 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_558056) is not StypyTypeError):

    if (import_558056 != 'pyd_module'):
        __import__(import_558056)
        sys_modules_558057 = sys.modules[import_558056]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_558057.module_type_store, module_type_store, ['assert_', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_558057, sys_modules_558057.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose'], [assert_, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_558056)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.special.orthogonal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_558058 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special.orthogonal')

if (type(import_558058) is not StypyTypeError):

    if (import_558058 != 'pyd_module'):
        __import__(import_558058)
        sys_modules_558059 = sys.modules[import_558058]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'orth', sys_modules_558059.module_type_store, module_type_store)
    else:
        import scipy.special.orthogonal as orth

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'orth', scipy.special.orthogonal, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special.orthogonal' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special.orthogonal', import_558058)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special._testutils import FuncData' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_558060 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._testutils')

if (type(import_558060) is not StypyTypeError):

    if (import_558060 != 'pyd_module'):
        __import__(import_558060)
        sys_modules_558061 = sys.modules[import_558060]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._testutils', sys_modules_558061.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_558061, sys_modules_558061.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._testutils', import_558060)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_eval_chebyt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_eval_chebyt'
    module_type_store = module_type_store.open_function_context('test_eval_chebyt', 10, 0, False)
    
    # Passed parameters checking function
    test_eval_chebyt.stypy_localization = localization
    test_eval_chebyt.stypy_type_of_self = None
    test_eval_chebyt.stypy_type_store = module_type_store
    test_eval_chebyt.stypy_function_name = 'test_eval_chebyt'
    test_eval_chebyt.stypy_param_names_list = []
    test_eval_chebyt.stypy_varargs_param_name = None
    test_eval_chebyt.stypy_kwargs_param_name = None
    test_eval_chebyt.stypy_call_defaults = defaults
    test_eval_chebyt.stypy_call_varargs = varargs
    test_eval_chebyt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_eval_chebyt', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_eval_chebyt', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_eval_chebyt(...)' code ##################

    
    # Assigning a Call to a Name (line 11):
    
    # Call to arange(...): (line 11)
    # Processing the call arguments (line 11)
    int_558064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
    int_558065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
    int_558066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 28), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_558067 = {}
    # Getting the type of 'np' (line 11)
    np_558062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 11)
    arange_558063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), np_558062, 'arange')
    # Calling arange(args, kwargs) (line 11)
    arange_call_result_558068 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), arange_558063, *[int_558064, int_558065, int_558066], **kwargs_558067)
    
    # Assigning a type to the variable 'n' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'n', arange_call_result_558068)
    
    # Assigning a BinOp to a Name (line 12):
    int_558069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'int')
    
    # Call to rand(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_558073 = {}
    # Getting the type of 'np' (line 12)
    np_558070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'np', False)
    # Obtaining the member 'random' of a type (line 12)
    random_558071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 10), np_558070, 'random')
    # Obtaining the member 'rand' of a type (line 12)
    rand_558072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 10), random_558071, 'rand')
    # Calling rand(args, kwargs) (line 12)
    rand_call_result_558074 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), rand_558072, *[], **kwargs_558073)
    
    # Applying the binary operator '*' (line 12)
    result_mul_558075 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 8), '*', int_558069, rand_call_result_558074)
    
    int_558076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 29), 'int')
    # Applying the binary operator '-' (line 12)
    result_sub_558077 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 8), '-', result_mul_558075, int_558076)
    
    # Assigning a type to the variable 'x' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'x', result_sub_558077)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to cos(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'n' (line 13)
    n_558080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'n', False)
    
    # Call to arccos(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'x' (line 13)
    x_558083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'x', False)
    # Processing the call keyword arguments (line 13)
    kwargs_558084 = {}
    # Getting the type of 'np' (line 13)
    np_558081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 18), 'np', False)
    # Obtaining the member 'arccos' of a type (line 13)
    arccos_558082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 18), np_558081, 'arccos')
    # Calling arccos(args, kwargs) (line 13)
    arccos_call_result_558085 = invoke(stypy.reporting.localization.Localization(__file__, 13, 18), arccos_558082, *[x_558083], **kwargs_558084)
    
    # Applying the binary operator '*' (line 13)
    result_mul_558086 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 16), '*', n_558080, arccos_call_result_558085)
    
    # Processing the call keyword arguments (line 13)
    kwargs_558087 = {}
    # Getting the type of 'np' (line 13)
    np_558078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'np', False)
    # Obtaining the member 'cos' of a type (line 13)
    cos_558079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), np_558078, 'cos')
    # Calling cos(args, kwargs) (line 13)
    cos_call_result_558088 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), cos_558079, *[result_mul_558086], **kwargs_558087)
    
    # Assigning a type to the variable 'v1' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'v1', cos_call_result_558088)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to eval_chebyt(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'n' (line 14)
    n_558091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 26), 'n', False)
    # Getting the type of 'x' (line 14)
    x_558092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 29), 'x', False)
    # Processing the call keyword arguments (line 14)
    kwargs_558093 = {}
    # Getting the type of 'orth' (line 14)
    orth_558089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'orth', False)
    # Obtaining the member 'eval_chebyt' of a type (line 14)
    eval_chebyt_558090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 9), orth_558089, 'eval_chebyt')
    # Calling eval_chebyt(args, kwargs) (line 14)
    eval_chebyt_call_result_558094 = invoke(stypy.reporting.localization.Localization(__file__, 14, 9), eval_chebyt_558090, *[n_558091, x_558092], **kwargs_558093)
    
    # Assigning a type to the variable 'v2' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'v2', eval_chebyt_call_result_558094)
    
    # Call to assert_(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to allclose(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'v1' (line 15)
    v1_558098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'v1', False)
    # Getting the type of 'v2' (line 15)
    v2_558099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 28), 'v2', False)
    # Processing the call keyword arguments (line 15)
    float_558100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'float')
    keyword_558101 = float_558100
    kwargs_558102 = {'rtol': keyword_558101}
    # Getting the type of 'np' (line 15)
    np_558096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'np', False)
    # Obtaining the member 'allclose' of a type (line 15)
    allclose_558097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 12), np_558096, 'allclose')
    # Calling allclose(args, kwargs) (line 15)
    allclose_call_result_558103 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), allclose_558097, *[v1_558098, v2_558099], **kwargs_558102)
    
    # Processing the call keyword arguments (line 15)
    kwargs_558104 = {}
    # Getting the type of 'assert_' (line 15)
    assert__558095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 15)
    assert__call_result_558105 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), assert__558095, *[allclose_call_result_558103], **kwargs_558104)
    
    
    # ################# End of 'test_eval_chebyt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_eval_chebyt' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_558106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_558106)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_eval_chebyt'
    return stypy_return_type_558106

# Assigning a type to the variable 'test_eval_chebyt' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test_eval_chebyt', test_eval_chebyt)

@norecursion
def test_eval_genlaguerre_restriction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_eval_genlaguerre_restriction'
    module_type_store = module_type_store.open_function_context('test_eval_genlaguerre_restriction', 18, 0, False)
    
    # Passed parameters checking function
    test_eval_genlaguerre_restriction.stypy_localization = localization
    test_eval_genlaguerre_restriction.stypy_type_of_self = None
    test_eval_genlaguerre_restriction.stypy_type_store = module_type_store
    test_eval_genlaguerre_restriction.stypy_function_name = 'test_eval_genlaguerre_restriction'
    test_eval_genlaguerre_restriction.stypy_param_names_list = []
    test_eval_genlaguerre_restriction.stypy_varargs_param_name = None
    test_eval_genlaguerre_restriction.stypy_kwargs_param_name = None
    test_eval_genlaguerre_restriction.stypy_call_defaults = defaults
    test_eval_genlaguerre_restriction.stypy_call_varargs = varargs
    test_eval_genlaguerre_restriction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_eval_genlaguerre_restriction', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_eval_genlaguerre_restriction', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_eval_genlaguerre_restriction(...)' code ##################

    
    # Call to assert_(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to isnan(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to eval_genlaguerre(...): (line 20)
    # Processing the call arguments (line 20)
    int_558112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 43), 'int')
    int_558113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 46), 'int')
    int_558114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 50), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_558115 = {}
    # Getting the type of 'orth' (line 20)
    orth_558110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'orth', False)
    # Obtaining the member 'eval_genlaguerre' of a type (line 20)
    eval_genlaguerre_558111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 21), orth_558110, 'eval_genlaguerre')
    # Calling eval_genlaguerre(args, kwargs) (line 20)
    eval_genlaguerre_call_result_558116 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), eval_genlaguerre_558111, *[int_558112, int_558113, int_558114], **kwargs_558115)
    
    # Processing the call keyword arguments (line 20)
    kwargs_558117 = {}
    # Getting the type of 'np' (line 20)
    np_558108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'np', False)
    # Obtaining the member 'isnan' of a type (line 20)
    isnan_558109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), np_558108, 'isnan')
    # Calling isnan(args, kwargs) (line 20)
    isnan_call_result_558118 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), isnan_558109, *[eval_genlaguerre_call_result_558116], **kwargs_558117)
    
    # Processing the call keyword arguments (line 20)
    kwargs_558119 = {}
    # Getting the type of 'assert_' (line 20)
    assert__558107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 20)
    assert__call_result_558120 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), assert__558107, *[isnan_call_result_558118], **kwargs_558119)
    
    
    # Call to assert_(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to isnan(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to eval_genlaguerre(...): (line 21)
    # Processing the call arguments (line 21)
    float_558126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 43), 'float')
    int_558127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 48), 'int')
    int_558128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 52), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_558129 = {}
    # Getting the type of 'orth' (line 21)
    orth_558124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'orth', False)
    # Obtaining the member 'eval_genlaguerre' of a type (line 21)
    eval_genlaguerre_558125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 21), orth_558124, 'eval_genlaguerre')
    # Calling eval_genlaguerre(args, kwargs) (line 21)
    eval_genlaguerre_call_result_558130 = invoke(stypy.reporting.localization.Localization(__file__, 21, 21), eval_genlaguerre_558125, *[float_558126, int_558127, int_558128], **kwargs_558129)
    
    # Processing the call keyword arguments (line 21)
    kwargs_558131 = {}
    # Getting the type of 'np' (line 21)
    np_558122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'np', False)
    # Obtaining the member 'isnan' of a type (line 21)
    isnan_558123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), np_558122, 'isnan')
    # Calling isnan(args, kwargs) (line 21)
    isnan_call_result_558132 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), isnan_558123, *[eval_genlaguerre_call_result_558130], **kwargs_558131)
    
    # Processing the call keyword arguments (line 21)
    kwargs_558133 = {}
    # Getting the type of 'assert_' (line 21)
    assert__558121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 21)
    assert__call_result_558134 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), assert__558121, *[isnan_call_result_558132], **kwargs_558133)
    
    
    # ################# End of 'test_eval_genlaguerre_restriction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_eval_genlaguerre_restriction' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_558135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_558135)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_eval_genlaguerre_restriction'
    return stypy_return_type_558135

# Assigning a type to the variable 'test_eval_genlaguerre_restriction' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test_eval_genlaguerre_restriction', test_eval_genlaguerre_restriction)

@norecursion
def test_warnings(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_warnings'
    module_type_store = module_type_store.open_function_context('test_warnings', 24, 0, False)
    
    # Passed parameters checking function
    test_warnings.stypy_localization = localization
    test_warnings.stypy_type_of_self = None
    test_warnings.stypy_type_store = module_type_store
    test_warnings.stypy_function_name = 'test_warnings'
    test_warnings.stypy_param_names_list = []
    test_warnings.stypy_varargs_param_name = None
    test_warnings.stypy_kwargs_param_name = None
    test_warnings.stypy_call_defaults = defaults
    test_warnings.stypy_call_varargs = varargs
    test_warnings.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_warnings', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_warnings', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_warnings(...)' code ##################

    
    # Assigning a Call to a Name (line 26):
    
    # Call to seterr(...): (line 26)
    # Processing the call keyword arguments (line 26)
    str_558138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'str', 'raise')
    keyword_558139 = str_558138
    kwargs_558140 = {'all': keyword_558139}
    # Getting the type of 'np' (line 26)
    np_558136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'np', False)
    # Obtaining the member 'seterr' of a type (line 26)
    seterr_558137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 13), np_558136, 'seterr')
    # Calling seterr(args, kwargs) (line 26)
    seterr_call_result_558141 = invoke(stypy.reporting.localization.Localization(__file__, 26, 13), seterr_558137, *[], **kwargs_558140)
    
    # Assigning a type to the variable 'olderr' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'olderr', seterr_call_result_558141)
    
    # Try-finally block (line 27)
    
    # Call to eval_legendre(...): (line 29)
    # Processing the call arguments (line 29)
    int_558144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'int')
    int_558145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 30), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_558146 = {}
    # Getting the type of 'orth' (line 29)
    orth_558142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'orth', False)
    # Obtaining the member 'eval_legendre' of a type (line 29)
    eval_legendre_558143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), orth_558142, 'eval_legendre')
    # Calling eval_legendre(args, kwargs) (line 29)
    eval_legendre_call_result_558147 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), eval_legendre_558143, *[int_558144, int_558145], **kwargs_558146)
    
    
    # Call to eval_laguerre(...): (line 30)
    # Processing the call arguments (line 30)
    int_558150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'int')
    int_558151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'int')
    # Processing the call keyword arguments (line 30)
    kwargs_558152 = {}
    # Getting the type of 'orth' (line 30)
    orth_558148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'orth', False)
    # Obtaining the member 'eval_laguerre' of a type (line 30)
    eval_laguerre_558149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), orth_558148, 'eval_laguerre')
    # Calling eval_laguerre(args, kwargs) (line 30)
    eval_laguerre_call_result_558153 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), eval_laguerre_558149, *[int_558150, int_558151], **kwargs_558152)
    
    
    # Call to eval_gegenbauer(...): (line 31)
    # Processing the call arguments (line 31)
    int_558156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'int')
    int_558157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'int')
    int_558158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_558159 = {}
    # Getting the type of 'orth' (line 31)
    orth_558154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'orth', False)
    # Obtaining the member 'eval_gegenbauer' of a type (line 31)
    eval_gegenbauer_558155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), orth_558154, 'eval_gegenbauer')
    # Calling eval_gegenbauer(args, kwargs) (line 31)
    eval_gegenbauer_call_result_558160 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), eval_gegenbauer_558155, *[int_558156, int_558157, int_558158], **kwargs_558159)
    
    
    # finally branch of the try-finally block (line 27)
    
    # Call to seterr(...): (line 33)
    # Processing the call keyword arguments (line 33)
    # Getting the type of 'olderr' (line 33)
    olderr_558163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'olderr', False)
    kwargs_558164 = {'olderr_558163': olderr_558163}
    # Getting the type of 'np' (line 33)
    np_558161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'np', False)
    # Obtaining the member 'seterr' of a type (line 33)
    seterr_558162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), np_558161, 'seterr')
    # Calling seterr(args, kwargs) (line 33)
    seterr_call_result_558165 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), seterr_558162, *[], **kwargs_558164)
    
    
    
    # ################# End of 'test_warnings(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_warnings' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_558166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_558166)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_warnings'
    return stypy_return_type_558166

# Assigning a type to the variable 'test_warnings' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'test_warnings', test_warnings)
# Declaration of the 'TestPolys' class

class TestPolys(object, ):
    str_558167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '\n    Check that the eval_* functions agree with the constructed polynomials\n\n    ')

    @norecursion
    def check_poly(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_558168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_558169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        
        int_558170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 68), 'int')
        int_558171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'int')
        int_558172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'int')
        float_558173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 42), 'float')
        defaults = [list_558168, list_558169, int_558170, int_558171, int_558172, float_558173]
        # Create a new context for function 'check_poly'
        module_type_store = module_type_store.open_function_context('check_poly', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.check_poly.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.check_poly.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.check_poly.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.check_poly.__dict__.__setitem__('stypy_function_name', 'TestPolys.check_poly')
        TestPolys.check_poly.__dict__.__setitem__('stypy_param_names_list', ['func', 'cls', 'param_ranges', 'x_range', 'nn', 'nparam', 'nx', 'rtol'])
        TestPolys.check_poly.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.check_poly.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.check_poly.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.check_poly.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.check_poly.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.check_poly.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.check_poly', ['func', 'cls', 'param_ranges', 'x_range', 'nn', 'nparam', 'nx', 'rtol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_poly', localization, ['func', 'cls', 'param_ranges', 'x_range', 'nn', 'nparam', 'nx', 'rtol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_poly(...)' code ##################

        
        # Call to seed(...): (line 44)
        # Processing the call arguments (line 44)
        int_558177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_558178 = {}
        # Getting the type of 'np' (line 44)
        np_558174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 44)
        random_558175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), np_558174, 'random')
        # Obtaining the member 'seed' of a type (line 44)
        seed_558176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), random_558175, 'seed')
        # Calling seed(args, kwargs) (line 44)
        seed_call_result_558179 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), seed_558176, *[int_558177], **kwargs_558178)
        
        
        # Assigning a List to a Name (line 46):
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_558180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        
        # Assigning a type to the variable 'dataset' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'dataset', list_558180)
        
        
        # Call to arange(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'nn' (line 47)
        nn_558183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'nn', False)
        # Processing the call keyword arguments (line 47)
        kwargs_558184 = {}
        # Getting the type of 'np' (line 47)
        np_558181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 47)
        arange_558182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), np_558181, 'arange')
        # Calling arange(args, kwargs) (line 47)
        arange_call_result_558185 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), arange_558182, *[nn_558183], **kwargs_558184)
        
        # Testing the type of a for loop iterable (line 47)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 8), arange_call_result_558185)
        # Getting the type of the for loop variable (line 47)
        for_loop_var_558186 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 8), arange_call_result_558185)
        # Assigning a type to the variable 'n' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'n', for_loop_var_558186)
        # SSA begins for a for statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Name (line 48):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'param_ranges' (line 48)
        param_ranges_558199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 66), 'param_ranges')
        comprehension_558200 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), param_ranges_558199)
        # Assigning a type to the variable 'a' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), comprehension_558200))
        # Assigning a type to the variable 'b' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), comprehension_558200))
        # Getting the type of 'a' (line 48)
        a_558187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'a')
        # Getting the type of 'b' (line 48)
        b_558188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'b')
        # Getting the type of 'a' (line 48)
        a_558189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'a')
        # Applying the binary operator '-' (line 48)
        result_sub_558190 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 27), '-', b_558188, a_558189)
        
        
        # Call to rand(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'nparam' (line 48)
        nparam_558194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 47), 'nparam', False)
        # Processing the call keyword arguments (line 48)
        kwargs_558195 = {}
        # Getting the type of 'np' (line 48)
        np_558191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'np', False)
        # Obtaining the member 'random' of a type (line 48)
        random_558192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 32), np_558191, 'random')
        # Obtaining the member 'rand' of a type (line 48)
        rand_558193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 32), random_558192, 'rand')
        # Calling rand(args, kwargs) (line 48)
        rand_call_result_558196 = invoke(stypy.reporting.localization.Localization(__file__, 48, 32), rand_558193, *[nparam_558194], **kwargs_558195)
        
        # Applying the binary operator '*' (line 48)
        result_mul_558197 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 26), '*', result_sub_558190, rand_call_result_558196)
        
        # Applying the binary operator '+' (line 48)
        result_add_558198 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 22), '+', a_558187, result_mul_558197)
        
        list_558201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), list_558201, result_add_558198)
        # Assigning a type to the variable 'params' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'params', list_558201)
        
        # Assigning a Attribute to a Name (line 49):
        
        # Call to asarray(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'params' (line 49)
        params_558204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 32), 'params', False)
        # Processing the call keyword arguments (line 49)
        kwargs_558205 = {}
        # Getting the type of 'np' (line 49)
        np_558202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'np', False)
        # Obtaining the member 'asarray' of a type (line 49)
        asarray_558203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 21), np_558202, 'asarray')
        # Calling asarray(args, kwargs) (line 49)
        asarray_call_result_558206 = invoke(stypy.reporting.localization.Localization(__file__, 49, 21), asarray_558203, *[params_558204], **kwargs_558205)
        
        # Obtaining the member 'T' of a type (line 49)
        T_558207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 21), asarray_call_result_558206, 'T')
        # Assigning a type to the variable 'params' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'params', T_558207)
        
        
        # Getting the type of 'param_ranges' (line 50)
        param_ranges_558208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'param_ranges')
        # Applying the 'not' unary operator (line 50)
        result_not__558209 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), 'not', param_ranges_558208)
        
        # Testing the type of an if condition (line 50)
        if_condition_558210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 12), result_not__558209)
        # Assigning a type to the variable 'if_condition_558210' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_condition_558210', if_condition_558210)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 51):
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_558211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        int_558212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), list_558211, int_558212)
        
        # Assigning a type to the variable 'params' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'params', list_558211)
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'params' (line 52)
        params_558213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'params')
        # Testing the type of a for loop iterable (line 52)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 12), params_558213)
        # Getting the type of the for loop variable (line 52)
        for_loop_var_558214 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 12), params_558213)
        # Assigning a type to the variable 'p' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'p', for_loop_var_558214)
        # SSA begins for a for statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'param_ranges' (line 53)
        param_ranges_558215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'param_ranges')
        # Testing the type of an if condition (line 53)
        if_condition_558216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 16), param_ranges_558215)
        # Assigning a type to the variable 'if_condition_558216' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'if_condition_558216', if_condition_558216)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 54):
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_558217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'n' (line 54)
        n_558218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 25), tuple_558217, n_558218)
        
        
        # Call to tuple(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'p' (line 54)
        p_558220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'p', False)
        # Processing the call keyword arguments (line 54)
        kwargs_558221 = {}
        # Getting the type of 'tuple' (line 54)
        tuple_558219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'tuple', False)
        # Calling tuple(args, kwargs) (line 54)
        tuple_call_result_558222 = invoke(stypy.reporting.localization.Localization(__file__, 54, 31), tuple_558219, *[p_558220], **kwargs_558221)
        
        # Applying the binary operator '+' (line 54)
        result_add_558223 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 24), '+', tuple_558217, tuple_call_result_558222)
        
        # Assigning a type to the variable 'p' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'p', result_add_558223)
        # SSA branch for the else part of an if statement (line 53)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Name (line 56):
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_558224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        # Getting the type of 'n' (line 56)
        n_558225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 25), tuple_558224, n_558225)
        
        # Assigning a type to the variable 'p' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'p', tuple_558224)
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 57):
        
        # Obtaining the type of the subscript
        int_558226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'int')
        # Getting the type of 'x_range' (line 57)
        x_range_558227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___558228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 20), x_range_558227, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_558229 = invoke(stypy.reporting.localization.Localization(__file__, 57, 20), getitem___558228, int_558226)
        
        
        # Obtaining the type of the subscript
        int_558230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 42), 'int')
        # Getting the type of 'x_range' (line 57)
        x_range_558231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___558232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 34), x_range_558231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_558233 = invoke(stypy.reporting.localization.Localization(__file__, 57, 34), getitem___558232, int_558230)
        
        
        # Obtaining the type of the subscript
        int_558234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 55), 'int')
        # Getting the type of 'x_range' (line 57)
        x_range_558235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 47), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___558236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 47), x_range_558235, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_558237 = invoke(stypy.reporting.localization.Localization(__file__, 57, 47), getitem___558236, int_558234)
        
        # Applying the binary operator '-' (line 57)
        result_sub_558238 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 34), '-', subscript_call_result_558233, subscript_call_result_558237)
        
        
        # Call to rand(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'nx' (line 57)
        nx_558242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 74), 'nx', False)
        # Processing the call keyword arguments (line 57)
        kwargs_558243 = {}
        # Getting the type of 'np' (line 57)
        np_558239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 59), 'np', False)
        # Obtaining the member 'random' of a type (line 57)
        random_558240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 59), np_558239, 'random')
        # Obtaining the member 'rand' of a type (line 57)
        rand_558241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 59), random_558240, 'rand')
        # Calling rand(args, kwargs) (line 57)
        rand_call_result_558244 = invoke(stypy.reporting.localization.Localization(__file__, 57, 59), rand_558241, *[nx_558242], **kwargs_558243)
        
        # Applying the binary operator '*' (line 57)
        result_mul_558245 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 33), '*', result_sub_558238, rand_call_result_558244)
        
        # Applying the binary operator '+' (line 57)
        result_add_558246 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 20), '+', subscript_call_result_558229, result_mul_558245)
        
        # Assigning a type to the variable 'x' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'x', result_add_558246)
        
        # Assigning a Subscript to a Subscript (line 58):
        
        # Obtaining the type of the subscript
        int_558247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'int')
        # Getting the type of 'x_range' (line 58)
        x_range_558248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 58)
        getitem___558249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 23), x_range_558248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 58)
        subscript_call_result_558250 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), getitem___558249, int_558247)
        
        # Getting the type of 'x' (line 58)
        x_558251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'x')
        int_558252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 18), 'int')
        # Storing an element on a container (line 58)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), x_558251, (int_558252, subscript_call_result_558250))
        
        # Assigning a Subscript to a Subscript (line 59):
        
        # Obtaining the type of the subscript
        int_558253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'int')
        # Getting the type of 'x_range' (line 59)
        x_range_558254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___558255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), x_range_558254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_558256 = invoke(stypy.reporting.localization.Localization(__file__, 59, 23), getitem___558255, int_558253)
        
        # Getting the type of 'x' (line 59)
        x_558257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'x')
        int_558258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 18), 'int')
        # Storing an element on a container (line 59)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), x_558257, (int_558258, subscript_call_result_558256))
        
        # Assigning a Call to a Name (line 60):
        
        # Call to poly1d(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to cls(...): (line 60)
        # Getting the type of 'p' (line 60)
        p_558262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'p', False)
        # Processing the call keyword arguments (line 60)
        kwargs_558263 = {}
        # Getting the type of 'cls' (line 60)
        cls_558261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 33), 'cls', False)
        # Calling cls(args, kwargs) (line 60)
        cls_call_result_558264 = invoke(stypy.reporting.localization.Localization(__file__, 60, 33), cls_558261, *[p_558262], **kwargs_558263)
        
        # Obtaining the member 'coef' of a type (line 60)
        coef_558265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 33), cls_call_result_558264, 'coef')
        # Processing the call keyword arguments (line 60)
        kwargs_558266 = {}
        # Getting the type of 'np' (line 60)
        np_558259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'np', False)
        # Obtaining the member 'poly1d' of a type (line 60)
        poly1d_558260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), np_558259, 'poly1d')
        # Calling poly1d(args, kwargs) (line 60)
        poly1d_call_result_558267 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), poly1d_558260, *[coef_558265], **kwargs_558266)
        
        # Assigning a type to the variable 'poly' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'poly', poly1d_call_result_558267)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 61)
        tuple_558268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 61)
        # Adding element type (line 61)
        
        # Call to tile(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'p' (line 61)
        p_558271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'p', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 61)
        tuple_558272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 61)
        # Adding element type (line 61)
        # Getting the type of 'nx' (line 61)
        nx_558273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'nx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 38), tuple_558272, nx_558273)
        # Adding element type (line 61)
        int_558274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 38), tuple_558272, int_558274)
        
        # Processing the call keyword arguments (line 61)
        kwargs_558275 = {}
        # Getting the type of 'np' (line 61)
        np_558269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'np', False)
        # Obtaining the member 'tile' of a type (line 61)
        tile_558270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 26), np_558269, 'tile')
        # Calling tile(args, kwargs) (line 61)
        tile_call_result_558276 = invoke(stypy.reporting.localization.Localization(__file__, 61, 26), tile_558270, *[p_558271, tuple_558272], **kwargs_558275)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), tuple_558268, tile_call_result_558276)
        # Adding element type (line 61)
        # Getting the type of 'x' (line 61)
        x_558277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), tuple_558268, x_558277)
        # Adding element type (line 61)
        
        # Call to poly(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'x' (line 61)
        x_558279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 54), 'x', False)
        # Processing the call keyword arguments (line 61)
        kwargs_558280 = {}
        # Getting the type of 'poly' (line 61)
        poly_558278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 49), 'poly', False)
        # Calling poly(args, kwargs) (line 61)
        poly_call_result_558281 = invoke(stypy.reporting.localization.Localization(__file__, 61, 49), poly_558278, *[x_558279], **kwargs_558280)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 26), tuple_558268, poly_call_result_558281)
        
        # Getting the type of 'np' (line 61)
        np_558282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'np')
        # Obtaining the member 'c_' of a type (line 61)
        c__558283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), np_558282, 'c_')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___558284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), c__558283, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_558285 = invoke(stypy.reporting.localization.Localization(__file__, 61, 20), getitem___558284, tuple_558268)
        
        # Assigning a type to the variable 'z' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'z', subscript_call_result_558285)
        
        # Call to append(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'z' (line 62)
        z_558288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'z', False)
        # Processing the call keyword arguments (line 62)
        kwargs_558289 = {}
        # Getting the type of 'dataset' (line 62)
        dataset_558286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'dataset', False)
        # Obtaining the member 'append' of a type (line 62)
        append_558287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), dataset_558286, 'append')
        # Calling append(args, kwargs) (line 62)
        append_call_result_558290 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), append_558287, *[z_558288], **kwargs_558289)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 64):
        
        # Call to concatenate(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'dataset' (line 64)
        dataset_558293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'dataset', False)
        # Processing the call keyword arguments (line 64)
        int_558294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 47), 'int')
        keyword_558295 = int_558294
        kwargs_558296 = {'axis': keyword_558295}
        # Getting the type of 'np' (line 64)
        np_558291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 64)
        concatenate_558292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 18), np_558291, 'concatenate')
        # Calling concatenate(args, kwargs) (line 64)
        concatenate_call_result_558297 = invoke(stypy.reporting.localization.Localization(__file__, 64, 18), concatenate_558292, *[dataset_558293], **kwargs_558296)
        
        # Assigning a type to the variable 'dataset' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'dataset', concatenate_call_result_558297)

        @norecursion
        def polyfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'polyfunc'
            module_type_store = module_type_store.open_function_context('polyfunc', 66, 8, False)
            
            # Passed parameters checking function
            polyfunc.stypy_localization = localization
            polyfunc.stypy_type_of_self = None
            polyfunc.stypy_type_store = module_type_store
            polyfunc.stypy_function_name = 'polyfunc'
            polyfunc.stypy_param_names_list = []
            polyfunc.stypy_varargs_param_name = 'p'
            polyfunc.stypy_kwargs_param_name = None
            polyfunc.stypy_call_defaults = defaults
            polyfunc.stypy_call_varargs = varargs
            polyfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'polyfunc', [], 'p', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'polyfunc', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'polyfunc(...)' code ##################

            
            # Assigning a BinOp to a Name (line 67):
            
            # Obtaining an instance of the builtin type 'tuple' (line 67)
            tuple_558298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 17), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 67)
            # Adding element type (line 67)
            
            # Call to astype(...): (line 67)
            # Processing the call arguments (line 67)
            # Getting the type of 'int' (line 67)
            int_558304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'int', False)
            # Processing the call keyword arguments (line 67)
            kwargs_558305 = {}
            
            # Obtaining the type of the subscript
            int_558299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 19), 'int')
            # Getting the type of 'p' (line 67)
            p_558300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'p', False)
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___558301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), p_558300, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_558302 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), getitem___558301, int_558299)
            
            # Obtaining the member 'astype' of a type (line 67)
            astype_558303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), subscript_call_result_558302, 'astype')
            # Calling astype(args, kwargs) (line 67)
            astype_call_result_558306 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), astype_558303, *[int_558304], **kwargs_558305)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 17), tuple_558298, astype_call_result_558306)
            
            
            # Obtaining the type of the subscript
            int_558307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 40), 'int')
            slice_558308 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 67, 38), int_558307, None, None)
            # Getting the type of 'p' (line 67)
            p_558309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 38), 'p')
            # Obtaining the member '__getitem__' of a type (line 67)
            getitem___558310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 38), p_558309, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 67)
            subscript_call_result_558311 = invoke(stypy.reporting.localization.Localization(__file__, 67, 38), getitem___558310, slice_558308)
            
            # Applying the binary operator '+' (line 67)
            result_add_558312 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 16), '+', tuple_558298, subscript_call_result_558311)
            
            # Assigning a type to the variable 'p' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'p', result_add_558312)
            
            # Call to func(...): (line 68)
            # Getting the type of 'p' (line 68)
            p_558314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'p', False)
            # Processing the call keyword arguments (line 68)
            kwargs_558315 = {}
            # Getting the type of 'func' (line 68)
            func_558313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'func', False)
            # Calling func(args, kwargs) (line 68)
            func_call_result_558316 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), func_558313, *[p_558314], **kwargs_558315)
            
            # Assigning a type to the variable 'stypy_return_type' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'stypy_return_type', func_call_result_558316)
            
            # ################# End of 'polyfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'polyfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 66)
            stypy_return_type_558317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_558317)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'polyfunc'
            return stypy_return_type_558317

        # Assigning a type to the variable 'polyfunc' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'polyfunc', polyfunc)
        
        # Assigning a Call to a Name (line 70):
        
        # Call to seterr(...): (line 70)
        # Processing the call keyword arguments (line 70)
        str_558320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 31), 'str', 'raise')
        keyword_558321 = str_558320
        kwargs_558322 = {'all': keyword_558321}
        # Getting the type of 'np' (line 70)
        np_558318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'np', False)
        # Obtaining the member 'seterr' of a type (line 70)
        seterr_558319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 17), np_558318, 'seterr')
        # Calling seterr(args, kwargs) (line 70)
        seterr_call_result_558323 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), seterr_558319, *[], **kwargs_558322)
        
        # Assigning a type to the variable 'olderr' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'olderr', seterr_call_result_558323)
        
        # Try-finally block (line 71)
        
        # Assigning a Call to a Name (line 72):
        
        # Call to FuncData(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'polyfunc' (line 72)
        polyfunc_558325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'polyfunc', False)
        # Getting the type of 'dataset' (line 72)
        dataset_558326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'dataset', False)
        
        # Call to list(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to range(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to len(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'param_ranges' (line 72)
        param_ranges_558330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 60), 'param_ranges', False)
        # Processing the call keyword arguments (line 72)
        kwargs_558331 = {}
        # Getting the type of 'len' (line 72)
        len_558329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 56), 'len', False)
        # Calling len(args, kwargs) (line 72)
        len_call_result_558332 = invoke(stypy.reporting.localization.Localization(__file__, 72, 56), len_558329, *[param_ranges_558330], **kwargs_558331)
        
        int_558333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 74), 'int')
        # Applying the binary operator '+' (line 72)
        result_add_558334 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 56), '+', len_call_result_558332, int_558333)
        
        # Processing the call keyword arguments (line 72)
        kwargs_558335 = {}
        # Getting the type of 'range' (line 72)
        range_558328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 50), 'range', False)
        # Calling range(args, kwargs) (line 72)
        range_call_result_558336 = invoke(stypy.reporting.localization.Localization(__file__, 72, 50), range_558328, *[result_add_558334], **kwargs_558335)
        
        # Processing the call keyword arguments (line 72)
        kwargs_558337 = {}
        # Getting the type of 'list' (line 72)
        list_558327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 45), 'list', False)
        # Calling list(args, kwargs) (line 72)
        list_call_result_558338 = invoke(stypy.reporting.localization.Localization(__file__, 72, 45), list_558327, *[range_call_result_558336], **kwargs_558337)
        
        int_558339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 79), 'int')
        # Processing the call keyword arguments (line 72)
        # Getting the type of 'rtol' (line 73)
        rtol_558340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'rtol', False)
        keyword_558341 = rtol_558340
        kwargs_558342 = {'rtol': keyword_558341}
        # Getting the type of 'FuncData' (line 72)
        FuncData_558324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 72)
        FuncData_call_result_558343 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), FuncData_558324, *[polyfunc_558325, dataset_558326, list_call_result_558338, int_558339], **kwargs_558342)
        
        # Assigning a type to the variable 'ds' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'ds', FuncData_call_result_558343)
        
        # Call to check(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_558346 = {}
        # Getting the type of 'ds' (line 74)
        ds_558344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'ds', False)
        # Obtaining the member 'check' of a type (line 74)
        check_558345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), ds_558344, 'check')
        # Calling check(args, kwargs) (line 74)
        check_call_result_558347 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), check_558345, *[], **kwargs_558346)
        
        
        # finally branch of the try-finally block (line 71)
        
        # Call to seterr(...): (line 76)
        # Processing the call keyword arguments (line 76)
        # Getting the type of 'olderr' (line 76)
        olderr_558350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'olderr', False)
        kwargs_558351 = {'olderr_558350': olderr_558350}
        # Getting the type of 'np' (line 76)
        np_558348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'np', False)
        # Obtaining the member 'seterr' of a type (line 76)
        seterr_558349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), np_558348, 'seterr')
        # Calling seterr(args, kwargs) (line 76)
        seterr_call_result_558352 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), seterr_558349, *[], **kwargs_558351)
        
        
        
        # ################# End of 'check_poly(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_poly' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_558353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_poly'
        return stypy_return_type_558353


    @norecursion
    def test_jacobi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_jacobi'
        module_type_store = module_type_store.open_function_context('test_jacobi', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_jacobi')
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_jacobi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_jacobi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_jacobi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_jacobi(...)' code ##################

        
        # Call to check_poly(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'orth' (line 79)
        orth_558356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'orth', False)
        # Obtaining the member 'eval_jacobi' of a type (line 79)
        eval_jacobi_558357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), orth_558356, 'eval_jacobi')
        # Getting the type of 'orth' (line 79)
        orth_558358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'orth', False)
        # Obtaining the member 'jacobi' of a type (line 79)
        jacobi_558359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 42), orth_558358, 'jacobi')
        # Processing the call keyword arguments (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_558360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_558361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        float_558362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 34), tuple_558361, float_558362)
        # Adding element type (line 80)
        int_558363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 34), tuple_558361, int_558363)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 32), list_558360, tuple_558361)
        # Adding element type (line 80)
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_558364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        float_558365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 47), tuple_558364, float_558365)
        # Adding element type (line 80)
        int_558366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 47), tuple_558364, int_558366)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 32), list_558360, tuple_558364)
        
        keyword_558367 = list_558360
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_558368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_558369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 68), list_558368, int_558369)
        # Adding element type (line 80)
        int_558370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 73), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 68), list_558368, int_558370)
        
        keyword_558371 = list_558368
        float_558372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'float')
        keyword_558373 = float_558372
        kwargs_558374 = {'param_ranges': keyword_558367, 'x_range': keyword_558371, 'rtol': keyword_558373}
        # Getting the type of 'self' (line 79)
        self_558354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 79)
        check_poly_558355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_558354, 'check_poly')
        # Calling check_poly(args, kwargs) (line 79)
        check_poly_call_result_558375 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), check_poly_558355, *[eval_jacobi_558357, jacobi_558359], **kwargs_558374)
        
        
        # ################# End of 'test_jacobi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_jacobi' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_558376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_jacobi'
        return stypy_return_type_558376


    @norecursion
    def test_sh_jacobi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sh_jacobi'
        module_type_store = module_type_store.open_function_context('test_sh_jacobi', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_sh_jacobi')
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_sh_jacobi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_sh_jacobi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sh_jacobi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sh_jacobi(...)' code ##################

        
        # Call to check_poly(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'orth' (line 84)
        orth_558379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'orth', False)
        # Obtaining the member 'eval_sh_jacobi' of a type (line 84)
        eval_sh_jacobi_558380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 24), orth_558379, 'eval_sh_jacobi')
        # Getting the type of 'orth' (line 84)
        orth_558381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 45), 'orth', False)
        # Obtaining the member 'sh_jacobi' of a type (line 84)
        sh_jacobi_558382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 45), orth_558381, 'sh_jacobi')
        # Processing the call keyword arguments (line 84)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_558383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_558384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        int_558385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 34), tuple_558384, int_558385)
        # Adding element type (line 85)
        int_558386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 34), tuple_558384, int_558386)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), list_558383, tuple_558384)
        # Adding element type (line 85)
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_558387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        int_558388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 43), tuple_558387, int_558388)
        # Adding element type (line 85)
        int_558389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 43), tuple_558387, int_558389)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), list_558383, tuple_558387)
        
        keyword_558390 = list_558383
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_558391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        int_558392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 59), list_558391, int_558392)
        # Adding element type (line 85)
        int_558393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 59), list_558391, int_558393)
        
        keyword_558394 = list_558391
        float_558395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 24), 'float')
        keyword_558396 = float_558395
        kwargs_558397 = {'param_ranges': keyword_558390, 'x_range': keyword_558394, 'rtol': keyword_558396}
        # Getting the type of 'self' (line 84)
        self_558377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 84)
        check_poly_558378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_558377, 'check_poly')
        # Calling check_poly(args, kwargs) (line 84)
        check_poly_call_result_558398 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), check_poly_558378, *[eval_sh_jacobi_558380, sh_jacobi_558382], **kwargs_558397)
        
        
        # ################# End of 'test_sh_jacobi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sh_jacobi' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_558399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558399)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sh_jacobi'
        return stypy_return_type_558399


    @norecursion
    def test_gegenbauer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gegenbauer'
        module_type_store = module_type_store.open_function_context('test_gegenbauer', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_gegenbauer')
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_gegenbauer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_gegenbauer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gegenbauer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gegenbauer(...)' code ##################

        
        # Call to check_poly(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'orth' (line 89)
        orth_558402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'orth', False)
        # Obtaining the member 'eval_gegenbauer' of a type (line 89)
        eval_gegenbauer_558403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), orth_558402, 'eval_gegenbauer')
        # Getting the type of 'orth' (line 89)
        orth_558404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 46), 'orth', False)
        # Obtaining the member 'gegenbauer' of a type (line 89)
        gegenbauer_558405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 46), orth_558404, 'gegenbauer')
        # Processing the call keyword arguments (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_558406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'tuple' (line 90)
        tuple_558407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 90)
        # Adding element type (line 90)
        float_558408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 34), tuple_558407, float_558408)
        # Adding element type (line 90)
        int_558409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 34), tuple_558407, int_558409)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 32), list_558406, tuple_558407)
        
        keyword_558410 = list_558406
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_558411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        int_558412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 56), list_558411, int_558412)
        # Adding element type (line 90)
        int_558413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 56), list_558411, int_558413)
        
        keyword_558414 = list_558411
        float_558415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 24), 'float')
        keyword_558416 = float_558415
        kwargs_558417 = {'param_ranges': keyword_558410, 'x_range': keyword_558414, 'rtol': keyword_558416}
        # Getting the type of 'self' (line 89)
        self_558400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 89)
        check_poly_558401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_558400, 'check_poly')
        # Calling check_poly(args, kwargs) (line 89)
        check_poly_call_result_558418 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), check_poly_558401, *[eval_gegenbauer_558403, gegenbauer_558405], **kwargs_558417)
        
        
        # ################# End of 'test_gegenbauer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gegenbauer' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_558419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gegenbauer'
        return stypy_return_type_558419


    @norecursion
    def test_chebyt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chebyt'
        module_type_store = module_type_store.open_function_context('test_chebyt', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_chebyt')
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_chebyt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_chebyt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chebyt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chebyt(...)' code ##################

        
        # Call to check_poly(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'orth' (line 94)
        orth_558422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'orth', False)
        # Obtaining the member 'eval_chebyt' of a type (line 94)
        eval_chebyt_558423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 24), orth_558422, 'eval_chebyt')
        # Getting the type of 'orth' (line 94)
        orth_558424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'orth', False)
        # Obtaining the member 'chebyt' of a type (line 94)
        chebyt_558425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 42), orth_558424, 'chebyt')
        # Processing the call keyword arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_558426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        
        keyword_558427 = list_558426
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_558428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        int_558429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 44), list_558428, int_558429)
        # Adding element type (line 95)
        int_558430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 44), list_558428, int_558430)
        
        keyword_558431 = list_558428
        kwargs_558432 = {'param_ranges': keyword_558427, 'x_range': keyword_558431}
        # Getting the type of 'self' (line 94)
        self_558420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 94)
        check_poly_558421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_558420, 'check_poly')
        # Calling check_poly(args, kwargs) (line 94)
        check_poly_call_result_558433 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), check_poly_558421, *[eval_chebyt_558423, chebyt_558425], **kwargs_558432)
        
        
        # ################# End of 'test_chebyt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chebyt' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_558434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558434)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chebyt'
        return stypy_return_type_558434


    @norecursion
    def test_chebyu(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chebyu'
        module_type_store = module_type_store.open_function_context('test_chebyu', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_chebyu')
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_chebyu.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_chebyu', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chebyu', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chebyu(...)' code ##################

        
        # Call to check_poly(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'orth' (line 98)
        orth_558437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'orth', False)
        # Obtaining the member 'eval_chebyu' of a type (line 98)
        eval_chebyu_558438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), orth_558437, 'eval_chebyu')
        # Getting the type of 'orth' (line 98)
        orth_558439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'orth', False)
        # Obtaining the member 'chebyu' of a type (line 98)
        chebyu_558440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 42), orth_558439, 'chebyu')
        # Processing the call keyword arguments (line 98)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_558441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        
        keyword_558442 = list_558441
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_558443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        int_558444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 44), list_558443, int_558444)
        # Adding element type (line 99)
        int_558445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 44), list_558443, int_558445)
        
        keyword_558446 = list_558443
        kwargs_558447 = {'param_ranges': keyword_558442, 'x_range': keyword_558446}
        # Getting the type of 'self' (line 98)
        self_558435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 98)
        check_poly_558436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_558435, 'check_poly')
        # Calling check_poly(args, kwargs) (line 98)
        check_poly_call_result_558448 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), check_poly_558436, *[eval_chebyu_558438, chebyu_558440], **kwargs_558447)
        
        
        # ################# End of 'test_chebyu(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chebyu' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_558449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558449)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chebyu'
        return stypy_return_type_558449


    @norecursion
    def test_chebys(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chebys'
        module_type_store = module_type_store.open_function_context('test_chebys', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_chebys.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_chebys.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_chebys.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_chebys.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_chebys')
        TestPolys.test_chebys.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_chebys.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_chebys.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_chebys.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_chebys.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_chebys.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_chebys.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_chebys', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chebys', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chebys(...)' code ##################

        
        # Call to check_poly(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'orth' (line 102)
        orth_558452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'orth', False)
        # Obtaining the member 'eval_chebys' of a type (line 102)
        eval_chebys_558453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), orth_558452, 'eval_chebys')
        # Getting the type of 'orth' (line 102)
        orth_558454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 42), 'orth', False)
        # Obtaining the member 'chebys' of a type (line 102)
        chebys_558455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 42), orth_558454, 'chebys')
        # Processing the call keyword arguments (line 102)
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_558456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        
        keyword_558457 = list_558456
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_558458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        int_558459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 44), list_558458, int_558459)
        # Adding element type (line 103)
        int_558460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 44), list_558458, int_558460)
        
        keyword_558461 = list_558458
        kwargs_558462 = {'param_ranges': keyword_558457, 'x_range': keyword_558461}
        # Getting the type of 'self' (line 102)
        self_558450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 102)
        check_poly_558451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_558450, 'check_poly')
        # Calling check_poly(args, kwargs) (line 102)
        check_poly_call_result_558463 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), check_poly_558451, *[eval_chebys_558453, chebys_558455], **kwargs_558462)
        
        
        # ################# End of 'test_chebys(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chebys' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_558464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558464)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chebys'
        return stypy_return_type_558464


    @norecursion
    def test_chebyc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chebyc'
        module_type_store = module_type_store.open_function_context('test_chebyc', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_chebyc')
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_chebyc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_chebyc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chebyc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chebyc(...)' code ##################

        
        # Call to check_poly(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'orth' (line 106)
        orth_558467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'orth', False)
        # Obtaining the member 'eval_chebyc' of a type (line 106)
        eval_chebyc_558468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 24), orth_558467, 'eval_chebyc')
        # Getting the type of 'orth' (line 106)
        orth_558469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'orth', False)
        # Obtaining the member 'chebyc' of a type (line 106)
        chebyc_558470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 42), orth_558469, 'chebyc')
        # Processing the call keyword arguments (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_558471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        
        keyword_558472 = list_558471
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_558473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        int_558474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 44), list_558473, int_558474)
        # Adding element type (line 107)
        int_558475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 44), list_558473, int_558475)
        
        keyword_558476 = list_558473
        kwargs_558477 = {'param_ranges': keyword_558472, 'x_range': keyword_558476}
        # Getting the type of 'self' (line 106)
        self_558465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 106)
        check_poly_558466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_558465, 'check_poly')
        # Calling check_poly(args, kwargs) (line 106)
        check_poly_call_result_558478 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), check_poly_558466, *[eval_chebyc_558468, chebyc_558470], **kwargs_558477)
        
        
        # ################# End of 'test_chebyc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chebyc' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_558479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558479)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chebyc'
        return stypy_return_type_558479


    @norecursion
    def test_sh_chebyt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sh_chebyt'
        module_type_store = module_type_store.open_function_context('test_sh_chebyt', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_sh_chebyt')
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_sh_chebyt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_sh_chebyt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sh_chebyt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sh_chebyt(...)' code ##################

        
        # Assigning a Call to a Name (line 110):
        
        # Call to seterr(...): (line 110)
        # Processing the call keyword arguments (line 110)
        str_558482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'str', 'ignore')
        keyword_558483 = str_558482
        kwargs_558484 = {'all': keyword_558483}
        # Getting the type of 'np' (line 110)
        np_558480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'np', False)
        # Obtaining the member 'seterr' of a type (line 110)
        seterr_558481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), np_558480, 'seterr')
        # Calling seterr(args, kwargs) (line 110)
        seterr_call_result_558485 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), seterr_558481, *[], **kwargs_558484)
        
        # Assigning a type to the variable 'olderr' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'olderr', seterr_call_result_558485)
        
        # Try-finally block (line 111)
        
        # Call to check_poly(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'orth' (line 112)
        orth_558488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'orth', False)
        # Obtaining the member 'eval_sh_chebyt' of a type (line 112)
        eval_sh_chebyt_558489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), orth_558488, 'eval_sh_chebyt')
        # Getting the type of 'orth' (line 112)
        orth_558490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 49), 'orth', False)
        # Obtaining the member 'sh_chebyt' of a type (line 112)
        sh_chebyt_558491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 49), orth_558490, 'sh_chebyt')
        # Processing the call keyword arguments (line 112)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_558492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        
        keyword_558493 = list_558492
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_558494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        int_558495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 53), list_558494, int_558495)
        # Adding element type (line 113)
        int_558496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 53), list_558494, int_558496)
        
        keyword_558497 = list_558494
        kwargs_558498 = {'param_ranges': keyword_558493, 'x_range': keyword_558497}
        # Getting the type of 'self' (line 112)
        self_558486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 112)
        check_poly_558487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), self_558486, 'check_poly')
        # Calling check_poly(args, kwargs) (line 112)
        check_poly_call_result_558499 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), check_poly_558487, *[eval_sh_chebyt_558489, sh_chebyt_558491], **kwargs_558498)
        
        
        # finally branch of the try-finally block (line 111)
        
        # Call to seterr(...): (line 115)
        # Processing the call keyword arguments (line 115)
        # Getting the type of 'olderr' (line 115)
        olderr_558502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 24), 'olderr', False)
        kwargs_558503 = {'olderr_558502': olderr_558502}
        # Getting the type of 'np' (line 115)
        np_558500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'np', False)
        # Obtaining the member 'seterr' of a type (line 115)
        seterr_558501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), np_558500, 'seterr')
        # Calling seterr(args, kwargs) (line 115)
        seterr_call_result_558504 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), seterr_558501, *[], **kwargs_558503)
        
        
        
        # ################# End of 'test_sh_chebyt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sh_chebyt' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_558505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sh_chebyt'
        return stypy_return_type_558505


    @norecursion
    def test_sh_chebyu(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sh_chebyu'
        module_type_store = module_type_store.open_function_context('test_sh_chebyu', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_sh_chebyu')
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_sh_chebyu.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_sh_chebyu', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sh_chebyu', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sh_chebyu(...)' code ##################

        
        # Call to check_poly(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'orth' (line 118)
        orth_558508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'orth', False)
        # Obtaining the member 'eval_sh_chebyu' of a type (line 118)
        eval_sh_chebyu_558509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 24), orth_558508, 'eval_sh_chebyu')
        # Getting the type of 'orth' (line 118)
        orth_558510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 45), 'orth', False)
        # Obtaining the member 'sh_chebyu' of a type (line 118)
        sh_chebyu_558511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 45), orth_558510, 'sh_chebyu')
        # Processing the call keyword arguments (line 118)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_558512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        
        keyword_558513 = list_558512
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_558514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        int_558515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), list_558514, int_558515)
        # Adding element type (line 119)
        int_558516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), list_558514, int_558516)
        
        keyword_558517 = list_558514
        kwargs_558518 = {'param_ranges': keyword_558513, 'x_range': keyword_558517}
        # Getting the type of 'self' (line 118)
        self_558506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 118)
        check_poly_558507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_558506, 'check_poly')
        # Calling check_poly(args, kwargs) (line 118)
        check_poly_call_result_558519 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), check_poly_558507, *[eval_sh_chebyu_558509, sh_chebyu_558511], **kwargs_558518)
        
        
        # ################# End of 'test_sh_chebyu(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sh_chebyu' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_558520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558520)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sh_chebyu'
        return stypy_return_type_558520


    @norecursion
    def test_legendre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_legendre'
        module_type_store = module_type_store.open_function_context('test_legendre', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_legendre.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_legendre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_legendre.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_legendre.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_legendre')
        TestPolys.test_legendre.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_legendre.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_legendre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_legendre.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_legendre.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_legendre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_legendre.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_legendre', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_legendre', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_legendre(...)' code ##################

        
        # Call to check_poly(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'orth' (line 122)
        orth_558523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 'orth', False)
        # Obtaining the member 'eval_legendre' of a type (line 122)
        eval_legendre_558524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 24), orth_558523, 'eval_legendre')
        # Getting the type of 'orth' (line 122)
        orth_558525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 44), 'orth', False)
        # Obtaining the member 'legendre' of a type (line 122)
        legendre_558526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 44), orth_558525, 'legendre')
        # Processing the call keyword arguments (line 122)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_558527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        
        keyword_558528 = list_558527
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_558529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        int_558530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 44), list_558529, int_558530)
        # Adding element type (line 123)
        int_558531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 44), list_558529, int_558531)
        
        keyword_558532 = list_558529
        kwargs_558533 = {'param_ranges': keyword_558528, 'x_range': keyword_558532}
        # Getting the type of 'self' (line 122)
        self_558521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 122)
        check_poly_558522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_558521, 'check_poly')
        # Calling check_poly(args, kwargs) (line 122)
        check_poly_call_result_558534 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), check_poly_558522, *[eval_legendre_558524, legendre_558526], **kwargs_558533)
        
        
        # ################# End of 'test_legendre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_legendre' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_558535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558535)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_legendre'
        return stypy_return_type_558535


    @norecursion
    def test_sh_legendre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sh_legendre'
        module_type_store = module_type_store.open_function_context('test_sh_legendre', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_sh_legendre')
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_sh_legendre.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_sh_legendre', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sh_legendre', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sh_legendre(...)' code ##################

        
        # Assigning a Call to a Name (line 126):
        
        # Call to seterr(...): (line 126)
        # Processing the call keyword arguments (line 126)
        str_558538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 31), 'str', 'ignore')
        keyword_558539 = str_558538
        kwargs_558540 = {'all': keyword_558539}
        # Getting the type of 'np' (line 126)
        np_558536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'np', False)
        # Obtaining the member 'seterr' of a type (line 126)
        seterr_558537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), np_558536, 'seterr')
        # Calling seterr(args, kwargs) (line 126)
        seterr_call_result_558541 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), seterr_558537, *[], **kwargs_558540)
        
        # Assigning a type to the variable 'olderr' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'olderr', seterr_call_result_558541)
        
        # Try-finally block (line 127)
        
        # Call to check_poly(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'orth' (line 128)
        orth_558544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'orth', False)
        # Obtaining the member 'eval_sh_legendre' of a type (line 128)
        eval_sh_legendre_558545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 28), orth_558544, 'eval_sh_legendre')
        # Getting the type of 'orth' (line 128)
        orth_558546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 51), 'orth', False)
        # Obtaining the member 'sh_legendre' of a type (line 128)
        sh_legendre_558547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 51), orth_558546, 'sh_legendre')
        # Processing the call keyword arguments (line 128)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_558548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        
        keyword_558549 = list_558548
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_558550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        int_558551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 53), list_558550, int_558551)
        # Adding element type (line 129)
        int_558552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 53), list_558550, int_558552)
        
        keyword_558553 = list_558550
        kwargs_558554 = {'param_ranges': keyword_558549, 'x_range': keyword_558553}
        # Getting the type of 'self' (line 128)
        self_558542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 128)
        check_poly_558543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), self_558542, 'check_poly')
        # Calling check_poly(args, kwargs) (line 128)
        check_poly_call_result_558555 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), check_poly_558543, *[eval_sh_legendre_558545, sh_legendre_558547], **kwargs_558554)
        
        
        # finally branch of the try-finally block (line 127)
        
        # Call to seterr(...): (line 131)
        # Processing the call keyword arguments (line 131)
        # Getting the type of 'olderr' (line 131)
        olderr_558558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'olderr', False)
        kwargs_558559 = {'olderr_558558': olderr_558558}
        # Getting the type of 'np' (line 131)
        np_558556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'np', False)
        # Obtaining the member 'seterr' of a type (line 131)
        seterr_558557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), np_558556, 'seterr')
        # Calling seterr(args, kwargs) (line 131)
        seterr_call_result_558560 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), seterr_558557, *[], **kwargs_558559)
        
        
        
        # ################# End of 'test_sh_legendre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sh_legendre' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_558561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558561)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sh_legendre'
        return stypy_return_type_558561


    @norecursion
    def test_genlaguerre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_genlaguerre'
        module_type_store = module_type_store.open_function_context('test_genlaguerre', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_genlaguerre')
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_genlaguerre.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_genlaguerre', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_genlaguerre', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_genlaguerre(...)' code ##################

        
        # Call to check_poly(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'orth' (line 134)
        orth_558564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'orth', False)
        # Obtaining the member 'eval_genlaguerre' of a type (line 134)
        eval_genlaguerre_558565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 24), orth_558564, 'eval_genlaguerre')
        # Getting the type of 'orth' (line 134)
        orth_558566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 47), 'orth', False)
        # Obtaining the member 'genlaguerre' of a type (line 134)
        genlaguerre_558567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 47), orth_558566, 'genlaguerre')
        # Processing the call keyword arguments (line 134)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_558568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_558569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        float_558570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 34), tuple_558569, float_558570)
        # Adding element type (line 135)
        int_558571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 34), tuple_558569, int_558571)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 32), list_558568, tuple_558569)
        
        keyword_558572 = list_558568
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_558573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        int_558574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 55), list_558573, int_558574)
        # Adding element type (line 135)
        int_558575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 55), list_558573, int_558575)
        
        keyword_558576 = list_558573
        kwargs_558577 = {'param_ranges': keyword_558572, 'x_range': keyword_558576}
        # Getting the type of 'self' (line 134)
        self_558562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 134)
        check_poly_558563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_558562, 'check_poly')
        # Calling check_poly(args, kwargs) (line 134)
        check_poly_call_result_558578 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), check_poly_558563, *[eval_genlaguerre_558565, genlaguerre_558567], **kwargs_558577)
        
        
        # ################# End of 'test_genlaguerre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_genlaguerre' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_558579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_genlaguerre'
        return stypy_return_type_558579


    @norecursion
    def test_laguerre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_laguerre'
        module_type_store = module_type_store.open_function_context('test_laguerre', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_laguerre')
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_laguerre.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_laguerre', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_laguerre', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_laguerre(...)' code ##################

        
        # Call to check_poly(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'orth' (line 138)
        orth_558582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'orth', False)
        # Obtaining the member 'eval_laguerre' of a type (line 138)
        eval_laguerre_558583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 24), orth_558582, 'eval_laguerre')
        # Getting the type of 'orth' (line 138)
        orth_558584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 44), 'orth', False)
        # Obtaining the member 'laguerre' of a type (line 138)
        laguerre_558585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 44), orth_558584, 'laguerre')
        # Processing the call keyword arguments (line 138)
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_558586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        
        keyword_558587 = list_558586
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_558588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        int_558589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 44), list_558588, int_558589)
        # Adding element type (line 139)
        int_558590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 44), list_558588, int_558590)
        
        keyword_558591 = list_558588
        kwargs_558592 = {'param_ranges': keyword_558587, 'x_range': keyword_558591}
        # Getting the type of 'self' (line 138)
        self_558580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 138)
        check_poly_558581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_558580, 'check_poly')
        # Calling check_poly(args, kwargs) (line 138)
        check_poly_call_result_558593 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), check_poly_558581, *[eval_laguerre_558583, laguerre_558585], **kwargs_558592)
        
        
        # ################# End of 'test_laguerre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_laguerre' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_558594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_laguerre'
        return stypy_return_type_558594


    @norecursion
    def test_hermite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hermite'
        module_type_store = module_type_store.open_function_context('test_hermite', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_hermite.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_hermite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_hermite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_hermite.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_hermite')
        TestPolys.test_hermite.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_hermite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_hermite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_hermite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_hermite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_hermite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_hermite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_hermite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hermite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hermite(...)' code ##################

        
        # Call to check_poly(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'orth' (line 142)
        orth_558597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'orth', False)
        # Obtaining the member 'eval_hermite' of a type (line 142)
        eval_hermite_558598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 24), orth_558597, 'eval_hermite')
        # Getting the type of 'orth' (line 142)
        orth_558599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 43), 'orth', False)
        # Obtaining the member 'hermite' of a type (line 142)
        hermite_558600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 43), orth_558599, 'hermite')
        # Processing the call keyword arguments (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_558601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        
        keyword_558602 = list_558601
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_558603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        int_558604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 44), list_558603, int_558604)
        # Adding element type (line 143)
        int_558605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 44), list_558603, int_558605)
        
        keyword_558606 = list_558603
        kwargs_558607 = {'param_ranges': keyword_558602, 'x_range': keyword_558606}
        # Getting the type of 'self' (line 142)
        self_558595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 142)
        check_poly_558596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_558595, 'check_poly')
        # Calling check_poly(args, kwargs) (line 142)
        check_poly_call_result_558608 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), check_poly_558596, *[eval_hermite_558598, hermite_558600], **kwargs_558607)
        
        
        # ################# End of 'test_hermite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hermite' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_558609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558609)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hermite'
        return stypy_return_type_558609


    @norecursion
    def test_hermitenorm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hermitenorm'
        module_type_store = module_type_store.open_function_context('test_hermitenorm', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_localization', localization)
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_function_name', 'TestPolys.test_hermitenorm')
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_param_names_list', [])
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPolys.test_hermitenorm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.test_hermitenorm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hermitenorm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hermitenorm(...)' code ##################

        
        # Call to check_poly(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'orth' (line 146)
        orth_558612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'orth', False)
        # Obtaining the member 'eval_hermitenorm' of a type (line 146)
        eval_hermitenorm_558613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 24), orth_558612, 'eval_hermitenorm')
        # Getting the type of 'orth' (line 146)
        orth_558614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 47), 'orth', False)
        # Obtaining the member 'hermitenorm' of a type (line 146)
        hermitenorm_558615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 47), orth_558614, 'hermitenorm')
        # Processing the call keyword arguments (line 146)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_558616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        
        keyword_558617 = list_558616
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_558618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        int_558619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 49), list_558618, int_558619)
        # Adding element type (line 147)
        int_558620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 49), list_558618, int_558620)
        
        keyword_558621 = list_558618
        kwargs_558622 = {'param_ranges': keyword_558617, 'x_range': keyword_558621}
        # Getting the type of 'self' (line 146)
        self_558610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 146)
        check_poly_558611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_558610, 'check_poly')
        # Calling check_poly(args, kwargs) (line 146)
        check_poly_call_result_558623 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), check_poly_558611, *[eval_hermitenorm_558613, hermitenorm_558615], **kwargs_558622)
        
        
        # ################# End of 'test_hermitenorm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hermitenorm' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_558624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558624)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hermitenorm'
        return stypy_return_type_558624


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 36, 0, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPolys.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestPolys' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'TestPolys', TestPolys)
# Declaration of the 'TestRecurrence' class

class TestRecurrence(object, ):
    str_558625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, (-1)), 'str', "\n    Check that the eval_* functions sig='ld->d' and 'dd->d' agree.\n\n    ")

    @norecursion
    def check_poly(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_558626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_558627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        
        int_558628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 63), 'int')
        int_558629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 26), 'int')
        int_558630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 33), 'int')
        float_558631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 42), 'float')
        defaults = [list_558626, list_558627, int_558628, int_558629, int_558630, float_558631]
        # Create a new context for function 'check_poly'
        module_type_store = module_type_store.open_function_context('check_poly', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.check_poly')
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_param_names_list', ['func', 'param_ranges', 'x_range', 'nn', 'nparam', 'nx', 'rtol'])
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.check_poly.__dict__.__setitem__('stypy_declared_arg_number', 8)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.check_poly', ['func', 'param_ranges', 'x_range', 'nn', 'nparam', 'nx', 'rtol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_poly', localization, ['func', 'param_ranges', 'x_range', 'nn', 'nparam', 'nx', 'rtol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_poly(...)' code ##################

        
        # Call to seed(...): (line 158)
        # Processing the call arguments (line 158)
        int_558635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'int')
        # Processing the call keyword arguments (line 158)
        kwargs_558636 = {}
        # Getting the type of 'np' (line 158)
        np_558632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 158)
        random_558633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), np_558632, 'random')
        # Obtaining the member 'seed' of a type (line 158)
        seed_558634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), random_558633, 'seed')
        # Calling seed(args, kwargs) (line 158)
        seed_call_result_558637 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), seed_558634, *[int_558635], **kwargs_558636)
        
        
        # Assigning a List to a Name (line 160):
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_558638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        
        # Assigning a type to the variable 'dataset' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'dataset', list_558638)
        
        
        # Call to arange(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'nn' (line 161)
        nn_558641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'nn', False)
        # Processing the call keyword arguments (line 161)
        kwargs_558642 = {}
        # Getting the type of 'np' (line 161)
        np_558639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 161)
        arange_558640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 17), np_558639, 'arange')
        # Calling arange(args, kwargs) (line 161)
        arange_call_result_558643 = invoke(stypy.reporting.localization.Localization(__file__, 161, 17), arange_558640, *[nn_558641], **kwargs_558642)
        
        # Testing the type of a for loop iterable (line 161)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 161, 8), arange_call_result_558643)
        # Getting the type of the for loop variable (line 161)
        for_loop_var_558644 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 161, 8), arange_call_result_558643)
        # Assigning a type to the variable 'n' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'n', for_loop_var_558644)
        # SSA begins for a for statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Name (line 162):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'param_ranges' (line 162)
        param_ranges_558657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 66), 'param_ranges')
        comprehension_558658 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 22), param_ranges_558657)
        # Assigning a type to the variable 'a' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 22), comprehension_558658))
        # Assigning a type to the variable 'b' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 22), comprehension_558658))
        # Getting the type of 'a' (line 162)
        a_558645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'a')
        # Getting the type of 'b' (line 162)
        b_558646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'b')
        # Getting the type of 'a' (line 162)
        a_558647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'a')
        # Applying the binary operator '-' (line 162)
        result_sub_558648 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 27), '-', b_558646, a_558647)
        
        
        # Call to rand(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'nparam' (line 162)
        nparam_558652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 47), 'nparam', False)
        # Processing the call keyword arguments (line 162)
        kwargs_558653 = {}
        # Getting the type of 'np' (line 162)
        np_558649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'np', False)
        # Obtaining the member 'random' of a type (line 162)
        random_558650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 32), np_558649, 'random')
        # Obtaining the member 'rand' of a type (line 162)
        rand_558651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 32), random_558650, 'rand')
        # Calling rand(args, kwargs) (line 162)
        rand_call_result_558654 = invoke(stypy.reporting.localization.Localization(__file__, 162, 32), rand_558651, *[nparam_558652], **kwargs_558653)
        
        # Applying the binary operator '*' (line 162)
        result_mul_558655 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 26), '*', result_sub_558648, rand_call_result_558654)
        
        # Applying the binary operator '+' (line 162)
        result_add_558656 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 22), '+', a_558645, result_mul_558655)
        
        list_558659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 22), list_558659, result_add_558656)
        # Assigning a type to the variable 'params' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'params', list_558659)
        
        # Assigning a Attribute to a Name (line 163):
        
        # Call to asarray(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'params' (line 163)
        params_558662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 32), 'params', False)
        # Processing the call keyword arguments (line 163)
        kwargs_558663 = {}
        # Getting the type of 'np' (line 163)
        np_558660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'np', False)
        # Obtaining the member 'asarray' of a type (line 163)
        asarray_558661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 21), np_558660, 'asarray')
        # Calling asarray(args, kwargs) (line 163)
        asarray_call_result_558664 = invoke(stypy.reporting.localization.Localization(__file__, 163, 21), asarray_558661, *[params_558662], **kwargs_558663)
        
        # Obtaining the member 'T' of a type (line 163)
        T_558665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 21), asarray_call_result_558664, 'T')
        # Assigning a type to the variable 'params' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'params', T_558665)
        
        
        # Getting the type of 'param_ranges' (line 164)
        param_ranges_558666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'param_ranges')
        # Applying the 'not' unary operator (line 164)
        result_not__558667 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), 'not', param_ranges_558666)
        
        # Testing the type of an if condition (line 164)
        if_condition_558668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 12), result_not__558667)
        # Assigning a type to the variable 'if_condition_558668' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'if_condition_558668', if_condition_558668)
        # SSA begins for if statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 165):
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_558669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        int_558670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 25), list_558669, int_558670)
        
        # Assigning a type to the variable 'params' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'params', list_558669)
        # SSA join for if statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'params' (line 166)
        params_558671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'params')
        # Testing the type of a for loop iterable (line 166)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 166, 12), params_558671)
        # Getting the type of the for loop variable (line 166)
        for_loop_var_558672 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 166, 12), params_558671)
        # Assigning a type to the variable 'p' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'p', for_loop_var_558672)
        # SSA begins for a for statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'param_ranges' (line 167)
        param_ranges_558673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'param_ranges')
        # Testing the type of an if condition (line 167)
        if_condition_558674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 16), param_ranges_558673)
        # Assigning a type to the variable 'if_condition_558674' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'if_condition_558674', if_condition_558674)
        # SSA begins for if statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 168):
        
        # Obtaining an instance of the builtin type 'tuple' (line 168)
        tuple_558675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 168)
        # Adding element type (line 168)
        # Getting the type of 'n' (line 168)
        n_558676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 25), tuple_558675, n_558676)
        
        
        # Call to tuple(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'p' (line 168)
        p_558678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'p', False)
        # Processing the call keyword arguments (line 168)
        kwargs_558679 = {}
        # Getting the type of 'tuple' (line 168)
        tuple_558677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'tuple', False)
        # Calling tuple(args, kwargs) (line 168)
        tuple_call_result_558680 = invoke(stypy.reporting.localization.Localization(__file__, 168, 31), tuple_558677, *[p_558678], **kwargs_558679)
        
        # Applying the binary operator '+' (line 168)
        result_add_558681 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 24), '+', tuple_558675, tuple_call_result_558680)
        
        # Assigning a type to the variable 'p' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'p', result_add_558681)
        # SSA branch for the else part of an if statement (line 167)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Name (line 170):
        
        # Obtaining an instance of the builtin type 'tuple' (line 170)
        tuple_558682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 170)
        # Adding element type (line 170)
        # Getting the type of 'n' (line 170)
        n_558683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 25), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 25), tuple_558682, n_558683)
        
        # Assigning a type to the variable 'p' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'p', tuple_558682)
        # SSA join for if statement (line 167)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 171):
        
        # Obtaining the type of the subscript
        int_558684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 28), 'int')
        # Getting the type of 'x_range' (line 171)
        x_range_558685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___558686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 20), x_range_558685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_558687 = invoke(stypy.reporting.localization.Localization(__file__, 171, 20), getitem___558686, int_558684)
        
        
        # Obtaining the type of the subscript
        int_558688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 42), 'int')
        # Getting the type of 'x_range' (line 171)
        x_range_558689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 34), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___558690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 34), x_range_558689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_558691 = invoke(stypy.reporting.localization.Localization(__file__, 171, 34), getitem___558690, int_558688)
        
        
        # Obtaining the type of the subscript
        int_558692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 55), 'int')
        # Getting the type of 'x_range' (line 171)
        x_range_558693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 47), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___558694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 47), x_range_558693, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_558695 = invoke(stypy.reporting.localization.Localization(__file__, 171, 47), getitem___558694, int_558692)
        
        # Applying the binary operator '-' (line 171)
        result_sub_558696 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 34), '-', subscript_call_result_558691, subscript_call_result_558695)
        
        
        # Call to rand(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'nx' (line 171)
        nx_558700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 74), 'nx', False)
        # Processing the call keyword arguments (line 171)
        kwargs_558701 = {}
        # Getting the type of 'np' (line 171)
        np_558697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 59), 'np', False)
        # Obtaining the member 'random' of a type (line 171)
        random_558698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 59), np_558697, 'random')
        # Obtaining the member 'rand' of a type (line 171)
        rand_558699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 59), random_558698, 'rand')
        # Calling rand(args, kwargs) (line 171)
        rand_call_result_558702 = invoke(stypy.reporting.localization.Localization(__file__, 171, 59), rand_558699, *[nx_558700], **kwargs_558701)
        
        # Applying the binary operator '*' (line 171)
        result_mul_558703 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 33), '*', result_sub_558696, rand_call_result_558702)
        
        # Applying the binary operator '+' (line 171)
        result_add_558704 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 20), '+', subscript_call_result_558687, result_mul_558703)
        
        # Assigning a type to the variable 'x' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'x', result_add_558704)
        
        # Assigning a Subscript to a Subscript (line 172):
        
        # Obtaining the type of the subscript
        int_558705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 31), 'int')
        # Getting the type of 'x_range' (line 172)
        x_range_558706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___558707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 23), x_range_558706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_558708 = invoke(stypy.reporting.localization.Localization(__file__, 172, 23), getitem___558707, int_558705)
        
        # Getting the type of 'x' (line 172)
        x_558709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'x')
        int_558710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'int')
        # Storing an element on a container (line 172)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 16), x_558709, (int_558710, subscript_call_result_558708))
        
        # Assigning a Subscript to a Subscript (line 173):
        
        # Obtaining the type of the subscript
        int_558711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 31), 'int')
        # Getting the type of 'x_range' (line 173)
        x_range_558712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'x_range')
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___558713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 23), x_range_558712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_558714 = invoke(stypy.reporting.localization.Localization(__file__, 173, 23), getitem___558713, int_558711)
        
        # Getting the type of 'x' (line 173)
        x_558715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'x')
        int_558716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 18), 'int')
        # Storing an element on a container (line 173)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 16), x_558715, (int_558716, subscript_call_result_558714))
        
        # Assigning a Call to a Name (line 174):
        
        # Call to dict(...): (line 174)
        # Processing the call keyword arguments (line 174)
        
        # Call to len(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'p' (line 174)
        p_558719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 35), 'p', False)
        # Processing the call keyword arguments (line 174)
        kwargs_558720 = {}
        # Getting the type of 'len' (line 174)
        len_558718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'len', False)
        # Calling len(args, kwargs) (line 174)
        len_call_result_558721 = invoke(stypy.reporting.localization.Localization(__file__, 174, 31), len_558718, *[p_558719], **kwargs_558720)
        
        int_558722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 38), 'int')
        # Applying the binary operator '+' (line 174)
        result_add_558723 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 31), '+', len_call_result_558721, int_558722)
        
        str_558724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 41), 'str', 'd')
        # Applying the binary operator '*' (line 174)
        result_mul_558725 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 30), '*', result_add_558723, str_558724)
        
        str_558726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 45), 'str', '->d')
        # Applying the binary operator '+' (line 174)
        result_add_558727 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 30), '+', result_mul_558725, str_558726)
        
        keyword_558728 = result_add_558727
        kwargs_558729 = {'sig': keyword_558728}
        # Getting the type of 'dict' (line 174)
        dict_558717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'dict', False)
        # Calling dict(args, kwargs) (line 174)
        dict_call_result_558730 = invoke(stypy.reporting.localization.Localization(__file__, 174, 21), dict_558717, *[], **kwargs_558729)
        
        # Assigning a type to the variable 'kw' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'kw', dict_call_result_558730)
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 175)
        tuple_558731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 175)
        # Adding element type (line 175)
        
        # Call to tile(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'p' (line 175)
        p_558734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 34), 'p', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 175)
        tuple_558735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 175)
        # Adding element type (line 175)
        # Getting the type of 'nx' (line 175)
        nx_558736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 38), 'nx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 38), tuple_558735, nx_558736)
        # Adding element type (line 175)
        int_558737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 38), tuple_558735, int_558737)
        
        # Processing the call keyword arguments (line 175)
        kwargs_558738 = {}
        # Getting the type of 'np' (line 175)
        np_558732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'np', False)
        # Obtaining the member 'tile' of a type (line 175)
        tile_558733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 26), np_558732, 'tile')
        # Calling tile(args, kwargs) (line 175)
        tile_call_result_558739 = invoke(stypy.reporting.localization.Localization(__file__, 175, 26), tile_558733, *[p_558734, tuple_558735], **kwargs_558738)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 26), tuple_558731, tile_call_result_558739)
        # Adding element type (line 175)
        # Getting the type of 'x' (line 175)
        x_558740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 46), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 26), tuple_558731, x_558740)
        # Adding element type (line 175)
        
        # Call to func(...): (line 175)
        # Getting the type of 'p' (line 175)
        p_558742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 56), 'p', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 175)
        tuple_558743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 175)
        # Adding element type (line 175)
        # Getting the type of 'x' (line 175)
        x_558744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 61), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 61), tuple_558743, x_558744)
        
        # Applying the binary operator '+' (line 175)
        result_add_558745 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 56), '+', p_558742, tuple_558743)
        
        # Processing the call keyword arguments (line 175)
        # Getting the type of 'kw' (line 175)
        kw_558746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 69), 'kw', False)
        kwargs_558747 = {'kw_558746': kw_558746}
        # Getting the type of 'func' (line 175)
        func_558741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 49), 'func', False)
        # Calling func(args, kwargs) (line 175)
        func_call_result_558748 = invoke(stypy.reporting.localization.Localization(__file__, 175, 49), func_558741, *[result_add_558745], **kwargs_558747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 26), tuple_558731, func_call_result_558748)
        
        # Getting the type of 'np' (line 175)
        np_558749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'np')
        # Obtaining the member 'c_' of a type (line 175)
        c__558750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 20), np_558749, 'c_')
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___558751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 20), c__558750, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_558752 = invoke(stypy.reporting.localization.Localization(__file__, 175, 20), getitem___558751, tuple_558731)
        
        # Assigning a type to the variable 'z' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'z', subscript_call_result_558752)
        
        # Call to append(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'z' (line 176)
        z_558755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 31), 'z', False)
        # Processing the call keyword arguments (line 176)
        kwargs_558756 = {}
        # Getting the type of 'dataset' (line 176)
        dataset_558753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'dataset', False)
        # Obtaining the member 'append' of a type (line 176)
        append_558754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), dataset_558753, 'append')
        # Calling append(args, kwargs) (line 176)
        append_call_result_558757 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), append_558754, *[z_558755], **kwargs_558756)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 178):
        
        # Call to concatenate(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'dataset' (line 178)
        dataset_558760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 33), 'dataset', False)
        # Processing the call keyword arguments (line 178)
        int_558761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 47), 'int')
        keyword_558762 = int_558761
        kwargs_558763 = {'axis': keyword_558762}
        # Getting the type of 'np' (line 178)
        np_558758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 178)
        concatenate_558759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 18), np_558758, 'concatenate')
        # Calling concatenate(args, kwargs) (line 178)
        concatenate_call_result_558764 = invoke(stypy.reporting.localization.Localization(__file__, 178, 18), concatenate_558759, *[dataset_558760], **kwargs_558763)
        
        # Assigning a type to the variable 'dataset' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'dataset', concatenate_call_result_558764)

        @norecursion
        def polyfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'polyfunc'
            module_type_store = module_type_store.open_function_context('polyfunc', 180, 8, False)
            
            # Passed parameters checking function
            polyfunc.stypy_localization = localization
            polyfunc.stypy_type_of_self = None
            polyfunc.stypy_type_store = module_type_store
            polyfunc.stypy_function_name = 'polyfunc'
            polyfunc.stypy_param_names_list = []
            polyfunc.stypy_varargs_param_name = 'p'
            polyfunc.stypy_kwargs_param_name = None
            polyfunc.stypy_call_defaults = defaults
            polyfunc.stypy_call_varargs = varargs
            polyfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'polyfunc', [], 'p', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'polyfunc', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'polyfunc(...)' code ##################

            
            # Assigning a BinOp to a Name (line 181):
            
            # Obtaining an instance of the builtin type 'tuple' (line 181)
            tuple_558765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 17), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 181)
            # Adding element type (line 181)
            
            # Call to astype(...): (line 181)
            # Processing the call arguments (line 181)
            # Getting the type of 'int' (line 181)
            int_558771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'int', False)
            # Processing the call keyword arguments (line 181)
            kwargs_558772 = {}
            
            # Obtaining the type of the subscript
            int_558766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 19), 'int')
            # Getting the type of 'p' (line 181)
            p_558767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 17), 'p', False)
            # Obtaining the member '__getitem__' of a type (line 181)
            getitem___558768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 17), p_558767, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 181)
            subscript_call_result_558769 = invoke(stypy.reporting.localization.Localization(__file__, 181, 17), getitem___558768, int_558766)
            
            # Obtaining the member 'astype' of a type (line 181)
            astype_558770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 17), subscript_call_result_558769, 'astype')
            # Calling astype(args, kwargs) (line 181)
            astype_call_result_558773 = invoke(stypy.reporting.localization.Localization(__file__, 181, 17), astype_558770, *[int_558771], **kwargs_558772)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), tuple_558765, astype_call_result_558773)
            
            
            # Obtaining the type of the subscript
            int_558774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 40), 'int')
            slice_558775 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 181, 38), int_558774, None, None)
            # Getting the type of 'p' (line 181)
            p_558776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 38), 'p')
            # Obtaining the member '__getitem__' of a type (line 181)
            getitem___558777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 38), p_558776, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 181)
            subscript_call_result_558778 = invoke(stypy.reporting.localization.Localization(__file__, 181, 38), getitem___558777, slice_558775)
            
            # Applying the binary operator '+' (line 181)
            result_add_558779 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 16), '+', tuple_558765, subscript_call_result_558778)
            
            # Assigning a type to the variable 'p' (line 181)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'p', result_add_558779)
            
            # Assigning a Call to a Name (line 182):
            
            # Call to dict(...): (line 182)
            # Processing the call keyword arguments (line 182)
            str_558781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 26), 'str', 'l')
            
            # Call to len(...): (line 182)
            # Processing the call arguments (line 182)
            # Getting the type of 'p' (line 182)
            p_558783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 35), 'p', False)
            # Processing the call keyword arguments (line 182)
            kwargs_558784 = {}
            # Getting the type of 'len' (line 182)
            len_558782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'len', False)
            # Calling len(args, kwargs) (line 182)
            len_call_result_558785 = invoke(stypy.reporting.localization.Localization(__file__, 182, 31), len_558782, *[p_558783], **kwargs_558784)
            
            int_558786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 38), 'int')
            # Applying the binary operator '-' (line 182)
            result_sub_558787 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 31), '-', len_call_result_558785, int_558786)
            
            str_558788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 41), 'str', 'd')
            # Applying the binary operator '*' (line 182)
            result_mul_558789 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 30), '*', result_sub_558787, str_558788)
            
            # Applying the binary operator '+' (line 182)
            result_add_558790 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 26), '+', str_558781, result_mul_558789)
            
            str_558791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 45), 'str', '->d')
            # Applying the binary operator '+' (line 182)
            result_add_558792 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 44), '+', result_add_558790, str_558791)
            
            keyword_558793 = result_add_558792
            kwargs_558794 = {'sig': keyword_558793}
            # Getting the type of 'dict' (line 182)
            dict_558780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'dict', False)
            # Calling dict(args, kwargs) (line 182)
            dict_call_result_558795 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), dict_558780, *[], **kwargs_558794)
            
            # Assigning a type to the variable 'kw' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'kw', dict_call_result_558795)
            
            # Call to func(...): (line 183)
            # Getting the type of 'p' (line 183)
            p_558797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 25), 'p', False)
            # Processing the call keyword arguments (line 183)
            # Getting the type of 'kw' (line 183)
            kw_558798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 30), 'kw', False)
            kwargs_558799 = {'kw_558798': kw_558798}
            # Getting the type of 'func' (line 183)
            func_558796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'func', False)
            # Calling func(args, kwargs) (line 183)
            func_call_result_558800 = invoke(stypy.reporting.localization.Localization(__file__, 183, 19), func_558796, *[p_558797], **kwargs_558799)
            
            # Assigning a type to the variable 'stypy_return_type' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'stypy_return_type', func_call_result_558800)
            
            # ################# End of 'polyfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'polyfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 180)
            stypy_return_type_558801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_558801)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'polyfunc'
            return stypy_return_type_558801

        # Assigning a type to the variable 'polyfunc' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'polyfunc', polyfunc)
        
        # Assigning a Call to a Name (line 185):
        
        # Call to seterr(...): (line 185)
        # Processing the call keyword arguments (line 185)
        str_558804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 31), 'str', 'raise')
        keyword_558805 = str_558804
        kwargs_558806 = {'all': keyword_558805}
        # Getting the type of 'np' (line 185)
        np_558802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'np', False)
        # Obtaining the member 'seterr' of a type (line 185)
        seterr_558803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 17), np_558802, 'seterr')
        # Calling seterr(args, kwargs) (line 185)
        seterr_call_result_558807 = invoke(stypy.reporting.localization.Localization(__file__, 185, 17), seterr_558803, *[], **kwargs_558806)
        
        # Assigning a type to the variable 'olderr' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'olderr', seterr_call_result_558807)
        
        # Try-finally block (line 186)
        
        # Assigning a Call to a Name (line 187):
        
        # Call to FuncData(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'polyfunc' (line 187)
        polyfunc_558809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 26), 'polyfunc', False)
        # Getting the type of 'dataset' (line 187)
        dataset_558810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 36), 'dataset', False)
        
        # Call to list(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Call to range(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Call to len(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'param_ranges' (line 187)
        param_ranges_558814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 60), 'param_ranges', False)
        # Processing the call keyword arguments (line 187)
        kwargs_558815 = {}
        # Getting the type of 'len' (line 187)
        len_558813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 56), 'len', False)
        # Calling len(args, kwargs) (line 187)
        len_call_result_558816 = invoke(stypy.reporting.localization.Localization(__file__, 187, 56), len_558813, *[param_ranges_558814], **kwargs_558815)
        
        int_558817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 74), 'int')
        # Applying the binary operator '+' (line 187)
        result_add_558818 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 56), '+', len_call_result_558816, int_558817)
        
        # Processing the call keyword arguments (line 187)
        kwargs_558819 = {}
        # Getting the type of 'range' (line 187)
        range_558812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 50), 'range', False)
        # Calling range(args, kwargs) (line 187)
        range_call_result_558820 = invoke(stypy.reporting.localization.Localization(__file__, 187, 50), range_558812, *[result_add_558818], **kwargs_558819)
        
        # Processing the call keyword arguments (line 187)
        kwargs_558821 = {}
        # Getting the type of 'list' (line 187)
        list_558811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 45), 'list', False)
        # Calling list(args, kwargs) (line 187)
        list_call_result_558822 = invoke(stypy.reporting.localization.Localization(__file__, 187, 45), list_558811, *[range_call_result_558820], **kwargs_558821)
        
        int_558823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 79), 'int')
        # Processing the call keyword arguments (line 187)
        # Getting the type of 'rtol' (line 188)
        rtol_558824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'rtol', False)
        keyword_558825 = rtol_558824
        kwargs_558826 = {'rtol': keyword_558825}
        # Getting the type of 'FuncData' (line 187)
        FuncData_558808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 17), 'FuncData', False)
        # Calling FuncData(args, kwargs) (line 187)
        FuncData_call_result_558827 = invoke(stypy.reporting.localization.Localization(__file__, 187, 17), FuncData_558808, *[polyfunc_558809, dataset_558810, list_call_result_558822, int_558823], **kwargs_558826)
        
        # Assigning a type to the variable 'ds' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'ds', FuncData_call_result_558827)
        
        # Call to check(...): (line 189)
        # Processing the call keyword arguments (line 189)
        kwargs_558830 = {}
        # Getting the type of 'ds' (line 189)
        ds_558828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'ds', False)
        # Obtaining the member 'check' of a type (line 189)
        check_558829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), ds_558828, 'check')
        # Calling check(args, kwargs) (line 189)
        check_call_result_558831 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), check_558829, *[], **kwargs_558830)
        
        
        # finally branch of the try-finally block (line 186)
        
        # Call to seterr(...): (line 191)
        # Processing the call keyword arguments (line 191)
        # Getting the type of 'olderr' (line 191)
        olderr_558834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'olderr', False)
        kwargs_558835 = {'olderr_558834': olderr_558834}
        # Getting the type of 'np' (line 191)
        np_558832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'np', False)
        # Obtaining the member 'seterr' of a type (line 191)
        seterr_558833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), np_558832, 'seterr')
        # Calling seterr(args, kwargs) (line 191)
        seterr_call_result_558836 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), seterr_558833, *[], **kwargs_558835)
        
        
        
        # ################# End of 'check_poly(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_poly' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_558837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_poly'
        return stypy_return_type_558837


    @norecursion
    def test_jacobi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_jacobi'
        module_type_store = module_type_store.open_function_context('test_jacobi', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_jacobi')
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_jacobi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_jacobi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_jacobi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_jacobi(...)' code ##################

        
        # Call to check_poly(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'orth' (line 194)
        orth_558840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'orth', False)
        # Obtaining the member 'eval_jacobi' of a type (line 194)
        eval_jacobi_558841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 24), orth_558840, 'eval_jacobi')
        # Processing the call keyword arguments (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_558842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        
        # Obtaining an instance of the builtin type 'tuple' (line 195)
        tuple_558843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 195)
        # Adding element type (line 195)
        float_558844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 34), tuple_558843, float_558844)
        # Adding element type (line 195)
        int_558845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 34), tuple_558843, int_558845)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 32), list_558842, tuple_558843)
        # Adding element type (line 195)
        
        # Obtaining an instance of the builtin type 'tuple' (line 195)
        tuple_558846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 195)
        # Adding element type (line 195)
        float_558847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 47), tuple_558846, float_558847)
        # Adding element type (line 195)
        int_558848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 47), tuple_558846, int_558848)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 32), list_558842, tuple_558846)
        
        keyword_558849 = list_558842
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_558850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 68), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        int_558851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 68), list_558850, int_558851)
        # Adding element type (line 195)
        int_558852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 73), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 68), list_558850, int_558852)
        
        keyword_558853 = list_558850
        kwargs_558854 = {'param_ranges': keyword_558849, 'x_range': keyword_558853}
        # Getting the type of 'self' (line 194)
        self_558838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 194)
        check_poly_558839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_558838, 'check_poly')
        # Calling check_poly(args, kwargs) (line 194)
        check_poly_call_result_558855 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), check_poly_558839, *[eval_jacobi_558841], **kwargs_558854)
        
        
        # ################# End of 'test_jacobi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_jacobi' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_558856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558856)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_jacobi'
        return stypy_return_type_558856


    @norecursion
    def test_sh_jacobi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sh_jacobi'
        module_type_store = module_type_store.open_function_context('test_sh_jacobi', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_sh_jacobi')
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_sh_jacobi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_sh_jacobi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sh_jacobi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sh_jacobi(...)' code ##################

        
        # Call to check_poly(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'orth' (line 198)
        orth_558859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'orth', False)
        # Obtaining the member 'eval_sh_jacobi' of a type (line 198)
        eval_sh_jacobi_558860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 24), orth_558859, 'eval_sh_jacobi')
        # Processing the call keyword arguments (line 198)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_558861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        
        # Obtaining an instance of the builtin type 'tuple' (line 199)
        tuple_558862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 199)
        # Adding element type (line 199)
        int_558863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 34), tuple_558862, int_558863)
        # Adding element type (line 199)
        int_558864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 34), tuple_558862, int_558864)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 32), list_558861, tuple_558862)
        # Adding element type (line 199)
        
        # Obtaining an instance of the builtin type 'tuple' (line 199)
        tuple_558865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 199)
        # Adding element type (line 199)
        int_558866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 43), tuple_558865, int_558866)
        # Adding element type (line 199)
        int_558867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 43), tuple_558865, int_558867)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 32), list_558861, tuple_558865)
        
        keyword_558868 = list_558861
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_558869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        int_558870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 59), list_558869, int_558870)
        # Adding element type (line 199)
        int_558871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 59), list_558869, int_558871)
        
        keyword_558872 = list_558869
        kwargs_558873 = {'param_ranges': keyword_558868, 'x_range': keyword_558872}
        # Getting the type of 'self' (line 198)
        self_558857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 198)
        check_poly_558858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_558857, 'check_poly')
        # Calling check_poly(args, kwargs) (line 198)
        check_poly_call_result_558874 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), check_poly_558858, *[eval_sh_jacobi_558860], **kwargs_558873)
        
        
        # ################# End of 'test_sh_jacobi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sh_jacobi' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_558875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sh_jacobi'
        return stypy_return_type_558875


    @norecursion
    def test_gegenbauer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gegenbauer'
        module_type_store = module_type_store.open_function_context('test_gegenbauer', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_gegenbauer')
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_gegenbauer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_gegenbauer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gegenbauer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gegenbauer(...)' code ##################

        
        # Call to check_poly(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'orth' (line 202)
        orth_558878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'orth', False)
        # Obtaining the member 'eval_gegenbauer' of a type (line 202)
        eval_gegenbauer_558879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 24), orth_558878, 'eval_gegenbauer')
        # Processing the call keyword arguments (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_558880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        
        # Obtaining an instance of the builtin type 'tuple' (line 203)
        tuple_558881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 203)
        # Adding element type (line 203)
        float_558882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 34), tuple_558881, float_558882)
        # Adding element type (line 203)
        int_558883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 34), tuple_558881, int_558883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 32), list_558880, tuple_558881)
        
        keyword_558884 = list_558880
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_558885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        int_558886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 56), list_558885, int_558886)
        # Adding element type (line 203)
        int_558887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 56), list_558885, int_558887)
        
        keyword_558888 = list_558885
        kwargs_558889 = {'param_ranges': keyword_558884, 'x_range': keyword_558888}
        # Getting the type of 'self' (line 202)
        self_558876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 202)
        check_poly_558877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_558876, 'check_poly')
        # Calling check_poly(args, kwargs) (line 202)
        check_poly_call_result_558890 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), check_poly_558877, *[eval_gegenbauer_558879], **kwargs_558889)
        
        
        # ################# End of 'test_gegenbauer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gegenbauer' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_558891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558891)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gegenbauer'
        return stypy_return_type_558891


    @norecursion
    def test_chebyt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chebyt'
        module_type_store = module_type_store.open_function_context('test_chebyt', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_chebyt')
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_chebyt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_chebyt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chebyt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chebyt(...)' code ##################

        
        # Call to check_poly(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'orth' (line 206)
        orth_558894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'orth', False)
        # Obtaining the member 'eval_chebyt' of a type (line 206)
        eval_chebyt_558895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 24), orth_558894, 'eval_chebyt')
        # Processing the call keyword arguments (line 206)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_558896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        
        keyword_558897 = list_558896
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_558898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        int_558899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 44), list_558898, int_558899)
        # Adding element type (line 207)
        int_558900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 44), list_558898, int_558900)
        
        keyword_558901 = list_558898
        kwargs_558902 = {'param_ranges': keyword_558897, 'x_range': keyword_558901}
        # Getting the type of 'self' (line 206)
        self_558892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 206)
        check_poly_558893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_558892, 'check_poly')
        # Calling check_poly(args, kwargs) (line 206)
        check_poly_call_result_558903 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), check_poly_558893, *[eval_chebyt_558895], **kwargs_558902)
        
        
        # ################# End of 'test_chebyt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chebyt' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_558904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558904)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chebyt'
        return stypy_return_type_558904


    @norecursion
    def test_chebyu(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chebyu'
        module_type_store = module_type_store.open_function_context('test_chebyu', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_chebyu')
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_chebyu.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_chebyu', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chebyu', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chebyu(...)' code ##################

        
        # Call to check_poly(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'orth' (line 210)
        orth_558907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), 'orth', False)
        # Obtaining the member 'eval_chebyu' of a type (line 210)
        eval_chebyu_558908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 24), orth_558907, 'eval_chebyu')
        # Processing the call keyword arguments (line 210)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_558909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        
        keyword_558910 = list_558909
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_558911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        int_558912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 44), list_558911, int_558912)
        # Adding element type (line 211)
        int_558913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 44), list_558911, int_558913)
        
        keyword_558914 = list_558911
        kwargs_558915 = {'param_ranges': keyword_558910, 'x_range': keyword_558914}
        # Getting the type of 'self' (line 210)
        self_558905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 210)
        check_poly_558906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_558905, 'check_poly')
        # Calling check_poly(args, kwargs) (line 210)
        check_poly_call_result_558916 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), check_poly_558906, *[eval_chebyu_558908], **kwargs_558915)
        
        
        # ################# End of 'test_chebyu(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chebyu' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_558917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558917)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chebyu'
        return stypy_return_type_558917


    @norecursion
    def test_chebys(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chebys'
        module_type_store = module_type_store.open_function_context('test_chebys', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_chebys')
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_chebys.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_chebys', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chebys', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chebys(...)' code ##################

        
        # Call to check_poly(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'orth' (line 214)
        orth_558920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'orth', False)
        # Obtaining the member 'eval_chebys' of a type (line 214)
        eval_chebys_558921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 24), orth_558920, 'eval_chebys')
        # Processing the call keyword arguments (line 214)
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_558922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        
        keyword_558923 = list_558922
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_558924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        # Adding element type (line 215)
        int_558925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 44), list_558924, int_558925)
        # Adding element type (line 215)
        int_558926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 44), list_558924, int_558926)
        
        keyword_558927 = list_558924
        kwargs_558928 = {'param_ranges': keyword_558923, 'x_range': keyword_558927}
        # Getting the type of 'self' (line 214)
        self_558918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 214)
        check_poly_558919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), self_558918, 'check_poly')
        # Calling check_poly(args, kwargs) (line 214)
        check_poly_call_result_558929 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), check_poly_558919, *[eval_chebys_558921], **kwargs_558928)
        
        
        # ################# End of 'test_chebys(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chebys' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_558930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chebys'
        return stypy_return_type_558930


    @norecursion
    def test_chebyc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_chebyc'
        module_type_store = module_type_store.open_function_context('test_chebyc', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_chebyc')
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_chebyc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_chebyc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_chebyc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_chebyc(...)' code ##################

        
        # Call to check_poly(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'orth' (line 218)
        orth_558933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'orth', False)
        # Obtaining the member 'eval_chebyc' of a type (line 218)
        eval_chebyc_558934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 24), orth_558933, 'eval_chebyc')
        # Processing the call keyword arguments (line 218)
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_558935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        
        keyword_558936 = list_558935
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_558937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        int_558938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 44), list_558937, int_558938)
        # Adding element type (line 219)
        int_558939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 44), list_558937, int_558939)
        
        keyword_558940 = list_558937
        kwargs_558941 = {'param_ranges': keyword_558936, 'x_range': keyword_558940}
        # Getting the type of 'self' (line 218)
        self_558931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 218)
        check_poly_558932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_558931, 'check_poly')
        # Calling check_poly(args, kwargs) (line 218)
        check_poly_call_result_558942 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), check_poly_558932, *[eval_chebyc_558934], **kwargs_558941)
        
        
        # ################# End of 'test_chebyc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_chebyc' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_558943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558943)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_chebyc'
        return stypy_return_type_558943


    @norecursion
    def test_sh_chebyt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sh_chebyt'
        module_type_store = module_type_store.open_function_context('test_sh_chebyt', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_sh_chebyt')
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_sh_chebyt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_sh_chebyt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sh_chebyt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sh_chebyt(...)' code ##################

        
        # Call to check_poly(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'orth' (line 222)
        orth_558946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'orth', False)
        # Obtaining the member 'eval_sh_chebyt' of a type (line 222)
        eval_sh_chebyt_558947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 24), orth_558946, 'eval_sh_chebyt')
        # Processing the call keyword arguments (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_558948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        
        keyword_558949 = list_558948
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_558950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        int_558951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 44), list_558950, int_558951)
        # Adding element type (line 223)
        int_558952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 44), list_558950, int_558952)
        
        keyword_558953 = list_558950
        kwargs_558954 = {'param_ranges': keyword_558949, 'x_range': keyword_558953}
        # Getting the type of 'self' (line 222)
        self_558944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 222)
        check_poly_558945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_558944, 'check_poly')
        # Calling check_poly(args, kwargs) (line 222)
        check_poly_call_result_558955 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), check_poly_558945, *[eval_sh_chebyt_558947], **kwargs_558954)
        
        
        # ################# End of 'test_sh_chebyt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sh_chebyt' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_558956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558956)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sh_chebyt'
        return stypy_return_type_558956


    @norecursion
    def test_sh_chebyu(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sh_chebyu'
        module_type_store = module_type_store.open_function_context('test_sh_chebyu', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_sh_chebyu')
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_sh_chebyu.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_sh_chebyu', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sh_chebyu', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sh_chebyu(...)' code ##################

        
        # Call to check_poly(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'orth' (line 226)
        orth_558959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'orth', False)
        # Obtaining the member 'eval_sh_chebyu' of a type (line 226)
        eval_sh_chebyu_558960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 24), orth_558959, 'eval_sh_chebyu')
        # Processing the call keyword arguments (line 226)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_558961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        
        keyword_558962 = list_558961
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_558963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        int_558964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 44), list_558963, int_558964)
        # Adding element type (line 227)
        int_558965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 44), list_558963, int_558965)
        
        keyword_558966 = list_558963
        kwargs_558967 = {'param_ranges': keyword_558962, 'x_range': keyword_558966}
        # Getting the type of 'self' (line 226)
        self_558957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 226)
        check_poly_558958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), self_558957, 'check_poly')
        # Calling check_poly(args, kwargs) (line 226)
        check_poly_call_result_558968 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), check_poly_558958, *[eval_sh_chebyu_558960], **kwargs_558967)
        
        
        # ################# End of 'test_sh_chebyu(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sh_chebyu' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_558969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558969)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sh_chebyu'
        return stypy_return_type_558969


    @norecursion
    def test_legendre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_legendre'
        module_type_store = module_type_store.open_function_context('test_legendre', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_legendre')
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_legendre.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_legendre', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_legendre', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_legendre(...)' code ##################

        
        # Call to check_poly(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'orth' (line 230)
        orth_558972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), 'orth', False)
        # Obtaining the member 'eval_legendre' of a type (line 230)
        eval_legendre_558973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 24), orth_558972, 'eval_legendre')
        # Processing the call keyword arguments (line 230)
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_558974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        
        keyword_558975 = list_558974
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_558976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        int_558977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 44), list_558976, int_558977)
        # Adding element type (line 231)
        int_558978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 44), list_558976, int_558978)
        
        keyword_558979 = list_558976
        kwargs_558980 = {'param_ranges': keyword_558975, 'x_range': keyword_558979}
        # Getting the type of 'self' (line 230)
        self_558970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 230)
        check_poly_558971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_558970, 'check_poly')
        # Calling check_poly(args, kwargs) (line 230)
        check_poly_call_result_558981 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), check_poly_558971, *[eval_legendre_558973], **kwargs_558980)
        
        
        # ################# End of 'test_legendre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_legendre' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_558982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558982)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_legendre'
        return stypy_return_type_558982


    @norecursion
    def test_sh_legendre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sh_legendre'
        module_type_store = module_type_store.open_function_context('test_sh_legendre', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_sh_legendre')
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_sh_legendre.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_sh_legendre', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sh_legendre', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sh_legendre(...)' code ##################

        
        # Call to check_poly(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'orth' (line 234)
        orth_558985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'orth', False)
        # Obtaining the member 'eval_sh_legendre' of a type (line 234)
        eval_sh_legendre_558986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 24), orth_558985, 'eval_sh_legendre')
        # Processing the call keyword arguments (line 234)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_558987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        
        keyword_558988 = list_558987
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_558989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        int_558990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 44), list_558989, int_558990)
        # Adding element type (line 235)
        int_558991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 44), list_558989, int_558991)
        
        keyword_558992 = list_558989
        kwargs_558993 = {'param_ranges': keyword_558988, 'x_range': keyword_558992}
        # Getting the type of 'self' (line 234)
        self_558983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 234)
        check_poly_558984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_558983, 'check_poly')
        # Calling check_poly(args, kwargs) (line 234)
        check_poly_call_result_558994 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), check_poly_558984, *[eval_sh_legendre_558986], **kwargs_558993)
        
        
        # ################# End of 'test_sh_legendre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sh_legendre' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_558995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_558995)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sh_legendre'
        return stypy_return_type_558995


    @norecursion
    def test_genlaguerre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_genlaguerre'
        module_type_store = module_type_store.open_function_context('test_genlaguerre', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_genlaguerre')
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_genlaguerre.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_genlaguerre', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_genlaguerre', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_genlaguerre(...)' code ##################

        
        # Call to check_poly(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'orth' (line 238)
        orth_558998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'orth', False)
        # Obtaining the member 'eval_genlaguerre' of a type (line 238)
        eval_genlaguerre_558999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 24), orth_558998, 'eval_genlaguerre')
        # Processing the call keyword arguments (line 238)
        
        # Obtaining an instance of the builtin type 'list' (line 239)
        list_559000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 239)
        # Adding element type (line 239)
        
        # Obtaining an instance of the builtin type 'tuple' (line 239)
        tuple_559001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 239)
        # Adding element type (line 239)
        float_559002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 34), tuple_559001, float_559002)
        # Adding element type (line 239)
        int_559003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 34), tuple_559001, int_559003)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 32), list_559000, tuple_559001)
        
        keyword_559004 = list_559000
        
        # Obtaining an instance of the builtin type 'list' (line 239)
        list_559005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 239)
        # Adding element type (line 239)
        int_559006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 55), list_559005, int_559006)
        # Adding element type (line 239)
        int_559007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 55), list_559005, int_559007)
        
        keyword_559008 = list_559005
        kwargs_559009 = {'param_ranges': keyword_559004, 'x_range': keyword_559008}
        # Getting the type of 'self' (line 238)
        self_558996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 238)
        check_poly_558997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_558996, 'check_poly')
        # Calling check_poly(args, kwargs) (line 238)
        check_poly_call_result_559010 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), check_poly_558997, *[eval_genlaguerre_558999], **kwargs_559009)
        
        
        # ################# End of 'test_genlaguerre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_genlaguerre' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_559011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559011)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_genlaguerre'
        return stypy_return_type_559011


    @norecursion
    def test_laguerre(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_laguerre'
        module_type_store = module_type_store.open_function_context('test_laguerre', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_laguerre')
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_laguerre.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_laguerre', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_laguerre', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_laguerre(...)' code ##################

        
        # Call to check_poly(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'orth' (line 242)
        orth_559014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'orth', False)
        # Obtaining the member 'eval_laguerre' of a type (line 242)
        eval_laguerre_559015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 24), orth_559014, 'eval_laguerre')
        # Processing the call keyword arguments (line 242)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_559016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        
        keyword_559017 = list_559016
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_559018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        int_559019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), list_559018, int_559019)
        # Adding element type (line 243)
        int_559020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), list_559018, int_559020)
        
        keyword_559021 = list_559018
        kwargs_559022 = {'param_ranges': keyword_559017, 'x_range': keyword_559021}
        # Getting the type of 'self' (line 242)
        self_559012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self', False)
        # Obtaining the member 'check_poly' of a type (line 242)
        check_poly_559013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_559012, 'check_poly')
        # Calling check_poly(args, kwargs) (line 242)
        check_poly_call_result_559023 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), check_poly_559013, *[eval_laguerre_559015], **kwargs_559022)
        
        
        # ################# End of 'test_laguerre(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_laguerre' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_559024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559024)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_laguerre'
        return stypy_return_type_559024


    @norecursion
    def test_hermite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hermite'
        module_type_store = module_type_store.open_function_context('test_hermite', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_localization', localization)
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_function_name', 'TestRecurrence.test_hermite')
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_param_names_list', [])
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRecurrence.test_hermite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.test_hermite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hermite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hermite(...)' code ##################

        
        # Assigning a Call to a Name (line 246):
        
        # Call to eval_hermite(...): (line 246)
        # Processing the call arguments (line 246)
        int_559027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 30), 'int')
        float_559028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 34), 'float')
        # Processing the call keyword arguments (line 246)
        kwargs_559029 = {}
        # Getting the type of 'orth' (line 246)
        orth_559025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'orth', False)
        # Obtaining the member 'eval_hermite' of a type (line 246)
        eval_hermite_559026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), orth_559025, 'eval_hermite')
        # Calling eval_hermite(args, kwargs) (line 246)
        eval_hermite_call_result_559030 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), eval_hermite_559026, *[int_559027, float_559028], **kwargs_559029)
        
        # Assigning a type to the variable 'v' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'v', eval_hermite_call_result_559030)
        
        # Assigning a Num to a Name (line 247):
        float_559031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 12), 'float')
        # Assigning a type to the variable 'a' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'a', float_559031)
        
        # Call to assert_allclose(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'v' (line 248)
        v_559033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'v', False)
        # Getting the type of 'a' (line 248)
        a_559034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 26), 'a', False)
        # Processing the call keyword arguments (line 248)
        kwargs_559035 = {}
        # Getting the type of 'assert_allclose' (line 248)
        assert_allclose_559032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 248)
        assert_allclose_call_result_559036 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), assert_allclose_559032, *[v_559033, a_559034], **kwargs_559035)
        
        
        # ################# End of 'test_hermite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hermite' in the type store
        # Getting the type of 'stypy_return_type' (line 245)
        stypy_return_type_559037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559037)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hermite'
        return stypy_return_type_559037


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 150, 0, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRecurrence.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestRecurrence' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'TestRecurrence', TestRecurrence)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

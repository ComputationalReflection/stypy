
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import itertools
4: 
5: import numpy as np
6: from numpy import exp
7: from numpy.testing import assert_, assert_equal
8: 
9: from scipy.optimize import root
10: 
11: 
12: def test_performance():
13:     # Compare performance results to those listed in
14:     # [Cheng & Li, IMA J. Num. An. 29, 814 (2008)]
15:     # and
16:     # [W. La Cruz, J.M. Martinez, M. Raydan, Math. Comp. 75, 1429 (2006)].
17:     # and those produced by dfsane.f from M. Raydan's website.
18:     #
19:     # Where the results disagree, the largest limits are taken.
20: 
21:     e_a = 1e-5
22:     e_r = 1e-4
23: 
24:     table_1 = [
25:         dict(F=F_1, x0=x0_1, n=1000, nit=5, nfev=5),
26:         dict(F=F_1, x0=x0_1, n=10000, nit=2, nfev=2),
27:         dict(F=F_2, x0=x0_2, n=500, nit=11, nfev=11),
28:         dict(F=F_2, x0=x0_2, n=2000, nit=11, nfev=11),
29:         # dict(F=F_4, x0=x0_4, n=999, nit=243, nfev=1188),  removed: too sensitive to rounding errors
30:         dict(F=F_6, x0=x0_6, n=100, nit=6, nfev=6),  # Results from dfsane.f; papers list nit=3, nfev=3
31:         dict(F=F_7, x0=x0_7, n=99, nit=23, nfev=29),  # Must have n%3==0, typo in papers?
32:         dict(F=F_7, x0=x0_7, n=999, nit=23, nfev=29),  # Must have n%3==0, typo in papers?
33:         dict(F=F_9, x0=x0_9, n=100, nit=12, nfev=18),  # Results from dfsane.f; papers list nit=nfev=6?
34:         dict(F=F_9, x0=x0_9, n=1000, nit=12, nfev=18),
35:         dict(F=F_10, x0=x0_10, n=1000, nit=5, nfev=5),  # Results from dfsane.f; papers list nit=2, nfev=12
36:     ]
37: 
38:     # Check also scaling invariance
39:     for xscale, yscale, line_search in itertools.product([1.0, 1e-10, 1e10], [1.0, 1e-10, 1e10],
40:                                                          ['cruz', 'cheng']):
41:         for problem in table_1:
42:             n = problem['n']
43:             func = lambda x, n: yscale*problem['F'](x/xscale, n)
44:             args = (n,)
45:             x0 = problem['x0'](n) * xscale
46: 
47:             fatol = np.sqrt(n) * e_a * yscale + e_r * np.linalg.norm(func(x0, n))
48: 
49:             sigma_eps = 1e-10 * min(yscale/xscale, xscale/yscale)
50:             sigma_0 = xscale/yscale
51: 
52:             with np.errstate(over='ignore'):
53:                 sol = root(func, x0, args=args,
54:                            options=dict(ftol=0, fatol=fatol, maxfev=problem['nfev'] + 1,
55:                                         sigma_0=sigma_0, sigma_eps=sigma_eps,
56:                                         line_search=line_search),
57:                            method='DF-SANE')
58: 
59:             err_msg = repr([xscale, yscale, line_search, problem, np.linalg.norm(func(sol.x, n)),
60:                             fatol, sol.success, sol.nit, sol.nfev])
61:             assert_(sol.success, err_msg)
62:             assert_(sol.nfev <= problem['nfev'] + 1, err_msg)  # nfev+1: dfsane.f doesn't count first eval
63:             assert_(sol.nit <= problem['nit'], err_msg)
64:             assert_(np.linalg.norm(func(sol.x, n)) <= fatol, err_msg)
65: 
66: 
67: def test_complex():
68:     def func(z):
69:         return z**2 - 1 + 2j
70:     x0 = 2.0j
71: 
72:     ftol = 1e-4
73:     sol = root(func, x0, tol=ftol, method='DF-SANE')
74: 
75:     assert_(sol.success)
76: 
77:     f0 = np.linalg.norm(func(x0))
78:     fx = np.linalg.norm(func(sol.x))
79:     assert_(fx <= ftol*f0)
80: 
81: 
82: def test_linear_definite():
83:     # The DF-SANE paper proves convergence for "strongly isolated"
84:     # solutions.
85:     #
86:     # For linear systems F(x) = A x - b = 0, with A positive or
87:     # negative definite, the solution is strongly isolated.
88: 
89:     def check_solvability(A, b, line_search='cruz'):
90:         func = lambda x: A.dot(x) - b
91:         xp = np.linalg.solve(A, b)
92:         eps = np.linalg.norm(func(xp)) * 1e3
93:         sol = root(func, b, options=dict(fatol=eps, ftol=0, maxfev=17523, line_search=line_search),
94:                    method='DF-SANE')
95:         assert_(sol.success)
96:         assert_(np.linalg.norm(func(sol.x)) <= eps)
97: 
98:     n = 90
99: 
100:     # Test linear pos.def. system
101:     np.random.seed(1234)
102:     A = np.arange(n*n).reshape(n, n)
103:     A = A + n*n * np.diag(1 + np.arange(n))
104:     assert_(np.linalg.eigvals(A).min() > 0)
105:     b = np.arange(n) * 1.0
106:     check_solvability(A, b, 'cruz')
107:     check_solvability(A, b, 'cheng')
108: 
109:     # Test linear neg.def. system
110:     check_solvability(-A, b, 'cruz')
111:     check_solvability(-A, b, 'cheng')
112: 
113: 
114: def test_shape():
115:     def f(x, arg):
116:         return x - arg
117: 
118:     for dt in [float, complex]:
119:         x = np.zeros([2,2])
120:         arg = np.ones([2,2], dtype=dt)
121: 
122:         sol = root(f, x, args=(arg,), method='DF-SANE')
123:         assert_(sol.success)
124:         assert_equal(sol.x.shape, x.shape)
125: 
126: 
127: # Some of the test functions and initial guesses listed in
128: # [W. La Cruz, M. Raydan. Optimization Methods and Software, 18, 583 (2003)]
129: 
130: def F_1(x, n):
131:     g = np.zeros([n])
132:     i = np.arange(2, n+1)
133:     g[0] = exp(x[0] - 1) - 1
134:     g[1:] = i*(exp(x[1:] - 1) - x[1:])
135:     return g
136: 
137: def x0_1(n):
138:     x0 = np.empty([n])
139:     x0.fill(n/(n-1))
140:     return x0
141: 
142: def F_2(x, n):
143:     g = np.zeros([n])
144:     i = np.arange(2, n+1)
145:     g[0] = exp(x[0]) - 1
146:     g[1:] = 0.1*i*(exp(x[1:]) + x[:-1] - 1)
147:     return g
148: 
149: def x0_2(n):
150:     x0 = np.empty([n])
151:     x0.fill(1/n**2)
152:     return x0
153: 
154: def F_4(x, n):
155:     assert_equal(n % 3, 0)
156:     g = np.zeros([n])
157:     # Note: the first line is typoed in some of the references;
158:     # correct in original [Gasparo, Optimization Meth. 13, 79 (2000)]
159:     g[::3] = 0.6 * x[::3] + 1.6 * x[1::3]**3 - 7.2 * x[1::3]**2 + 9.6 * x[1::3] - 4.8
160:     g[1::3] = 0.48 * x[::3] - 0.72 * x[1::3]**3 + 3.24 * x[1::3]**2 - 4.32 * x[1::3] - x[2::3] + 0.2 * x[2::3]**3 + 2.16
161:     g[2::3] = 1.25 * x[2::3] - 0.25*x[2::3]**3
162:     return g
163: 
164: def x0_4(n):
165:     assert_equal(n % 3, 0)
166:     x0 = np.array([-1, 1/2, -1] * (n//3))
167:     return x0
168: 
169: def F_6(x, n):
170:     c = 0.9
171:     mu = (np.arange(1, n+1) - 0.5)/n
172:     return x - 1/(1 - c/(2*n) * (mu[:,None]*x / (mu[:,None] + mu)).sum(axis=1))
173: 
174: def x0_6(n):
175:     return np.ones([n])
176: 
177: def F_7(x, n):
178:     assert_equal(n % 3, 0)
179: 
180:     def phi(t):
181:         v = 0.5*t - 2
182:         v[t > -1] = ((-592*t**3 + 888*t**2 + 4551*t - 1924)/1998)[t > -1]
183:         v[t >= 2] = (0.5*t + 2)[t >= 2]
184:         return v
185:     g = np.zeros([n])
186:     g[::3] = 1e4 * x[1::3]**2 - 1
187:     g[1::3] = exp(-x[::3]) + exp(-x[1::3]) - 1.0001
188:     g[2::3] = phi(x[2::3])
189:     return g
190: 
191: def x0_7(n):
192:     assert_equal(n % 3, 0)
193:     return np.array([1e-3, 18, 1] * (n//3))
194: 
195: def F_9(x, n):
196:     g = np.zeros([n])
197:     i = np.arange(2, n)
198:     g[0] = x[0]**3/3 + x[1]**2/2
199:     g[1:-1] = -x[1:-1]**2/2 + i*x[1:-1]**3/3 + x[2:]**2/2
200:     g[-1] = -x[-1]**2/2 + n*x[-1]**3/3
201:     return g
202: 
203: def x0_9(n):
204:     return np.ones([n])
205: 
206: def F_10(x, n):
207:     return np.log(1 + x) - x/n
208: 
209: def x0_10(n):
210:     return np.ones([n])
211: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import itertools' statement (line 3)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_245882 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_245882) is not StypyTypeError):

    if (import_245882 != 'pyd_module'):
        __import__(import_245882)
        sys_modules_245883 = sys.modules[import_245882]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_245883.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_245882)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import exp' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_245884 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_245884) is not StypyTypeError):

    if (import_245884 != 'pyd_module'):
        __import__(import_245884)
        sys_modules_245885 = sys.modules[import_245884]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_245885.module_type_store, module_type_store, ['exp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_245885, sys_modules_245885.module_type_store, module_type_store)
    else:
        from numpy import exp

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['exp'], [exp])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_245884)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_, assert_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_245886 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_245886) is not StypyTypeError):

    if (import_245886 != 'pyd_module'):
        __import__(import_245886)
        sys_modules_245887 = sys.modules[import_245886]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_245887.module_type_store, module_type_store, ['assert_', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_245887, sys_modules_245887.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal'], [assert_, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_245886)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.optimize import root' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_245888 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize')

if (type(import_245888) is not StypyTypeError):

    if (import_245888 != 'pyd_module'):
        __import__(import_245888)
        sys_modules_245889 = sys.modules[import_245888]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', sys_modules_245889.module_type_store, module_type_store, ['root'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_245889, sys_modules_245889.module_type_store, module_type_store)
    else:
        from scipy.optimize import root

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', None, module_type_store, ['root'], [root])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', import_245888)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def test_performance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_performance'
    module_type_store = module_type_store.open_function_context('test_performance', 12, 0, False)
    
    # Passed parameters checking function
    test_performance.stypy_localization = localization
    test_performance.stypy_type_of_self = None
    test_performance.stypy_type_store = module_type_store
    test_performance.stypy_function_name = 'test_performance'
    test_performance.stypy_param_names_list = []
    test_performance.stypy_varargs_param_name = None
    test_performance.stypy_kwargs_param_name = None
    test_performance.stypy_call_defaults = defaults
    test_performance.stypy_call_varargs = varargs
    test_performance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_performance', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_performance', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_performance(...)' code ##################

    
    # Assigning a Num to a Name (line 21):
    float_245890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'float')
    # Assigning a type to the variable 'e_a' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'e_a', float_245890)
    
    # Assigning a Num to a Name (line 22):
    float_245891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'float')
    # Assigning a type to the variable 'e_r' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'e_r', float_245891)
    
    # Assigning a List to a Name (line 24):
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_245892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 25)
    # Processing the call keyword arguments (line 25)
    # Getting the type of 'F_1' (line 25)
    F_1_245894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'F_1', False)
    keyword_245895 = F_1_245894
    # Getting the type of 'x0_1' (line 25)
    x0_1_245896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'x0_1', False)
    keyword_245897 = x0_1_245896
    int_245898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'int')
    keyword_245899 = int_245898
    int_245900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'int')
    keyword_245901 = int_245900
    int_245902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 49), 'int')
    keyword_245903 = int_245902
    kwargs_245904 = {'nfev': keyword_245903, 'x0': keyword_245897, 'n': keyword_245899, 'nit': keyword_245901, 'F': keyword_245895}
    # Getting the type of 'dict' (line 25)
    dict_245893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 25)
    dict_call_result_245905 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), dict_245893, *[], **kwargs_245904)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_245905)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 26)
    # Processing the call keyword arguments (line 26)
    # Getting the type of 'F_1' (line 26)
    F_1_245907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'F_1', False)
    keyword_245908 = F_1_245907
    # Getting the type of 'x0_1' (line 26)
    x0_1_245909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'x0_1', False)
    keyword_245910 = x0_1_245909
    int_245911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'int')
    keyword_245912 = int_245911
    int_245913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'int')
    keyword_245914 = int_245913
    int_245915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 50), 'int')
    keyword_245916 = int_245915
    kwargs_245917 = {'nfev': keyword_245916, 'x0': keyword_245910, 'n': keyword_245912, 'nit': keyword_245914, 'F': keyword_245908}
    # Getting the type of 'dict' (line 26)
    dict_245906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 26)
    dict_call_result_245918 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), dict_245906, *[], **kwargs_245917)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_245918)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 27)
    # Processing the call keyword arguments (line 27)
    # Getting the type of 'F_2' (line 27)
    F_2_245920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'F_2', False)
    keyword_245921 = F_2_245920
    # Getting the type of 'x0_2' (line 27)
    x0_2_245922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'x0_2', False)
    keyword_245923 = x0_2_245922
    int_245924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'int')
    keyword_245925 = int_245924
    int_245926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 40), 'int')
    keyword_245927 = int_245926
    int_245928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 49), 'int')
    keyword_245929 = int_245928
    kwargs_245930 = {'nfev': keyword_245929, 'x0': keyword_245923, 'n': keyword_245925, 'nit': keyword_245927, 'F': keyword_245921}
    # Getting the type of 'dict' (line 27)
    dict_245919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 27)
    dict_call_result_245931 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), dict_245919, *[], **kwargs_245930)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_245931)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 28)
    # Processing the call keyword arguments (line 28)
    # Getting the type of 'F_2' (line 28)
    F_2_245933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'F_2', False)
    keyword_245934 = F_2_245933
    # Getting the type of 'x0_2' (line 28)
    x0_2_245935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'x0_2', False)
    keyword_245936 = x0_2_245935
    int_245937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'int')
    keyword_245938 = int_245937
    int_245939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 41), 'int')
    keyword_245940 = int_245939
    int_245941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 50), 'int')
    keyword_245942 = int_245941
    kwargs_245943 = {'nfev': keyword_245942, 'x0': keyword_245936, 'n': keyword_245938, 'nit': keyword_245940, 'F': keyword_245934}
    # Getting the type of 'dict' (line 28)
    dict_245932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 28)
    dict_call_result_245944 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), dict_245932, *[], **kwargs_245943)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_245944)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 30)
    # Processing the call keyword arguments (line 30)
    # Getting the type of 'F_6' (line 30)
    F_6_245946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'F_6', False)
    keyword_245947 = F_6_245946
    # Getting the type of 'x0_6' (line 30)
    x0_6_245948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'x0_6', False)
    keyword_245949 = x0_6_245948
    int_245950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'int')
    keyword_245951 = int_245950
    int_245952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 40), 'int')
    keyword_245953 = int_245952
    int_245954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 48), 'int')
    keyword_245955 = int_245954
    kwargs_245956 = {'nfev': keyword_245955, 'x0': keyword_245949, 'n': keyword_245951, 'nit': keyword_245953, 'F': keyword_245947}
    # Getting the type of 'dict' (line 30)
    dict_245945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 30)
    dict_call_result_245957 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), dict_245945, *[], **kwargs_245956)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_245957)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 31)
    # Processing the call keyword arguments (line 31)
    # Getting the type of 'F_7' (line 31)
    F_7_245959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'F_7', False)
    keyword_245960 = F_7_245959
    # Getting the type of 'x0_7' (line 31)
    x0_7_245961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'x0_7', False)
    keyword_245962 = x0_7_245961
    int_245963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'int')
    keyword_245964 = int_245963
    int_245965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 39), 'int')
    keyword_245966 = int_245965
    int_245967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 48), 'int')
    keyword_245968 = int_245967
    kwargs_245969 = {'nfev': keyword_245968, 'x0': keyword_245962, 'n': keyword_245964, 'nit': keyword_245966, 'F': keyword_245960}
    # Getting the type of 'dict' (line 31)
    dict_245958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 31)
    dict_call_result_245970 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), dict_245958, *[], **kwargs_245969)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_245970)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 32)
    # Processing the call keyword arguments (line 32)
    # Getting the type of 'F_7' (line 32)
    F_7_245972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'F_7', False)
    keyword_245973 = F_7_245972
    # Getting the type of 'x0_7' (line 32)
    x0_7_245974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'x0_7', False)
    keyword_245975 = x0_7_245974
    int_245976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'int')
    keyword_245977 = int_245976
    int_245978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 40), 'int')
    keyword_245979 = int_245978
    int_245980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 49), 'int')
    keyword_245981 = int_245980
    kwargs_245982 = {'nfev': keyword_245981, 'x0': keyword_245975, 'n': keyword_245977, 'nit': keyword_245979, 'F': keyword_245973}
    # Getting the type of 'dict' (line 32)
    dict_245971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 32)
    dict_call_result_245983 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), dict_245971, *[], **kwargs_245982)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_245983)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 33)
    # Processing the call keyword arguments (line 33)
    # Getting the type of 'F_9' (line 33)
    F_9_245985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'F_9', False)
    keyword_245986 = F_9_245985
    # Getting the type of 'x0_9' (line 33)
    x0_9_245987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'x0_9', False)
    keyword_245988 = x0_9_245987
    int_245989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'int')
    keyword_245990 = int_245989
    int_245991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 40), 'int')
    keyword_245992 = int_245991
    int_245993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 49), 'int')
    keyword_245994 = int_245993
    kwargs_245995 = {'nfev': keyword_245994, 'x0': keyword_245988, 'n': keyword_245990, 'nit': keyword_245992, 'F': keyword_245986}
    # Getting the type of 'dict' (line 33)
    dict_245984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 33)
    dict_call_result_245996 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), dict_245984, *[], **kwargs_245995)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_245996)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 34)
    # Processing the call keyword arguments (line 34)
    # Getting the type of 'F_9' (line 34)
    F_9_245998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'F_9', False)
    keyword_245999 = F_9_245998
    # Getting the type of 'x0_9' (line 34)
    x0_9_246000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'x0_9', False)
    keyword_246001 = x0_9_246000
    int_246002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
    keyword_246003 = int_246002
    int_246004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 41), 'int')
    keyword_246005 = int_246004
    int_246006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 50), 'int')
    keyword_246007 = int_246006
    kwargs_246008 = {'nfev': keyword_246007, 'x0': keyword_246001, 'n': keyword_246003, 'nit': keyword_246005, 'F': keyword_245999}
    # Getting the type of 'dict' (line 34)
    dict_245997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 34)
    dict_call_result_246009 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), dict_245997, *[], **kwargs_246008)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_246009)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 35)
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'F_10' (line 35)
    F_10_246011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'F_10', False)
    keyword_246012 = F_10_246011
    # Getting the type of 'x0_10' (line 35)
    x0_10_246013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'x0_10', False)
    keyword_246014 = x0_10_246013
    int_246015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'int')
    keyword_246016 = int_246015
    int_246017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'int')
    keyword_246018 = int_246017
    int_246019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 51), 'int')
    keyword_246020 = int_246019
    kwargs_246021 = {'nfev': keyword_246020, 'x0': keyword_246014, 'n': keyword_246016, 'nit': keyword_246018, 'F': keyword_246012}
    # Getting the type of 'dict' (line 35)
    dict_246010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 35)
    dict_call_result_246022 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), dict_246010, *[], **kwargs_246021)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_245892, dict_call_result_246022)
    
    # Assigning a type to the variable 'table_1' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'table_1', list_245892)
    
    
    # Call to product(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_246025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    float_246026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 58), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 57), list_246025, float_246026)
    # Adding element type (line 39)
    float_246027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 63), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 57), list_246025, float_246027)
    # Adding element type (line 39)
    float_246028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 70), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 57), list_246025, float_246028)
    
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_246029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 77), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    float_246030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 78), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 77), list_246029, float_246030)
    # Adding element type (line 39)
    float_246031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 83), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 77), list_246029, float_246031)
    # Adding element type (line 39)
    float_246032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 90), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 77), list_246029, float_246032)
    
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_246033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    str_246034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 58), 'str', 'cruz')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 57), list_246033, str_246034)
    # Adding element type (line 40)
    str_246035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 66), 'str', 'cheng')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 57), list_246033, str_246035)
    
    # Processing the call keyword arguments (line 39)
    kwargs_246036 = {}
    # Getting the type of 'itertools' (line 39)
    itertools_246023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'itertools', False)
    # Obtaining the member 'product' of a type (line 39)
    product_246024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 39), itertools_246023, 'product')
    # Calling product(args, kwargs) (line 39)
    product_call_result_246037 = invoke(stypy.reporting.localization.Localization(__file__, 39, 39), product_246024, *[list_246025, list_246029, list_246033], **kwargs_246036)
    
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 4), product_call_result_246037)
    # Getting the type of the for loop variable (line 39)
    for_loop_var_246038 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 4), product_call_result_246037)
    # Assigning a type to the variable 'xscale' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'xscale', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), for_loop_var_246038))
    # Assigning a type to the variable 'yscale' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'yscale', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), for_loop_var_246038))
    # Assigning a type to the variable 'line_search' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'line_search', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), for_loop_var_246038))
    # SSA begins for a for statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'table_1' (line 41)
    table_1_246039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'table_1')
    # Testing the type of a for loop iterable (line 41)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 8), table_1_246039)
    # Getting the type of the for loop variable (line 41)
    for_loop_var_246040 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 8), table_1_246039)
    # Assigning a type to the variable 'problem' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'problem', for_loop_var_246040)
    # SSA begins for a for statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 42):
    
    # Obtaining the type of the subscript
    str_246041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'str', 'n')
    # Getting the type of 'problem' (line 42)
    problem_246042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'problem')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___246043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), problem_246042, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_246044 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), getitem___246043, str_246041)
    
    # Assigning a type to the variable 'n' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'n', subscript_call_result_246044)
    
    # Assigning a Lambda to a Name (line 43):

    @norecursion
    def _stypy_temp_lambda_165(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_165'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_165', 43, 19, True)
        # Passed parameters checking function
        _stypy_temp_lambda_165.stypy_localization = localization
        _stypy_temp_lambda_165.stypy_type_of_self = None
        _stypy_temp_lambda_165.stypy_type_store = module_type_store
        _stypy_temp_lambda_165.stypy_function_name = '_stypy_temp_lambda_165'
        _stypy_temp_lambda_165.stypy_param_names_list = ['x', 'n']
        _stypy_temp_lambda_165.stypy_varargs_param_name = None
        _stypy_temp_lambda_165.stypy_kwargs_param_name = None
        _stypy_temp_lambda_165.stypy_call_defaults = defaults
        _stypy_temp_lambda_165.stypy_call_varargs = varargs
        _stypy_temp_lambda_165.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_165', ['x', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_165', ['x', 'n'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'yscale' (line 43)
        yscale_246045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'yscale')
        
        # Call to (...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'x' (line 43)
        x_246050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 52), 'x', False)
        # Getting the type of 'xscale' (line 43)
        xscale_246051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 54), 'xscale', False)
        # Applying the binary operator 'div' (line 43)
        result_div_246052 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 52), 'div', x_246050, xscale_246051)
        
        # Getting the type of 'n' (line 43)
        n_246053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 62), 'n', False)
        # Processing the call keyword arguments (line 43)
        kwargs_246054 = {}
        
        # Obtaining the type of the subscript
        str_246046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 47), 'str', 'F')
        # Getting the type of 'problem' (line 43)
        problem_246047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 39), 'problem', False)
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___246048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 39), problem_246047, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_246049 = invoke(stypy.reporting.localization.Localization(__file__, 43, 39), getitem___246048, str_246046)
        
        # Calling (args, kwargs) (line 43)
        _call_result_246055 = invoke(stypy.reporting.localization.Localization(__file__, 43, 39), subscript_call_result_246049, *[result_div_246052, n_246053], **kwargs_246054)
        
        # Applying the binary operator '*' (line 43)
        result_mul_246056 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 32), '*', yscale_246045, _call_result_246055)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'stypy_return_type', result_mul_246056)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_165' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_246057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_246057)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_165'
        return stypy_return_type_246057

    # Assigning a type to the variable '_stypy_temp_lambda_165' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), '_stypy_temp_lambda_165', _stypy_temp_lambda_165)
    # Getting the type of '_stypy_temp_lambda_165' (line 43)
    _stypy_temp_lambda_165_246058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), '_stypy_temp_lambda_165')
    # Assigning a type to the variable 'func' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'func', _stypy_temp_lambda_165_246058)
    
    # Assigning a Tuple to a Name (line 44):
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_246059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'n' (line 44)
    n_246060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 20), tuple_246059, n_246060)
    
    # Assigning a type to the variable 'args' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'args', tuple_246059)
    
    # Assigning a BinOp to a Name (line 45):
    
    # Call to (...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'n' (line 45)
    n_246065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'n', False)
    # Processing the call keyword arguments (line 45)
    kwargs_246066 = {}
    
    # Obtaining the type of the subscript
    str_246061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'str', 'x0')
    # Getting the type of 'problem' (line 45)
    problem_246062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'problem', False)
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___246063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), problem_246062, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_246064 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), getitem___246063, str_246061)
    
    # Calling (args, kwargs) (line 45)
    _call_result_246067 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), subscript_call_result_246064, *[n_246065], **kwargs_246066)
    
    # Getting the type of 'xscale' (line 45)
    xscale_246068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'xscale')
    # Applying the binary operator '*' (line 45)
    result_mul_246069 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 17), '*', _call_result_246067, xscale_246068)
    
    # Assigning a type to the variable 'x0' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'x0', result_mul_246069)
    
    # Assigning a BinOp to a Name (line 47):
    
    # Call to sqrt(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'n' (line 47)
    n_246072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 28), 'n', False)
    # Processing the call keyword arguments (line 47)
    kwargs_246073 = {}
    # Getting the type of 'np' (line 47)
    np_246070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 47)
    sqrt_246071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 20), np_246070, 'sqrt')
    # Calling sqrt(args, kwargs) (line 47)
    sqrt_call_result_246074 = invoke(stypy.reporting.localization.Localization(__file__, 47, 20), sqrt_246071, *[n_246072], **kwargs_246073)
    
    # Getting the type of 'e_a' (line 47)
    e_a_246075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'e_a')
    # Applying the binary operator '*' (line 47)
    result_mul_246076 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 20), '*', sqrt_call_result_246074, e_a_246075)
    
    # Getting the type of 'yscale' (line 47)
    yscale_246077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 39), 'yscale')
    # Applying the binary operator '*' (line 47)
    result_mul_246078 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 37), '*', result_mul_246076, yscale_246077)
    
    # Getting the type of 'e_r' (line 47)
    e_r_246079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 48), 'e_r')
    
    # Call to norm(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Call to func(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'x0' (line 47)
    x0_246084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 74), 'x0', False)
    # Getting the type of 'n' (line 47)
    n_246085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 78), 'n', False)
    # Processing the call keyword arguments (line 47)
    kwargs_246086 = {}
    # Getting the type of 'func' (line 47)
    func_246083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 69), 'func', False)
    # Calling func(args, kwargs) (line 47)
    func_call_result_246087 = invoke(stypy.reporting.localization.Localization(__file__, 47, 69), func_246083, *[x0_246084, n_246085], **kwargs_246086)
    
    # Processing the call keyword arguments (line 47)
    kwargs_246088 = {}
    # Getting the type of 'np' (line 47)
    np_246080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 54), 'np', False)
    # Obtaining the member 'linalg' of a type (line 47)
    linalg_246081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 54), np_246080, 'linalg')
    # Obtaining the member 'norm' of a type (line 47)
    norm_246082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 54), linalg_246081, 'norm')
    # Calling norm(args, kwargs) (line 47)
    norm_call_result_246089 = invoke(stypy.reporting.localization.Localization(__file__, 47, 54), norm_246082, *[func_call_result_246087], **kwargs_246088)
    
    # Applying the binary operator '*' (line 47)
    result_mul_246090 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 48), '*', e_r_246079, norm_call_result_246089)
    
    # Applying the binary operator '+' (line 47)
    result_add_246091 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 20), '+', result_mul_246078, result_mul_246090)
    
    # Assigning a type to the variable 'fatol' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'fatol', result_add_246091)
    
    # Assigning a BinOp to a Name (line 49):
    float_246092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'float')
    
    # Call to min(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'yscale' (line 49)
    yscale_246094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'yscale', False)
    # Getting the type of 'xscale' (line 49)
    xscale_246095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 43), 'xscale', False)
    # Applying the binary operator 'div' (line 49)
    result_div_246096 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 36), 'div', yscale_246094, xscale_246095)
    
    # Getting the type of 'xscale' (line 49)
    xscale_246097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 51), 'xscale', False)
    # Getting the type of 'yscale' (line 49)
    yscale_246098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 58), 'yscale', False)
    # Applying the binary operator 'div' (line 49)
    result_div_246099 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 51), 'div', xscale_246097, yscale_246098)
    
    # Processing the call keyword arguments (line 49)
    kwargs_246100 = {}
    # Getting the type of 'min' (line 49)
    min_246093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 32), 'min', False)
    # Calling min(args, kwargs) (line 49)
    min_call_result_246101 = invoke(stypy.reporting.localization.Localization(__file__, 49, 32), min_246093, *[result_div_246096, result_div_246099], **kwargs_246100)
    
    # Applying the binary operator '*' (line 49)
    result_mul_246102 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 24), '*', float_246092, min_call_result_246101)
    
    # Assigning a type to the variable 'sigma_eps' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'sigma_eps', result_mul_246102)
    
    # Assigning a BinOp to a Name (line 50):
    # Getting the type of 'xscale' (line 50)
    xscale_246103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'xscale')
    # Getting the type of 'yscale' (line 50)
    yscale_246104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'yscale')
    # Applying the binary operator 'div' (line 50)
    result_div_246105 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 22), 'div', xscale_246103, yscale_246104)
    
    # Assigning a type to the variable 'sigma_0' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'sigma_0', result_div_246105)
    
    # Call to errstate(...): (line 52)
    # Processing the call keyword arguments (line 52)
    str_246108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 34), 'str', 'ignore')
    keyword_246109 = str_246108
    kwargs_246110 = {'over': keyword_246109}
    # Getting the type of 'np' (line 52)
    np_246106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'np', False)
    # Obtaining the member 'errstate' of a type (line 52)
    errstate_246107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 17), np_246106, 'errstate')
    # Calling errstate(args, kwargs) (line 52)
    errstate_call_result_246111 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), errstate_246107, *[], **kwargs_246110)
    
    with_246112 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 52, 17), errstate_call_result_246111, 'with parameter', '__enter__', '__exit__')

    if with_246112:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 52)
        enter___246113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 17), errstate_call_result_246111, '__enter__')
        with_enter_246114 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), enter___246113)
        
        # Assigning a Call to a Name (line 53):
        
        # Call to root(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'func' (line 53)
        func_246116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'func', False)
        # Getting the type of 'x0' (line 53)
        x0_246117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'x0', False)
        # Processing the call keyword arguments (line 53)
        # Getting the type of 'args' (line 53)
        args_246118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 42), 'args', False)
        keyword_246119 = args_246118
        
        # Call to dict(...): (line 54)
        # Processing the call keyword arguments (line 54)
        int_246121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 45), 'int')
        keyword_246122 = int_246121
        # Getting the type of 'fatol' (line 54)
        fatol_246123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 54), 'fatol', False)
        keyword_246124 = fatol_246123
        
        # Obtaining the type of the subscript
        str_246125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 76), 'str', 'nfev')
        # Getting the type of 'problem' (line 54)
        problem_246126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 68), 'problem', False)
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___246127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 68), problem_246126, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_246128 = invoke(stypy.reporting.localization.Localization(__file__, 54, 68), getitem___246127, str_246125)
        
        int_246129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 86), 'int')
        # Applying the binary operator '+' (line 54)
        result_add_246130 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 68), '+', subscript_call_result_246128, int_246129)
        
        keyword_246131 = result_add_246130
        # Getting the type of 'sigma_0' (line 55)
        sigma_0_246132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 48), 'sigma_0', False)
        keyword_246133 = sigma_0_246132
        # Getting the type of 'sigma_eps' (line 55)
        sigma_eps_246134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 67), 'sigma_eps', False)
        keyword_246135 = sigma_eps_246134
        # Getting the type of 'line_search' (line 56)
        line_search_246136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 52), 'line_search', False)
        keyword_246137 = line_search_246136
        kwargs_246138 = {'sigma_eps': keyword_246135, 'sigma_0': keyword_246133, 'line_search': keyword_246137, 'fatol': keyword_246124, 'maxfev': keyword_246131, 'ftol': keyword_246122}
        # Getting the type of 'dict' (line 54)
        dict_246120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 35), 'dict', False)
        # Calling dict(args, kwargs) (line 54)
        dict_call_result_246139 = invoke(stypy.reporting.localization.Localization(__file__, 54, 35), dict_246120, *[], **kwargs_246138)
        
        keyword_246140 = dict_call_result_246139
        str_246141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'str', 'DF-SANE')
        keyword_246142 = str_246141
        kwargs_246143 = {'args': keyword_246119, 'options': keyword_246140, 'method': keyword_246142}
        # Getting the type of 'root' (line 53)
        root_246115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'root', False)
        # Calling root(args, kwargs) (line 53)
        root_call_result_246144 = invoke(stypy.reporting.localization.Localization(__file__, 53, 22), root_246115, *[func_246116, x0_246117], **kwargs_246143)
        
        # Assigning a type to the variable 'sol' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'sol', root_call_result_246144)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 52)
        exit___246145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 17), errstate_call_result_246111, '__exit__')
        with_exit_246146 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), exit___246145, None, None, None)

    
    # Assigning a Call to a Name (line 59):
    
    # Call to repr(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_246148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    # Getting the type of 'xscale' (line 59)
    xscale_246149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'xscale', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, xscale_246149)
    # Adding element type (line 59)
    # Getting the type of 'yscale' (line 59)
    yscale_246150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'yscale', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, yscale_246150)
    # Adding element type (line 59)
    # Getting the type of 'line_search' (line 59)
    line_search_246151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), 'line_search', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, line_search_246151)
    # Adding element type (line 59)
    # Getting the type of 'problem' (line 59)
    problem_246152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 57), 'problem', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, problem_246152)
    # Adding element type (line 59)
    
    # Call to norm(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Call to func(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'sol' (line 59)
    sol_246157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 86), 'sol', False)
    # Obtaining the member 'x' of a type (line 59)
    x_246158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 86), sol_246157, 'x')
    # Getting the type of 'n' (line 59)
    n_246159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 93), 'n', False)
    # Processing the call keyword arguments (line 59)
    kwargs_246160 = {}
    # Getting the type of 'func' (line 59)
    func_246156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 81), 'func', False)
    # Calling func(args, kwargs) (line 59)
    func_call_result_246161 = invoke(stypy.reporting.localization.Localization(__file__, 59, 81), func_246156, *[x_246158, n_246159], **kwargs_246160)
    
    # Processing the call keyword arguments (line 59)
    kwargs_246162 = {}
    # Getting the type of 'np' (line 59)
    np_246153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 66), 'np', False)
    # Obtaining the member 'linalg' of a type (line 59)
    linalg_246154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 66), np_246153, 'linalg')
    # Obtaining the member 'norm' of a type (line 59)
    norm_246155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 66), linalg_246154, 'norm')
    # Calling norm(args, kwargs) (line 59)
    norm_call_result_246163 = invoke(stypy.reporting.localization.Localization(__file__, 59, 66), norm_246155, *[func_call_result_246161], **kwargs_246162)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, norm_call_result_246163)
    # Adding element type (line 59)
    # Getting the type of 'fatol' (line 60)
    fatol_246164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 28), 'fatol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, fatol_246164)
    # Adding element type (line 59)
    # Getting the type of 'sol' (line 60)
    sol_246165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'sol', False)
    # Obtaining the member 'success' of a type (line 60)
    success_246166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 35), sol_246165, 'success')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, success_246166)
    # Adding element type (line 59)
    # Getting the type of 'sol' (line 60)
    sol_246167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 48), 'sol', False)
    # Obtaining the member 'nit' of a type (line 60)
    nit_246168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 48), sol_246167, 'nit')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, nit_246168)
    # Adding element type (line 59)
    # Getting the type of 'sol' (line 60)
    sol_246169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 57), 'sol', False)
    # Obtaining the member 'nfev' of a type (line 60)
    nfev_246170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 57), sol_246169, 'nfev')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 27), list_246148, nfev_246170)
    
    # Processing the call keyword arguments (line 59)
    kwargs_246171 = {}
    # Getting the type of 'repr' (line 59)
    repr_246147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'repr', False)
    # Calling repr(args, kwargs) (line 59)
    repr_call_result_246172 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), repr_246147, *[list_246148], **kwargs_246171)
    
    # Assigning a type to the variable 'err_msg' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'err_msg', repr_call_result_246172)
    
    # Call to assert_(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'sol' (line 61)
    sol_246174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'sol', False)
    # Obtaining the member 'success' of a type (line 61)
    success_246175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), sol_246174, 'success')
    # Getting the type of 'err_msg' (line 61)
    err_msg_246176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'err_msg', False)
    # Processing the call keyword arguments (line 61)
    kwargs_246177 = {}
    # Getting the type of 'assert_' (line 61)
    assert__246173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 61)
    assert__call_result_246178 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), assert__246173, *[success_246175, err_msg_246176], **kwargs_246177)
    
    
    # Call to assert_(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Getting the type of 'sol' (line 62)
    sol_246180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'sol', False)
    # Obtaining the member 'nfev' of a type (line 62)
    nfev_246181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), sol_246180, 'nfev')
    
    # Obtaining the type of the subscript
    str_246182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 40), 'str', 'nfev')
    # Getting the type of 'problem' (line 62)
    problem_246183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'problem', False)
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___246184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 32), problem_246183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_246185 = invoke(stypy.reporting.localization.Localization(__file__, 62, 32), getitem___246184, str_246182)
    
    int_246186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 50), 'int')
    # Applying the binary operator '+' (line 62)
    result_add_246187 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 32), '+', subscript_call_result_246185, int_246186)
    
    # Applying the binary operator '<=' (line 62)
    result_le_246188 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 20), '<=', nfev_246181, result_add_246187)
    
    # Getting the type of 'err_msg' (line 62)
    err_msg_246189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 53), 'err_msg', False)
    # Processing the call keyword arguments (line 62)
    kwargs_246190 = {}
    # Getting the type of 'assert_' (line 62)
    assert__246179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 62)
    assert__call_result_246191 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), assert__246179, *[result_le_246188, err_msg_246189], **kwargs_246190)
    
    
    # Call to assert_(...): (line 63)
    # Processing the call arguments (line 63)
    
    # Getting the type of 'sol' (line 63)
    sol_246193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'sol', False)
    # Obtaining the member 'nit' of a type (line 63)
    nit_246194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), sol_246193, 'nit')
    
    # Obtaining the type of the subscript
    str_246195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'str', 'nit')
    # Getting the type of 'problem' (line 63)
    problem_246196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'problem', False)
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___246197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 31), problem_246196, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_246198 = invoke(stypy.reporting.localization.Localization(__file__, 63, 31), getitem___246197, str_246195)
    
    # Applying the binary operator '<=' (line 63)
    result_le_246199 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 20), '<=', nit_246194, subscript_call_result_246198)
    
    # Getting the type of 'err_msg' (line 63)
    err_msg_246200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'err_msg', False)
    # Processing the call keyword arguments (line 63)
    kwargs_246201 = {}
    # Getting the type of 'assert_' (line 63)
    assert__246192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 63)
    assert__call_result_246202 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), assert__246192, *[result_le_246199, err_msg_246200], **kwargs_246201)
    
    
    # Call to assert_(...): (line 64)
    # Processing the call arguments (line 64)
    
    
    # Call to norm(...): (line 64)
    # Processing the call arguments (line 64)
    
    # Call to func(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'sol' (line 64)
    sol_246208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'sol', False)
    # Obtaining the member 'x' of a type (line 64)
    x_246209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 40), sol_246208, 'x')
    # Getting the type of 'n' (line 64)
    n_246210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 47), 'n', False)
    # Processing the call keyword arguments (line 64)
    kwargs_246211 = {}
    # Getting the type of 'func' (line 64)
    func_246207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 'func', False)
    # Calling func(args, kwargs) (line 64)
    func_call_result_246212 = invoke(stypy.reporting.localization.Localization(__file__, 64, 35), func_246207, *[x_246209, n_246210], **kwargs_246211)
    
    # Processing the call keyword arguments (line 64)
    kwargs_246213 = {}
    # Getting the type of 'np' (line 64)
    np_246204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'np', False)
    # Obtaining the member 'linalg' of a type (line 64)
    linalg_246205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 20), np_246204, 'linalg')
    # Obtaining the member 'norm' of a type (line 64)
    norm_246206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 20), linalg_246205, 'norm')
    # Calling norm(args, kwargs) (line 64)
    norm_call_result_246214 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), norm_246206, *[func_call_result_246212], **kwargs_246213)
    
    # Getting the type of 'fatol' (line 64)
    fatol_246215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 54), 'fatol', False)
    # Applying the binary operator '<=' (line 64)
    result_le_246216 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 20), '<=', norm_call_result_246214, fatol_246215)
    
    # Getting the type of 'err_msg' (line 64)
    err_msg_246217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 61), 'err_msg', False)
    # Processing the call keyword arguments (line 64)
    kwargs_246218 = {}
    # Getting the type of 'assert_' (line 64)
    assert__246203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'assert_', False)
    # Calling assert_(args, kwargs) (line 64)
    assert__call_result_246219 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), assert__246203, *[result_le_246216, err_msg_246217], **kwargs_246218)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_performance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_performance' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_246220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246220)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_performance'
    return stypy_return_type_246220

# Assigning a type to the variable 'test_performance' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'test_performance', test_performance)

@norecursion
def test_complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_complex'
    module_type_store = module_type_store.open_function_context('test_complex', 67, 0, False)
    
    # Passed parameters checking function
    test_complex.stypy_localization = localization
    test_complex.stypy_type_of_self = None
    test_complex.stypy_type_store = module_type_store
    test_complex.stypy_function_name = 'test_complex'
    test_complex.stypy_param_names_list = []
    test_complex.stypy_varargs_param_name = None
    test_complex.stypy_kwargs_param_name = None
    test_complex.stypy_call_defaults = defaults
    test_complex.stypy_call_varargs = varargs
    test_complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_complex', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_complex', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_complex(...)' code ##################


    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 68, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['z']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        # Getting the type of 'z' (line 69)
        z_246221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'z')
        int_246222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'int')
        # Applying the binary operator '**' (line 69)
        result_pow_246223 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 15), '**', z_246221, int_246222)
        
        int_246224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'int')
        # Applying the binary operator '-' (line 69)
        result_sub_246225 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 15), '-', result_pow_246223, int_246224)
        
        complex_246226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'complex')
        # Applying the binary operator '+' (line 69)
        result_add_246227 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 24), '+', result_sub_246225, complex_246226)
        
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', result_add_246227)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_246228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_246228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_246228

    # Assigning a type to the variable 'func' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'func', func)
    
    # Assigning a Num to a Name (line 70):
    complex_246229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 9), 'complex')
    # Assigning a type to the variable 'x0' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'x0', complex_246229)
    
    # Assigning a Num to a Name (line 72):
    float_246230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'float')
    # Assigning a type to the variable 'ftol' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'ftol', float_246230)
    
    # Assigning a Call to a Name (line 73):
    
    # Call to root(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'func' (line 73)
    func_246232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'func', False)
    # Getting the type of 'x0' (line 73)
    x0_246233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 21), 'x0', False)
    # Processing the call keyword arguments (line 73)
    # Getting the type of 'ftol' (line 73)
    ftol_246234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 29), 'ftol', False)
    keyword_246235 = ftol_246234
    str_246236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 42), 'str', 'DF-SANE')
    keyword_246237 = str_246236
    kwargs_246238 = {'method': keyword_246237, 'tol': keyword_246235}
    # Getting the type of 'root' (line 73)
    root_246231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 10), 'root', False)
    # Calling root(args, kwargs) (line 73)
    root_call_result_246239 = invoke(stypy.reporting.localization.Localization(__file__, 73, 10), root_246231, *[func_246232, x0_246233], **kwargs_246238)
    
    # Assigning a type to the variable 'sol' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'sol', root_call_result_246239)
    
    # Call to assert_(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'sol' (line 75)
    sol_246241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'sol', False)
    # Obtaining the member 'success' of a type (line 75)
    success_246242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), sol_246241, 'success')
    # Processing the call keyword arguments (line 75)
    kwargs_246243 = {}
    # Getting the type of 'assert_' (line 75)
    assert__246240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 75)
    assert__call_result_246244 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), assert__246240, *[success_246242], **kwargs_246243)
    
    
    # Assigning a Call to a Name (line 77):
    
    # Call to norm(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Call to func(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'x0' (line 77)
    x0_246249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'x0', False)
    # Processing the call keyword arguments (line 77)
    kwargs_246250 = {}
    # Getting the type of 'func' (line 77)
    func_246248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'func', False)
    # Calling func(args, kwargs) (line 77)
    func_call_result_246251 = invoke(stypy.reporting.localization.Localization(__file__, 77, 24), func_246248, *[x0_246249], **kwargs_246250)
    
    # Processing the call keyword arguments (line 77)
    kwargs_246252 = {}
    # Getting the type of 'np' (line 77)
    np_246245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 9), 'np', False)
    # Obtaining the member 'linalg' of a type (line 77)
    linalg_246246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 9), np_246245, 'linalg')
    # Obtaining the member 'norm' of a type (line 77)
    norm_246247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 9), linalg_246246, 'norm')
    # Calling norm(args, kwargs) (line 77)
    norm_call_result_246253 = invoke(stypy.reporting.localization.Localization(__file__, 77, 9), norm_246247, *[func_call_result_246251], **kwargs_246252)
    
    # Assigning a type to the variable 'f0' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'f0', norm_call_result_246253)
    
    # Assigning a Call to a Name (line 78):
    
    # Call to norm(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Call to func(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'sol' (line 78)
    sol_246258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'sol', False)
    # Obtaining the member 'x' of a type (line 78)
    x_246259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 29), sol_246258, 'x')
    # Processing the call keyword arguments (line 78)
    kwargs_246260 = {}
    # Getting the type of 'func' (line 78)
    func_246257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'func', False)
    # Calling func(args, kwargs) (line 78)
    func_call_result_246261 = invoke(stypy.reporting.localization.Localization(__file__, 78, 24), func_246257, *[x_246259], **kwargs_246260)
    
    # Processing the call keyword arguments (line 78)
    kwargs_246262 = {}
    # Getting the type of 'np' (line 78)
    np_246254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 9), 'np', False)
    # Obtaining the member 'linalg' of a type (line 78)
    linalg_246255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 9), np_246254, 'linalg')
    # Obtaining the member 'norm' of a type (line 78)
    norm_246256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 9), linalg_246255, 'norm')
    # Calling norm(args, kwargs) (line 78)
    norm_call_result_246263 = invoke(stypy.reporting.localization.Localization(__file__, 78, 9), norm_246256, *[func_call_result_246261], **kwargs_246262)
    
    # Assigning a type to the variable 'fx' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'fx', norm_call_result_246263)
    
    # Call to assert_(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Getting the type of 'fx' (line 79)
    fx_246265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'fx', False)
    # Getting the type of 'ftol' (line 79)
    ftol_246266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'ftol', False)
    # Getting the type of 'f0' (line 79)
    f0_246267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'f0', False)
    # Applying the binary operator '*' (line 79)
    result_mul_246268 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 18), '*', ftol_246266, f0_246267)
    
    # Applying the binary operator '<=' (line 79)
    result_le_246269 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 12), '<=', fx_246265, result_mul_246268)
    
    # Processing the call keyword arguments (line 79)
    kwargs_246270 = {}
    # Getting the type of 'assert_' (line 79)
    assert__246264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 79)
    assert__call_result_246271 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), assert__246264, *[result_le_246269], **kwargs_246270)
    
    
    # ################# End of 'test_complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_complex' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_246272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246272)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_complex'
    return stypy_return_type_246272

# Assigning a type to the variable 'test_complex' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'test_complex', test_complex)

@norecursion
def test_linear_definite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_linear_definite'
    module_type_store = module_type_store.open_function_context('test_linear_definite', 82, 0, False)
    
    # Passed parameters checking function
    test_linear_definite.stypy_localization = localization
    test_linear_definite.stypy_type_of_self = None
    test_linear_definite.stypy_type_store = module_type_store
    test_linear_definite.stypy_function_name = 'test_linear_definite'
    test_linear_definite.stypy_param_names_list = []
    test_linear_definite.stypy_varargs_param_name = None
    test_linear_definite.stypy_kwargs_param_name = None
    test_linear_definite.stypy_call_defaults = defaults
    test_linear_definite.stypy_call_varargs = varargs
    test_linear_definite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_linear_definite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_linear_definite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_linear_definite(...)' code ##################


    @norecursion
    def check_solvability(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_246273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 44), 'str', 'cruz')
        defaults = [str_246273]
        # Create a new context for function 'check_solvability'
        module_type_store = module_type_store.open_function_context('check_solvability', 89, 4, False)
        
        # Passed parameters checking function
        check_solvability.stypy_localization = localization
        check_solvability.stypy_type_of_self = None
        check_solvability.stypy_type_store = module_type_store
        check_solvability.stypy_function_name = 'check_solvability'
        check_solvability.stypy_param_names_list = ['A', 'b', 'line_search']
        check_solvability.stypy_varargs_param_name = None
        check_solvability.stypy_kwargs_param_name = None
        check_solvability.stypy_call_defaults = defaults
        check_solvability.stypy_call_varargs = varargs
        check_solvability.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_solvability', ['A', 'b', 'line_search'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_solvability', localization, ['A', 'b', 'line_search'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_solvability(...)' code ##################

        
        # Assigning a Lambda to a Name (line 90):

        @norecursion
        def _stypy_temp_lambda_166(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_166'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_166', 90, 15, True)
            # Passed parameters checking function
            _stypy_temp_lambda_166.stypy_localization = localization
            _stypy_temp_lambda_166.stypy_type_of_self = None
            _stypy_temp_lambda_166.stypy_type_store = module_type_store
            _stypy_temp_lambda_166.stypy_function_name = '_stypy_temp_lambda_166'
            _stypy_temp_lambda_166.stypy_param_names_list = ['x']
            _stypy_temp_lambda_166.stypy_varargs_param_name = None
            _stypy_temp_lambda_166.stypy_kwargs_param_name = None
            _stypy_temp_lambda_166.stypy_call_defaults = defaults
            _stypy_temp_lambda_166.stypy_call_varargs = varargs
            _stypy_temp_lambda_166.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_166', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_166', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to dot(...): (line 90)
            # Processing the call arguments (line 90)
            # Getting the type of 'x' (line 90)
            x_246276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'x', False)
            # Processing the call keyword arguments (line 90)
            kwargs_246277 = {}
            # Getting the type of 'A' (line 90)
            A_246274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'A', False)
            # Obtaining the member 'dot' of a type (line 90)
            dot_246275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 25), A_246274, 'dot')
            # Calling dot(args, kwargs) (line 90)
            dot_call_result_246278 = invoke(stypy.reporting.localization.Localization(__file__, 90, 25), dot_246275, *[x_246276], **kwargs_246277)
            
            # Getting the type of 'b' (line 90)
            b_246279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 36), 'b')
            # Applying the binary operator '-' (line 90)
            result_sub_246280 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 25), '-', dot_call_result_246278, b_246279)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'stypy_return_type', result_sub_246280)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_166' in the type store
            # Getting the type of 'stypy_return_type' (line 90)
            stypy_return_type_246281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_246281)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_166'
            return stypy_return_type_246281

        # Assigning a type to the variable '_stypy_temp_lambda_166' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), '_stypy_temp_lambda_166', _stypy_temp_lambda_166)
        # Getting the type of '_stypy_temp_lambda_166' (line 90)
        _stypy_temp_lambda_166_246282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), '_stypy_temp_lambda_166')
        # Assigning a type to the variable 'func' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'func', _stypy_temp_lambda_166_246282)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to solve(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'A' (line 91)
        A_246286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'A', False)
        # Getting the type of 'b' (line 91)
        b_246287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'b', False)
        # Processing the call keyword arguments (line 91)
        kwargs_246288 = {}
        # Getting the type of 'np' (line 91)
        np_246283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'np', False)
        # Obtaining the member 'linalg' of a type (line 91)
        linalg_246284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), np_246283, 'linalg')
        # Obtaining the member 'solve' of a type (line 91)
        solve_246285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), linalg_246284, 'solve')
        # Calling solve(args, kwargs) (line 91)
        solve_call_result_246289 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), solve_246285, *[A_246286, b_246287], **kwargs_246288)
        
        # Assigning a type to the variable 'xp' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'xp', solve_call_result_246289)
        
        # Assigning a BinOp to a Name (line 92):
        
        # Call to norm(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to func(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'xp' (line 92)
        xp_246294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 34), 'xp', False)
        # Processing the call keyword arguments (line 92)
        kwargs_246295 = {}
        # Getting the type of 'func' (line 92)
        func_246293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'func', False)
        # Calling func(args, kwargs) (line 92)
        func_call_result_246296 = invoke(stypy.reporting.localization.Localization(__file__, 92, 29), func_246293, *[xp_246294], **kwargs_246295)
        
        # Processing the call keyword arguments (line 92)
        kwargs_246297 = {}
        # Getting the type of 'np' (line 92)
        np_246290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'np', False)
        # Obtaining the member 'linalg' of a type (line 92)
        linalg_246291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 14), np_246290, 'linalg')
        # Obtaining the member 'norm' of a type (line 92)
        norm_246292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 14), linalg_246291, 'norm')
        # Calling norm(args, kwargs) (line 92)
        norm_call_result_246298 = invoke(stypy.reporting.localization.Localization(__file__, 92, 14), norm_246292, *[func_call_result_246296], **kwargs_246297)
        
        float_246299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 41), 'float')
        # Applying the binary operator '*' (line 92)
        result_mul_246300 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 14), '*', norm_call_result_246298, float_246299)
        
        # Assigning a type to the variable 'eps' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'eps', result_mul_246300)
        
        # Assigning a Call to a Name (line 93):
        
        # Call to root(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'func' (line 93)
        func_246302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'func', False)
        # Getting the type of 'b' (line 93)
        b_246303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'b', False)
        # Processing the call keyword arguments (line 93)
        
        # Call to dict(...): (line 93)
        # Processing the call keyword arguments (line 93)
        # Getting the type of 'eps' (line 93)
        eps_246305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 47), 'eps', False)
        keyword_246306 = eps_246305
        int_246307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 57), 'int')
        keyword_246308 = int_246307
        int_246309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 67), 'int')
        keyword_246310 = int_246309
        # Getting the type of 'line_search' (line 93)
        line_search_246311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 86), 'line_search', False)
        keyword_246312 = line_search_246311
        kwargs_246313 = {'fatol': keyword_246306, 'maxfev': keyword_246310, 'line_search': keyword_246312, 'ftol': keyword_246308}
        # Getting the type of 'dict' (line 93)
        dict_246304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'dict', False)
        # Calling dict(args, kwargs) (line 93)
        dict_call_result_246314 = invoke(stypy.reporting.localization.Localization(__file__, 93, 36), dict_246304, *[], **kwargs_246313)
        
        keyword_246315 = dict_call_result_246314
        str_246316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 26), 'str', 'DF-SANE')
        keyword_246317 = str_246316
        kwargs_246318 = {'options': keyword_246315, 'method': keyword_246317}
        # Getting the type of 'root' (line 93)
        root_246301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'root', False)
        # Calling root(args, kwargs) (line 93)
        root_call_result_246319 = invoke(stypy.reporting.localization.Localization(__file__, 93, 14), root_246301, *[func_246302, b_246303], **kwargs_246318)
        
        # Assigning a type to the variable 'sol' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'sol', root_call_result_246319)
        
        # Call to assert_(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'sol' (line 95)
        sol_246321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'sol', False)
        # Obtaining the member 'success' of a type (line 95)
        success_246322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), sol_246321, 'success')
        # Processing the call keyword arguments (line 95)
        kwargs_246323 = {}
        # Getting the type of 'assert_' (line 95)
        assert__246320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 95)
        assert__call_result_246324 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assert__246320, *[success_246322], **kwargs_246323)
        
        
        # Call to assert_(...): (line 96)
        # Processing the call arguments (line 96)
        
        
        # Call to norm(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to func(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'sol' (line 96)
        sol_246330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 36), 'sol', False)
        # Obtaining the member 'x' of a type (line 96)
        x_246331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 36), sol_246330, 'x')
        # Processing the call keyword arguments (line 96)
        kwargs_246332 = {}
        # Getting the type of 'func' (line 96)
        func_246329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'func', False)
        # Calling func(args, kwargs) (line 96)
        func_call_result_246333 = invoke(stypy.reporting.localization.Localization(__file__, 96, 31), func_246329, *[x_246331], **kwargs_246332)
        
        # Processing the call keyword arguments (line 96)
        kwargs_246334 = {}
        # Getting the type of 'np' (line 96)
        np_246326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'np', False)
        # Obtaining the member 'linalg' of a type (line 96)
        linalg_246327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), np_246326, 'linalg')
        # Obtaining the member 'norm' of a type (line 96)
        norm_246328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), linalg_246327, 'norm')
        # Calling norm(args, kwargs) (line 96)
        norm_call_result_246335 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), norm_246328, *[func_call_result_246333], **kwargs_246334)
        
        # Getting the type of 'eps' (line 96)
        eps_246336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 47), 'eps', False)
        # Applying the binary operator '<=' (line 96)
        result_le_246337 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 16), '<=', norm_call_result_246335, eps_246336)
        
        # Processing the call keyword arguments (line 96)
        kwargs_246338 = {}
        # Getting the type of 'assert_' (line 96)
        assert__246325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 96)
        assert__call_result_246339 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), assert__246325, *[result_le_246337], **kwargs_246338)
        
        
        # ################# End of 'check_solvability(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_solvability' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_246340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_246340)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_solvability'
        return stypy_return_type_246340

    # Assigning a type to the variable 'check_solvability' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'check_solvability', check_solvability)
    
    # Assigning a Num to a Name (line 98):
    int_246341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'int')
    # Assigning a type to the variable 'n' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'n', int_246341)
    
    # Call to seed(...): (line 101)
    # Processing the call arguments (line 101)
    int_246345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 19), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_246346 = {}
    # Getting the type of 'np' (line 101)
    np_246342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 101)
    random_246343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), np_246342, 'random')
    # Obtaining the member 'seed' of a type (line 101)
    seed_246344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), random_246343, 'seed')
    # Calling seed(args, kwargs) (line 101)
    seed_call_result_246347 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), seed_246344, *[int_246345], **kwargs_246346)
    
    
    # Assigning a Call to a Name (line 102):
    
    # Call to reshape(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'n' (line 102)
    n_246356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'n', False)
    # Getting the type of 'n' (line 102)
    n_246357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'n', False)
    # Processing the call keyword arguments (line 102)
    kwargs_246358 = {}
    
    # Call to arange(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'n' (line 102)
    n_246350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 18), 'n', False)
    # Getting the type of 'n' (line 102)
    n_246351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'n', False)
    # Applying the binary operator '*' (line 102)
    result_mul_246352 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 18), '*', n_246350, n_246351)
    
    # Processing the call keyword arguments (line 102)
    kwargs_246353 = {}
    # Getting the type of 'np' (line 102)
    np_246348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 102)
    arange_246349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), np_246348, 'arange')
    # Calling arange(args, kwargs) (line 102)
    arange_call_result_246354 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), arange_246349, *[result_mul_246352], **kwargs_246353)
    
    # Obtaining the member 'reshape' of a type (line 102)
    reshape_246355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), arange_call_result_246354, 'reshape')
    # Calling reshape(args, kwargs) (line 102)
    reshape_call_result_246359 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), reshape_246355, *[n_246356, n_246357], **kwargs_246358)
    
    # Assigning a type to the variable 'A' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'A', reshape_call_result_246359)
    
    # Assigning a BinOp to a Name (line 103):
    # Getting the type of 'A' (line 103)
    A_246360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'A')
    # Getting the type of 'n' (line 103)
    n_246361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'n')
    # Getting the type of 'n' (line 103)
    n_246362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 14), 'n')
    # Applying the binary operator '*' (line 103)
    result_mul_246363 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 12), '*', n_246361, n_246362)
    
    
    # Call to diag(...): (line 103)
    # Processing the call arguments (line 103)
    int_246366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'int')
    
    # Call to arange(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'n' (line 103)
    n_246369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'n', False)
    # Processing the call keyword arguments (line 103)
    kwargs_246370 = {}
    # Getting the type of 'np' (line 103)
    np_246367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'np', False)
    # Obtaining the member 'arange' of a type (line 103)
    arange_246368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 30), np_246367, 'arange')
    # Calling arange(args, kwargs) (line 103)
    arange_call_result_246371 = invoke(stypy.reporting.localization.Localization(__file__, 103, 30), arange_246368, *[n_246369], **kwargs_246370)
    
    # Applying the binary operator '+' (line 103)
    result_add_246372 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 26), '+', int_246366, arange_call_result_246371)
    
    # Processing the call keyword arguments (line 103)
    kwargs_246373 = {}
    # Getting the type of 'np' (line 103)
    np_246364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'np', False)
    # Obtaining the member 'diag' of a type (line 103)
    diag_246365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 18), np_246364, 'diag')
    # Calling diag(args, kwargs) (line 103)
    diag_call_result_246374 = invoke(stypy.reporting.localization.Localization(__file__, 103, 18), diag_246365, *[result_add_246372], **kwargs_246373)
    
    # Applying the binary operator '*' (line 103)
    result_mul_246375 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 16), '*', result_mul_246363, diag_call_result_246374)
    
    # Applying the binary operator '+' (line 103)
    result_add_246376 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 8), '+', A_246360, result_mul_246375)
    
    # Assigning a type to the variable 'A' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'A', result_add_246376)
    
    # Call to assert_(...): (line 104)
    # Processing the call arguments (line 104)
    
    
    # Call to min(...): (line 104)
    # Processing the call keyword arguments (line 104)
    kwargs_246385 = {}
    
    # Call to eigvals(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'A' (line 104)
    A_246381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'A', False)
    # Processing the call keyword arguments (line 104)
    kwargs_246382 = {}
    # Getting the type of 'np' (line 104)
    np_246378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'np', False)
    # Obtaining the member 'linalg' of a type (line 104)
    linalg_246379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), np_246378, 'linalg')
    # Obtaining the member 'eigvals' of a type (line 104)
    eigvals_246380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), linalg_246379, 'eigvals')
    # Calling eigvals(args, kwargs) (line 104)
    eigvals_call_result_246383 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), eigvals_246380, *[A_246381], **kwargs_246382)
    
    # Obtaining the member 'min' of a type (line 104)
    min_246384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), eigvals_call_result_246383, 'min')
    # Calling min(args, kwargs) (line 104)
    min_call_result_246386 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), min_246384, *[], **kwargs_246385)
    
    int_246387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 41), 'int')
    # Applying the binary operator '>' (line 104)
    result_gt_246388 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 12), '>', min_call_result_246386, int_246387)
    
    # Processing the call keyword arguments (line 104)
    kwargs_246389 = {}
    # Getting the type of 'assert_' (line 104)
    assert__246377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 104)
    assert__call_result_246390 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), assert__246377, *[result_gt_246388], **kwargs_246389)
    
    
    # Assigning a BinOp to a Name (line 105):
    
    # Call to arange(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'n' (line 105)
    n_246393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'n', False)
    # Processing the call keyword arguments (line 105)
    kwargs_246394 = {}
    # Getting the type of 'np' (line 105)
    np_246391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 105)
    arange_246392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), np_246391, 'arange')
    # Calling arange(args, kwargs) (line 105)
    arange_call_result_246395 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), arange_246392, *[n_246393], **kwargs_246394)
    
    float_246396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'float')
    # Applying the binary operator '*' (line 105)
    result_mul_246397 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 8), '*', arange_call_result_246395, float_246396)
    
    # Assigning a type to the variable 'b' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'b', result_mul_246397)
    
    # Call to check_solvability(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'A' (line 106)
    A_246399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'A', False)
    # Getting the type of 'b' (line 106)
    b_246400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'b', False)
    str_246401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'str', 'cruz')
    # Processing the call keyword arguments (line 106)
    kwargs_246402 = {}
    # Getting the type of 'check_solvability' (line 106)
    check_solvability_246398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'check_solvability', False)
    # Calling check_solvability(args, kwargs) (line 106)
    check_solvability_call_result_246403 = invoke(stypy.reporting.localization.Localization(__file__, 106, 4), check_solvability_246398, *[A_246399, b_246400, str_246401], **kwargs_246402)
    
    
    # Call to check_solvability(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'A' (line 107)
    A_246405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'A', False)
    # Getting the type of 'b' (line 107)
    b_246406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'b', False)
    str_246407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 28), 'str', 'cheng')
    # Processing the call keyword arguments (line 107)
    kwargs_246408 = {}
    # Getting the type of 'check_solvability' (line 107)
    check_solvability_246404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'check_solvability', False)
    # Calling check_solvability(args, kwargs) (line 107)
    check_solvability_call_result_246409 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), check_solvability_246404, *[A_246405, b_246406, str_246407], **kwargs_246408)
    
    
    # Call to check_solvability(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Getting the type of 'A' (line 110)
    A_246411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'A', False)
    # Applying the 'usub' unary operator (line 110)
    result___neg___246412 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 22), 'usub', A_246411)
    
    # Getting the type of 'b' (line 110)
    b_246413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'b', False)
    str_246414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'str', 'cruz')
    # Processing the call keyword arguments (line 110)
    kwargs_246415 = {}
    # Getting the type of 'check_solvability' (line 110)
    check_solvability_246410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'check_solvability', False)
    # Calling check_solvability(args, kwargs) (line 110)
    check_solvability_call_result_246416 = invoke(stypy.reporting.localization.Localization(__file__, 110, 4), check_solvability_246410, *[result___neg___246412, b_246413, str_246414], **kwargs_246415)
    
    
    # Call to check_solvability(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Getting the type of 'A' (line 111)
    A_246418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'A', False)
    # Applying the 'usub' unary operator (line 111)
    result___neg___246419 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 22), 'usub', A_246418)
    
    # Getting the type of 'b' (line 111)
    b_246420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'b', False)
    str_246421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'str', 'cheng')
    # Processing the call keyword arguments (line 111)
    kwargs_246422 = {}
    # Getting the type of 'check_solvability' (line 111)
    check_solvability_246417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'check_solvability', False)
    # Calling check_solvability(args, kwargs) (line 111)
    check_solvability_call_result_246423 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), check_solvability_246417, *[result___neg___246419, b_246420, str_246421], **kwargs_246422)
    
    
    # ################# End of 'test_linear_definite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_linear_definite' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_246424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246424)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_linear_definite'
    return stypy_return_type_246424

# Assigning a type to the variable 'test_linear_definite' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'test_linear_definite', test_linear_definite)

@norecursion
def test_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_shape'
    module_type_store = module_type_store.open_function_context('test_shape', 114, 0, False)
    
    # Passed parameters checking function
    test_shape.stypy_localization = localization
    test_shape.stypy_type_of_self = None
    test_shape.stypy_type_store = module_type_store
    test_shape.stypy_function_name = 'test_shape'
    test_shape.stypy_param_names_list = []
    test_shape.stypy_varargs_param_name = None
    test_shape.stypy_kwargs_param_name = None
    test_shape.stypy_call_defaults = defaults
    test_shape.stypy_call_varargs = varargs
    test_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_shape', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_shape', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_shape(...)' code ##################


    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 115, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['x', 'arg']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['x', 'arg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['x', 'arg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        # Getting the type of 'x' (line 116)
        x_246425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'x')
        # Getting the type of 'arg' (line 116)
        arg_246426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'arg')
        # Applying the binary operator '-' (line 116)
        result_sub_246427 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), '-', x_246425, arg_246426)
        
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', result_sub_246427)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_246428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_246428)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_246428

    # Assigning a type to the variable 'f' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'f', f)
    
    
    # Obtaining an instance of the builtin type 'list' (line 118)
    list_246429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 118)
    # Adding element type (line 118)
    # Getting the type of 'float' (line 118)
    float_246430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 14), list_246429, float_246430)
    # Adding element type (line 118)
    # Getting the type of 'complex' (line 118)
    complex_246431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 14), list_246429, complex_246431)
    
    # Testing the type of a for loop iterable (line 118)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 4), list_246429)
    # Getting the type of the for loop variable (line 118)
    for_loop_var_246432 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 4), list_246429)
    # Assigning a type to the variable 'dt' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'dt', for_loop_var_246432)
    # SSA begins for a for statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 119):
    
    # Call to zeros(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Obtaining an instance of the builtin type 'list' (line 119)
    list_246435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 119)
    # Adding element type (line 119)
    int_246436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 21), list_246435, int_246436)
    # Adding element type (line 119)
    int_246437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 21), list_246435, int_246437)
    
    # Processing the call keyword arguments (line 119)
    kwargs_246438 = {}
    # Getting the type of 'np' (line 119)
    np_246433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 119)
    zeros_246434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), np_246433, 'zeros')
    # Calling zeros(args, kwargs) (line 119)
    zeros_call_result_246439 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), zeros_246434, *[list_246435], **kwargs_246438)
    
    # Assigning a type to the variable 'x' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'x', zeros_call_result_246439)
    
    # Assigning a Call to a Name (line 120):
    
    # Call to ones(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Obtaining an instance of the builtin type 'list' (line 120)
    list_246442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 120)
    # Adding element type (line 120)
    int_246443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 22), list_246442, int_246443)
    # Adding element type (line 120)
    int_246444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 22), list_246442, int_246444)
    
    # Processing the call keyword arguments (line 120)
    # Getting the type of 'dt' (line 120)
    dt_246445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'dt', False)
    keyword_246446 = dt_246445
    kwargs_246447 = {'dtype': keyword_246446}
    # Getting the type of 'np' (line 120)
    np_246440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'np', False)
    # Obtaining the member 'ones' of a type (line 120)
    ones_246441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 14), np_246440, 'ones')
    # Calling ones(args, kwargs) (line 120)
    ones_call_result_246448 = invoke(stypy.reporting.localization.Localization(__file__, 120, 14), ones_246441, *[list_246442], **kwargs_246447)
    
    # Assigning a type to the variable 'arg' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'arg', ones_call_result_246448)
    
    # Assigning a Call to a Name (line 122):
    
    # Call to root(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'f' (line 122)
    f_246450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'f', False)
    # Getting the type of 'x' (line 122)
    x_246451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'x', False)
    # Processing the call keyword arguments (line 122)
    
    # Obtaining an instance of the builtin type 'tuple' (line 122)
    tuple_246452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 122)
    # Adding element type (line 122)
    # Getting the type of 'arg' (line 122)
    arg_246453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 31), tuple_246452, arg_246453)
    
    keyword_246454 = tuple_246452
    str_246455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 45), 'str', 'DF-SANE')
    keyword_246456 = str_246455
    kwargs_246457 = {'args': keyword_246454, 'method': keyword_246456}
    # Getting the type of 'root' (line 122)
    root_246449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'root', False)
    # Calling root(args, kwargs) (line 122)
    root_call_result_246458 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), root_246449, *[f_246450, x_246451], **kwargs_246457)
    
    # Assigning a type to the variable 'sol' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'sol', root_call_result_246458)
    
    # Call to assert_(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'sol' (line 123)
    sol_246460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'sol', False)
    # Obtaining the member 'success' of a type (line 123)
    success_246461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), sol_246460, 'success')
    # Processing the call keyword arguments (line 123)
    kwargs_246462 = {}
    # Getting the type of 'assert_' (line 123)
    assert__246459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 123)
    assert__call_result_246463 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), assert__246459, *[success_246461], **kwargs_246462)
    
    
    # Call to assert_equal(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'sol' (line 124)
    sol_246465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'sol', False)
    # Obtaining the member 'x' of a type (line 124)
    x_246466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 21), sol_246465, 'x')
    # Obtaining the member 'shape' of a type (line 124)
    shape_246467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 21), x_246466, 'shape')
    # Getting the type of 'x' (line 124)
    x_246468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'x', False)
    # Obtaining the member 'shape' of a type (line 124)
    shape_246469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 34), x_246468, 'shape')
    # Processing the call keyword arguments (line 124)
    kwargs_246470 = {}
    # Getting the type of 'assert_equal' (line 124)
    assert_equal_246464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 124)
    assert_equal_call_result_246471 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assert_equal_246464, *[shape_246467, shape_246469], **kwargs_246470)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_246472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_shape'
    return stypy_return_type_246472

# Assigning a type to the variable 'test_shape' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'test_shape', test_shape)

@norecursion
def F_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F_1'
    module_type_store = module_type_store.open_function_context('F_1', 130, 0, False)
    
    # Passed parameters checking function
    F_1.stypy_localization = localization
    F_1.stypy_type_of_self = None
    F_1.stypy_type_store = module_type_store
    F_1.stypy_function_name = 'F_1'
    F_1.stypy_param_names_list = ['x', 'n']
    F_1.stypy_varargs_param_name = None
    F_1.stypy_kwargs_param_name = None
    F_1.stypy_call_defaults = defaults
    F_1.stypy_call_varargs = varargs
    F_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F_1', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F_1', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F_1(...)' code ##################

    
    # Assigning a Call to a Name (line 131):
    
    # Call to zeros(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Obtaining an instance of the builtin type 'list' (line 131)
    list_246475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 131)
    # Adding element type (line 131)
    # Getting the type of 'n' (line 131)
    n_246476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 17), list_246475, n_246476)
    
    # Processing the call keyword arguments (line 131)
    kwargs_246477 = {}
    # Getting the type of 'np' (line 131)
    np_246473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 131)
    zeros_246474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), np_246473, 'zeros')
    # Calling zeros(args, kwargs) (line 131)
    zeros_call_result_246478 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), zeros_246474, *[list_246475], **kwargs_246477)
    
    # Assigning a type to the variable 'g' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'g', zeros_call_result_246478)
    
    # Assigning a Call to a Name (line 132):
    
    # Call to arange(...): (line 132)
    # Processing the call arguments (line 132)
    int_246481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 18), 'int')
    # Getting the type of 'n' (line 132)
    n_246482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 21), 'n', False)
    int_246483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'int')
    # Applying the binary operator '+' (line 132)
    result_add_246484 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 21), '+', n_246482, int_246483)
    
    # Processing the call keyword arguments (line 132)
    kwargs_246485 = {}
    # Getting the type of 'np' (line 132)
    np_246479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 132)
    arange_246480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), np_246479, 'arange')
    # Calling arange(args, kwargs) (line 132)
    arange_call_result_246486 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), arange_246480, *[int_246481, result_add_246484], **kwargs_246485)
    
    # Assigning a type to the variable 'i' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'i', arange_call_result_246486)
    
    # Assigning a BinOp to a Subscript (line 133):
    
    # Call to exp(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Obtaining the type of the subscript
    int_246488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 17), 'int')
    # Getting the type of 'x' (line 133)
    x_246489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___246490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), x_246489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_246491 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), getitem___246490, int_246488)
    
    int_246492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 22), 'int')
    # Applying the binary operator '-' (line 133)
    result_sub_246493 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 15), '-', subscript_call_result_246491, int_246492)
    
    # Processing the call keyword arguments (line 133)
    kwargs_246494 = {}
    # Getting the type of 'exp' (line 133)
    exp_246487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'exp', False)
    # Calling exp(args, kwargs) (line 133)
    exp_call_result_246495 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), exp_246487, *[result_sub_246493], **kwargs_246494)
    
    int_246496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'int')
    # Applying the binary operator '-' (line 133)
    result_sub_246497 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 11), '-', exp_call_result_246495, int_246496)
    
    # Getting the type of 'g' (line 133)
    g_246498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'g')
    int_246499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 6), 'int')
    # Storing an element on a container (line 133)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 4), g_246498, (int_246499, result_sub_246497))
    
    # Assigning a BinOp to a Subscript (line 134):
    # Getting the type of 'i' (line 134)
    i_246500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'i')
    
    # Call to exp(...): (line 134)
    # Processing the call arguments (line 134)
    
    # Obtaining the type of the subscript
    int_246502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 21), 'int')
    slice_246503 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 19), int_246502, None, None)
    # Getting the type of 'x' (line 134)
    x_246504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 19), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___246505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 19), x_246504, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_246506 = invoke(stypy.reporting.localization.Localization(__file__, 134, 19), getitem___246505, slice_246503)
    
    int_246507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 27), 'int')
    # Applying the binary operator '-' (line 134)
    result_sub_246508 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 19), '-', subscript_call_result_246506, int_246507)
    
    # Processing the call keyword arguments (line 134)
    kwargs_246509 = {}
    # Getting the type of 'exp' (line 134)
    exp_246501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'exp', False)
    # Calling exp(args, kwargs) (line 134)
    exp_call_result_246510 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), exp_246501, *[result_sub_246508], **kwargs_246509)
    
    
    # Obtaining the type of the subscript
    int_246511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'int')
    slice_246512 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 32), int_246511, None, None)
    # Getting the type of 'x' (line 134)
    x_246513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 32), 'x')
    # Obtaining the member '__getitem__' of a type (line 134)
    getitem___246514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 32), x_246513, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 134)
    subscript_call_result_246515 = invoke(stypy.reporting.localization.Localization(__file__, 134, 32), getitem___246514, slice_246512)
    
    # Applying the binary operator '-' (line 134)
    result_sub_246516 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '-', exp_call_result_246510, subscript_call_result_246515)
    
    # Applying the binary operator '*' (line 134)
    result_mul_246517 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 12), '*', i_246500, result_sub_246516)
    
    # Getting the type of 'g' (line 134)
    g_246518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'g')
    int_246519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 6), 'int')
    slice_246520 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 4), int_246519, None, None)
    # Storing an element on a container (line 134)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 4), g_246518, (slice_246520, result_mul_246517))
    # Getting the type of 'g' (line 135)
    g_246521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'g')
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type', g_246521)
    
    # ################# End of 'F_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F_1' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_246522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246522)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F_1'
    return stypy_return_type_246522

# Assigning a type to the variable 'F_1' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'F_1', F_1)

@norecursion
def x0_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'x0_1'
    module_type_store = module_type_store.open_function_context('x0_1', 137, 0, False)
    
    # Passed parameters checking function
    x0_1.stypy_localization = localization
    x0_1.stypy_type_of_self = None
    x0_1.stypy_type_store = module_type_store
    x0_1.stypy_function_name = 'x0_1'
    x0_1.stypy_param_names_list = ['n']
    x0_1.stypy_varargs_param_name = None
    x0_1.stypy_kwargs_param_name = None
    x0_1.stypy_call_defaults = defaults
    x0_1.stypy_call_varargs = varargs
    x0_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'x0_1', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'x0_1', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'x0_1(...)' code ##################

    
    # Assigning a Call to a Name (line 138):
    
    # Call to empty(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Obtaining an instance of the builtin type 'list' (line 138)
    list_246525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 138)
    # Adding element type (line 138)
    # Getting the type of 'n' (line 138)
    n_246526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 18), list_246525, n_246526)
    
    # Processing the call keyword arguments (line 138)
    kwargs_246527 = {}
    # Getting the type of 'np' (line 138)
    np_246523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 9), 'np', False)
    # Obtaining the member 'empty' of a type (line 138)
    empty_246524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 9), np_246523, 'empty')
    # Calling empty(args, kwargs) (line 138)
    empty_call_result_246528 = invoke(stypy.reporting.localization.Localization(__file__, 138, 9), empty_246524, *[list_246525], **kwargs_246527)
    
    # Assigning a type to the variable 'x0' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'x0', empty_call_result_246528)
    
    # Call to fill(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'n' (line 139)
    n_246531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'n', False)
    # Getting the type of 'n' (line 139)
    n_246532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'n', False)
    int_246533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 17), 'int')
    # Applying the binary operator '-' (line 139)
    result_sub_246534 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 15), '-', n_246532, int_246533)
    
    # Applying the binary operator 'div' (line 139)
    result_div_246535 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 12), 'div', n_246531, result_sub_246534)
    
    # Processing the call keyword arguments (line 139)
    kwargs_246536 = {}
    # Getting the type of 'x0' (line 139)
    x0_246529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'x0', False)
    # Obtaining the member 'fill' of a type (line 139)
    fill_246530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 4), x0_246529, 'fill')
    # Calling fill(args, kwargs) (line 139)
    fill_call_result_246537 = invoke(stypy.reporting.localization.Localization(__file__, 139, 4), fill_246530, *[result_div_246535], **kwargs_246536)
    
    # Getting the type of 'x0' (line 140)
    x0_246538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'x0')
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type', x0_246538)
    
    # ################# End of 'x0_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'x0_1' in the type store
    # Getting the type of 'stypy_return_type' (line 137)
    stypy_return_type_246539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246539)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'x0_1'
    return stypy_return_type_246539

# Assigning a type to the variable 'x0_1' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'x0_1', x0_1)

@norecursion
def F_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F_2'
    module_type_store = module_type_store.open_function_context('F_2', 142, 0, False)
    
    # Passed parameters checking function
    F_2.stypy_localization = localization
    F_2.stypy_type_of_self = None
    F_2.stypy_type_store = module_type_store
    F_2.stypy_function_name = 'F_2'
    F_2.stypy_param_names_list = ['x', 'n']
    F_2.stypy_varargs_param_name = None
    F_2.stypy_kwargs_param_name = None
    F_2.stypy_call_defaults = defaults
    F_2.stypy_call_varargs = varargs
    F_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F_2', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F_2', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F_2(...)' code ##################

    
    # Assigning a Call to a Name (line 143):
    
    # Call to zeros(...): (line 143)
    # Processing the call arguments (line 143)
    
    # Obtaining an instance of the builtin type 'list' (line 143)
    list_246542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 143)
    # Adding element type (line 143)
    # Getting the type of 'n' (line 143)
    n_246543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 18), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 17), list_246542, n_246543)
    
    # Processing the call keyword arguments (line 143)
    kwargs_246544 = {}
    # Getting the type of 'np' (line 143)
    np_246540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 143)
    zeros_246541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), np_246540, 'zeros')
    # Calling zeros(args, kwargs) (line 143)
    zeros_call_result_246545 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), zeros_246541, *[list_246542], **kwargs_246544)
    
    # Assigning a type to the variable 'g' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'g', zeros_call_result_246545)
    
    # Assigning a Call to a Name (line 144):
    
    # Call to arange(...): (line 144)
    # Processing the call arguments (line 144)
    int_246548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 18), 'int')
    # Getting the type of 'n' (line 144)
    n_246549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'n', False)
    int_246550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'int')
    # Applying the binary operator '+' (line 144)
    result_add_246551 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 21), '+', n_246549, int_246550)
    
    # Processing the call keyword arguments (line 144)
    kwargs_246552 = {}
    # Getting the type of 'np' (line 144)
    np_246546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 144)
    arange_246547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), np_246546, 'arange')
    # Calling arange(args, kwargs) (line 144)
    arange_call_result_246553 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), arange_246547, *[int_246548, result_add_246551], **kwargs_246552)
    
    # Assigning a type to the variable 'i' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'i', arange_call_result_246553)
    
    # Assigning a BinOp to a Subscript (line 145):
    
    # Call to exp(...): (line 145)
    # Processing the call arguments (line 145)
    
    # Obtaining the type of the subscript
    int_246555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 17), 'int')
    # Getting the type of 'x' (line 145)
    x_246556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 145)
    getitem___246557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 15), x_246556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 145)
    subscript_call_result_246558 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), getitem___246557, int_246555)
    
    # Processing the call keyword arguments (line 145)
    kwargs_246559 = {}
    # Getting the type of 'exp' (line 145)
    exp_246554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'exp', False)
    # Calling exp(args, kwargs) (line 145)
    exp_call_result_246560 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), exp_246554, *[subscript_call_result_246558], **kwargs_246559)
    
    int_246561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'int')
    # Applying the binary operator '-' (line 145)
    result_sub_246562 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), '-', exp_call_result_246560, int_246561)
    
    # Getting the type of 'g' (line 145)
    g_246563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'g')
    int_246564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 6), 'int')
    # Storing an element on a container (line 145)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 4), g_246563, (int_246564, result_sub_246562))
    
    # Assigning a BinOp to a Subscript (line 146):
    float_246565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 12), 'float')
    # Getting the type of 'i' (line 146)
    i_246566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'i')
    # Applying the binary operator '*' (line 146)
    result_mul_246567 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 12), '*', float_246565, i_246566)
    
    
    # Call to exp(...): (line 146)
    # Processing the call arguments (line 146)
    
    # Obtaining the type of the subscript
    int_246569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 25), 'int')
    slice_246570 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 23), int_246569, None, None)
    # Getting the type of 'x' (line 146)
    x_246571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 146)
    getitem___246572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 23), x_246571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 146)
    subscript_call_result_246573 = invoke(stypy.reporting.localization.Localization(__file__, 146, 23), getitem___246572, slice_246570)
    
    # Processing the call keyword arguments (line 146)
    kwargs_246574 = {}
    # Getting the type of 'exp' (line 146)
    exp_246568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'exp', False)
    # Calling exp(args, kwargs) (line 146)
    exp_call_result_246575 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), exp_246568, *[subscript_call_result_246573], **kwargs_246574)
    
    
    # Obtaining the type of the subscript
    int_246576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 35), 'int')
    slice_246577 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 32), None, int_246576, None)
    # Getting the type of 'x' (line 146)
    x_246578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'x')
    # Obtaining the member '__getitem__' of a type (line 146)
    getitem___246579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 32), x_246578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 146)
    subscript_call_result_246580 = invoke(stypy.reporting.localization.Localization(__file__, 146, 32), getitem___246579, slice_246577)
    
    # Applying the binary operator '+' (line 146)
    result_add_246581 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 19), '+', exp_call_result_246575, subscript_call_result_246580)
    
    int_246582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 41), 'int')
    # Applying the binary operator '-' (line 146)
    result_sub_246583 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 39), '-', result_add_246581, int_246582)
    
    # Applying the binary operator '*' (line 146)
    result_mul_246584 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 17), '*', result_mul_246567, result_sub_246583)
    
    # Getting the type of 'g' (line 146)
    g_246585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'g')
    int_246586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 6), 'int')
    slice_246587 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 4), int_246586, None, None)
    # Storing an element on a container (line 146)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 4), g_246585, (slice_246587, result_mul_246584))
    # Getting the type of 'g' (line 147)
    g_246588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'g')
    # Assigning a type to the variable 'stypy_return_type' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type', g_246588)
    
    # ################# End of 'F_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F_2' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_246589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246589)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F_2'
    return stypy_return_type_246589

# Assigning a type to the variable 'F_2' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'F_2', F_2)

@norecursion
def x0_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'x0_2'
    module_type_store = module_type_store.open_function_context('x0_2', 149, 0, False)
    
    # Passed parameters checking function
    x0_2.stypy_localization = localization
    x0_2.stypy_type_of_self = None
    x0_2.stypy_type_store = module_type_store
    x0_2.stypy_function_name = 'x0_2'
    x0_2.stypy_param_names_list = ['n']
    x0_2.stypy_varargs_param_name = None
    x0_2.stypy_kwargs_param_name = None
    x0_2.stypy_call_defaults = defaults
    x0_2.stypy_call_varargs = varargs
    x0_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'x0_2', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'x0_2', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'x0_2(...)' code ##################

    
    # Assigning a Call to a Name (line 150):
    
    # Call to empty(...): (line 150)
    # Processing the call arguments (line 150)
    
    # Obtaining an instance of the builtin type 'list' (line 150)
    list_246592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 150)
    # Adding element type (line 150)
    # Getting the type of 'n' (line 150)
    n_246593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), list_246592, n_246593)
    
    # Processing the call keyword arguments (line 150)
    kwargs_246594 = {}
    # Getting the type of 'np' (line 150)
    np_246590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 9), 'np', False)
    # Obtaining the member 'empty' of a type (line 150)
    empty_246591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 9), np_246590, 'empty')
    # Calling empty(args, kwargs) (line 150)
    empty_call_result_246595 = invoke(stypy.reporting.localization.Localization(__file__, 150, 9), empty_246591, *[list_246592], **kwargs_246594)
    
    # Assigning a type to the variable 'x0' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'x0', empty_call_result_246595)
    
    # Call to fill(...): (line 151)
    # Processing the call arguments (line 151)
    int_246598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
    # Getting the type of 'n' (line 151)
    n_246599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'n', False)
    int_246600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 17), 'int')
    # Applying the binary operator '**' (line 151)
    result_pow_246601 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 14), '**', n_246599, int_246600)
    
    # Applying the binary operator 'div' (line 151)
    result_div_246602 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 12), 'div', int_246598, result_pow_246601)
    
    # Processing the call keyword arguments (line 151)
    kwargs_246603 = {}
    # Getting the type of 'x0' (line 151)
    x0_246596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'x0', False)
    # Obtaining the member 'fill' of a type (line 151)
    fill_246597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 4), x0_246596, 'fill')
    # Calling fill(args, kwargs) (line 151)
    fill_call_result_246604 = invoke(stypy.reporting.localization.Localization(__file__, 151, 4), fill_246597, *[result_div_246602], **kwargs_246603)
    
    # Getting the type of 'x0' (line 152)
    x0_246605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'x0')
    # Assigning a type to the variable 'stypy_return_type' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type', x0_246605)
    
    # ################# End of 'x0_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'x0_2' in the type store
    # Getting the type of 'stypy_return_type' (line 149)
    stypy_return_type_246606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246606)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'x0_2'
    return stypy_return_type_246606

# Assigning a type to the variable 'x0_2' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'x0_2', x0_2)

@norecursion
def F_4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F_4'
    module_type_store = module_type_store.open_function_context('F_4', 154, 0, False)
    
    # Passed parameters checking function
    F_4.stypy_localization = localization
    F_4.stypy_type_of_self = None
    F_4.stypy_type_store = module_type_store
    F_4.stypy_function_name = 'F_4'
    F_4.stypy_param_names_list = ['x', 'n']
    F_4.stypy_varargs_param_name = None
    F_4.stypy_kwargs_param_name = None
    F_4.stypy_call_defaults = defaults
    F_4.stypy_call_varargs = varargs
    F_4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F_4', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F_4', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F_4(...)' code ##################

    
    # Call to assert_equal(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'n' (line 155)
    n_246608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'n', False)
    int_246609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'int')
    # Applying the binary operator '%' (line 155)
    result_mod_246610 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 17), '%', n_246608, int_246609)
    
    int_246611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'int')
    # Processing the call keyword arguments (line 155)
    kwargs_246612 = {}
    # Getting the type of 'assert_equal' (line 155)
    assert_equal_246607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 155)
    assert_equal_call_result_246613 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), assert_equal_246607, *[result_mod_246610, int_246611], **kwargs_246612)
    
    
    # Assigning a Call to a Name (line 156):
    
    # Call to zeros(...): (line 156)
    # Processing the call arguments (line 156)
    
    # Obtaining an instance of the builtin type 'list' (line 156)
    list_246616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 156)
    # Adding element type (line 156)
    # Getting the type of 'n' (line 156)
    n_246617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 18), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 17), list_246616, n_246617)
    
    # Processing the call keyword arguments (line 156)
    kwargs_246618 = {}
    # Getting the type of 'np' (line 156)
    np_246614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 156)
    zeros_246615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), np_246614, 'zeros')
    # Calling zeros(args, kwargs) (line 156)
    zeros_call_result_246619 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), zeros_246615, *[list_246616], **kwargs_246618)
    
    # Assigning a type to the variable 'g' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'g', zeros_call_result_246619)
    
    # Assigning a BinOp to a Subscript (line 159):
    float_246620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 13), 'float')
    
    # Obtaining the type of the subscript
    int_246621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 23), 'int')
    slice_246622 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 19), None, None, int_246621)
    # Getting the type of 'x' (line 159)
    x_246623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'x')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___246624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 19), x_246623, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_246625 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), getitem___246624, slice_246622)
    
    # Applying the binary operator '*' (line 159)
    result_mul_246626 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 13), '*', float_246620, subscript_call_result_246625)
    
    float_246627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 28), 'float')
    
    # Obtaining the type of the subscript
    int_246628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 36), 'int')
    int_246629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 39), 'int')
    slice_246630 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 34), int_246628, None, int_246629)
    # Getting the type of 'x' (line 159)
    x_246631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'x')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___246632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 34), x_246631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_246633 = invoke(stypy.reporting.localization.Localization(__file__, 159, 34), getitem___246632, slice_246630)
    
    int_246634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'int')
    # Applying the binary operator '**' (line 159)
    result_pow_246635 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 34), '**', subscript_call_result_246633, int_246634)
    
    # Applying the binary operator '*' (line 159)
    result_mul_246636 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 28), '*', float_246627, result_pow_246635)
    
    # Applying the binary operator '+' (line 159)
    result_add_246637 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 13), '+', result_mul_246626, result_mul_246636)
    
    float_246638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 47), 'float')
    
    # Obtaining the type of the subscript
    int_246639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 55), 'int')
    int_246640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 58), 'int')
    slice_246641 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 53), int_246639, None, int_246640)
    # Getting the type of 'x' (line 159)
    x_246642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 53), 'x')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___246643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 53), x_246642, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_246644 = invoke(stypy.reporting.localization.Localization(__file__, 159, 53), getitem___246643, slice_246641)
    
    int_246645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 62), 'int')
    # Applying the binary operator '**' (line 159)
    result_pow_246646 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 53), '**', subscript_call_result_246644, int_246645)
    
    # Applying the binary operator '*' (line 159)
    result_mul_246647 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 47), '*', float_246638, result_pow_246646)
    
    # Applying the binary operator '-' (line 159)
    result_sub_246648 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 45), '-', result_add_246637, result_mul_246647)
    
    float_246649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 66), 'float')
    
    # Obtaining the type of the subscript
    int_246650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 74), 'int')
    int_246651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 77), 'int')
    slice_246652 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 72), int_246650, None, int_246651)
    # Getting the type of 'x' (line 159)
    x_246653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 72), 'x')
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___246654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 72), x_246653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_246655 = invoke(stypy.reporting.localization.Localization(__file__, 159, 72), getitem___246654, slice_246652)
    
    # Applying the binary operator '*' (line 159)
    result_mul_246656 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 66), '*', float_246649, subscript_call_result_246655)
    
    # Applying the binary operator '+' (line 159)
    result_add_246657 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 64), '+', result_sub_246648, result_mul_246656)
    
    float_246658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 82), 'float')
    # Applying the binary operator '-' (line 159)
    result_sub_246659 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 80), '-', result_add_246657, float_246658)
    
    # Getting the type of 'g' (line 159)
    g_246660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'g')
    int_246661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
    slice_246662 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 159, 4), None, None, int_246661)
    # Storing an element on a container (line 159)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 4), g_246660, (slice_246662, result_sub_246659))
    
    # Assigning a BinOp to a Subscript (line 160):
    float_246663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 14), 'float')
    
    # Obtaining the type of the subscript
    int_246664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'int')
    slice_246665 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 21), None, None, int_246664)
    # Getting the type of 'x' (line 160)
    x_246666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'x')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___246667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 21), x_246666, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_246668 = invoke(stypy.reporting.localization.Localization(__file__, 160, 21), getitem___246667, slice_246665)
    
    # Applying the binary operator '*' (line 160)
    result_mul_246669 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 14), '*', float_246663, subscript_call_result_246668)
    
    float_246670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 30), 'float')
    
    # Obtaining the type of the subscript
    int_246671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 39), 'int')
    int_246672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 42), 'int')
    slice_246673 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 37), int_246671, None, int_246672)
    # Getting the type of 'x' (line 160)
    x_246674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'x')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___246675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 37), x_246674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_246676 = invoke(stypy.reporting.localization.Localization(__file__, 160, 37), getitem___246675, slice_246673)
    
    int_246677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 46), 'int')
    # Applying the binary operator '**' (line 160)
    result_pow_246678 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 37), '**', subscript_call_result_246676, int_246677)
    
    # Applying the binary operator '*' (line 160)
    result_mul_246679 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 30), '*', float_246670, result_pow_246678)
    
    # Applying the binary operator '-' (line 160)
    result_sub_246680 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 14), '-', result_mul_246669, result_mul_246679)
    
    float_246681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 50), 'float')
    
    # Obtaining the type of the subscript
    int_246682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 59), 'int')
    int_246683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 62), 'int')
    slice_246684 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 57), int_246682, None, int_246683)
    # Getting the type of 'x' (line 160)
    x_246685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 57), 'x')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___246686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 57), x_246685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_246687 = invoke(stypy.reporting.localization.Localization(__file__, 160, 57), getitem___246686, slice_246684)
    
    int_246688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 66), 'int')
    # Applying the binary operator '**' (line 160)
    result_pow_246689 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 57), '**', subscript_call_result_246687, int_246688)
    
    # Applying the binary operator '*' (line 160)
    result_mul_246690 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 50), '*', float_246681, result_pow_246689)
    
    # Applying the binary operator '+' (line 160)
    result_add_246691 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 48), '+', result_sub_246680, result_mul_246690)
    
    float_246692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 70), 'float')
    
    # Obtaining the type of the subscript
    int_246693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 79), 'int')
    int_246694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 82), 'int')
    slice_246695 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 77), int_246693, None, int_246694)
    # Getting the type of 'x' (line 160)
    x_246696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 77), 'x')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___246697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 77), x_246696, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_246698 = invoke(stypy.reporting.localization.Localization(__file__, 160, 77), getitem___246697, slice_246695)
    
    # Applying the binary operator '*' (line 160)
    result_mul_246699 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 70), '*', float_246692, subscript_call_result_246698)
    
    # Applying the binary operator '-' (line 160)
    result_sub_246700 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 68), '-', result_add_246691, result_mul_246699)
    
    
    # Obtaining the type of the subscript
    int_246701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 89), 'int')
    int_246702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 92), 'int')
    slice_246703 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 87), int_246701, None, int_246702)
    # Getting the type of 'x' (line 160)
    x_246704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 87), 'x')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___246705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 87), x_246704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_246706 = invoke(stypy.reporting.localization.Localization(__file__, 160, 87), getitem___246705, slice_246703)
    
    # Applying the binary operator '-' (line 160)
    result_sub_246707 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 85), '-', result_sub_246700, subscript_call_result_246706)
    
    float_246708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 97), 'float')
    
    # Obtaining the type of the subscript
    int_246709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 105), 'int')
    int_246710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 108), 'int')
    slice_246711 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 103), int_246709, None, int_246710)
    # Getting the type of 'x' (line 160)
    x_246712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 103), 'x')
    # Obtaining the member '__getitem__' of a type (line 160)
    getitem___246713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 103), x_246712, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
    subscript_call_result_246714 = invoke(stypy.reporting.localization.Localization(__file__, 160, 103), getitem___246713, slice_246711)
    
    int_246715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 112), 'int')
    # Applying the binary operator '**' (line 160)
    result_pow_246716 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 103), '**', subscript_call_result_246714, int_246715)
    
    # Applying the binary operator '*' (line 160)
    result_mul_246717 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 97), '*', float_246708, result_pow_246716)
    
    # Applying the binary operator '+' (line 160)
    result_add_246718 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 95), '+', result_sub_246707, result_mul_246717)
    
    float_246719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 116), 'float')
    # Applying the binary operator '+' (line 160)
    result_add_246720 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 114), '+', result_add_246718, float_246719)
    
    # Getting the type of 'g' (line 160)
    g_246721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'g')
    int_246722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 6), 'int')
    int_246723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 9), 'int')
    slice_246724 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 4), int_246722, None, int_246723)
    # Storing an element on a container (line 160)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 4), g_246721, (slice_246724, result_add_246720))
    
    # Assigning a BinOp to a Subscript (line 161):
    float_246725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 14), 'float')
    
    # Obtaining the type of the subscript
    int_246726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 23), 'int')
    int_246727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 26), 'int')
    slice_246728 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 161, 21), int_246726, None, int_246727)
    # Getting the type of 'x' (line 161)
    x_246729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'x')
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___246730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 21), x_246729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_246731 = invoke(stypy.reporting.localization.Localization(__file__, 161, 21), getitem___246730, slice_246728)
    
    # Applying the binary operator '*' (line 161)
    result_mul_246732 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 14), '*', float_246725, subscript_call_result_246731)
    
    float_246733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 31), 'float')
    
    # Obtaining the type of the subscript
    int_246734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 38), 'int')
    int_246735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 41), 'int')
    slice_246736 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 161, 36), int_246734, None, int_246735)
    # Getting the type of 'x' (line 161)
    x_246737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'x')
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___246738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 36), x_246737, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_246739 = invoke(stypy.reporting.localization.Localization(__file__, 161, 36), getitem___246738, slice_246736)
    
    int_246740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 45), 'int')
    # Applying the binary operator '**' (line 161)
    result_pow_246741 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 36), '**', subscript_call_result_246739, int_246740)
    
    # Applying the binary operator '*' (line 161)
    result_mul_246742 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 31), '*', float_246733, result_pow_246741)
    
    # Applying the binary operator '-' (line 161)
    result_sub_246743 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 14), '-', result_mul_246732, result_mul_246742)
    
    # Getting the type of 'g' (line 161)
    g_246744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'g')
    int_246745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 6), 'int')
    int_246746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 9), 'int')
    slice_246747 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 161, 4), int_246745, None, int_246746)
    # Storing an element on a container (line 161)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 4), g_246744, (slice_246747, result_sub_246743))
    # Getting the type of 'g' (line 162)
    g_246748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'g')
    # Assigning a type to the variable 'stypy_return_type' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type', g_246748)
    
    # ################# End of 'F_4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F_4' in the type store
    # Getting the type of 'stypy_return_type' (line 154)
    stypy_return_type_246749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F_4'
    return stypy_return_type_246749

# Assigning a type to the variable 'F_4' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'F_4', F_4)

@norecursion
def x0_4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'x0_4'
    module_type_store = module_type_store.open_function_context('x0_4', 164, 0, False)
    
    # Passed parameters checking function
    x0_4.stypy_localization = localization
    x0_4.stypy_type_of_self = None
    x0_4.stypy_type_store = module_type_store
    x0_4.stypy_function_name = 'x0_4'
    x0_4.stypy_param_names_list = ['n']
    x0_4.stypy_varargs_param_name = None
    x0_4.stypy_kwargs_param_name = None
    x0_4.stypy_call_defaults = defaults
    x0_4.stypy_call_varargs = varargs
    x0_4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'x0_4', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'x0_4', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'x0_4(...)' code ##################

    
    # Call to assert_equal(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'n' (line 165)
    n_246751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'n', False)
    int_246752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 21), 'int')
    # Applying the binary operator '%' (line 165)
    result_mod_246753 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 17), '%', n_246751, int_246752)
    
    int_246754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 24), 'int')
    # Processing the call keyword arguments (line 165)
    kwargs_246755 = {}
    # Getting the type of 'assert_equal' (line 165)
    assert_equal_246750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 165)
    assert_equal_call_result_246756 = invoke(stypy.reporting.localization.Localization(__file__, 165, 4), assert_equal_246750, *[result_mod_246753, int_246754], **kwargs_246755)
    
    
    # Assigning a Call to a Name (line 166):
    
    # Call to array(...): (line 166)
    # Processing the call arguments (line 166)
    
    # Obtaining an instance of the builtin type 'list' (line 166)
    list_246759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 166)
    # Adding element type (line 166)
    int_246760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), list_246759, int_246760)
    # Adding element type (line 166)
    int_246761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 23), 'int')
    int_246762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 25), 'int')
    # Applying the binary operator 'div' (line 166)
    result_div_246763 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 23), 'div', int_246761, int_246762)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), list_246759, result_div_246763)
    # Adding element type (line 166)
    int_246764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), list_246759, int_246764)
    
    # Getting the type of 'n' (line 166)
    n_246765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 35), 'n', False)
    int_246766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 38), 'int')
    # Applying the binary operator '//' (line 166)
    result_floordiv_246767 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 35), '//', n_246765, int_246766)
    
    # Applying the binary operator '*' (line 166)
    result_mul_246768 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 18), '*', list_246759, result_floordiv_246767)
    
    # Processing the call keyword arguments (line 166)
    kwargs_246769 = {}
    # Getting the type of 'np' (line 166)
    np_246757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 166)
    array_246758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 9), np_246757, 'array')
    # Calling array(args, kwargs) (line 166)
    array_call_result_246770 = invoke(stypy.reporting.localization.Localization(__file__, 166, 9), array_246758, *[result_mul_246768], **kwargs_246769)
    
    # Assigning a type to the variable 'x0' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'x0', array_call_result_246770)
    # Getting the type of 'x0' (line 167)
    x0_246771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'x0')
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', x0_246771)
    
    # ################# End of 'x0_4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'x0_4' in the type store
    # Getting the type of 'stypy_return_type' (line 164)
    stypy_return_type_246772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246772)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'x0_4'
    return stypy_return_type_246772

# Assigning a type to the variable 'x0_4' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'x0_4', x0_4)

@norecursion
def F_6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F_6'
    module_type_store = module_type_store.open_function_context('F_6', 169, 0, False)
    
    # Passed parameters checking function
    F_6.stypy_localization = localization
    F_6.stypy_type_of_self = None
    F_6.stypy_type_store = module_type_store
    F_6.stypy_function_name = 'F_6'
    F_6.stypy_param_names_list = ['x', 'n']
    F_6.stypy_varargs_param_name = None
    F_6.stypy_kwargs_param_name = None
    F_6.stypy_call_defaults = defaults
    F_6.stypy_call_varargs = varargs
    F_6.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F_6', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F_6', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F_6(...)' code ##################

    
    # Assigning a Num to a Name (line 170):
    float_246773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'float')
    # Assigning a type to the variable 'c' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'c', float_246773)
    
    # Assigning a BinOp to a Name (line 171):
    
    # Call to arange(...): (line 171)
    # Processing the call arguments (line 171)
    int_246776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'int')
    # Getting the type of 'n' (line 171)
    n_246777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'n', False)
    int_246778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 25), 'int')
    # Applying the binary operator '+' (line 171)
    result_add_246779 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 23), '+', n_246777, int_246778)
    
    # Processing the call keyword arguments (line 171)
    kwargs_246780 = {}
    # Getting the type of 'np' (line 171)
    np_246774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 10), 'np', False)
    # Obtaining the member 'arange' of a type (line 171)
    arange_246775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 10), np_246774, 'arange')
    # Calling arange(args, kwargs) (line 171)
    arange_call_result_246781 = invoke(stypy.reporting.localization.Localization(__file__, 171, 10), arange_246775, *[int_246776, result_add_246779], **kwargs_246780)
    
    float_246782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'float')
    # Applying the binary operator '-' (line 171)
    result_sub_246783 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 10), '-', arange_call_result_246781, float_246782)
    
    # Getting the type of 'n' (line 171)
    n_246784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'n')
    # Applying the binary operator 'div' (line 171)
    result_div_246785 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 9), 'div', result_sub_246783, n_246784)
    
    # Assigning a type to the variable 'mu' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'mu', result_div_246785)
    # Getting the type of 'x' (line 172)
    x_246786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'x')
    int_246787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'int')
    int_246788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'int')
    # Getting the type of 'c' (line 172)
    c_246789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 'c')
    int_246790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'int')
    # Getting the type of 'n' (line 172)
    n_246791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 27), 'n')
    # Applying the binary operator '*' (line 172)
    result_mul_246792 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 25), '*', int_246790, n_246791)
    
    # Applying the binary operator 'div' (line 172)
    result_div_246793 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 22), 'div', c_246789, result_mul_246792)
    
    
    # Call to sum(...): (line 172)
    # Processing the call keyword arguments (line 172)
    int_246810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 76), 'int')
    keyword_246811 = int_246810
    kwargs_246812 = {'axis': keyword_246811}
    
    # Obtaining the type of the subscript
    slice_246794 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 172, 33), None, None, None)
    # Getting the type of 'None' (line 172)
    None_246795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 38), 'None', False)
    # Getting the type of 'mu' (line 172)
    mu_246796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 33), 'mu', False)
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___246797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 33), mu_246796, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_246798 = invoke(stypy.reporting.localization.Localization(__file__, 172, 33), getitem___246797, (slice_246794, None_246795))
    
    # Getting the type of 'x' (line 172)
    x_246799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 44), 'x', False)
    # Applying the binary operator '*' (line 172)
    result_mul_246800 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 33), '*', subscript_call_result_246798, x_246799)
    
    
    # Obtaining the type of the subscript
    slice_246801 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 172, 49), None, None, None)
    # Getting the type of 'None' (line 172)
    None_246802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 54), 'None', False)
    # Getting the type of 'mu' (line 172)
    mu_246803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 49), 'mu', False)
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___246804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 49), mu_246803, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_246805 = invoke(stypy.reporting.localization.Localization(__file__, 172, 49), getitem___246804, (slice_246801, None_246802))
    
    # Getting the type of 'mu' (line 172)
    mu_246806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 62), 'mu', False)
    # Applying the binary operator '+' (line 172)
    result_add_246807 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 49), '+', subscript_call_result_246805, mu_246806)
    
    # Applying the binary operator 'div' (line 172)
    result_div_246808 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 46), 'div', result_mul_246800, result_add_246807)
    
    # Obtaining the member 'sum' of a type (line 172)
    sum_246809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 46), result_div_246808, 'sum')
    # Calling sum(args, kwargs) (line 172)
    sum_call_result_246813 = invoke(stypy.reporting.localization.Localization(__file__, 172, 46), sum_246809, *[], **kwargs_246812)
    
    # Applying the binary operator '*' (line 172)
    result_mul_246814 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 30), '*', result_div_246793, sum_call_result_246813)
    
    # Applying the binary operator '-' (line 172)
    result_sub_246815 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 18), '-', int_246788, result_mul_246814)
    
    # Applying the binary operator 'div' (line 172)
    result_div_246816 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), 'div', int_246787, result_sub_246815)
    
    # Applying the binary operator '-' (line 172)
    result_sub_246817 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), '-', x_246786, result_div_246816)
    
    # Assigning a type to the variable 'stypy_return_type' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type', result_sub_246817)
    
    # ################# End of 'F_6(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F_6' in the type store
    # Getting the type of 'stypy_return_type' (line 169)
    stypy_return_type_246818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246818)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F_6'
    return stypy_return_type_246818

# Assigning a type to the variable 'F_6' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'F_6', F_6)

@norecursion
def x0_6(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'x0_6'
    module_type_store = module_type_store.open_function_context('x0_6', 174, 0, False)
    
    # Passed parameters checking function
    x0_6.stypy_localization = localization
    x0_6.stypy_type_of_self = None
    x0_6.stypy_type_store = module_type_store
    x0_6.stypy_function_name = 'x0_6'
    x0_6.stypy_param_names_list = ['n']
    x0_6.stypy_varargs_param_name = None
    x0_6.stypy_kwargs_param_name = None
    x0_6.stypy_call_defaults = defaults
    x0_6.stypy_call_varargs = varargs
    x0_6.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'x0_6', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'x0_6', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'x0_6(...)' code ##################

    
    # Call to ones(...): (line 175)
    # Processing the call arguments (line 175)
    
    # Obtaining an instance of the builtin type 'list' (line 175)
    list_246821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 175)
    # Adding element type (line 175)
    # Getting the type of 'n' (line 175)
    n_246822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 19), list_246821, n_246822)
    
    # Processing the call keyword arguments (line 175)
    kwargs_246823 = {}
    # Getting the type of 'np' (line 175)
    np_246819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'np', False)
    # Obtaining the member 'ones' of a type (line 175)
    ones_246820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 11), np_246819, 'ones')
    # Calling ones(args, kwargs) (line 175)
    ones_call_result_246824 = invoke(stypy.reporting.localization.Localization(__file__, 175, 11), ones_246820, *[list_246821], **kwargs_246823)
    
    # Assigning a type to the variable 'stypy_return_type' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type', ones_call_result_246824)
    
    # ################# End of 'x0_6(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'x0_6' in the type store
    # Getting the type of 'stypy_return_type' (line 174)
    stypy_return_type_246825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246825)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'x0_6'
    return stypy_return_type_246825

# Assigning a type to the variable 'x0_6' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'x0_6', x0_6)

@norecursion
def F_7(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F_7'
    module_type_store = module_type_store.open_function_context('F_7', 177, 0, False)
    
    # Passed parameters checking function
    F_7.stypy_localization = localization
    F_7.stypy_type_of_self = None
    F_7.stypy_type_store = module_type_store
    F_7.stypy_function_name = 'F_7'
    F_7.stypy_param_names_list = ['x', 'n']
    F_7.stypy_varargs_param_name = None
    F_7.stypy_kwargs_param_name = None
    F_7.stypy_call_defaults = defaults
    F_7.stypy_call_varargs = varargs
    F_7.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F_7', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F_7', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F_7(...)' code ##################

    
    # Call to assert_equal(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'n' (line 178)
    n_246827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'n', False)
    int_246828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 21), 'int')
    # Applying the binary operator '%' (line 178)
    result_mod_246829 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 17), '%', n_246827, int_246828)
    
    int_246830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 24), 'int')
    # Processing the call keyword arguments (line 178)
    kwargs_246831 = {}
    # Getting the type of 'assert_equal' (line 178)
    assert_equal_246826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 178)
    assert_equal_call_result_246832 = invoke(stypy.reporting.localization.Localization(__file__, 178, 4), assert_equal_246826, *[result_mod_246829, int_246830], **kwargs_246831)
    

    @norecursion
    def phi(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'phi'
        module_type_store = module_type_store.open_function_context('phi', 180, 4, False)
        
        # Passed parameters checking function
        phi.stypy_localization = localization
        phi.stypy_type_of_self = None
        phi.stypy_type_store = module_type_store
        phi.stypy_function_name = 'phi'
        phi.stypy_param_names_list = ['t']
        phi.stypy_varargs_param_name = None
        phi.stypy_kwargs_param_name = None
        phi.stypy_call_defaults = defaults
        phi.stypy_call_varargs = varargs
        phi.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'phi', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'phi', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'phi(...)' code ##################

        
        # Assigning a BinOp to a Name (line 181):
        float_246833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 12), 'float')
        # Getting the type of 't' (line 181)
        t_246834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 't')
        # Applying the binary operator '*' (line 181)
        result_mul_246835 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 12), '*', float_246833, t_246834)
        
        int_246836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 20), 'int')
        # Applying the binary operator '-' (line 181)
        result_sub_246837 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 12), '-', result_mul_246835, int_246836)
        
        # Assigning a type to the variable 'v' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'v', result_sub_246837)
        
        # Assigning a Subscript to a Subscript (line 182):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 't' (line 182)
        t_246838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 66), 't')
        int_246839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 70), 'int')
        # Applying the binary operator '>' (line 182)
        result_gt_246840 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 66), '>', t_246838, int_246839)
        
        int_246841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 22), 'int')
        # Getting the type of 't' (line 182)
        t_246842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 27), 't')
        int_246843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 30), 'int')
        # Applying the binary operator '**' (line 182)
        result_pow_246844 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 27), '**', t_246842, int_246843)
        
        # Applying the binary operator '*' (line 182)
        result_mul_246845 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 22), '*', int_246841, result_pow_246844)
        
        int_246846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 34), 'int')
        # Getting the type of 't' (line 182)
        t_246847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 38), 't')
        int_246848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 41), 'int')
        # Applying the binary operator '**' (line 182)
        result_pow_246849 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 38), '**', t_246847, int_246848)
        
        # Applying the binary operator '*' (line 182)
        result_mul_246850 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 34), '*', int_246846, result_pow_246849)
        
        # Applying the binary operator '+' (line 182)
        result_add_246851 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 22), '+', result_mul_246845, result_mul_246850)
        
        int_246852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 45), 'int')
        # Getting the type of 't' (line 182)
        t_246853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 50), 't')
        # Applying the binary operator '*' (line 182)
        result_mul_246854 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 45), '*', int_246852, t_246853)
        
        # Applying the binary operator '+' (line 182)
        result_add_246855 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 43), '+', result_add_246851, result_mul_246854)
        
        int_246856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 54), 'int')
        # Applying the binary operator '-' (line 182)
        result_sub_246857 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 52), '-', result_add_246855, int_246856)
        
        int_246858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 60), 'int')
        # Applying the binary operator 'div' (line 182)
        result_div_246859 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 21), 'div', result_sub_246857, int_246858)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___246860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 21), result_div_246859, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_246861 = invoke(stypy.reporting.localization.Localization(__file__, 182, 21), getitem___246860, result_gt_246840)
        
        # Getting the type of 'v' (line 182)
        v_246862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'v')
        
        # Getting the type of 't' (line 182)
        t_246863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 10), 't')
        int_246864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 14), 'int')
        # Applying the binary operator '>' (line 182)
        result_gt_246865 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 10), '>', t_246863, int_246864)
        
        # Storing an element on a container (line 182)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 8), v_246862, (result_gt_246865, subscript_call_result_246861))
        
        # Assigning a Subscript to a Subscript (line 183):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 't' (line 183)
        t_246866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 't')
        int_246867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 37), 'int')
        # Applying the binary operator '>=' (line 183)
        result_ge_246868 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 32), '>=', t_246866, int_246867)
        
        float_246869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 21), 'float')
        # Getting the type of 't' (line 183)
        t_246870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 25), 't')
        # Applying the binary operator '*' (line 183)
        result_mul_246871 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 21), '*', float_246869, t_246870)
        
        int_246872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 29), 'int')
        # Applying the binary operator '+' (line 183)
        result_add_246873 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 21), '+', result_mul_246871, int_246872)
        
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___246874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 21), result_add_246873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_246875 = invoke(stypy.reporting.localization.Localization(__file__, 183, 21), getitem___246874, result_ge_246868)
        
        # Getting the type of 'v' (line 183)
        v_246876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'v')
        
        # Getting the type of 't' (line 183)
        t_246877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 10), 't')
        int_246878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 15), 'int')
        # Applying the binary operator '>=' (line 183)
        result_ge_246879 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 10), '>=', t_246877, int_246878)
        
        # Storing an element on a container (line 183)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 8), v_246876, (result_ge_246879, subscript_call_result_246875))
        # Getting the type of 'v' (line 184)
        v_246880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'stypy_return_type', v_246880)
        
        # ################# End of 'phi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'phi' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_246881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_246881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'phi'
        return stypy_return_type_246881

    # Assigning a type to the variable 'phi' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'phi', phi)
    
    # Assigning a Call to a Name (line 185):
    
    # Call to zeros(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Obtaining an instance of the builtin type 'list' (line 185)
    list_246884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 185)
    # Adding element type (line 185)
    # Getting the type of 'n' (line 185)
    n_246885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 18), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 17), list_246884, n_246885)
    
    # Processing the call keyword arguments (line 185)
    kwargs_246886 = {}
    # Getting the type of 'np' (line 185)
    np_246882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 185)
    zeros_246883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), np_246882, 'zeros')
    # Calling zeros(args, kwargs) (line 185)
    zeros_call_result_246887 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), zeros_246883, *[list_246884], **kwargs_246886)
    
    # Assigning a type to the variable 'g' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'g', zeros_call_result_246887)
    
    # Assigning a BinOp to a Subscript (line 186):
    float_246888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 13), 'float')
    
    # Obtaining the type of the subscript
    int_246889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'int')
    int_246890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 24), 'int')
    slice_246891 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 186, 19), int_246889, None, int_246890)
    # Getting the type of 'x' (line 186)
    x_246892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'x')
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___246893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 19), x_246892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_246894 = invoke(stypy.reporting.localization.Localization(__file__, 186, 19), getitem___246893, slice_246891)
    
    int_246895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 28), 'int')
    # Applying the binary operator '**' (line 186)
    result_pow_246896 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 19), '**', subscript_call_result_246894, int_246895)
    
    # Applying the binary operator '*' (line 186)
    result_mul_246897 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 13), '*', float_246888, result_pow_246896)
    
    int_246898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 32), 'int')
    # Applying the binary operator '-' (line 186)
    result_sub_246899 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 13), '-', result_mul_246897, int_246898)
    
    # Getting the type of 'g' (line 186)
    g_246900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'g')
    int_246901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 8), 'int')
    slice_246902 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 186, 4), None, None, int_246901)
    # Storing an element on a container (line 186)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 4), g_246900, (slice_246902, result_sub_246899))
    
    # Assigning a BinOp to a Subscript (line 187):
    
    # Call to exp(...): (line 187)
    # Processing the call arguments (line 187)
    
    
    # Obtaining the type of the subscript
    int_246904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 23), 'int')
    slice_246905 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 19), None, None, int_246904)
    # Getting the type of 'x' (line 187)
    x_246906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___246907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 19), x_246906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_246908 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), getitem___246907, slice_246905)
    
    # Applying the 'usub' unary operator (line 187)
    result___neg___246909 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 18), 'usub', subscript_call_result_246908)
    
    # Processing the call keyword arguments (line 187)
    kwargs_246910 = {}
    # Getting the type of 'exp' (line 187)
    exp_246903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'exp', False)
    # Calling exp(args, kwargs) (line 187)
    exp_call_result_246911 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), exp_246903, *[result___neg___246909], **kwargs_246910)
    
    
    # Call to exp(...): (line 187)
    # Processing the call arguments (line 187)
    
    
    # Obtaining the type of the subscript
    int_246913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 36), 'int')
    int_246914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 39), 'int')
    slice_246915 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 34), int_246913, None, int_246914)
    # Getting the type of 'x' (line 187)
    x_246916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 187)
    getitem___246917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 34), x_246916, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 187)
    subscript_call_result_246918 = invoke(stypy.reporting.localization.Localization(__file__, 187, 34), getitem___246917, slice_246915)
    
    # Applying the 'usub' unary operator (line 187)
    result___neg___246919 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 33), 'usub', subscript_call_result_246918)
    
    # Processing the call keyword arguments (line 187)
    kwargs_246920 = {}
    # Getting the type of 'exp' (line 187)
    exp_246912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 29), 'exp', False)
    # Calling exp(args, kwargs) (line 187)
    exp_call_result_246921 = invoke(stypy.reporting.localization.Localization(__file__, 187, 29), exp_246912, *[result___neg___246919], **kwargs_246920)
    
    # Applying the binary operator '+' (line 187)
    result_add_246922 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 14), '+', exp_call_result_246911, exp_call_result_246921)
    
    float_246923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 45), 'float')
    # Applying the binary operator '-' (line 187)
    result_sub_246924 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 43), '-', result_add_246922, float_246923)
    
    # Getting the type of 'g' (line 187)
    g_246925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'g')
    int_246926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 6), 'int')
    int_246927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 9), 'int')
    slice_246928 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 4), int_246926, None, int_246927)
    # Storing an element on a container (line 187)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 4), g_246925, (slice_246928, result_sub_246924))
    
    # Assigning a Call to a Subscript (line 188):
    
    # Call to phi(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Obtaining the type of the subscript
    int_246930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 20), 'int')
    int_246931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 23), 'int')
    slice_246932 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 188, 18), int_246930, None, int_246931)
    # Getting the type of 'x' (line 188)
    x_246933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___246934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 18), x_246933, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_246935 = invoke(stypy.reporting.localization.Localization(__file__, 188, 18), getitem___246934, slice_246932)
    
    # Processing the call keyword arguments (line 188)
    kwargs_246936 = {}
    # Getting the type of 'phi' (line 188)
    phi_246929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 14), 'phi', False)
    # Calling phi(args, kwargs) (line 188)
    phi_call_result_246937 = invoke(stypy.reporting.localization.Localization(__file__, 188, 14), phi_246929, *[subscript_call_result_246935], **kwargs_246936)
    
    # Getting the type of 'g' (line 188)
    g_246938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'g')
    int_246939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 6), 'int')
    int_246940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 9), 'int')
    slice_246941 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 188, 4), int_246939, None, int_246940)
    # Storing an element on a container (line 188)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 4), g_246938, (slice_246941, phi_call_result_246937))
    # Getting the type of 'g' (line 189)
    g_246942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'g')
    # Assigning a type to the variable 'stypy_return_type' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type', g_246942)
    
    # ################# End of 'F_7(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F_7' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_246943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F_7'
    return stypy_return_type_246943

# Assigning a type to the variable 'F_7' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'F_7', F_7)

@norecursion
def x0_7(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'x0_7'
    module_type_store = module_type_store.open_function_context('x0_7', 191, 0, False)
    
    # Passed parameters checking function
    x0_7.stypy_localization = localization
    x0_7.stypy_type_of_self = None
    x0_7.stypy_type_store = module_type_store
    x0_7.stypy_function_name = 'x0_7'
    x0_7.stypy_param_names_list = ['n']
    x0_7.stypy_varargs_param_name = None
    x0_7.stypy_kwargs_param_name = None
    x0_7.stypy_call_defaults = defaults
    x0_7.stypy_call_varargs = varargs
    x0_7.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'x0_7', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'x0_7', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'x0_7(...)' code ##################

    
    # Call to assert_equal(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'n' (line 192)
    n_246945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'n', False)
    int_246946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 21), 'int')
    # Applying the binary operator '%' (line 192)
    result_mod_246947 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 17), '%', n_246945, int_246946)
    
    int_246948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 24), 'int')
    # Processing the call keyword arguments (line 192)
    kwargs_246949 = {}
    # Getting the type of 'assert_equal' (line 192)
    assert_equal_246944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 192)
    assert_equal_call_result_246950 = invoke(stypy.reporting.localization.Localization(__file__, 192, 4), assert_equal_246944, *[result_mod_246947, int_246948], **kwargs_246949)
    
    
    # Call to array(...): (line 193)
    # Processing the call arguments (line 193)
    
    # Obtaining an instance of the builtin type 'list' (line 193)
    list_246953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 193)
    # Adding element type (line 193)
    float_246954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 20), list_246953, float_246954)
    # Adding element type (line 193)
    int_246955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 20), list_246953, int_246955)
    # Adding element type (line 193)
    int_246956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 20), list_246953, int_246956)
    
    # Getting the type of 'n' (line 193)
    n_246957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 37), 'n', False)
    int_246958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 40), 'int')
    # Applying the binary operator '//' (line 193)
    result_floordiv_246959 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 37), '//', n_246957, int_246958)
    
    # Applying the binary operator '*' (line 193)
    result_mul_246960 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 20), '*', list_246953, result_floordiv_246959)
    
    # Processing the call keyword arguments (line 193)
    kwargs_246961 = {}
    # Getting the type of 'np' (line 193)
    np_246951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 193)
    array_246952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 11), np_246951, 'array')
    # Calling array(args, kwargs) (line 193)
    array_call_result_246962 = invoke(stypy.reporting.localization.Localization(__file__, 193, 11), array_246952, *[result_mul_246960], **kwargs_246961)
    
    # Assigning a type to the variable 'stypy_return_type' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type', array_call_result_246962)
    
    # ################# End of 'x0_7(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'x0_7' in the type store
    # Getting the type of 'stypy_return_type' (line 191)
    stypy_return_type_246963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_246963)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'x0_7'
    return stypy_return_type_246963

# Assigning a type to the variable 'x0_7' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'x0_7', x0_7)

@norecursion
def F_9(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F_9'
    module_type_store = module_type_store.open_function_context('F_9', 195, 0, False)
    
    # Passed parameters checking function
    F_9.stypy_localization = localization
    F_9.stypy_type_of_self = None
    F_9.stypy_type_store = module_type_store
    F_9.stypy_function_name = 'F_9'
    F_9.stypy_param_names_list = ['x', 'n']
    F_9.stypy_varargs_param_name = None
    F_9.stypy_kwargs_param_name = None
    F_9.stypy_call_defaults = defaults
    F_9.stypy_call_varargs = varargs
    F_9.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F_9', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F_9', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F_9(...)' code ##################

    
    # Assigning a Call to a Name (line 196):
    
    # Call to zeros(...): (line 196)
    # Processing the call arguments (line 196)
    
    # Obtaining an instance of the builtin type 'list' (line 196)
    list_246966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 196)
    # Adding element type (line 196)
    # Getting the type of 'n' (line 196)
    n_246967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 18), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 17), list_246966, n_246967)
    
    # Processing the call keyword arguments (line 196)
    kwargs_246968 = {}
    # Getting the type of 'np' (line 196)
    np_246964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 196)
    zeros_246965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), np_246964, 'zeros')
    # Calling zeros(args, kwargs) (line 196)
    zeros_call_result_246969 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), zeros_246965, *[list_246966], **kwargs_246968)
    
    # Assigning a type to the variable 'g' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'g', zeros_call_result_246969)
    
    # Assigning a Call to a Name (line 197):
    
    # Call to arange(...): (line 197)
    # Processing the call arguments (line 197)
    int_246972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'int')
    # Getting the type of 'n' (line 197)
    n_246973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 21), 'n', False)
    # Processing the call keyword arguments (line 197)
    kwargs_246974 = {}
    # Getting the type of 'np' (line 197)
    np_246970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 197)
    arange_246971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), np_246970, 'arange')
    # Calling arange(args, kwargs) (line 197)
    arange_call_result_246975 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), arange_246971, *[int_246972, n_246973], **kwargs_246974)
    
    # Assigning a type to the variable 'i' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'i', arange_call_result_246975)
    
    # Assigning a BinOp to a Subscript (line 198):
    
    # Obtaining the type of the subscript
    int_246976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 13), 'int')
    # Getting the type of 'x' (line 198)
    x_246977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'x')
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___246978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), x_246977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_246979 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), getitem___246978, int_246976)
    
    int_246980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 17), 'int')
    # Applying the binary operator '**' (line 198)
    result_pow_246981 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '**', subscript_call_result_246979, int_246980)
    
    int_246982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 19), 'int')
    # Applying the binary operator 'div' (line 198)
    result_div_246983 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), 'div', result_pow_246981, int_246982)
    
    
    # Obtaining the type of the subscript
    int_246984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 25), 'int')
    # Getting the type of 'x' (line 198)
    x_246985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'x')
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___246986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 23), x_246985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_246987 = invoke(stypy.reporting.localization.Localization(__file__, 198, 23), getitem___246986, int_246984)
    
    int_246988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 29), 'int')
    # Applying the binary operator '**' (line 198)
    result_pow_246989 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 23), '**', subscript_call_result_246987, int_246988)
    
    int_246990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 31), 'int')
    # Applying the binary operator 'div' (line 198)
    result_div_246991 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 23), 'div', result_pow_246989, int_246990)
    
    # Applying the binary operator '+' (line 198)
    result_add_246992 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), '+', result_div_246983, result_div_246991)
    
    # Getting the type of 'g' (line 198)
    g_246993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'g')
    int_246994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 6), 'int')
    # Storing an element on a container (line 198)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 4), g_246993, (int_246994, result_add_246992))
    
    # Assigning a BinOp to a Subscript (line 199):
    
    
    # Obtaining the type of the subscript
    int_246995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 17), 'int')
    int_246996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 19), 'int')
    slice_246997 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 15), int_246995, int_246996, None)
    # Getting the type of 'x' (line 199)
    x_246998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'x')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___246999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 15), x_246998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_247000 = invoke(stypy.reporting.localization.Localization(__file__, 199, 15), getitem___246999, slice_246997)
    
    int_247001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 24), 'int')
    # Applying the binary operator '**' (line 199)
    result_pow_247002 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 15), '**', subscript_call_result_247000, int_247001)
    
    # Applying the 'usub' unary operator (line 199)
    result___neg___247003 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 14), 'usub', result_pow_247002)
    
    int_247004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 26), 'int')
    # Applying the binary operator 'div' (line 199)
    result_div_247005 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 14), 'div', result___neg___247003, int_247004)
    
    # Getting the type of 'i' (line 199)
    i_247006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'i')
    
    # Obtaining the type of the subscript
    int_247007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'int')
    int_247008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 36), 'int')
    slice_247009 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 32), int_247007, int_247008, None)
    # Getting the type of 'x' (line 199)
    x_247010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'x')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___247011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 32), x_247010, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_247012 = invoke(stypy.reporting.localization.Localization(__file__, 199, 32), getitem___247011, slice_247009)
    
    int_247013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 41), 'int')
    # Applying the binary operator '**' (line 199)
    result_pow_247014 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 32), '**', subscript_call_result_247012, int_247013)
    
    # Applying the binary operator '*' (line 199)
    result_mul_247015 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 30), '*', i_247006, result_pow_247014)
    
    int_247016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 43), 'int')
    # Applying the binary operator 'div' (line 199)
    result_div_247017 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 42), 'div', result_mul_247015, int_247016)
    
    # Applying the binary operator '+' (line 199)
    result_add_247018 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 14), '+', result_div_247005, result_div_247017)
    
    
    # Obtaining the type of the subscript
    int_247019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 49), 'int')
    slice_247020 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 47), int_247019, None, None)
    # Getting the type of 'x' (line 199)
    x_247021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 47), 'x')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___247022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 47), x_247021, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_247023 = invoke(stypy.reporting.localization.Localization(__file__, 199, 47), getitem___247022, slice_247020)
    
    int_247024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 54), 'int')
    # Applying the binary operator '**' (line 199)
    result_pow_247025 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 47), '**', subscript_call_result_247023, int_247024)
    
    int_247026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 56), 'int')
    # Applying the binary operator 'div' (line 199)
    result_div_247027 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 47), 'div', result_pow_247025, int_247026)
    
    # Applying the binary operator '+' (line 199)
    result_add_247028 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 45), '+', result_add_247018, result_div_247027)
    
    # Getting the type of 'g' (line 199)
    g_247029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'g')
    int_247030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 6), 'int')
    int_247031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
    slice_247032 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 4), int_247030, int_247031, None)
    # Storing an element on a container (line 199)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), g_247029, (slice_247032, result_add_247028))
    
    # Assigning a BinOp to a Subscript (line 200):
    
    
    # Obtaining the type of the subscript
    int_247033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 15), 'int')
    # Getting the type of 'x' (line 200)
    x_247034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'x')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___247035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 13), x_247034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_247036 = invoke(stypy.reporting.localization.Localization(__file__, 200, 13), getitem___247035, int_247033)
    
    int_247037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'int')
    # Applying the binary operator '**' (line 200)
    result_pow_247038 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 13), '**', subscript_call_result_247036, int_247037)
    
    # Applying the 'usub' unary operator (line 200)
    result___neg___247039 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 12), 'usub', result_pow_247038)
    
    int_247040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 22), 'int')
    # Applying the binary operator 'div' (line 200)
    result_div_247041 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 12), 'div', result___neg___247039, int_247040)
    
    # Getting the type of 'n' (line 200)
    n_247042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'n')
    
    # Obtaining the type of the subscript
    int_247043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 30), 'int')
    # Getting the type of 'x' (line 200)
    x_247044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'x')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___247045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 28), x_247044, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_247046 = invoke(stypy.reporting.localization.Localization(__file__, 200, 28), getitem___247045, int_247043)
    
    int_247047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 35), 'int')
    # Applying the binary operator '**' (line 200)
    result_pow_247048 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 28), '**', subscript_call_result_247046, int_247047)
    
    # Applying the binary operator '*' (line 200)
    result_mul_247049 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 26), '*', n_247042, result_pow_247048)
    
    int_247050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 37), 'int')
    # Applying the binary operator 'div' (line 200)
    result_div_247051 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 36), 'div', result_mul_247049, int_247050)
    
    # Applying the binary operator '+' (line 200)
    result_add_247052 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 12), '+', result_div_247041, result_div_247051)
    
    # Getting the type of 'g' (line 200)
    g_247053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'g')
    int_247054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 6), 'int')
    # Storing an element on a container (line 200)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 4), g_247053, (int_247054, result_add_247052))
    # Getting the type of 'g' (line 201)
    g_247055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'g')
    # Assigning a type to the variable 'stypy_return_type' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type', g_247055)
    
    # ################# End of 'F_9(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F_9' in the type store
    # Getting the type of 'stypy_return_type' (line 195)
    stypy_return_type_247056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_247056)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F_9'
    return stypy_return_type_247056

# Assigning a type to the variable 'F_9' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'F_9', F_9)

@norecursion
def x0_9(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'x0_9'
    module_type_store = module_type_store.open_function_context('x0_9', 203, 0, False)
    
    # Passed parameters checking function
    x0_9.stypy_localization = localization
    x0_9.stypy_type_of_self = None
    x0_9.stypy_type_store = module_type_store
    x0_9.stypy_function_name = 'x0_9'
    x0_9.stypy_param_names_list = ['n']
    x0_9.stypy_varargs_param_name = None
    x0_9.stypy_kwargs_param_name = None
    x0_9.stypy_call_defaults = defaults
    x0_9.stypy_call_varargs = varargs
    x0_9.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'x0_9', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'x0_9', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'x0_9(...)' code ##################

    
    # Call to ones(...): (line 204)
    # Processing the call arguments (line 204)
    
    # Obtaining an instance of the builtin type 'list' (line 204)
    list_247059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 204)
    # Adding element type (line 204)
    # Getting the type of 'n' (line 204)
    n_247060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 19), list_247059, n_247060)
    
    # Processing the call keyword arguments (line 204)
    kwargs_247061 = {}
    # Getting the type of 'np' (line 204)
    np_247057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'np', False)
    # Obtaining the member 'ones' of a type (line 204)
    ones_247058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), np_247057, 'ones')
    # Calling ones(args, kwargs) (line 204)
    ones_call_result_247062 = invoke(stypy.reporting.localization.Localization(__file__, 204, 11), ones_247058, *[list_247059], **kwargs_247061)
    
    # Assigning a type to the variable 'stypy_return_type' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type', ones_call_result_247062)
    
    # ################# End of 'x0_9(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'x0_9' in the type store
    # Getting the type of 'stypy_return_type' (line 203)
    stypy_return_type_247063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_247063)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'x0_9'
    return stypy_return_type_247063

# Assigning a type to the variable 'x0_9' (line 203)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'x0_9', x0_9)

@norecursion
def F_10(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'F_10'
    module_type_store = module_type_store.open_function_context('F_10', 206, 0, False)
    
    # Passed parameters checking function
    F_10.stypy_localization = localization
    F_10.stypy_type_of_self = None
    F_10.stypy_type_store = module_type_store
    F_10.stypy_function_name = 'F_10'
    F_10.stypy_param_names_list = ['x', 'n']
    F_10.stypy_varargs_param_name = None
    F_10.stypy_kwargs_param_name = None
    F_10.stypy_call_defaults = defaults
    F_10.stypy_call_varargs = varargs
    F_10.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'F_10', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'F_10', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'F_10(...)' code ##################

    
    # Call to log(...): (line 207)
    # Processing the call arguments (line 207)
    int_247066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 18), 'int')
    # Getting the type of 'x' (line 207)
    x_247067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'x', False)
    # Applying the binary operator '+' (line 207)
    result_add_247068 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 18), '+', int_247066, x_247067)
    
    # Processing the call keyword arguments (line 207)
    kwargs_247069 = {}
    # Getting the type of 'np' (line 207)
    np_247064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'np', False)
    # Obtaining the member 'log' of a type (line 207)
    log_247065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 11), np_247064, 'log')
    # Calling log(args, kwargs) (line 207)
    log_call_result_247070 = invoke(stypy.reporting.localization.Localization(__file__, 207, 11), log_247065, *[result_add_247068], **kwargs_247069)
    
    # Getting the type of 'x' (line 207)
    x_247071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'x')
    # Getting the type of 'n' (line 207)
    n_247072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'n')
    # Applying the binary operator 'div' (line 207)
    result_div_247073 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 27), 'div', x_247071, n_247072)
    
    # Applying the binary operator '-' (line 207)
    result_sub_247074 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), '-', log_call_result_247070, result_div_247073)
    
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type', result_sub_247074)
    
    # ################# End of 'F_10(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'F_10' in the type store
    # Getting the type of 'stypy_return_type' (line 206)
    stypy_return_type_247075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_247075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'F_10'
    return stypy_return_type_247075

# Assigning a type to the variable 'F_10' (line 206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'F_10', F_10)

@norecursion
def x0_10(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'x0_10'
    module_type_store = module_type_store.open_function_context('x0_10', 209, 0, False)
    
    # Passed parameters checking function
    x0_10.stypy_localization = localization
    x0_10.stypy_type_of_self = None
    x0_10.stypy_type_store = module_type_store
    x0_10.stypy_function_name = 'x0_10'
    x0_10.stypy_param_names_list = ['n']
    x0_10.stypy_varargs_param_name = None
    x0_10.stypy_kwargs_param_name = None
    x0_10.stypy_call_defaults = defaults
    x0_10.stypy_call_varargs = varargs
    x0_10.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'x0_10', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'x0_10', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'x0_10(...)' code ##################

    
    # Call to ones(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_247078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of 'n' (line 210)
    n_247079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 19), list_247078, n_247079)
    
    # Processing the call keyword arguments (line 210)
    kwargs_247080 = {}
    # Getting the type of 'np' (line 210)
    np_247076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'np', False)
    # Obtaining the member 'ones' of a type (line 210)
    ones_247077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 11), np_247076, 'ones')
    # Calling ones(args, kwargs) (line 210)
    ones_call_result_247081 = invoke(stypy.reporting.localization.Localization(__file__, 210, 11), ones_247077, *[list_247078], **kwargs_247080)
    
    # Assigning a type to the variable 'stypy_return_type' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type', ones_call_result_247081)
    
    # ################# End of 'x0_10(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'x0_10' in the type store
    # Getting the type of 'stypy_return_type' (line 209)
    stypy_return_type_247082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_247082)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'x0_10'
    return stypy_return_type_247082

# Assigning a type to the variable 'x0_10' (line 209)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'x0_10', x0_10)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

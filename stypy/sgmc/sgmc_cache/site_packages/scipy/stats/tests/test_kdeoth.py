
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from scipy import stats
4: import numpy as np
5: from numpy.testing import (assert_almost_equal, assert_,
6:     assert_array_almost_equal, assert_array_almost_equal_nulp)
7: from pytest import raises as assert_raises
8: 
9: 
10: def test_kde_1d():
11:     #some basic tests comparing to normal distribution
12:     np.random.seed(8765678)
13:     n_basesample = 500
14:     xn = np.random.randn(n_basesample)
15:     xnmean = xn.mean()
16:     xnstd = xn.std(ddof=1)
17: 
18:     # get kde for original sample
19:     gkde = stats.gaussian_kde(xn)
20: 
21:     # evaluate the density function for the kde for some points
22:     xs = np.linspace(-7,7,501)
23:     kdepdf = gkde.evaluate(xs)
24:     normpdf = stats.norm.pdf(xs, loc=xnmean, scale=xnstd)
25:     intervall = xs[1] - xs[0]
26: 
27:     assert_(np.sum((kdepdf - normpdf)**2)*intervall < 0.01)
28:     prob1 = gkde.integrate_box_1d(xnmean, np.inf)
29:     prob2 = gkde.integrate_box_1d(-np.inf, xnmean)
30:     assert_almost_equal(prob1, 0.5, decimal=1)
31:     assert_almost_equal(prob2, 0.5, decimal=1)
32:     assert_almost_equal(gkde.integrate_box(xnmean, np.inf), prob1, decimal=13)
33:     assert_almost_equal(gkde.integrate_box(-np.inf, xnmean), prob2, decimal=13)
34: 
35:     assert_almost_equal(gkde.integrate_kde(gkde),
36:                         (kdepdf**2).sum()*intervall, decimal=2)
37:     assert_almost_equal(gkde.integrate_gaussian(xnmean, xnstd**2),
38:                         (kdepdf*normpdf).sum()*intervall, decimal=2)
39: 
40: 
41: def test_kde_2d():
42:     #some basic tests comparing to normal distribution
43:     np.random.seed(8765678)
44:     n_basesample = 500
45: 
46:     mean = np.array([1.0, 3.0])
47:     covariance = np.array([[1.0, 2.0], [2.0, 6.0]])
48: 
49:     # Need transpose (shape (2, 500)) for kde
50:     xn = np.random.multivariate_normal(mean, covariance, size=n_basesample).T
51: 
52:     # get kde for original sample
53:     gkde = stats.gaussian_kde(xn)
54: 
55:     # evaluate the density function for the kde for some points
56:     x, y = np.mgrid[-7:7:500j, -7:7:500j]
57:     grid_coords = np.vstack([x.ravel(), y.ravel()])
58:     kdepdf = gkde.evaluate(grid_coords)
59:     kdepdf = kdepdf.reshape(500, 500)
60: 
61:     normpdf = stats.multivariate_normal.pdf(np.dstack([x, y]), mean=mean, cov=covariance)
62:     intervall = y.ravel()[1] - y.ravel()[0]
63: 
64:     assert_(np.sum((kdepdf - normpdf)**2) * (intervall**2) < 0.01)
65: 
66:     small = -1e100
67:     large = 1e100
68:     prob1 = gkde.integrate_box([small, mean[1]], [large, large])
69:     prob2 = gkde.integrate_box([small, small], [large, mean[1]])
70: 
71:     assert_almost_equal(prob1, 0.5, decimal=1)
72:     assert_almost_equal(prob2, 0.5, decimal=1)
73:     assert_almost_equal(gkde.integrate_kde(gkde),
74:                         (kdepdf**2).sum()*(intervall**2), decimal=2)
75:     assert_almost_equal(gkde.integrate_gaussian(mean, covariance),
76:                         (kdepdf*normpdf).sum()*(intervall**2), decimal=2)
77: 
78: 
79: def test_kde_bandwidth_method():
80:     def scotts_factor(kde_obj):
81:         '''Same as default, just check that it works.'''
82:         return np.power(kde_obj.n, -1./(kde_obj.d+4))
83: 
84:     np.random.seed(8765678)
85:     n_basesample = 50
86:     xn = np.random.randn(n_basesample)
87: 
88:     # Default
89:     gkde = stats.gaussian_kde(xn)
90:     # Supply a callable
91:     gkde2 = stats.gaussian_kde(xn, bw_method=scotts_factor)
92:     # Supply a scalar
93:     gkde3 = stats.gaussian_kde(xn, bw_method=gkde.factor)
94: 
95:     xs = np.linspace(-7,7,51)
96:     kdepdf = gkde.evaluate(xs)
97:     kdepdf2 = gkde2.evaluate(xs)
98:     assert_almost_equal(kdepdf, kdepdf2)
99:     kdepdf3 = gkde3.evaluate(xs)
100:     assert_almost_equal(kdepdf, kdepdf3)
101: 
102:     assert_raises(ValueError, stats.gaussian_kde, xn, bw_method='wrongstring')
103: 
104: 
105: # Subclasses that should stay working (extracted from various sources).
106: # Unfortunately the earlier design of gaussian_kde made it necessary for users
107: # to create these kinds of subclasses, or call _compute_covariance() directly.
108: 
109: class _kde_subclass1(stats.gaussian_kde):
110:     def __init__(self, dataset):
111:         self.dataset = np.atleast_2d(dataset)
112:         self.d, self.n = self.dataset.shape
113:         self.covariance_factor = self.scotts_factor
114:         self._compute_covariance()
115: 
116: 
117: class _kde_subclass2(stats.gaussian_kde):
118:     def __init__(self, dataset):
119:         self.covariance_factor = self.scotts_factor
120:         super(_kde_subclass2, self).__init__(dataset)
121: 
122: 
123: class _kde_subclass3(stats.gaussian_kde):
124:     def __init__(self, dataset, covariance):
125:         self.covariance = covariance
126:         stats.gaussian_kde.__init__(self, dataset)
127: 
128:     def _compute_covariance(self):
129:         self.inv_cov = np.linalg.inv(self.covariance)
130:         self._norm_factor = np.sqrt(np.linalg.det(2*np.pi * self.covariance)) \
131:                                    * self.n
132: 
133: 
134: class _kde_subclass4(stats.gaussian_kde):
135:     def covariance_factor(self):
136:         return 0.5 * self.silverman_factor()
137: 
138: 
139: def test_gaussian_kde_subclassing():
140:     x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
141:     xs = np.linspace(-10, 10, num=50)
142: 
143:     # gaussian_kde itself
144:     kde = stats.gaussian_kde(x1)
145:     ys = kde(xs)
146: 
147:     # subclass 1
148:     kde1 = _kde_subclass1(x1)
149:     y1 = kde1(xs)
150:     assert_array_almost_equal_nulp(ys, y1, nulp=10)
151: 
152:     # subclass 2
153:     kde2 = _kde_subclass2(x1)
154:     y2 = kde2(xs)
155:     assert_array_almost_equal_nulp(ys, y2, nulp=10)
156: 
157:     # subclass 3
158:     kde3 = _kde_subclass3(x1, kde.covariance)
159:     y3 = kde3(xs)
160:     assert_array_almost_equal_nulp(ys, y3, nulp=10)
161: 
162:     # subclass 4
163:     kde4 = _kde_subclass4(x1)
164:     y4 = kde4(x1)
165:     y_expected = [0.06292987, 0.06346938, 0.05860291, 0.08657652, 0.07904017]
166: 
167:     assert_array_almost_equal(y_expected, y4, decimal=6)
168: 
169:     # Not a subclass, but check for use of _compute_covariance()
170:     kde5 = kde
171:     kde5.covariance_factor = lambda: kde.factor
172:     kde5._compute_covariance()
173:     y5 = kde5(xs)
174:     assert_array_almost_equal_nulp(ys, y5, nulp=10)
175: 
176: 
177: def test_gaussian_kde_covariance_caching():
178:     x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
179:     xs = np.linspace(-10, 10, num=5)
180:     # These expected values are from scipy 0.10, before some changes to
181:     # gaussian_kde.  They were not compared with any external reference.
182:     y_expected = [0.02463386, 0.04689208, 0.05395444, 0.05337754, 0.01664475]
183: 
184:     # Set the bandwidth, then reset it to the default.
185:     kde = stats.gaussian_kde(x1)
186:     kde.set_bandwidth(bw_method=0.5)
187:     kde.set_bandwidth(bw_method='scott')
188:     y2 = kde(xs)
189: 
190:     assert_array_almost_equal(y_expected, y2, decimal=7)
191: 
192: 
193: def test_gaussian_kde_monkeypatch():
194:     '''Ugly, but people may rely on this.  See scipy pull request 123,
195:     specifically the linked ML thread "Width of the Gaussian in stats.kde".
196:     If it is necessary to break this later on, that is to be discussed on ML.
197:     '''
198:     x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
199:     xs = np.linspace(-10, 10, num=50)
200: 
201:     # The old monkeypatched version to get at Silverman's Rule.
202:     kde = stats.gaussian_kde(x1)
203:     kde.covariance_factor = kde.silverman_factor
204:     kde._compute_covariance()
205:     y1 = kde(xs)
206: 
207:     # The new saner version.
208:     kde2 = stats.gaussian_kde(x1, bw_method='silverman')
209:     y2 = kde2(xs)
210: 
211:     assert_array_almost_equal_nulp(y1, y2, nulp=10)
212: 
213: 
214: def test_kde_integer_input():
215:     '''Regression test for #1181.'''
216:     x1 = np.arange(5)
217:     kde = stats.gaussian_kde(x1)
218:     y_expected = [0.13480721, 0.18222869, 0.19514935, 0.18222869, 0.13480721]
219:     assert_array_almost_equal(kde(x1), y_expected, decimal=6)
220: 
221: 
222: def test_pdf_logpdf():
223:     np.random.seed(1)
224:     n_basesample = 50
225:     xn = np.random.randn(n_basesample)
226: 
227:     # Default
228:     gkde = stats.gaussian_kde(xn)
229: 
230:     xs = np.linspace(-15, 12, 25)
231:     pdf = gkde.evaluate(xs)
232:     pdf2 = gkde.pdf(xs)
233:     assert_almost_equal(pdf, pdf2, decimal=12)
234: 
235:     logpdf = np.log(pdf)
236:     logpdf2 = gkde.logpdf(xs)
237:     assert_almost_equal(logpdf, logpdf2, decimal=12)
238: 
239:     # There are more points than data
240:     gkde = stats.gaussian_kde(xs)
241:     pdf = np.log(gkde.evaluate(xn))
242:     pdf2 = gkde.logpdf(xn)
243:     assert_almost_equal(pdf, pdf2, decimal=12)
244: 
245: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy import stats' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_652287 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy')

if (type(import_652287) is not StypyTypeError):

    if (import_652287 != 'pyd_module'):
        __import__(import_652287)
        sys_modules_652288 = sys.modules[import_652287]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy', sys_modules_652288.module_type_store, module_type_store, ['stats'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_652288, sys_modules_652288.module_type_store, module_type_store)
    else:
        from scipy import stats

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy', None, module_type_store, ['stats'], [stats])

else:
    # Assigning a type to the variable 'scipy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy', import_652287)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_652289 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_652289) is not StypyTypeError):

    if (import_652289 != 'pyd_module'):
        __import__(import_652289)
        sys_modules_652290 = sys.modules[import_652289]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_652290.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_652289)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_almost_equal, assert_, assert_array_almost_equal, assert_array_almost_equal_nulp' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_652291 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_652291) is not StypyTypeError):

    if (import_652291 != 'pyd_module'):
        __import__(import_652291)
        sys_modules_652292 = sys.modules[import_652291]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_652292.module_type_store, module_type_store, ['assert_almost_equal', 'assert_', 'assert_array_almost_equal', 'assert_array_almost_equal_nulp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_652292, sys_modules_652292.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_, assert_array_almost_equal, assert_array_almost_equal_nulp

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_', 'assert_array_almost_equal', 'assert_array_almost_equal_nulp'], [assert_almost_equal, assert_, assert_array_almost_equal, assert_array_almost_equal_nulp])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_652291)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from pytest import assert_raises' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_652293 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_652293) is not StypyTypeError):

    if (import_652293 != 'pyd_module'):
        __import__(import_652293)
        sys_modules_652294 = sys.modules[import_652293]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_652294.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_652294, sys_modules_652294.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_652293)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')


@norecursion
def test_kde_1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_kde_1d'
    module_type_store = module_type_store.open_function_context('test_kde_1d', 10, 0, False)
    
    # Passed parameters checking function
    test_kde_1d.stypy_localization = localization
    test_kde_1d.stypy_type_of_self = None
    test_kde_1d.stypy_type_store = module_type_store
    test_kde_1d.stypy_function_name = 'test_kde_1d'
    test_kde_1d.stypy_param_names_list = []
    test_kde_1d.stypy_varargs_param_name = None
    test_kde_1d.stypy_kwargs_param_name = None
    test_kde_1d.stypy_call_defaults = defaults
    test_kde_1d.stypy_call_varargs = varargs
    test_kde_1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_kde_1d', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_kde_1d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_kde_1d(...)' code ##################

    
    # Call to seed(...): (line 12)
    # Processing the call arguments (line 12)
    int_652298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_652299 = {}
    # Getting the type of 'np' (line 12)
    np_652295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 12)
    random_652296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), np_652295, 'random')
    # Obtaining the member 'seed' of a type (line 12)
    seed_652297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), random_652296, 'seed')
    # Calling seed(args, kwargs) (line 12)
    seed_call_result_652300 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), seed_652297, *[int_652298], **kwargs_652299)
    
    
    # Assigning a Num to a Name (line 13):
    
    # Assigning a Num to a Name (line 13):
    int_652301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'int')
    # Assigning a type to the variable 'n_basesample' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'n_basesample', int_652301)
    
    # Assigning a Call to a Name (line 14):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to randn(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'n_basesample' (line 14)
    n_basesample_652305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 25), 'n_basesample', False)
    # Processing the call keyword arguments (line 14)
    kwargs_652306 = {}
    # Getting the type of 'np' (line 14)
    np_652302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 14)
    random_652303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 9), np_652302, 'random')
    # Obtaining the member 'randn' of a type (line 14)
    randn_652304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 9), random_652303, 'randn')
    # Calling randn(args, kwargs) (line 14)
    randn_call_result_652307 = invoke(stypy.reporting.localization.Localization(__file__, 14, 9), randn_652304, *[n_basesample_652305], **kwargs_652306)
    
    # Assigning a type to the variable 'xn' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'xn', randn_call_result_652307)
    
    # Assigning a Call to a Name (line 15):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to mean(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_652310 = {}
    # Getting the type of 'xn' (line 15)
    xn_652308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'xn', False)
    # Obtaining the member 'mean' of a type (line 15)
    mean_652309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 13), xn_652308, 'mean')
    # Calling mean(args, kwargs) (line 15)
    mean_call_result_652311 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), mean_652309, *[], **kwargs_652310)
    
    # Assigning a type to the variable 'xnmean' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'xnmean', mean_call_result_652311)
    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to std(...): (line 16)
    # Processing the call keyword arguments (line 16)
    int_652314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'int')
    keyword_652315 = int_652314
    kwargs_652316 = {'ddof': keyword_652315}
    # Getting the type of 'xn' (line 16)
    xn_652312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'xn', False)
    # Obtaining the member 'std' of a type (line 16)
    std_652313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), xn_652312, 'std')
    # Calling std(args, kwargs) (line 16)
    std_call_result_652317 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), std_652313, *[], **kwargs_652316)
    
    # Assigning a type to the variable 'xnstd' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'xnstd', std_call_result_652317)
    
    # Assigning a Call to a Name (line 19):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to gaussian_kde(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'xn' (line 19)
    xn_652320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 30), 'xn', False)
    # Processing the call keyword arguments (line 19)
    kwargs_652321 = {}
    # Getting the type of 'stats' (line 19)
    stats_652318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 19)
    gaussian_kde_652319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 11), stats_652318, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 19)
    gaussian_kde_call_result_652322 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), gaussian_kde_652319, *[xn_652320], **kwargs_652321)
    
    # Assigning a type to the variable 'gkde' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'gkde', gaussian_kde_call_result_652322)
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to linspace(...): (line 22)
    # Processing the call arguments (line 22)
    int_652325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    int_652326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'int')
    int_652327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_652328 = {}
    # Getting the type of 'np' (line 22)
    np_652323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'np', False)
    # Obtaining the member 'linspace' of a type (line 22)
    linspace_652324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 9), np_652323, 'linspace')
    # Calling linspace(args, kwargs) (line 22)
    linspace_call_result_652329 = invoke(stypy.reporting.localization.Localization(__file__, 22, 9), linspace_652324, *[int_652325, int_652326, int_652327], **kwargs_652328)
    
    # Assigning a type to the variable 'xs' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'xs', linspace_call_result_652329)
    
    # Assigning a Call to a Name (line 23):
    
    # Assigning a Call to a Name (line 23):
    
    # Call to evaluate(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'xs' (line 23)
    xs_652332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'xs', False)
    # Processing the call keyword arguments (line 23)
    kwargs_652333 = {}
    # Getting the type of 'gkde' (line 23)
    gkde_652330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'gkde', False)
    # Obtaining the member 'evaluate' of a type (line 23)
    evaluate_652331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), gkde_652330, 'evaluate')
    # Calling evaluate(args, kwargs) (line 23)
    evaluate_call_result_652334 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), evaluate_652331, *[xs_652332], **kwargs_652333)
    
    # Assigning a type to the variable 'kdepdf' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'kdepdf', evaluate_call_result_652334)
    
    # Assigning a Call to a Name (line 24):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to pdf(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'xs' (line 24)
    xs_652338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'xs', False)
    # Processing the call keyword arguments (line 24)
    # Getting the type of 'xnmean' (line 24)
    xnmean_652339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'xnmean', False)
    keyword_652340 = xnmean_652339
    # Getting the type of 'xnstd' (line 24)
    xnstd_652341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 51), 'xnstd', False)
    keyword_652342 = xnstd_652341
    kwargs_652343 = {'loc': keyword_652340, 'scale': keyword_652342}
    # Getting the type of 'stats' (line 24)
    stats_652335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'stats', False)
    # Obtaining the member 'norm' of a type (line 24)
    norm_652336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 14), stats_652335, 'norm')
    # Obtaining the member 'pdf' of a type (line 24)
    pdf_652337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 14), norm_652336, 'pdf')
    # Calling pdf(args, kwargs) (line 24)
    pdf_call_result_652344 = invoke(stypy.reporting.localization.Localization(__file__, 24, 14), pdf_652337, *[xs_652338], **kwargs_652343)
    
    # Assigning a type to the variable 'normpdf' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'normpdf', pdf_call_result_652344)
    
    # Assigning a BinOp to a Name (line 25):
    
    # Assigning a BinOp to a Name (line 25):
    
    # Obtaining the type of the subscript
    int_652345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
    # Getting the type of 'xs' (line 25)
    xs_652346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'xs')
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___652347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), xs_652346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_652348 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), getitem___652347, int_652345)
    
    
    # Obtaining the type of the subscript
    int_652349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'int')
    # Getting the type of 'xs' (line 25)
    xs_652350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'xs')
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___652351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), xs_652350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_652352 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), getitem___652351, int_652349)
    
    # Applying the binary operator '-' (line 25)
    result_sub_652353 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 16), '-', subscript_call_result_652348, subscript_call_result_652352)
    
    # Assigning a type to the variable 'intervall' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'intervall', result_sub_652353)
    
    # Call to assert_(...): (line 27)
    # Processing the call arguments (line 27)
    
    
    # Call to sum(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'kdepdf' (line 27)
    kdepdf_652357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'kdepdf', False)
    # Getting the type of 'normpdf' (line 27)
    normpdf_652358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'normpdf', False)
    # Applying the binary operator '-' (line 27)
    result_sub_652359 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 20), '-', kdepdf_652357, normpdf_652358)
    
    int_652360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 39), 'int')
    # Applying the binary operator '**' (line 27)
    result_pow_652361 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 19), '**', result_sub_652359, int_652360)
    
    # Processing the call keyword arguments (line 27)
    kwargs_652362 = {}
    # Getting the type of 'np' (line 27)
    np_652355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'np', False)
    # Obtaining the member 'sum' of a type (line 27)
    sum_652356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), np_652355, 'sum')
    # Calling sum(args, kwargs) (line 27)
    sum_call_result_652363 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), sum_652356, *[result_pow_652361], **kwargs_652362)
    
    # Getting the type of 'intervall' (line 27)
    intervall_652364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 42), 'intervall', False)
    # Applying the binary operator '*' (line 27)
    result_mul_652365 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 12), '*', sum_call_result_652363, intervall_652364)
    
    float_652366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 54), 'float')
    # Applying the binary operator '<' (line 27)
    result_lt_652367 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 12), '<', result_mul_652365, float_652366)
    
    # Processing the call keyword arguments (line 27)
    kwargs_652368 = {}
    # Getting the type of 'assert_' (line 27)
    assert__652354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 27)
    assert__call_result_652369 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), assert__652354, *[result_lt_652367], **kwargs_652368)
    
    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to integrate_box_1d(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'xnmean' (line 28)
    xnmean_652372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'xnmean', False)
    # Getting the type of 'np' (line 28)
    np_652373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'np', False)
    # Obtaining the member 'inf' of a type (line 28)
    inf_652374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 42), np_652373, 'inf')
    # Processing the call keyword arguments (line 28)
    kwargs_652375 = {}
    # Getting the type of 'gkde' (line 28)
    gkde_652370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'gkde', False)
    # Obtaining the member 'integrate_box_1d' of a type (line 28)
    integrate_box_1d_652371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), gkde_652370, 'integrate_box_1d')
    # Calling integrate_box_1d(args, kwargs) (line 28)
    integrate_box_1d_call_result_652376 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), integrate_box_1d_652371, *[xnmean_652372, inf_652374], **kwargs_652375)
    
    # Assigning a type to the variable 'prob1' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'prob1', integrate_box_1d_call_result_652376)
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to integrate_box_1d(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Getting the type of 'np' (line 29)
    np_652379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'np', False)
    # Obtaining the member 'inf' of a type (line 29)
    inf_652380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 35), np_652379, 'inf')
    # Applying the 'usub' unary operator (line 29)
    result___neg___652381 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 34), 'usub', inf_652380)
    
    # Getting the type of 'xnmean' (line 29)
    xnmean_652382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 43), 'xnmean', False)
    # Processing the call keyword arguments (line 29)
    kwargs_652383 = {}
    # Getting the type of 'gkde' (line 29)
    gkde_652377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'gkde', False)
    # Obtaining the member 'integrate_box_1d' of a type (line 29)
    integrate_box_1d_652378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), gkde_652377, 'integrate_box_1d')
    # Calling integrate_box_1d(args, kwargs) (line 29)
    integrate_box_1d_call_result_652384 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), integrate_box_1d_652378, *[result___neg___652381, xnmean_652382], **kwargs_652383)
    
    # Assigning a type to the variable 'prob2' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'prob2', integrate_box_1d_call_result_652384)
    
    # Call to assert_almost_equal(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'prob1' (line 30)
    prob1_652386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'prob1', False)
    float_652387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'float')
    # Processing the call keyword arguments (line 30)
    int_652388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 44), 'int')
    keyword_652389 = int_652388
    kwargs_652390 = {'decimal': keyword_652389}
    # Getting the type of 'assert_almost_equal' (line 30)
    assert_almost_equal_652385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 30)
    assert_almost_equal_call_result_652391 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_almost_equal_652385, *[prob1_652386, float_652387], **kwargs_652390)
    
    
    # Call to assert_almost_equal(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'prob2' (line 31)
    prob2_652393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'prob2', False)
    float_652394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'float')
    # Processing the call keyword arguments (line 31)
    int_652395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 44), 'int')
    keyword_652396 = int_652395
    kwargs_652397 = {'decimal': keyword_652396}
    # Getting the type of 'assert_almost_equal' (line 31)
    assert_almost_equal_652392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 31)
    assert_almost_equal_call_result_652398 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_almost_equal_652392, *[prob2_652393, float_652394], **kwargs_652397)
    
    
    # Call to assert_almost_equal(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to integrate_box(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'xnmean' (line 32)
    xnmean_652402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 43), 'xnmean', False)
    # Getting the type of 'np' (line 32)
    np_652403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 51), 'np', False)
    # Obtaining the member 'inf' of a type (line 32)
    inf_652404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 51), np_652403, 'inf')
    # Processing the call keyword arguments (line 32)
    kwargs_652405 = {}
    # Getting the type of 'gkde' (line 32)
    gkde_652400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'gkde', False)
    # Obtaining the member 'integrate_box' of a type (line 32)
    integrate_box_652401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 24), gkde_652400, 'integrate_box')
    # Calling integrate_box(args, kwargs) (line 32)
    integrate_box_call_result_652406 = invoke(stypy.reporting.localization.Localization(__file__, 32, 24), integrate_box_652401, *[xnmean_652402, inf_652404], **kwargs_652405)
    
    # Getting the type of 'prob1' (line 32)
    prob1_652407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 60), 'prob1', False)
    # Processing the call keyword arguments (line 32)
    int_652408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 75), 'int')
    keyword_652409 = int_652408
    kwargs_652410 = {'decimal': keyword_652409}
    # Getting the type of 'assert_almost_equal' (line 32)
    assert_almost_equal_652399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 32)
    assert_almost_equal_call_result_652411 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), assert_almost_equal_652399, *[integrate_box_call_result_652406, prob1_652407], **kwargs_652410)
    
    
    # Call to assert_almost_equal(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to integrate_box(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Getting the type of 'np' (line 33)
    np_652415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 44), 'np', False)
    # Obtaining the member 'inf' of a type (line 33)
    inf_652416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 44), np_652415, 'inf')
    # Applying the 'usub' unary operator (line 33)
    result___neg___652417 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 43), 'usub', inf_652416)
    
    # Getting the type of 'xnmean' (line 33)
    xnmean_652418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 52), 'xnmean', False)
    # Processing the call keyword arguments (line 33)
    kwargs_652419 = {}
    # Getting the type of 'gkde' (line 33)
    gkde_652413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'gkde', False)
    # Obtaining the member 'integrate_box' of a type (line 33)
    integrate_box_652414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 24), gkde_652413, 'integrate_box')
    # Calling integrate_box(args, kwargs) (line 33)
    integrate_box_call_result_652420 = invoke(stypy.reporting.localization.Localization(__file__, 33, 24), integrate_box_652414, *[result___neg___652417, xnmean_652418], **kwargs_652419)
    
    # Getting the type of 'prob2' (line 33)
    prob2_652421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 61), 'prob2', False)
    # Processing the call keyword arguments (line 33)
    int_652422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 76), 'int')
    keyword_652423 = int_652422
    kwargs_652424 = {'decimal': keyword_652423}
    # Getting the type of 'assert_almost_equal' (line 33)
    assert_almost_equal_652412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 33)
    assert_almost_equal_call_result_652425 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), assert_almost_equal_652412, *[integrate_box_call_result_652420, prob2_652421], **kwargs_652424)
    
    
    # Call to assert_almost_equal(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to integrate_kde(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'gkde' (line 35)
    gkde_652429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 'gkde', False)
    # Processing the call keyword arguments (line 35)
    kwargs_652430 = {}
    # Getting the type of 'gkde' (line 35)
    gkde_652427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'gkde', False)
    # Obtaining the member 'integrate_kde' of a type (line 35)
    integrate_kde_652428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 24), gkde_652427, 'integrate_kde')
    # Calling integrate_kde(args, kwargs) (line 35)
    integrate_kde_call_result_652431 = invoke(stypy.reporting.localization.Localization(__file__, 35, 24), integrate_kde_652428, *[gkde_652429], **kwargs_652430)
    
    
    # Call to sum(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_652436 = {}
    # Getting the type of 'kdepdf' (line 36)
    kdepdf_652432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'kdepdf', False)
    int_652433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 33), 'int')
    # Applying the binary operator '**' (line 36)
    result_pow_652434 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 25), '**', kdepdf_652432, int_652433)
    
    # Obtaining the member 'sum' of a type (line 36)
    sum_652435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 25), result_pow_652434, 'sum')
    # Calling sum(args, kwargs) (line 36)
    sum_call_result_652437 = invoke(stypy.reporting.localization.Localization(__file__, 36, 25), sum_652435, *[], **kwargs_652436)
    
    # Getting the type of 'intervall' (line 36)
    intervall_652438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 42), 'intervall', False)
    # Applying the binary operator '*' (line 36)
    result_mul_652439 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 24), '*', sum_call_result_652437, intervall_652438)
    
    # Processing the call keyword arguments (line 35)
    int_652440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 61), 'int')
    keyword_652441 = int_652440
    kwargs_652442 = {'decimal': keyword_652441}
    # Getting the type of 'assert_almost_equal' (line 35)
    assert_almost_equal_652426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 35)
    assert_almost_equal_call_result_652443 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), assert_almost_equal_652426, *[integrate_kde_call_result_652431, result_mul_652439], **kwargs_652442)
    
    
    # Call to assert_almost_equal(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to integrate_gaussian(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'xnmean' (line 37)
    xnmean_652447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 48), 'xnmean', False)
    # Getting the type of 'xnstd' (line 37)
    xnstd_652448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 56), 'xnstd', False)
    int_652449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 63), 'int')
    # Applying the binary operator '**' (line 37)
    result_pow_652450 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 56), '**', xnstd_652448, int_652449)
    
    # Processing the call keyword arguments (line 37)
    kwargs_652451 = {}
    # Getting the type of 'gkde' (line 37)
    gkde_652445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'gkde', False)
    # Obtaining the member 'integrate_gaussian' of a type (line 37)
    integrate_gaussian_652446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 24), gkde_652445, 'integrate_gaussian')
    # Calling integrate_gaussian(args, kwargs) (line 37)
    integrate_gaussian_call_result_652452 = invoke(stypy.reporting.localization.Localization(__file__, 37, 24), integrate_gaussian_652446, *[xnmean_652447, result_pow_652450], **kwargs_652451)
    
    
    # Call to sum(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_652457 = {}
    # Getting the type of 'kdepdf' (line 38)
    kdepdf_652453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'kdepdf', False)
    # Getting the type of 'normpdf' (line 38)
    normpdf_652454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'normpdf', False)
    # Applying the binary operator '*' (line 38)
    result_mul_652455 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 25), '*', kdepdf_652453, normpdf_652454)
    
    # Obtaining the member 'sum' of a type (line 38)
    sum_652456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 25), result_mul_652455, 'sum')
    # Calling sum(args, kwargs) (line 38)
    sum_call_result_652458 = invoke(stypy.reporting.localization.Localization(__file__, 38, 25), sum_652456, *[], **kwargs_652457)
    
    # Getting the type of 'intervall' (line 38)
    intervall_652459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 47), 'intervall', False)
    # Applying the binary operator '*' (line 38)
    result_mul_652460 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 24), '*', sum_call_result_652458, intervall_652459)
    
    # Processing the call keyword arguments (line 37)
    int_652461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 66), 'int')
    keyword_652462 = int_652461
    kwargs_652463 = {'decimal': keyword_652462}
    # Getting the type of 'assert_almost_equal' (line 37)
    assert_almost_equal_652444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 37)
    assert_almost_equal_call_result_652464 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), assert_almost_equal_652444, *[integrate_gaussian_call_result_652452, result_mul_652460], **kwargs_652463)
    
    
    # ################# End of 'test_kde_1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_kde_1d' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_652465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652465)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_kde_1d'
    return stypy_return_type_652465

# Assigning a type to the variable 'test_kde_1d' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test_kde_1d', test_kde_1d)

@norecursion
def test_kde_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_kde_2d'
    module_type_store = module_type_store.open_function_context('test_kde_2d', 41, 0, False)
    
    # Passed parameters checking function
    test_kde_2d.stypy_localization = localization
    test_kde_2d.stypy_type_of_self = None
    test_kde_2d.stypy_type_store = module_type_store
    test_kde_2d.stypy_function_name = 'test_kde_2d'
    test_kde_2d.stypy_param_names_list = []
    test_kde_2d.stypy_varargs_param_name = None
    test_kde_2d.stypy_kwargs_param_name = None
    test_kde_2d.stypy_call_defaults = defaults
    test_kde_2d.stypy_call_varargs = varargs
    test_kde_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_kde_2d', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_kde_2d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_kde_2d(...)' code ##################

    
    # Call to seed(...): (line 43)
    # Processing the call arguments (line 43)
    int_652469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_652470 = {}
    # Getting the type of 'np' (line 43)
    np_652466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 43)
    random_652467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), np_652466, 'random')
    # Obtaining the member 'seed' of a type (line 43)
    seed_652468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), random_652467, 'seed')
    # Calling seed(args, kwargs) (line 43)
    seed_call_result_652471 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), seed_652468, *[int_652469], **kwargs_652470)
    
    
    # Assigning a Num to a Name (line 44):
    
    # Assigning a Num to a Name (line 44):
    int_652472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'int')
    # Assigning a type to the variable 'n_basesample' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'n_basesample', int_652472)
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to array(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_652475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    float_652476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 20), list_652475, float_652476)
    # Adding element type (line 46)
    float_652477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 20), list_652475, float_652477)
    
    # Processing the call keyword arguments (line 46)
    kwargs_652478 = {}
    # Getting the type of 'np' (line 46)
    np_652473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 46)
    array_652474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 11), np_652473, 'array')
    # Calling array(args, kwargs) (line 46)
    array_call_result_652479 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), array_652474, *[list_652475], **kwargs_652478)
    
    # Assigning a type to the variable 'mean' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'mean', array_call_result_652479)
    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to array(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_652482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_652483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    float_652484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 27), list_652483, float_652484)
    # Adding element type (line 47)
    float_652485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 27), list_652483, float_652485)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 26), list_652482, list_652483)
    # Adding element type (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_652486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    float_652487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 39), list_652486, float_652487)
    # Adding element type (line 47)
    float_652488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 39), list_652486, float_652488)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 26), list_652482, list_652486)
    
    # Processing the call keyword arguments (line 47)
    kwargs_652489 = {}
    # Getting the type of 'np' (line 47)
    np_652480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'np', False)
    # Obtaining the member 'array' of a type (line 47)
    array_652481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), np_652480, 'array')
    # Calling array(args, kwargs) (line 47)
    array_call_result_652490 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), array_652481, *[list_652482], **kwargs_652489)
    
    # Assigning a type to the variable 'covariance' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'covariance', array_call_result_652490)
    
    # Assigning a Attribute to a Name (line 50):
    
    # Assigning a Attribute to a Name (line 50):
    
    # Call to multivariate_normal(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'mean' (line 50)
    mean_652494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 39), 'mean', False)
    # Getting the type of 'covariance' (line 50)
    covariance_652495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 45), 'covariance', False)
    # Processing the call keyword arguments (line 50)
    # Getting the type of 'n_basesample' (line 50)
    n_basesample_652496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 62), 'n_basesample', False)
    keyword_652497 = n_basesample_652496
    kwargs_652498 = {'size': keyword_652497}
    # Getting the type of 'np' (line 50)
    np_652491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 50)
    random_652492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 9), np_652491, 'random')
    # Obtaining the member 'multivariate_normal' of a type (line 50)
    multivariate_normal_652493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 9), random_652492, 'multivariate_normal')
    # Calling multivariate_normal(args, kwargs) (line 50)
    multivariate_normal_call_result_652499 = invoke(stypy.reporting.localization.Localization(__file__, 50, 9), multivariate_normal_652493, *[mean_652494, covariance_652495], **kwargs_652498)
    
    # Obtaining the member 'T' of a type (line 50)
    T_652500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 9), multivariate_normal_call_result_652499, 'T')
    # Assigning a type to the variable 'xn' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'xn', T_652500)
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to gaussian_kde(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'xn' (line 53)
    xn_652503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'xn', False)
    # Processing the call keyword arguments (line 53)
    kwargs_652504 = {}
    # Getting the type of 'stats' (line 53)
    stats_652501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 53)
    gaussian_kde_652502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), stats_652501, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 53)
    gaussian_kde_call_result_652505 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), gaussian_kde_652502, *[xn_652503], **kwargs_652504)
    
    # Assigning a type to the variable 'gkde' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'gkde', gaussian_kde_call_result_652505)
    
    # Assigning a Subscript to a Tuple (line 56):
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_652506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Obtaining the type of the subscript
    int_652507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'int')
    int_652508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'int')
    complex_652509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'complex')
    slice_652510 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 11), int_652507, int_652508, complex_652509)
    int_652511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'int')
    int_652512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'int')
    complex_652513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 36), 'complex')
    slice_652514 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 11), int_652511, int_652512, complex_652513)
    # Getting the type of 'np' (line 56)
    np_652515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'np')
    # Obtaining the member 'mgrid' of a type (line 56)
    mgrid_652516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), np_652515, 'mgrid')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___652517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), mgrid_652516, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_652518 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), getitem___652517, (slice_652510, slice_652514))
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___652519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), subscript_call_result_652518, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_652520 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___652519, int_652506)
    
    # Assigning a type to the variable 'tuple_var_assignment_652283' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_652283', subscript_call_result_652520)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_652521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Obtaining the type of the subscript
    int_652522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'int')
    int_652523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'int')
    complex_652524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'complex')
    slice_652525 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 11), int_652522, int_652523, complex_652524)
    int_652526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'int')
    int_652527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'int')
    complex_652528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 36), 'complex')
    slice_652529 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 11), int_652526, int_652527, complex_652528)
    # Getting the type of 'np' (line 56)
    np_652530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'np')
    # Obtaining the member 'mgrid' of a type (line 56)
    mgrid_652531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), np_652530, 'mgrid')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___652532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), mgrid_652531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_652533 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), getitem___652532, (slice_652525, slice_652529))
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___652534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), subscript_call_result_652533, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_652535 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___652534, int_652521)
    
    # Assigning a type to the variable 'tuple_var_assignment_652284' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_652284', subscript_call_result_652535)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_652283' (line 56)
    tuple_var_assignment_652283_652536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_652283')
    # Assigning a type to the variable 'x' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'x', tuple_var_assignment_652283_652536)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_652284' (line 56)
    tuple_var_assignment_652284_652537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_652284')
    # Assigning a type to the variable 'y' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 7), 'y', tuple_var_assignment_652284_652537)
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to vstack(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Obtaining an instance of the builtin type 'list' (line 57)
    list_652540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 57)
    # Adding element type (line 57)
    
    # Call to ravel(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_652543 = {}
    # Getting the type of 'x' (line 57)
    x_652541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'x', False)
    # Obtaining the member 'ravel' of a type (line 57)
    ravel_652542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 29), x_652541, 'ravel')
    # Calling ravel(args, kwargs) (line 57)
    ravel_call_result_652544 = invoke(stypy.reporting.localization.Localization(__file__, 57, 29), ravel_652542, *[], **kwargs_652543)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), list_652540, ravel_call_result_652544)
    # Adding element type (line 57)
    
    # Call to ravel(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_652547 = {}
    # Getting the type of 'y' (line 57)
    y_652545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 40), 'y', False)
    # Obtaining the member 'ravel' of a type (line 57)
    ravel_652546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 40), y_652545, 'ravel')
    # Calling ravel(args, kwargs) (line 57)
    ravel_call_result_652548 = invoke(stypy.reporting.localization.Localization(__file__, 57, 40), ravel_652546, *[], **kwargs_652547)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 28), list_652540, ravel_call_result_652548)
    
    # Processing the call keyword arguments (line 57)
    kwargs_652549 = {}
    # Getting the type of 'np' (line 57)
    np_652538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'np', False)
    # Obtaining the member 'vstack' of a type (line 57)
    vstack_652539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), np_652538, 'vstack')
    # Calling vstack(args, kwargs) (line 57)
    vstack_call_result_652550 = invoke(stypy.reporting.localization.Localization(__file__, 57, 18), vstack_652539, *[list_652540], **kwargs_652549)
    
    # Assigning a type to the variable 'grid_coords' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'grid_coords', vstack_call_result_652550)
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to evaluate(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'grid_coords' (line 58)
    grid_coords_652553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'grid_coords', False)
    # Processing the call keyword arguments (line 58)
    kwargs_652554 = {}
    # Getting the type of 'gkde' (line 58)
    gkde_652551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'gkde', False)
    # Obtaining the member 'evaluate' of a type (line 58)
    evaluate_652552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 13), gkde_652551, 'evaluate')
    # Calling evaluate(args, kwargs) (line 58)
    evaluate_call_result_652555 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), evaluate_652552, *[grid_coords_652553], **kwargs_652554)
    
    # Assigning a type to the variable 'kdepdf' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'kdepdf', evaluate_call_result_652555)
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to reshape(...): (line 59)
    # Processing the call arguments (line 59)
    int_652558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'int')
    int_652559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 33), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_652560 = {}
    # Getting the type of 'kdepdf' (line 59)
    kdepdf_652556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'kdepdf', False)
    # Obtaining the member 'reshape' of a type (line 59)
    reshape_652557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 13), kdepdf_652556, 'reshape')
    # Calling reshape(args, kwargs) (line 59)
    reshape_call_result_652561 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), reshape_652557, *[int_652558, int_652559], **kwargs_652560)
    
    # Assigning a type to the variable 'kdepdf' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'kdepdf', reshape_call_result_652561)
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to pdf(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Call to dstack(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_652567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 54), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    # Adding element type (line 61)
    # Getting the type of 'x' (line 61)
    x_652568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 55), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 54), list_652567, x_652568)
    # Adding element type (line 61)
    # Getting the type of 'y' (line 61)
    y_652569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 58), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 54), list_652567, y_652569)
    
    # Processing the call keyword arguments (line 61)
    kwargs_652570 = {}
    # Getting the type of 'np' (line 61)
    np_652565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 44), 'np', False)
    # Obtaining the member 'dstack' of a type (line 61)
    dstack_652566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 44), np_652565, 'dstack')
    # Calling dstack(args, kwargs) (line 61)
    dstack_call_result_652571 = invoke(stypy.reporting.localization.Localization(__file__, 61, 44), dstack_652566, *[list_652567], **kwargs_652570)
    
    # Processing the call keyword arguments (line 61)
    # Getting the type of 'mean' (line 61)
    mean_652572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 68), 'mean', False)
    keyword_652573 = mean_652572
    # Getting the type of 'covariance' (line 61)
    covariance_652574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 78), 'covariance', False)
    keyword_652575 = covariance_652574
    kwargs_652576 = {'cov': keyword_652575, 'mean': keyword_652573}
    # Getting the type of 'stats' (line 61)
    stats_652562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'stats', False)
    # Obtaining the member 'multivariate_normal' of a type (line 61)
    multivariate_normal_652563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 14), stats_652562, 'multivariate_normal')
    # Obtaining the member 'pdf' of a type (line 61)
    pdf_652564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 14), multivariate_normal_652563, 'pdf')
    # Calling pdf(args, kwargs) (line 61)
    pdf_call_result_652577 = invoke(stypy.reporting.localization.Localization(__file__, 61, 14), pdf_652564, *[dstack_call_result_652571], **kwargs_652576)
    
    # Assigning a type to the variable 'normpdf' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'normpdf', pdf_call_result_652577)
    
    # Assigning a BinOp to a Name (line 62):
    
    # Assigning a BinOp to a Name (line 62):
    
    # Obtaining the type of the subscript
    int_652578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 26), 'int')
    
    # Call to ravel(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_652581 = {}
    # Getting the type of 'y' (line 62)
    y_652579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'y', False)
    # Obtaining the member 'ravel' of a type (line 62)
    ravel_652580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), y_652579, 'ravel')
    # Calling ravel(args, kwargs) (line 62)
    ravel_call_result_652582 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), ravel_652580, *[], **kwargs_652581)
    
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___652583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), ravel_call_result_652582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_652584 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), getitem___652583, int_652578)
    
    
    # Obtaining the type of the subscript
    int_652585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 41), 'int')
    
    # Call to ravel(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_652588 = {}
    # Getting the type of 'y' (line 62)
    y_652586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'y', False)
    # Obtaining the member 'ravel' of a type (line 62)
    ravel_652587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 31), y_652586, 'ravel')
    # Calling ravel(args, kwargs) (line 62)
    ravel_call_result_652589 = invoke(stypy.reporting.localization.Localization(__file__, 62, 31), ravel_652587, *[], **kwargs_652588)
    
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___652590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 31), ravel_call_result_652589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_652591 = invoke(stypy.reporting.localization.Localization(__file__, 62, 31), getitem___652590, int_652585)
    
    # Applying the binary operator '-' (line 62)
    result_sub_652592 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 16), '-', subscript_call_result_652584, subscript_call_result_652591)
    
    # Assigning a type to the variable 'intervall' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'intervall', result_sub_652592)
    
    # Call to assert_(...): (line 64)
    # Processing the call arguments (line 64)
    
    
    # Call to sum(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'kdepdf' (line 64)
    kdepdf_652596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'kdepdf', False)
    # Getting the type of 'normpdf' (line 64)
    normpdf_652597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'normpdf', False)
    # Applying the binary operator '-' (line 64)
    result_sub_652598 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 20), '-', kdepdf_652596, normpdf_652597)
    
    int_652599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 39), 'int')
    # Applying the binary operator '**' (line 64)
    result_pow_652600 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 19), '**', result_sub_652598, int_652599)
    
    # Processing the call keyword arguments (line 64)
    kwargs_652601 = {}
    # Getting the type of 'np' (line 64)
    np_652594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'np', False)
    # Obtaining the member 'sum' of a type (line 64)
    sum_652595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), np_652594, 'sum')
    # Calling sum(args, kwargs) (line 64)
    sum_call_result_652602 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), sum_652595, *[result_pow_652600], **kwargs_652601)
    
    # Getting the type of 'intervall' (line 64)
    intervall_652603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 45), 'intervall', False)
    int_652604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 56), 'int')
    # Applying the binary operator '**' (line 64)
    result_pow_652605 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 45), '**', intervall_652603, int_652604)
    
    # Applying the binary operator '*' (line 64)
    result_mul_652606 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '*', sum_call_result_652602, result_pow_652605)
    
    float_652607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 61), 'float')
    # Applying the binary operator '<' (line 64)
    result_lt_652608 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '<', result_mul_652606, float_652607)
    
    # Processing the call keyword arguments (line 64)
    kwargs_652609 = {}
    # Getting the type of 'assert_' (line 64)
    assert__652593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 64)
    assert__call_result_652610 = invoke(stypy.reporting.localization.Localization(__file__, 64, 4), assert__652593, *[result_lt_652608], **kwargs_652609)
    
    
    # Assigning a Num to a Name (line 66):
    
    # Assigning a Num to a Name (line 66):
    float_652611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'float')
    # Assigning a type to the variable 'small' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'small', float_652611)
    
    # Assigning a Num to a Name (line 67):
    
    # Assigning a Num to a Name (line 67):
    float_652612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 12), 'float')
    # Assigning a type to the variable 'large' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'large', float_652612)
    
    # Assigning a Call to a Name (line 68):
    
    # Assigning a Call to a Name (line 68):
    
    # Call to integrate_box(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_652615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'small' (line 68)
    small_652616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'small', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 31), list_652615, small_652616)
    # Adding element type (line 68)
    
    # Obtaining the type of the subscript
    int_652617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 44), 'int')
    # Getting the type of 'mean' (line 68)
    mean_652618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'mean', False)
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___652619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 39), mean_652618, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_652620 = invoke(stypy.reporting.localization.Localization(__file__, 68, 39), getitem___652619, int_652617)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 31), list_652615, subscript_call_result_652620)
    
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_652621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'large' (line 68)
    large_652622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 50), 'large', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 49), list_652621, large_652622)
    # Adding element type (line 68)
    # Getting the type of 'large' (line 68)
    large_652623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 57), 'large', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 49), list_652621, large_652623)
    
    # Processing the call keyword arguments (line 68)
    kwargs_652624 = {}
    # Getting the type of 'gkde' (line 68)
    gkde_652613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'gkde', False)
    # Obtaining the member 'integrate_box' of a type (line 68)
    integrate_box_652614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), gkde_652613, 'integrate_box')
    # Calling integrate_box(args, kwargs) (line 68)
    integrate_box_call_result_652625 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), integrate_box_652614, *[list_652615, list_652621], **kwargs_652624)
    
    # Assigning a type to the variable 'prob1' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'prob1', integrate_box_call_result_652625)
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to integrate_box(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_652628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'small' (line 69)
    small_652629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 32), 'small', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_652628, small_652629)
    # Adding element type (line 69)
    # Getting the type of 'small' (line 69)
    small_652630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 39), 'small', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_652628, small_652630)
    
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_652631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'large' (line 69)
    large_652632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 48), 'large', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 47), list_652631, large_652632)
    # Adding element type (line 69)
    
    # Obtaining the type of the subscript
    int_652633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 60), 'int')
    # Getting the type of 'mean' (line 69)
    mean_652634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 55), 'mean', False)
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___652635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 55), mean_652634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_652636 = invoke(stypy.reporting.localization.Localization(__file__, 69, 55), getitem___652635, int_652633)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 47), list_652631, subscript_call_result_652636)
    
    # Processing the call keyword arguments (line 69)
    kwargs_652637 = {}
    # Getting the type of 'gkde' (line 69)
    gkde_652626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'gkde', False)
    # Obtaining the member 'integrate_box' of a type (line 69)
    integrate_box_652627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), gkde_652626, 'integrate_box')
    # Calling integrate_box(args, kwargs) (line 69)
    integrate_box_call_result_652638 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), integrate_box_652627, *[list_652628, list_652631], **kwargs_652637)
    
    # Assigning a type to the variable 'prob2' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'prob2', integrate_box_call_result_652638)
    
    # Call to assert_almost_equal(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'prob1' (line 71)
    prob1_652640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'prob1', False)
    float_652641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 31), 'float')
    # Processing the call keyword arguments (line 71)
    int_652642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 44), 'int')
    keyword_652643 = int_652642
    kwargs_652644 = {'decimal': keyword_652643}
    # Getting the type of 'assert_almost_equal' (line 71)
    assert_almost_equal_652639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 71)
    assert_almost_equal_call_result_652645 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), assert_almost_equal_652639, *[prob1_652640, float_652641], **kwargs_652644)
    
    
    # Call to assert_almost_equal(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'prob2' (line 72)
    prob2_652647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'prob2', False)
    float_652648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'float')
    # Processing the call keyword arguments (line 72)
    int_652649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 44), 'int')
    keyword_652650 = int_652649
    kwargs_652651 = {'decimal': keyword_652650}
    # Getting the type of 'assert_almost_equal' (line 72)
    assert_almost_equal_652646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 72)
    assert_almost_equal_call_result_652652 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), assert_almost_equal_652646, *[prob2_652647, float_652648], **kwargs_652651)
    
    
    # Call to assert_almost_equal(...): (line 73)
    # Processing the call arguments (line 73)
    
    # Call to integrate_kde(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'gkde' (line 73)
    gkde_652656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 43), 'gkde', False)
    # Processing the call keyword arguments (line 73)
    kwargs_652657 = {}
    # Getting the type of 'gkde' (line 73)
    gkde_652654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'gkde', False)
    # Obtaining the member 'integrate_kde' of a type (line 73)
    integrate_kde_652655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 24), gkde_652654, 'integrate_kde')
    # Calling integrate_kde(args, kwargs) (line 73)
    integrate_kde_call_result_652658 = invoke(stypy.reporting.localization.Localization(__file__, 73, 24), integrate_kde_652655, *[gkde_652656], **kwargs_652657)
    
    
    # Call to sum(...): (line 74)
    # Processing the call keyword arguments (line 74)
    kwargs_652663 = {}
    # Getting the type of 'kdepdf' (line 74)
    kdepdf_652659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'kdepdf', False)
    int_652660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'int')
    # Applying the binary operator '**' (line 74)
    result_pow_652661 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 25), '**', kdepdf_652659, int_652660)
    
    # Obtaining the member 'sum' of a type (line 74)
    sum_652662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 25), result_pow_652661, 'sum')
    # Calling sum(args, kwargs) (line 74)
    sum_call_result_652664 = invoke(stypy.reporting.localization.Localization(__file__, 74, 25), sum_652662, *[], **kwargs_652663)
    
    # Getting the type of 'intervall' (line 74)
    intervall_652665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 43), 'intervall', False)
    int_652666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 54), 'int')
    # Applying the binary operator '**' (line 74)
    result_pow_652667 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 43), '**', intervall_652665, int_652666)
    
    # Applying the binary operator '*' (line 74)
    result_mul_652668 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 24), '*', sum_call_result_652664, result_pow_652667)
    
    # Processing the call keyword arguments (line 73)
    int_652669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 66), 'int')
    keyword_652670 = int_652669
    kwargs_652671 = {'decimal': keyword_652670}
    # Getting the type of 'assert_almost_equal' (line 73)
    assert_almost_equal_652653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 73)
    assert_almost_equal_call_result_652672 = invoke(stypy.reporting.localization.Localization(__file__, 73, 4), assert_almost_equal_652653, *[integrate_kde_call_result_652658, result_mul_652668], **kwargs_652671)
    
    
    # Call to assert_almost_equal(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Call to integrate_gaussian(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'mean' (line 75)
    mean_652676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 48), 'mean', False)
    # Getting the type of 'covariance' (line 75)
    covariance_652677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 54), 'covariance', False)
    # Processing the call keyword arguments (line 75)
    kwargs_652678 = {}
    # Getting the type of 'gkde' (line 75)
    gkde_652674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'gkde', False)
    # Obtaining the member 'integrate_gaussian' of a type (line 75)
    integrate_gaussian_652675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 24), gkde_652674, 'integrate_gaussian')
    # Calling integrate_gaussian(args, kwargs) (line 75)
    integrate_gaussian_call_result_652679 = invoke(stypy.reporting.localization.Localization(__file__, 75, 24), integrate_gaussian_652675, *[mean_652676, covariance_652677], **kwargs_652678)
    
    
    # Call to sum(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_652684 = {}
    # Getting the type of 'kdepdf' (line 76)
    kdepdf_652680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'kdepdf', False)
    # Getting the type of 'normpdf' (line 76)
    normpdf_652681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'normpdf', False)
    # Applying the binary operator '*' (line 76)
    result_mul_652682 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 25), '*', kdepdf_652680, normpdf_652681)
    
    # Obtaining the member 'sum' of a type (line 76)
    sum_652683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), result_mul_652682, 'sum')
    # Calling sum(args, kwargs) (line 76)
    sum_call_result_652685 = invoke(stypy.reporting.localization.Localization(__file__, 76, 25), sum_652683, *[], **kwargs_652684)
    
    # Getting the type of 'intervall' (line 76)
    intervall_652686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'intervall', False)
    int_652687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 59), 'int')
    # Applying the binary operator '**' (line 76)
    result_pow_652688 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 48), '**', intervall_652686, int_652687)
    
    # Applying the binary operator '*' (line 76)
    result_mul_652689 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 24), '*', sum_call_result_652685, result_pow_652688)
    
    # Processing the call keyword arguments (line 75)
    int_652690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 71), 'int')
    keyword_652691 = int_652690
    kwargs_652692 = {'decimal': keyword_652691}
    # Getting the type of 'assert_almost_equal' (line 75)
    assert_almost_equal_652673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 75)
    assert_almost_equal_call_result_652693 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), assert_almost_equal_652673, *[integrate_gaussian_call_result_652679, result_mul_652689], **kwargs_652692)
    
    
    # ################# End of 'test_kde_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_kde_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_652694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652694)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_kde_2d'
    return stypy_return_type_652694

# Assigning a type to the variable 'test_kde_2d' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'test_kde_2d', test_kde_2d)

@norecursion
def test_kde_bandwidth_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_kde_bandwidth_method'
    module_type_store = module_type_store.open_function_context('test_kde_bandwidth_method', 79, 0, False)
    
    # Passed parameters checking function
    test_kde_bandwidth_method.stypy_localization = localization
    test_kde_bandwidth_method.stypy_type_of_self = None
    test_kde_bandwidth_method.stypy_type_store = module_type_store
    test_kde_bandwidth_method.stypy_function_name = 'test_kde_bandwidth_method'
    test_kde_bandwidth_method.stypy_param_names_list = []
    test_kde_bandwidth_method.stypy_varargs_param_name = None
    test_kde_bandwidth_method.stypy_kwargs_param_name = None
    test_kde_bandwidth_method.stypy_call_defaults = defaults
    test_kde_bandwidth_method.stypy_call_varargs = varargs
    test_kde_bandwidth_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_kde_bandwidth_method', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_kde_bandwidth_method', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_kde_bandwidth_method(...)' code ##################


    @norecursion
    def scotts_factor(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'scotts_factor'
        module_type_store = module_type_store.open_function_context('scotts_factor', 80, 4, False)
        
        # Passed parameters checking function
        scotts_factor.stypy_localization = localization
        scotts_factor.stypy_type_of_self = None
        scotts_factor.stypy_type_store = module_type_store
        scotts_factor.stypy_function_name = 'scotts_factor'
        scotts_factor.stypy_param_names_list = ['kde_obj']
        scotts_factor.stypy_varargs_param_name = None
        scotts_factor.stypy_kwargs_param_name = None
        scotts_factor.stypy_call_defaults = defaults
        scotts_factor.stypy_call_varargs = varargs
        scotts_factor.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'scotts_factor', ['kde_obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'scotts_factor', localization, ['kde_obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'scotts_factor(...)' code ##################

        str_652695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 8), 'str', 'Same as default, just check that it works.')
        
        # Call to power(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'kde_obj' (line 82)
        kde_obj_652698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'kde_obj', False)
        # Obtaining the member 'n' of a type (line 82)
        n_652699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 24), kde_obj_652698, 'n')
        float_652700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 35), 'float')
        # Getting the type of 'kde_obj' (line 82)
        kde_obj_652701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'kde_obj', False)
        # Obtaining the member 'd' of a type (line 82)
        d_652702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), kde_obj_652701, 'd')
        int_652703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 50), 'int')
        # Applying the binary operator '+' (line 82)
        result_add_652704 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 40), '+', d_652702, int_652703)
        
        # Applying the binary operator 'div' (line 82)
        result_div_652705 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 35), 'div', float_652700, result_add_652704)
        
        # Processing the call keyword arguments (line 82)
        kwargs_652706 = {}
        # Getting the type of 'np' (line 82)
        np_652696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'np', False)
        # Obtaining the member 'power' of a type (line 82)
        power_652697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), np_652696, 'power')
        # Calling power(args, kwargs) (line 82)
        power_call_result_652707 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), power_652697, *[n_652699, result_div_652705], **kwargs_652706)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', power_call_result_652707)
        
        # ################# End of 'scotts_factor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'scotts_factor' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_652708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_652708)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'scotts_factor'
        return stypy_return_type_652708

    # Assigning a type to the variable 'scotts_factor' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'scotts_factor', scotts_factor)
    
    # Call to seed(...): (line 84)
    # Processing the call arguments (line 84)
    int_652712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 19), 'int')
    # Processing the call keyword arguments (line 84)
    kwargs_652713 = {}
    # Getting the type of 'np' (line 84)
    np_652709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 84)
    random_652710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 4), np_652709, 'random')
    # Obtaining the member 'seed' of a type (line 84)
    seed_652711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 4), random_652710, 'seed')
    # Calling seed(args, kwargs) (line 84)
    seed_call_result_652714 = invoke(stypy.reporting.localization.Localization(__file__, 84, 4), seed_652711, *[int_652712], **kwargs_652713)
    
    
    # Assigning a Num to a Name (line 85):
    
    # Assigning a Num to a Name (line 85):
    int_652715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 19), 'int')
    # Assigning a type to the variable 'n_basesample' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'n_basesample', int_652715)
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to randn(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'n_basesample' (line 86)
    n_basesample_652719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'n_basesample', False)
    # Processing the call keyword arguments (line 86)
    kwargs_652720 = {}
    # Getting the type of 'np' (line 86)
    np_652716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 86)
    random_652717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 9), np_652716, 'random')
    # Obtaining the member 'randn' of a type (line 86)
    randn_652718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 9), random_652717, 'randn')
    # Calling randn(args, kwargs) (line 86)
    randn_call_result_652721 = invoke(stypy.reporting.localization.Localization(__file__, 86, 9), randn_652718, *[n_basesample_652719], **kwargs_652720)
    
    # Assigning a type to the variable 'xn' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'xn', randn_call_result_652721)
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to gaussian_kde(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'xn' (line 89)
    xn_652724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'xn', False)
    # Processing the call keyword arguments (line 89)
    kwargs_652725 = {}
    # Getting the type of 'stats' (line 89)
    stats_652722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 89)
    gaussian_kde_652723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 11), stats_652722, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 89)
    gaussian_kde_call_result_652726 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), gaussian_kde_652723, *[xn_652724], **kwargs_652725)
    
    # Assigning a type to the variable 'gkde' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'gkde', gaussian_kde_call_result_652726)
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to gaussian_kde(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'xn' (line 91)
    xn_652729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'xn', False)
    # Processing the call keyword arguments (line 91)
    # Getting the type of 'scotts_factor' (line 91)
    scotts_factor_652730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 45), 'scotts_factor', False)
    keyword_652731 = scotts_factor_652730
    kwargs_652732 = {'bw_method': keyword_652731}
    # Getting the type of 'stats' (line 91)
    stats_652727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 91)
    gaussian_kde_652728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), stats_652727, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 91)
    gaussian_kde_call_result_652733 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), gaussian_kde_652728, *[xn_652729], **kwargs_652732)
    
    # Assigning a type to the variable 'gkde2' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'gkde2', gaussian_kde_call_result_652733)
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to gaussian_kde(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'xn' (line 93)
    xn_652736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'xn', False)
    # Processing the call keyword arguments (line 93)
    # Getting the type of 'gkde' (line 93)
    gkde_652737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 45), 'gkde', False)
    # Obtaining the member 'factor' of a type (line 93)
    factor_652738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 45), gkde_652737, 'factor')
    keyword_652739 = factor_652738
    kwargs_652740 = {'bw_method': keyword_652739}
    # Getting the type of 'stats' (line 93)
    stats_652734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 93)
    gaussian_kde_652735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), stats_652734, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 93)
    gaussian_kde_call_result_652741 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), gaussian_kde_652735, *[xn_652736], **kwargs_652740)
    
    # Assigning a type to the variable 'gkde3' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'gkde3', gaussian_kde_call_result_652741)
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to linspace(...): (line 95)
    # Processing the call arguments (line 95)
    int_652744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'int')
    int_652745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 24), 'int')
    int_652746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'int')
    # Processing the call keyword arguments (line 95)
    kwargs_652747 = {}
    # Getting the type of 'np' (line 95)
    np_652742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 9), 'np', False)
    # Obtaining the member 'linspace' of a type (line 95)
    linspace_652743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 9), np_652742, 'linspace')
    # Calling linspace(args, kwargs) (line 95)
    linspace_call_result_652748 = invoke(stypy.reporting.localization.Localization(__file__, 95, 9), linspace_652743, *[int_652744, int_652745, int_652746], **kwargs_652747)
    
    # Assigning a type to the variable 'xs' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'xs', linspace_call_result_652748)
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to evaluate(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'xs' (line 96)
    xs_652751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'xs', False)
    # Processing the call keyword arguments (line 96)
    kwargs_652752 = {}
    # Getting the type of 'gkde' (line 96)
    gkde_652749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'gkde', False)
    # Obtaining the member 'evaluate' of a type (line 96)
    evaluate_652750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), gkde_652749, 'evaluate')
    # Calling evaluate(args, kwargs) (line 96)
    evaluate_call_result_652753 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), evaluate_652750, *[xs_652751], **kwargs_652752)
    
    # Assigning a type to the variable 'kdepdf' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'kdepdf', evaluate_call_result_652753)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to evaluate(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'xs' (line 97)
    xs_652756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'xs', False)
    # Processing the call keyword arguments (line 97)
    kwargs_652757 = {}
    # Getting the type of 'gkde2' (line 97)
    gkde2_652754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'gkde2', False)
    # Obtaining the member 'evaluate' of a type (line 97)
    evaluate_652755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 14), gkde2_652754, 'evaluate')
    # Calling evaluate(args, kwargs) (line 97)
    evaluate_call_result_652758 = invoke(stypy.reporting.localization.Localization(__file__, 97, 14), evaluate_652755, *[xs_652756], **kwargs_652757)
    
    # Assigning a type to the variable 'kdepdf2' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'kdepdf2', evaluate_call_result_652758)
    
    # Call to assert_almost_equal(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'kdepdf' (line 98)
    kdepdf_652760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'kdepdf', False)
    # Getting the type of 'kdepdf2' (line 98)
    kdepdf2_652761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 32), 'kdepdf2', False)
    # Processing the call keyword arguments (line 98)
    kwargs_652762 = {}
    # Getting the type of 'assert_almost_equal' (line 98)
    assert_almost_equal_652759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 98)
    assert_almost_equal_call_result_652763 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), assert_almost_equal_652759, *[kdepdf_652760, kdepdf2_652761], **kwargs_652762)
    
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to evaluate(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'xs' (line 99)
    xs_652766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'xs', False)
    # Processing the call keyword arguments (line 99)
    kwargs_652767 = {}
    # Getting the type of 'gkde3' (line 99)
    gkde3_652764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 14), 'gkde3', False)
    # Obtaining the member 'evaluate' of a type (line 99)
    evaluate_652765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 14), gkde3_652764, 'evaluate')
    # Calling evaluate(args, kwargs) (line 99)
    evaluate_call_result_652768 = invoke(stypy.reporting.localization.Localization(__file__, 99, 14), evaluate_652765, *[xs_652766], **kwargs_652767)
    
    # Assigning a type to the variable 'kdepdf3' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'kdepdf3', evaluate_call_result_652768)
    
    # Call to assert_almost_equal(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'kdepdf' (line 100)
    kdepdf_652770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'kdepdf', False)
    # Getting the type of 'kdepdf3' (line 100)
    kdepdf3_652771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'kdepdf3', False)
    # Processing the call keyword arguments (line 100)
    kwargs_652772 = {}
    # Getting the type of 'assert_almost_equal' (line 100)
    assert_almost_equal_652769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 100)
    assert_almost_equal_call_result_652773 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), assert_almost_equal_652769, *[kdepdf_652770, kdepdf3_652771], **kwargs_652772)
    
    
    # Call to assert_raises(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'ValueError' (line 102)
    ValueError_652775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 18), 'ValueError', False)
    # Getting the type of 'stats' (line 102)
    stats_652776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 102)
    gaussian_kde_652777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 30), stats_652776, 'gaussian_kde')
    # Getting the type of 'xn' (line 102)
    xn_652778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 50), 'xn', False)
    # Processing the call keyword arguments (line 102)
    str_652779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 64), 'str', 'wrongstring')
    keyword_652780 = str_652779
    kwargs_652781 = {'bw_method': keyword_652780}
    # Getting the type of 'assert_raises' (line 102)
    assert_raises_652774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 102)
    assert_raises_call_result_652782 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), assert_raises_652774, *[ValueError_652775, gaussian_kde_652777, xn_652778], **kwargs_652781)
    
    
    # ################# End of 'test_kde_bandwidth_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_kde_bandwidth_method' in the type store
    # Getting the type of 'stypy_return_type' (line 79)
    stypy_return_type_652783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652783)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_kde_bandwidth_method'
    return stypy_return_type_652783

# Assigning a type to the variable 'test_kde_bandwidth_method' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'test_kde_bandwidth_method', test_kde_bandwidth_method)
# Declaration of the '_kde_subclass1' class
# Getting the type of 'stats' (line 109)
stats_652784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'stats')
# Obtaining the member 'gaussian_kde' of a type (line 109)
gaussian_kde_652785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 21), stats_652784, 'gaussian_kde')

class _kde_subclass1(gaussian_kde_652785, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 110, 4, False)
        # Assigning a type to the variable 'self' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_kde_subclass1.__init__', ['dataset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dataset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 111):
        
        # Assigning a Call to a Attribute (line 111):
        
        # Call to atleast_2d(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'dataset' (line 111)
        dataset_652788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 37), 'dataset', False)
        # Processing the call keyword arguments (line 111)
        kwargs_652789 = {}
        # Getting the type of 'np' (line 111)
        np_652786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 111)
        atleast_2d_652787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 23), np_652786, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 111)
        atleast_2d_call_result_652790 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), atleast_2d_652787, *[dataset_652788], **kwargs_652789)
        
        # Getting the type of 'self' (line 111)
        self_652791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member 'dataset' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_652791, 'dataset', atleast_2d_call_result_652790)
        
        # Assigning a Attribute to a Tuple (line 112):
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_652792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        # Getting the type of 'self' (line 112)
        self_652793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'self')
        # Obtaining the member 'dataset' of a type (line 112)
        dataset_652794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), self_652793, 'dataset')
        # Obtaining the member 'shape' of a type (line 112)
        shape_652795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), dataset_652794, 'shape')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___652796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), shape_652795, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_652797 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), getitem___652796, int_652792)
        
        # Assigning a type to the variable 'tuple_var_assignment_652285' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_652285', subscript_call_result_652797)
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_652798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        # Getting the type of 'self' (line 112)
        self_652799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'self')
        # Obtaining the member 'dataset' of a type (line 112)
        dataset_652800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), self_652799, 'dataset')
        # Obtaining the member 'shape' of a type (line 112)
        shape_652801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), dataset_652800, 'shape')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___652802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), shape_652801, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_652803 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), getitem___652802, int_652798)
        
        # Assigning a type to the variable 'tuple_var_assignment_652286' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_652286', subscript_call_result_652803)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'tuple_var_assignment_652285' (line 112)
        tuple_var_assignment_652285_652804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_652285')
        # Getting the type of 'self' (line 112)
        self_652805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member 'd' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_652805, 'd', tuple_var_assignment_652285_652804)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'tuple_var_assignment_652286' (line 112)
        tuple_var_assignment_652286_652806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'tuple_var_assignment_652286')
        # Getting the type of 'self' (line 112)
        self_652807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'self')
        # Setting the type of the member 'n' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), self_652807, 'n', tuple_var_assignment_652286_652806)
        
        # Assigning a Attribute to a Attribute (line 113):
        
        # Assigning a Attribute to a Attribute (line 113):
        # Getting the type of 'self' (line 113)
        self_652808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'self')
        # Obtaining the member 'scotts_factor' of a type (line 113)
        scotts_factor_652809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 33), self_652808, 'scotts_factor')
        # Getting the type of 'self' (line 113)
        self_652810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'covariance_factor' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_652810, 'covariance_factor', scotts_factor_652809)
        
        # Call to _compute_covariance(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_652813 = {}
        # Getting the type of 'self' (line 114)
        self_652811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self', False)
        # Obtaining the member '_compute_covariance' of a type (line 114)
        _compute_covariance_652812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_652811, '_compute_covariance')
        # Calling _compute_covariance(args, kwargs) (line 114)
        _compute_covariance_call_result_652814 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), _compute_covariance_652812, *[], **kwargs_652813)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable '_kde_subclass1' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), '_kde_subclass1', _kde_subclass1)
# Declaration of the '_kde_subclass2' class
# Getting the type of 'stats' (line 117)
stats_652815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'stats')
# Obtaining the member 'gaussian_kde' of a type (line 117)
gaussian_kde_652816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 21), stats_652815, 'gaussian_kde')

class _kde_subclass2(gaussian_kde_652816, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_kde_subclass2.__init__', ['dataset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dataset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 119):
        
        # Assigning a Attribute to a Attribute (line 119):
        # Getting the type of 'self' (line 119)
        self_652817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'self')
        # Obtaining the member 'scotts_factor' of a type (line 119)
        scotts_factor_652818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 33), self_652817, 'scotts_factor')
        # Getting the type of 'self' (line 119)
        self_652819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'covariance_factor' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_652819, 'covariance_factor', scotts_factor_652818)
        
        # Call to __init__(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'dataset' (line 120)
        dataset_652826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 45), 'dataset', False)
        # Processing the call keyword arguments (line 120)
        kwargs_652827 = {}
        
        # Call to super(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of '_kde_subclass2' (line 120)
        _kde_subclass2_652821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), '_kde_subclass2', False)
        # Getting the type of 'self' (line 120)
        self_652822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'self', False)
        # Processing the call keyword arguments (line 120)
        kwargs_652823 = {}
        # Getting the type of 'super' (line 120)
        super_652820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'super', False)
        # Calling super(args, kwargs) (line 120)
        super_call_result_652824 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), super_652820, *[_kde_subclass2_652821, self_652822], **kwargs_652823)
        
        # Obtaining the member '__init__' of a type (line 120)
        init___652825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), super_call_result_652824, '__init__')
        # Calling __init__(args, kwargs) (line 120)
        init___call_result_652828 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), init___652825, *[dataset_652826], **kwargs_652827)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable '_kde_subclass2' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), '_kde_subclass2', _kde_subclass2)
# Declaration of the '_kde_subclass3' class
# Getting the type of 'stats' (line 123)
stats_652829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 21), 'stats')
# Obtaining the member 'gaussian_kde' of a type (line 123)
gaussian_kde_652830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 21), stats_652829, 'gaussian_kde')

class _kde_subclass3(gaussian_kde_652830, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_kde_subclass3.__init__', ['dataset', 'covariance'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['dataset', 'covariance'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 125):
        
        # Assigning a Name to a Attribute (line 125):
        # Getting the type of 'covariance' (line 125)
        covariance_652831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'covariance')
        # Getting the type of 'self' (line 125)
        self_652832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self')
        # Setting the type of the member 'covariance' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_652832, 'covariance', covariance_652831)
        
        # Call to __init__(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'self' (line 126)
        self_652836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'self', False)
        # Getting the type of 'dataset' (line 126)
        dataset_652837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 42), 'dataset', False)
        # Processing the call keyword arguments (line 126)
        kwargs_652838 = {}
        # Getting the type of 'stats' (line 126)
        stats_652833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stats', False)
        # Obtaining the member 'gaussian_kde' of a type (line 126)
        gaussian_kde_652834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), stats_652833, 'gaussian_kde')
        # Obtaining the member '__init__' of a type (line 126)
        init___652835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), gaussian_kde_652834, '__init__')
        # Calling __init__(args, kwargs) (line 126)
        init___call_result_652839 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), init___652835, *[self_652836, dataset_652837], **kwargs_652838)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _compute_covariance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compute_covariance'
        module_type_store = module_type_store.open_function_context('_compute_covariance', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_localization', localization)
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_type_store', module_type_store)
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_function_name', '_kde_subclass3._compute_covariance')
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_param_names_list', [])
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_varargs_param_name', None)
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_call_defaults', defaults)
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_call_varargs', varargs)
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _kde_subclass3._compute_covariance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_kde_subclass3._compute_covariance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compute_covariance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compute_covariance(...)' code ##################

        
        # Assigning a Call to a Attribute (line 129):
        
        # Assigning a Call to a Attribute (line 129):
        
        # Call to inv(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'self' (line 129)
        self_652843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'self', False)
        # Obtaining the member 'covariance' of a type (line 129)
        covariance_652844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 37), self_652843, 'covariance')
        # Processing the call keyword arguments (line 129)
        kwargs_652845 = {}
        # Getting the type of 'np' (line 129)
        np_652840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'np', False)
        # Obtaining the member 'linalg' of a type (line 129)
        linalg_652841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 23), np_652840, 'linalg')
        # Obtaining the member 'inv' of a type (line 129)
        inv_652842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 23), linalg_652841, 'inv')
        # Calling inv(args, kwargs) (line 129)
        inv_call_result_652846 = invoke(stypy.reporting.localization.Localization(__file__, 129, 23), inv_652842, *[covariance_652844], **kwargs_652845)
        
        # Getting the type of 'self' (line 129)
        self_652847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'inv_cov' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_652847, 'inv_cov', inv_call_result_652846)
        
        # Assigning a BinOp to a Attribute (line 130):
        
        # Assigning a BinOp to a Attribute (line 130):
        
        # Call to sqrt(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Call to det(...): (line 130)
        # Processing the call arguments (line 130)
        int_652853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 50), 'int')
        # Getting the type of 'np' (line 130)
        np_652854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 52), 'np', False)
        # Obtaining the member 'pi' of a type (line 130)
        pi_652855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 52), np_652854, 'pi')
        # Applying the binary operator '*' (line 130)
        result_mul_652856 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 50), '*', int_652853, pi_652855)
        
        # Getting the type of 'self' (line 130)
        self_652857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 60), 'self', False)
        # Obtaining the member 'covariance' of a type (line 130)
        covariance_652858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 60), self_652857, 'covariance')
        # Applying the binary operator '*' (line 130)
        result_mul_652859 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 58), '*', result_mul_652856, covariance_652858)
        
        # Processing the call keyword arguments (line 130)
        kwargs_652860 = {}
        # Getting the type of 'np' (line 130)
        np_652850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 36), 'np', False)
        # Obtaining the member 'linalg' of a type (line 130)
        linalg_652851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 36), np_652850, 'linalg')
        # Obtaining the member 'det' of a type (line 130)
        det_652852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 36), linalg_652851, 'det')
        # Calling det(args, kwargs) (line 130)
        det_call_result_652861 = invoke(stypy.reporting.localization.Localization(__file__, 130, 36), det_652852, *[result_mul_652859], **kwargs_652860)
        
        # Processing the call keyword arguments (line 130)
        kwargs_652862 = {}
        # Getting the type of 'np' (line 130)
        np_652848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 130)
        sqrt_652849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 28), np_652848, 'sqrt')
        # Calling sqrt(args, kwargs) (line 130)
        sqrt_call_result_652863 = invoke(stypy.reporting.localization.Localization(__file__, 130, 28), sqrt_652849, *[det_call_result_652861], **kwargs_652862)
        
        # Getting the type of 'self' (line 131)
        self_652864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'self')
        # Obtaining the member 'n' of a type (line 131)
        n_652865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 37), self_652864, 'n')
        # Applying the binary operator '*' (line 130)
        result_mul_652866 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 28), '*', sqrt_call_result_652863, n_652865)
        
        # Getting the type of 'self' (line 130)
        self_652867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member '_norm_factor' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_652867, '_norm_factor', result_mul_652866)
        
        # ################# End of '_compute_covariance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compute_covariance' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_652868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_652868)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compute_covariance'
        return stypy_return_type_652868


# Assigning a type to the variable '_kde_subclass3' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), '_kde_subclass3', _kde_subclass3)
# Declaration of the '_kde_subclass4' class
# Getting the type of 'stats' (line 134)
stats_652869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 21), 'stats')
# Obtaining the member 'gaussian_kde' of a type (line 134)
gaussian_kde_652870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 21), stats_652869, 'gaussian_kde')

class _kde_subclass4(gaussian_kde_652870, ):

    @norecursion
    def covariance_factor(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'covariance_factor'
        module_type_store = module_type_store.open_function_context('covariance_factor', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_localization', localization)
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_type_store', module_type_store)
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_function_name', '_kde_subclass4.covariance_factor')
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_param_names_list', [])
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_varargs_param_name', None)
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_call_defaults', defaults)
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_call_varargs', varargs)
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _kde_subclass4.covariance_factor.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_kde_subclass4.covariance_factor', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'covariance_factor', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'covariance_factor(...)' code ##################

        float_652871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 15), 'float')
        
        # Call to silverman_factor(...): (line 136)
        # Processing the call keyword arguments (line 136)
        kwargs_652874 = {}
        # Getting the type of 'self' (line 136)
        self_652872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'self', False)
        # Obtaining the member 'silverman_factor' of a type (line 136)
        silverman_factor_652873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 21), self_652872, 'silverman_factor')
        # Calling silverman_factor(args, kwargs) (line 136)
        silverman_factor_call_result_652875 = invoke(stypy.reporting.localization.Localization(__file__, 136, 21), silverman_factor_652873, *[], **kwargs_652874)
        
        # Applying the binary operator '*' (line 136)
        result_mul_652876 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 15), '*', float_652871, silverman_factor_call_result_652875)
        
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type', result_mul_652876)
        
        # ################# End of 'covariance_factor(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'covariance_factor' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_652877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_652877)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'covariance_factor'
        return stypy_return_type_652877


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 134, 0, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_kde_subclass4.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_kde_subclass4' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), '_kde_subclass4', _kde_subclass4)

@norecursion
def test_gaussian_kde_subclassing(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gaussian_kde_subclassing'
    module_type_store = module_type_store.open_function_context('test_gaussian_kde_subclassing', 139, 0, False)
    
    # Passed parameters checking function
    test_gaussian_kde_subclassing.stypy_localization = localization
    test_gaussian_kde_subclassing.stypy_type_of_self = None
    test_gaussian_kde_subclassing.stypy_type_store = module_type_store
    test_gaussian_kde_subclassing.stypy_function_name = 'test_gaussian_kde_subclassing'
    test_gaussian_kde_subclassing.stypy_param_names_list = []
    test_gaussian_kde_subclassing.stypy_varargs_param_name = None
    test_gaussian_kde_subclassing.stypy_kwargs_param_name = None
    test_gaussian_kde_subclassing.stypy_call_defaults = defaults
    test_gaussian_kde_subclassing.stypy_call_varargs = varargs
    test_gaussian_kde_subclassing.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gaussian_kde_subclassing', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gaussian_kde_subclassing', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gaussian_kde_subclassing(...)' code ##################

    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to array(...): (line 140)
    # Processing the call arguments (line 140)
    
    # Obtaining an instance of the builtin type 'list' (line 140)
    list_652880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 140)
    # Adding element type (line 140)
    int_652881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 18), list_652880, int_652881)
    # Adding element type (line 140)
    int_652882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 18), list_652880, int_652882)
    # Adding element type (line 140)
    int_652883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 18), list_652880, int_652883)
    # Adding element type (line 140)
    int_652884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 18), list_652880, int_652884)
    # Adding element type (line 140)
    int_652885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 18), list_652880, int_652885)
    
    # Processing the call keyword arguments (line 140)
    # Getting the type of 'float' (line 140)
    float_652886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'float', False)
    keyword_652887 = float_652886
    kwargs_652888 = {'dtype': keyword_652887}
    # Getting the type of 'np' (line 140)
    np_652878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 140)
    array_652879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 9), np_652878, 'array')
    # Calling array(args, kwargs) (line 140)
    array_call_result_652889 = invoke(stypy.reporting.localization.Localization(__file__, 140, 9), array_652879, *[list_652880], **kwargs_652888)
    
    # Assigning a type to the variable 'x1' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'x1', array_call_result_652889)
    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 141):
    
    # Call to linspace(...): (line 141)
    # Processing the call arguments (line 141)
    int_652892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 21), 'int')
    int_652893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 26), 'int')
    # Processing the call keyword arguments (line 141)
    int_652894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 34), 'int')
    keyword_652895 = int_652894
    kwargs_652896 = {'num': keyword_652895}
    # Getting the type of 'np' (line 141)
    np_652890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'np', False)
    # Obtaining the member 'linspace' of a type (line 141)
    linspace_652891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 9), np_652890, 'linspace')
    # Calling linspace(args, kwargs) (line 141)
    linspace_call_result_652897 = invoke(stypy.reporting.localization.Localization(__file__, 141, 9), linspace_652891, *[int_652892, int_652893], **kwargs_652896)
    
    # Assigning a type to the variable 'xs' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'xs', linspace_call_result_652897)
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to gaussian_kde(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'x1' (line 144)
    x1_652900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'x1', False)
    # Processing the call keyword arguments (line 144)
    kwargs_652901 = {}
    # Getting the type of 'stats' (line 144)
    stats_652898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 10), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 144)
    gaussian_kde_652899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 10), stats_652898, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 144)
    gaussian_kde_call_result_652902 = invoke(stypy.reporting.localization.Localization(__file__, 144, 10), gaussian_kde_652899, *[x1_652900], **kwargs_652901)
    
    # Assigning a type to the variable 'kde' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'kde', gaussian_kde_call_result_652902)
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to kde(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'xs' (line 145)
    xs_652904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 13), 'xs', False)
    # Processing the call keyword arguments (line 145)
    kwargs_652905 = {}
    # Getting the type of 'kde' (line 145)
    kde_652903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 9), 'kde', False)
    # Calling kde(args, kwargs) (line 145)
    kde_call_result_652906 = invoke(stypy.reporting.localization.Localization(__file__, 145, 9), kde_652903, *[xs_652904], **kwargs_652905)
    
    # Assigning a type to the variable 'ys' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'ys', kde_call_result_652906)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to _kde_subclass1(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'x1' (line 148)
    x1_652908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'x1', False)
    # Processing the call keyword arguments (line 148)
    kwargs_652909 = {}
    # Getting the type of '_kde_subclass1' (line 148)
    _kde_subclass1_652907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), '_kde_subclass1', False)
    # Calling _kde_subclass1(args, kwargs) (line 148)
    _kde_subclass1_call_result_652910 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), _kde_subclass1_652907, *[x1_652908], **kwargs_652909)
    
    # Assigning a type to the variable 'kde1' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'kde1', _kde_subclass1_call_result_652910)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to kde1(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'xs' (line 149)
    xs_652912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'xs', False)
    # Processing the call keyword arguments (line 149)
    kwargs_652913 = {}
    # Getting the type of 'kde1' (line 149)
    kde1_652911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 9), 'kde1', False)
    # Calling kde1(args, kwargs) (line 149)
    kde1_call_result_652914 = invoke(stypy.reporting.localization.Localization(__file__, 149, 9), kde1_652911, *[xs_652912], **kwargs_652913)
    
    # Assigning a type to the variable 'y1' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'y1', kde1_call_result_652914)
    
    # Call to assert_array_almost_equal_nulp(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'ys' (line 150)
    ys_652916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 35), 'ys', False)
    # Getting the type of 'y1' (line 150)
    y1_652917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 39), 'y1', False)
    # Processing the call keyword arguments (line 150)
    int_652918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 48), 'int')
    keyword_652919 = int_652918
    kwargs_652920 = {'nulp': keyword_652919}
    # Getting the type of 'assert_array_almost_equal_nulp' (line 150)
    assert_array_almost_equal_nulp_652915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'assert_array_almost_equal_nulp', False)
    # Calling assert_array_almost_equal_nulp(args, kwargs) (line 150)
    assert_array_almost_equal_nulp_call_result_652921 = invoke(stypy.reporting.localization.Localization(__file__, 150, 4), assert_array_almost_equal_nulp_652915, *[ys_652916, y1_652917], **kwargs_652920)
    
    
    # Assigning a Call to a Name (line 153):
    
    # Assigning a Call to a Name (line 153):
    
    # Call to _kde_subclass2(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'x1' (line 153)
    x1_652923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'x1', False)
    # Processing the call keyword arguments (line 153)
    kwargs_652924 = {}
    # Getting the type of '_kde_subclass2' (line 153)
    _kde_subclass2_652922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), '_kde_subclass2', False)
    # Calling _kde_subclass2(args, kwargs) (line 153)
    _kde_subclass2_call_result_652925 = invoke(stypy.reporting.localization.Localization(__file__, 153, 11), _kde_subclass2_652922, *[x1_652923], **kwargs_652924)
    
    # Assigning a type to the variable 'kde2' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'kde2', _kde_subclass2_call_result_652925)
    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to kde2(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'xs' (line 154)
    xs_652927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 14), 'xs', False)
    # Processing the call keyword arguments (line 154)
    kwargs_652928 = {}
    # Getting the type of 'kde2' (line 154)
    kde2_652926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 9), 'kde2', False)
    # Calling kde2(args, kwargs) (line 154)
    kde2_call_result_652929 = invoke(stypy.reporting.localization.Localization(__file__, 154, 9), kde2_652926, *[xs_652927], **kwargs_652928)
    
    # Assigning a type to the variable 'y2' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'y2', kde2_call_result_652929)
    
    # Call to assert_array_almost_equal_nulp(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'ys' (line 155)
    ys_652931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'ys', False)
    # Getting the type of 'y2' (line 155)
    y2_652932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 39), 'y2', False)
    # Processing the call keyword arguments (line 155)
    int_652933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 48), 'int')
    keyword_652934 = int_652933
    kwargs_652935 = {'nulp': keyword_652934}
    # Getting the type of 'assert_array_almost_equal_nulp' (line 155)
    assert_array_almost_equal_nulp_652930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'assert_array_almost_equal_nulp', False)
    # Calling assert_array_almost_equal_nulp(args, kwargs) (line 155)
    assert_array_almost_equal_nulp_call_result_652936 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), assert_array_almost_equal_nulp_652930, *[ys_652931, y2_652932], **kwargs_652935)
    
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to _kde_subclass3(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'x1' (line 158)
    x1_652938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'x1', False)
    # Getting the type of 'kde' (line 158)
    kde_652939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 30), 'kde', False)
    # Obtaining the member 'covariance' of a type (line 158)
    covariance_652940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 30), kde_652939, 'covariance')
    # Processing the call keyword arguments (line 158)
    kwargs_652941 = {}
    # Getting the type of '_kde_subclass3' (line 158)
    _kde_subclass3_652937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), '_kde_subclass3', False)
    # Calling _kde_subclass3(args, kwargs) (line 158)
    _kde_subclass3_call_result_652942 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), _kde_subclass3_652937, *[x1_652938, covariance_652940], **kwargs_652941)
    
    # Assigning a type to the variable 'kde3' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'kde3', _kde_subclass3_call_result_652942)
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to kde3(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'xs' (line 159)
    xs_652944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'xs', False)
    # Processing the call keyword arguments (line 159)
    kwargs_652945 = {}
    # Getting the type of 'kde3' (line 159)
    kde3_652943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 9), 'kde3', False)
    # Calling kde3(args, kwargs) (line 159)
    kde3_call_result_652946 = invoke(stypy.reporting.localization.Localization(__file__, 159, 9), kde3_652943, *[xs_652944], **kwargs_652945)
    
    # Assigning a type to the variable 'y3' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'y3', kde3_call_result_652946)
    
    # Call to assert_array_almost_equal_nulp(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'ys' (line 160)
    ys_652948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 35), 'ys', False)
    # Getting the type of 'y3' (line 160)
    y3_652949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 39), 'y3', False)
    # Processing the call keyword arguments (line 160)
    int_652950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 48), 'int')
    keyword_652951 = int_652950
    kwargs_652952 = {'nulp': keyword_652951}
    # Getting the type of 'assert_array_almost_equal_nulp' (line 160)
    assert_array_almost_equal_nulp_652947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'assert_array_almost_equal_nulp', False)
    # Calling assert_array_almost_equal_nulp(args, kwargs) (line 160)
    assert_array_almost_equal_nulp_call_result_652953 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), assert_array_almost_equal_nulp_652947, *[ys_652948, y3_652949], **kwargs_652952)
    
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to _kde_subclass4(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'x1' (line 163)
    x1_652955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'x1', False)
    # Processing the call keyword arguments (line 163)
    kwargs_652956 = {}
    # Getting the type of '_kde_subclass4' (line 163)
    _kde_subclass4_652954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), '_kde_subclass4', False)
    # Calling _kde_subclass4(args, kwargs) (line 163)
    _kde_subclass4_call_result_652957 = invoke(stypy.reporting.localization.Localization(__file__, 163, 11), _kde_subclass4_652954, *[x1_652955], **kwargs_652956)
    
    # Assigning a type to the variable 'kde4' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'kde4', _kde_subclass4_call_result_652957)
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to kde4(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'x1' (line 164)
    x1_652959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'x1', False)
    # Processing the call keyword arguments (line 164)
    kwargs_652960 = {}
    # Getting the type of 'kde4' (line 164)
    kde4_652958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 9), 'kde4', False)
    # Calling kde4(args, kwargs) (line 164)
    kde4_call_result_652961 = invoke(stypy.reporting.localization.Localization(__file__, 164, 9), kde4_652958, *[x1_652959], **kwargs_652960)
    
    # Assigning a type to the variable 'y4' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'y4', kde4_call_result_652961)
    
    # Assigning a List to a Name (line 165):
    
    # Assigning a List to a Name (line 165):
    
    # Obtaining an instance of the builtin type 'list' (line 165)
    list_652962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 165)
    # Adding element type (line 165)
    float_652963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 17), list_652962, float_652963)
    # Adding element type (line 165)
    float_652964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 17), list_652962, float_652964)
    # Adding element type (line 165)
    float_652965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 17), list_652962, float_652965)
    # Adding element type (line 165)
    float_652966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 54), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 17), list_652962, float_652966)
    # Adding element type (line 165)
    float_652967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 66), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 17), list_652962, float_652967)
    
    # Assigning a type to the variable 'y_expected' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'y_expected', list_652962)
    
    # Call to assert_array_almost_equal(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'y_expected' (line 167)
    y_expected_652969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'y_expected', False)
    # Getting the type of 'y4' (line 167)
    y4_652970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 42), 'y4', False)
    # Processing the call keyword arguments (line 167)
    int_652971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 54), 'int')
    keyword_652972 = int_652971
    kwargs_652973 = {'decimal': keyword_652972}
    # Getting the type of 'assert_array_almost_equal' (line 167)
    assert_array_almost_equal_652968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 167)
    assert_array_almost_equal_call_result_652974 = invoke(stypy.reporting.localization.Localization(__file__, 167, 4), assert_array_almost_equal_652968, *[y_expected_652969, y4_652970], **kwargs_652973)
    
    
    # Assigning a Name to a Name (line 170):
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'kde' (line 170)
    kde_652975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'kde')
    # Assigning a type to the variable 'kde5' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'kde5', kde_652975)
    
    # Assigning a Lambda to a Attribute (line 171):
    
    # Assigning a Lambda to a Attribute (line 171):

    @norecursion
    def _stypy_temp_lambda_564(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_564'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_564', 171, 29, True)
        # Passed parameters checking function
        _stypy_temp_lambda_564.stypy_localization = localization
        _stypy_temp_lambda_564.stypy_type_of_self = None
        _stypy_temp_lambda_564.stypy_type_store = module_type_store
        _stypy_temp_lambda_564.stypy_function_name = '_stypy_temp_lambda_564'
        _stypy_temp_lambda_564.stypy_param_names_list = []
        _stypy_temp_lambda_564.stypy_varargs_param_name = None
        _stypy_temp_lambda_564.stypy_kwargs_param_name = None
        _stypy_temp_lambda_564.stypy_call_defaults = defaults
        _stypy_temp_lambda_564.stypy_call_varargs = varargs
        _stypy_temp_lambda_564.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_564', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_564', [], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'kde' (line 171)
        kde_652976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 37), 'kde')
        # Obtaining the member 'factor' of a type (line 171)
        factor_652977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 37), kde_652976, 'factor')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'stypy_return_type', factor_652977)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_564' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_652978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_652978)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_564'
        return stypy_return_type_652978

    # Assigning a type to the variable '_stypy_temp_lambda_564' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), '_stypy_temp_lambda_564', _stypy_temp_lambda_564)
    # Getting the type of '_stypy_temp_lambda_564' (line 171)
    _stypy_temp_lambda_564_652979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), '_stypy_temp_lambda_564')
    # Getting the type of 'kde5' (line 171)
    kde5_652980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'kde5')
    # Setting the type of the member 'covariance_factor' of a type (line 171)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 4), kde5_652980, 'covariance_factor', _stypy_temp_lambda_564_652979)
    
    # Call to _compute_covariance(...): (line 172)
    # Processing the call keyword arguments (line 172)
    kwargs_652983 = {}
    # Getting the type of 'kde5' (line 172)
    kde5_652981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'kde5', False)
    # Obtaining the member '_compute_covariance' of a type (line 172)
    _compute_covariance_652982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 4), kde5_652981, '_compute_covariance')
    # Calling _compute_covariance(args, kwargs) (line 172)
    _compute_covariance_call_result_652984 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), _compute_covariance_652982, *[], **kwargs_652983)
    
    
    # Assigning a Call to a Name (line 173):
    
    # Assigning a Call to a Name (line 173):
    
    # Call to kde5(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'xs' (line 173)
    xs_652986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 14), 'xs', False)
    # Processing the call keyword arguments (line 173)
    kwargs_652987 = {}
    # Getting the type of 'kde5' (line 173)
    kde5_652985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 9), 'kde5', False)
    # Calling kde5(args, kwargs) (line 173)
    kde5_call_result_652988 = invoke(stypy.reporting.localization.Localization(__file__, 173, 9), kde5_652985, *[xs_652986], **kwargs_652987)
    
    # Assigning a type to the variable 'y5' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'y5', kde5_call_result_652988)
    
    # Call to assert_array_almost_equal_nulp(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'ys' (line 174)
    ys_652990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 35), 'ys', False)
    # Getting the type of 'y5' (line 174)
    y5_652991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 39), 'y5', False)
    # Processing the call keyword arguments (line 174)
    int_652992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 48), 'int')
    keyword_652993 = int_652992
    kwargs_652994 = {'nulp': keyword_652993}
    # Getting the type of 'assert_array_almost_equal_nulp' (line 174)
    assert_array_almost_equal_nulp_652989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'assert_array_almost_equal_nulp', False)
    # Calling assert_array_almost_equal_nulp(args, kwargs) (line 174)
    assert_array_almost_equal_nulp_call_result_652995 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), assert_array_almost_equal_nulp_652989, *[ys_652990, y5_652991], **kwargs_652994)
    
    
    # ################# End of 'test_gaussian_kde_subclassing(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gaussian_kde_subclassing' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_652996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652996)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gaussian_kde_subclassing'
    return stypy_return_type_652996

# Assigning a type to the variable 'test_gaussian_kde_subclassing' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'test_gaussian_kde_subclassing', test_gaussian_kde_subclassing)

@norecursion
def test_gaussian_kde_covariance_caching(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gaussian_kde_covariance_caching'
    module_type_store = module_type_store.open_function_context('test_gaussian_kde_covariance_caching', 177, 0, False)
    
    # Passed parameters checking function
    test_gaussian_kde_covariance_caching.stypy_localization = localization
    test_gaussian_kde_covariance_caching.stypy_type_of_self = None
    test_gaussian_kde_covariance_caching.stypy_type_store = module_type_store
    test_gaussian_kde_covariance_caching.stypy_function_name = 'test_gaussian_kde_covariance_caching'
    test_gaussian_kde_covariance_caching.stypy_param_names_list = []
    test_gaussian_kde_covariance_caching.stypy_varargs_param_name = None
    test_gaussian_kde_covariance_caching.stypy_kwargs_param_name = None
    test_gaussian_kde_covariance_caching.stypy_call_defaults = defaults
    test_gaussian_kde_covariance_caching.stypy_call_varargs = varargs
    test_gaussian_kde_covariance_caching.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gaussian_kde_covariance_caching', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gaussian_kde_covariance_caching', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gaussian_kde_covariance_caching(...)' code ##################

    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to array(...): (line 178)
    # Processing the call arguments (line 178)
    
    # Obtaining an instance of the builtin type 'list' (line 178)
    list_652999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 178)
    # Adding element type (line 178)
    int_653000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 18), list_652999, int_653000)
    # Adding element type (line 178)
    int_653001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 18), list_652999, int_653001)
    # Adding element type (line 178)
    int_653002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 18), list_652999, int_653002)
    # Adding element type (line 178)
    int_653003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 18), list_652999, int_653003)
    # Adding element type (line 178)
    int_653004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 18), list_652999, int_653004)
    
    # Processing the call keyword arguments (line 178)
    # Getting the type of 'float' (line 178)
    float_653005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 43), 'float', False)
    keyword_653006 = float_653005
    kwargs_653007 = {'dtype': keyword_653006}
    # Getting the type of 'np' (line 178)
    np_652997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 178)
    array_652998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 9), np_652997, 'array')
    # Calling array(args, kwargs) (line 178)
    array_call_result_653008 = invoke(stypy.reporting.localization.Localization(__file__, 178, 9), array_652998, *[list_652999], **kwargs_653007)
    
    # Assigning a type to the variable 'x1' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'x1', array_call_result_653008)
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to linspace(...): (line 179)
    # Processing the call arguments (line 179)
    int_653011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'int')
    int_653012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 26), 'int')
    # Processing the call keyword arguments (line 179)
    int_653013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 34), 'int')
    keyword_653014 = int_653013
    kwargs_653015 = {'num': keyword_653014}
    # Getting the type of 'np' (line 179)
    np_653009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 9), 'np', False)
    # Obtaining the member 'linspace' of a type (line 179)
    linspace_653010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 9), np_653009, 'linspace')
    # Calling linspace(args, kwargs) (line 179)
    linspace_call_result_653016 = invoke(stypy.reporting.localization.Localization(__file__, 179, 9), linspace_653010, *[int_653011, int_653012], **kwargs_653015)
    
    # Assigning a type to the variable 'xs' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'xs', linspace_call_result_653016)
    
    # Assigning a List to a Name (line 182):
    
    # Assigning a List to a Name (line 182):
    
    # Obtaining an instance of the builtin type 'list' (line 182)
    list_653017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 182)
    # Adding element type (line 182)
    float_653018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 17), list_653017, float_653018)
    # Adding element type (line 182)
    float_653019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 17), list_653017, float_653019)
    # Adding element type (line 182)
    float_653020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 17), list_653017, float_653020)
    # Adding element type (line 182)
    float_653021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 54), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 17), list_653017, float_653021)
    # Adding element type (line 182)
    float_653022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 66), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 17), list_653017, float_653022)
    
    # Assigning a type to the variable 'y_expected' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'y_expected', list_653017)
    
    # Assigning a Call to a Name (line 185):
    
    # Assigning a Call to a Name (line 185):
    
    # Call to gaussian_kde(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'x1' (line 185)
    x1_653025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'x1', False)
    # Processing the call keyword arguments (line 185)
    kwargs_653026 = {}
    # Getting the type of 'stats' (line 185)
    stats_653023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 10), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 185)
    gaussian_kde_653024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 10), stats_653023, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 185)
    gaussian_kde_call_result_653027 = invoke(stypy.reporting.localization.Localization(__file__, 185, 10), gaussian_kde_653024, *[x1_653025], **kwargs_653026)
    
    # Assigning a type to the variable 'kde' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'kde', gaussian_kde_call_result_653027)
    
    # Call to set_bandwidth(...): (line 186)
    # Processing the call keyword arguments (line 186)
    float_653030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 32), 'float')
    keyword_653031 = float_653030
    kwargs_653032 = {'bw_method': keyword_653031}
    # Getting the type of 'kde' (line 186)
    kde_653028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'kde', False)
    # Obtaining the member 'set_bandwidth' of a type (line 186)
    set_bandwidth_653029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 4), kde_653028, 'set_bandwidth')
    # Calling set_bandwidth(args, kwargs) (line 186)
    set_bandwidth_call_result_653033 = invoke(stypy.reporting.localization.Localization(__file__, 186, 4), set_bandwidth_653029, *[], **kwargs_653032)
    
    
    # Call to set_bandwidth(...): (line 187)
    # Processing the call keyword arguments (line 187)
    str_653036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 32), 'str', 'scott')
    keyword_653037 = str_653036
    kwargs_653038 = {'bw_method': keyword_653037}
    # Getting the type of 'kde' (line 187)
    kde_653034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'kde', False)
    # Obtaining the member 'set_bandwidth' of a type (line 187)
    set_bandwidth_653035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 4), kde_653034, 'set_bandwidth')
    # Calling set_bandwidth(args, kwargs) (line 187)
    set_bandwidth_call_result_653039 = invoke(stypy.reporting.localization.Localization(__file__, 187, 4), set_bandwidth_653035, *[], **kwargs_653038)
    
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to kde(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'xs' (line 188)
    xs_653041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'xs', False)
    # Processing the call keyword arguments (line 188)
    kwargs_653042 = {}
    # Getting the type of 'kde' (line 188)
    kde_653040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 9), 'kde', False)
    # Calling kde(args, kwargs) (line 188)
    kde_call_result_653043 = invoke(stypy.reporting.localization.Localization(__file__, 188, 9), kde_653040, *[xs_653041], **kwargs_653042)
    
    # Assigning a type to the variable 'y2' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'y2', kde_call_result_653043)
    
    # Call to assert_array_almost_equal(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'y_expected' (line 190)
    y_expected_653045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 30), 'y_expected', False)
    # Getting the type of 'y2' (line 190)
    y2_653046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 42), 'y2', False)
    # Processing the call keyword arguments (line 190)
    int_653047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 54), 'int')
    keyword_653048 = int_653047
    kwargs_653049 = {'decimal': keyword_653048}
    # Getting the type of 'assert_array_almost_equal' (line 190)
    assert_array_almost_equal_653044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 190)
    assert_array_almost_equal_call_result_653050 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), assert_array_almost_equal_653044, *[y_expected_653045, y2_653046], **kwargs_653049)
    
    
    # ################# End of 'test_gaussian_kde_covariance_caching(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gaussian_kde_covariance_caching' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_653051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_653051)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gaussian_kde_covariance_caching'
    return stypy_return_type_653051

# Assigning a type to the variable 'test_gaussian_kde_covariance_caching' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'test_gaussian_kde_covariance_caching', test_gaussian_kde_covariance_caching)

@norecursion
def test_gaussian_kde_monkeypatch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gaussian_kde_monkeypatch'
    module_type_store = module_type_store.open_function_context('test_gaussian_kde_monkeypatch', 193, 0, False)
    
    # Passed parameters checking function
    test_gaussian_kde_monkeypatch.stypy_localization = localization
    test_gaussian_kde_monkeypatch.stypy_type_of_self = None
    test_gaussian_kde_monkeypatch.stypy_type_store = module_type_store
    test_gaussian_kde_monkeypatch.stypy_function_name = 'test_gaussian_kde_monkeypatch'
    test_gaussian_kde_monkeypatch.stypy_param_names_list = []
    test_gaussian_kde_monkeypatch.stypy_varargs_param_name = None
    test_gaussian_kde_monkeypatch.stypy_kwargs_param_name = None
    test_gaussian_kde_monkeypatch.stypy_call_defaults = defaults
    test_gaussian_kde_monkeypatch.stypy_call_varargs = varargs
    test_gaussian_kde_monkeypatch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gaussian_kde_monkeypatch', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gaussian_kde_monkeypatch', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gaussian_kde_monkeypatch(...)' code ##################

    str_653052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, (-1)), 'str', 'Ugly, but people may rely on this.  See scipy pull request 123,\n    specifically the linked ML thread "Width of the Gaussian in stats.kde".\n    If it is necessary to break this later on, that is to be discussed on ML.\n    ')
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to array(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining an instance of the builtin type 'list' (line 198)
    list_653055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 198)
    # Adding element type (line 198)
    int_653056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_653055, int_653056)
    # Adding element type (line 198)
    int_653057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_653055, int_653057)
    # Adding element type (line 198)
    int_653058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_653055, int_653058)
    # Adding element type (line 198)
    int_653059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_653055, int_653059)
    # Adding element type (line 198)
    int_653060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_653055, int_653060)
    
    # Processing the call keyword arguments (line 198)
    # Getting the type of 'float' (line 198)
    float_653061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 43), 'float', False)
    keyword_653062 = float_653061
    kwargs_653063 = {'dtype': keyword_653062}
    # Getting the type of 'np' (line 198)
    np_653053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 9), 'np', False)
    # Obtaining the member 'array' of a type (line 198)
    array_653054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 9), np_653053, 'array')
    # Calling array(args, kwargs) (line 198)
    array_call_result_653064 = invoke(stypy.reporting.localization.Localization(__file__, 198, 9), array_653054, *[list_653055], **kwargs_653063)
    
    # Assigning a type to the variable 'x1' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'x1', array_call_result_653064)
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to linspace(...): (line 199)
    # Processing the call arguments (line 199)
    int_653067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'int')
    int_653068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 26), 'int')
    # Processing the call keyword arguments (line 199)
    int_653069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'int')
    keyword_653070 = int_653069
    kwargs_653071 = {'num': keyword_653070}
    # Getting the type of 'np' (line 199)
    np_653065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 9), 'np', False)
    # Obtaining the member 'linspace' of a type (line 199)
    linspace_653066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 9), np_653065, 'linspace')
    # Calling linspace(args, kwargs) (line 199)
    linspace_call_result_653072 = invoke(stypy.reporting.localization.Localization(__file__, 199, 9), linspace_653066, *[int_653067, int_653068], **kwargs_653071)
    
    # Assigning a type to the variable 'xs' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'xs', linspace_call_result_653072)
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to gaussian_kde(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'x1' (line 202)
    x1_653075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'x1', False)
    # Processing the call keyword arguments (line 202)
    kwargs_653076 = {}
    # Getting the type of 'stats' (line 202)
    stats_653073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 10), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 202)
    gaussian_kde_653074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 10), stats_653073, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 202)
    gaussian_kde_call_result_653077 = invoke(stypy.reporting.localization.Localization(__file__, 202, 10), gaussian_kde_653074, *[x1_653075], **kwargs_653076)
    
    # Assigning a type to the variable 'kde' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'kde', gaussian_kde_call_result_653077)
    
    # Assigning a Attribute to a Attribute (line 203):
    
    # Assigning a Attribute to a Attribute (line 203):
    # Getting the type of 'kde' (line 203)
    kde_653078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 28), 'kde')
    # Obtaining the member 'silverman_factor' of a type (line 203)
    silverman_factor_653079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 28), kde_653078, 'silverman_factor')
    # Getting the type of 'kde' (line 203)
    kde_653080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'kde')
    # Setting the type of the member 'covariance_factor' of a type (line 203)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), kde_653080, 'covariance_factor', silverman_factor_653079)
    
    # Call to _compute_covariance(...): (line 204)
    # Processing the call keyword arguments (line 204)
    kwargs_653083 = {}
    # Getting the type of 'kde' (line 204)
    kde_653081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'kde', False)
    # Obtaining the member '_compute_covariance' of a type (line 204)
    _compute_covariance_653082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 4), kde_653081, '_compute_covariance')
    # Calling _compute_covariance(args, kwargs) (line 204)
    _compute_covariance_call_result_653084 = invoke(stypy.reporting.localization.Localization(__file__, 204, 4), _compute_covariance_653082, *[], **kwargs_653083)
    
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to kde(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'xs' (line 205)
    xs_653086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'xs', False)
    # Processing the call keyword arguments (line 205)
    kwargs_653087 = {}
    # Getting the type of 'kde' (line 205)
    kde_653085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 9), 'kde', False)
    # Calling kde(args, kwargs) (line 205)
    kde_call_result_653088 = invoke(stypy.reporting.localization.Localization(__file__, 205, 9), kde_653085, *[xs_653086], **kwargs_653087)
    
    # Assigning a type to the variable 'y1' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'y1', kde_call_result_653088)
    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to gaussian_kde(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'x1' (line 208)
    x1_653091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 30), 'x1', False)
    # Processing the call keyword arguments (line 208)
    str_653092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 44), 'str', 'silverman')
    keyword_653093 = str_653092
    kwargs_653094 = {'bw_method': keyword_653093}
    # Getting the type of 'stats' (line 208)
    stats_653089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 208)
    gaussian_kde_653090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), stats_653089, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 208)
    gaussian_kde_call_result_653095 = invoke(stypy.reporting.localization.Localization(__file__, 208, 11), gaussian_kde_653090, *[x1_653091], **kwargs_653094)
    
    # Assigning a type to the variable 'kde2' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'kde2', gaussian_kde_call_result_653095)
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to kde2(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'xs' (line 209)
    xs_653097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'xs', False)
    # Processing the call keyword arguments (line 209)
    kwargs_653098 = {}
    # Getting the type of 'kde2' (line 209)
    kde2_653096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 9), 'kde2', False)
    # Calling kde2(args, kwargs) (line 209)
    kde2_call_result_653099 = invoke(stypy.reporting.localization.Localization(__file__, 209, 9), kde2_653096, *[xs_653097], **kwargs_653098)
    
    # Assigning a type to the variable 'y2' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'y2', kde2_call_result_653099)
    
    # Call to assert_array_almost_equal_nulp(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'y1' (line 211)
    y1_653101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 35), 'y1', False)
    # Getting the type of 'y2' (line 211)
    y2_653102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 39), 'y2', False)
    # Processing the call keyword arguments (line 211)
    int_653103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 48), 'int')
    keyword_653104 = int_653103
    kwargs_653105 = {'nulp': keyword_653104}
    # Getting the type of 'assert_array_almost_equal_nulp' (line 211)
    assert_array_almost_equal_nulp_653100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'assert_array_almost_equal_nulp', False)
    # Calling assert_array_almost_equal_nulp(args, kwargs) (line 211)
    assert_array_almost_equal_nulp_call_result_653106 = invoke(stypy.reporting.localization.Localization(__file__, 211, 4), assert_array_almost_equal_nulp_653100, *[y1_653101, y2_653102], **kwargs_653105)
    
    
    # ################# End of 'test_gaussian_kde_monkeypatch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gaussian_kde_monkeypatch' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_653107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_653107)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gaussian_kde_monkeypatch'
    return stypy_return_type_653107

# Assigning a type to the variable 'test_gaussian_kde_monkeypatch' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'test_gaussian_kde_monkeypatch', test_gaussian_kde_monkeypatch)

@norecursion
def test_kde_integer_input(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_kde_integer_input'
    module_type_store = module_type_store.open_function_context('test_kde_integer_input', 214, 0, False)
    
    # Passed parameters checking function
    test_kde_integer_input.stypy_localization = localization
    test_kde_integer_input.stypy_type_of_self = None
    test_kde_integer_input.stypy_type_store = module_type_store
    test_kde_integer_input.stypy_function_name = 'test_kde_integer_input'
    test_kde_integer_input.stypy_param_names_list = []
    test_kde_integer_input.stypy_varargs_param_name = None
    test_kde_integer_input.stypy_kwargs_param_name = None
    test_kde_integer_input.stypy_call_defaults = defaults
    test_kde_integer_input.stypy_call_varargs = varargs
    test_kde_integer_input.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_kde_integer_input', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_kde_integer_input', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_kde_integer_input(...)' code ##################

    str_653108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 4), 'str', 'Regression test for #1181.')
    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to arange(...): (line 216)
    # Processing the call arguments (line 216)
    int_653111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 19), 'int')
    # Processing the call keyword arguments (line 216)
    kwargs_653112 = {}
    # Getting the type of 'np' (line 216)
    np_653109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 9), 'np', False)
    # Obtaining the member 'arange' of a type (line 216)
    arange_653110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 9), np_653109, 'arange')
    # Calling arange(args, kwargs) (line 216)
    arange_call_result_653113 = invoke(stypy.reporting.localization.Localization(__file__, 216, 9), arange_653110, *[int_653111], **kwargs_653112)
    
    # Assigning a type to the variable 'x1' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'x1', arange_call_result_653113)
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to gaussian_kde(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'x1' (line 217)
    x1_653116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'x1', False)
    # Processing the call keyword arguments (line 217)
    kwargs_653117 = {}
    # Getting the type of 'stats' (line 217)
    stats_653114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 10), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 217)
    gaussian_kde_653115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 10), stats_653114, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 217)
    gaussian_kde_call_result_653118 = invoke(stypy.reporting.localization.Localization(__file__, 217, 10), gaussian_kde_653115, *[x1_653116], **kwargs_653117)
    
    # Assigning a type to the variable 'kde' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'kde', gaussian_kde_call_result_653118)
    
    # Assigning a List to a Name (line 218):
    
    # Assigning a List to a Name (line 218):
    
    # Obtaining an instance of the builtin type 'list' (line 218)
    list_653119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 218)
    # Adding element type (line 218)
    float_653120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 17), list_653119, float_653120)
    # Adding element type (line 218)
    float_653121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 17), list_653119, float_653121)
    # Adding element type (line 218)
    float_653122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 17), list_653119, float_653122)
    # Adding element type (line 218)
    float_653123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 54), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 17), list_653119, float_653123)
    # Adding element type (line 218)
    float_653124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 66), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 17), list_653119, float_653124)
    
    # Assigning a type to the variable 'y_expected' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'y_expected', list_653119)
    
    # Call to assert_array_almost_equal(...): (line 219)
    # Processing the call arguments (line 219)
    
    # Call to kde(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'x1' (line 219)
    x1_653127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 34), 'x1', False)
    # Processing the call keyword arguments (line 219)
    kwargs_653128 = {}
    # Getting the type of 'kde' (line 219)
    kde_653126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'kde', False)
    # Calling kde(args, kwargs) (line 219)
    kde_call_result_653129 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), kde_653126, *[x1_653127], **kwargs_653128)
    
    # Getting the type of 'y_expected' (line 219)
    y_expected_653130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 39), 'y_expected', False)
    # Processing the call keyword arguments (line 219)
    int_653131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 59), 'int')
    keyword_653132 = int_653131
    kwargs_653133 = {'decimal': keyword_653132}
    # Getting the type of 'assert_array_almost_equal' (line 219)
    assert_array_almost_equal_653125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 219)
    assert_array_almost_equal_call_result_653134 = invoke(stypy.reporting.localization.Localization(__file__, 219, 4), assert_array_almost_equal_653125, *[kde_call_result_653129, y_expected_653130], **kwargs_653133)
    
    
    # ################# End of 'test_kde_integer_input(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_kde_integer_input' in the type store
    # Getting the type of 'stypy_return_type' (line 214)
    stypy_return_type_653135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_653135)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_kde_integer_input'
    return stypy_return_type_653135

# Assigning a type to the variable 'test_kde_integer_input' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'test_kde_integer_input', test_kde_integer_input)

@norecursion
def test_pdf_logpdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_pdf_logpdf'
    module_type_store = module_type_store.open_function_context('test_pdf_logpdf', 222, 0, False)
    
    # Passed parameters checking function
    test_pdf_logpdf.stypy_localization = localization
    test_pdf_logpdf.stypy_type_of_self = None
    test_pdf_logpdf.stypy_type_store = module_type_store
    test_pdf_logpdf.stypy_function_name = 'test_pdf_logpdf'
    test_pdf_logpdf.stypy_param_names_list = []
    test_pdf_logpdf.stypy_varargs_param_name = None
    test_pdf_logpdf.stypy_kwargs_param_name = None
    test_pdf_logpdf.stypy_call_defaults = defaults
    test_pdf_logpdf.stypy_call_varargs = varargs
    test_pdf_logpdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_pdf_logpdf', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_pdf_logpdf', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_pdf_logpdf(...)' code ##################

    
    # Call to seed(...): (line 223)
    # Processing the call arguments (line 223)
    int_653139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 19), 'int')
    # Processing the call keyword arguments (line 223)
    kwargs_653140 = {}
    # Getting the type of 'np' (line 223)
    np_653136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 223)
    random_653137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 4), np_653136, 'random')
    # Obtaining the member 'seed' of a type (line 223)
    seed_653138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 4), random_653137, 'seed')
    # Calling seed(args, kwargs) (line 223)
    seed_call_result_653141 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), seed_653138, *[int_653139], **kwargs_653140)
    
    
    # Assigning a Num to a Name (line 224):
    
    # Assigning a Num to a Name (line 224):
    int_653142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 19), 'int')
    # Assigning a type to the variable 'n_basesample' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'n_basesample', int_653142)
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to randn(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'n_basesample' (line 225)
    n_basesample_653146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 25), 'n_basesample', False)
    # Processing the call keyword arguments (line 225)
    kwargs_653147 = {}
    # Getting the type of 'np' (line 225)
    np_653143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 225)
    random_653144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 9), np_653143, 'random')
    # Obtaining the member 'randn' of a type (line 225)
    randn_653145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 9), random_653144, 'randn')
    # Calling randn(args, kwargs) (line 225)
    randn_call_result_653148 = invoke(stypy.reporting.localization.Localization(__file__, 225, 9), randn_653145, *[n_basesample_653146], **kwargs_653147)
    
    # Assigning a type to the variable 'xn' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'xn', randn_call_result_653148)
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to gaussian_kde(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'xn' (line 228)
    xn_653151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'xn', False)
    # Processing the call keyword arguments (line 228)
    kwargs_653152 = {}
    # Getting the type of 'stats' (line 228)
    stats_653149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 228)
    gaussian_kde_653150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 11), stats_653149, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 228)
    gaussian_kde_call_result_653153 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), gaussian_kde_653150, *[xn_653151], **kwargs_653152)
    
    # Assigning a type to the variable 'gkde' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'gkde', gaussian_kde_call_result_653153)
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to linspace(...): (line 230)
    # Processing the call arguments (line 230)
    int_653156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 21), 'int')
    int_653157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 26), 'int')
    int_653158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 30), 'int')
    # Processing the call keyword arguments (line 230)
    kwargs_653159 = {}
    # Getting the type of 'np' (line 230)
    np_653154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 9), 'np', False)
    # Obtaining the member 'linspace' of a type (line 230)
    linspace_653155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 9), np_653154, 'linspace')
    # Calling linspace(args, kwargs) (line 230)
    linspace_call_result_653160 = invoke(stypy.reporting.localization.Localization(__file__, 230, 9), linspace_653155, *[int_653156, int_653157, int_653158], **kwargs_653159)
    
    # Assigning a type to the variable 'xs' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'xs', linspace_call_result_653160)
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to evaluate(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'xs' (line 231)
    xs_653163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'xs', False)
    # Processing the call keyword arguments (line 231)
    kwargs_653164 = {}
    # Getting the type of 'gkde' (line 231)
    gkde_653161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 10), 'gkde', False)
    # Obtaining the member 'evaluate' of a type (line 231)
    evaluate_653162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 10), gkde_653161, 'evaluate')
    # Calling evaluate(args, kwargs) (line 231)
    evaluate_call_result_653165 = invoke(stypy.reporting.localization.Localization(__file__, 231, 10), evaluate_653162, *[xs_653163], **kwargs_653164)
    
    # Assigning a type to the variable 'pdf' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'pdf', evaluate_call_result_653165)
    
    # Assigning a Call to a Name (line 232):
    
    # Assigning a Call to a Name (line 232):
    
    # Call to pdf(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'xs' (line 232)
    xs_653168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'xs', False)
    # Processing the call keyword arguments (line 232)
    kwargs_653169 = {}
    # Getting the type of 'gkde' (line 232)
    gkde_653166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'gkde', False)
    # Obtaining the member 'pdf' of a type (line 232)
    pdf_653167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 11), gkde_653166, 'pdf')
    # Calling pdf(args, kwargs) (line 232)
    pdf_call_result_653170 = invoke(stypy.reporting.localization.Localization(__file__, 232, 11), pdf_653167, *[xs_653168], **kwargs_653169)
    
    # Assigning a type to the variable 'pdf2' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'pdf2', pdf_call_result_653170)
    
    # Call to assert_almost_equal(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'pdf' (line 233)
    pdf_653172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'pdf', False)
    # Getting the type of 'pdf2' (line 233)
    pdf2_653173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 29), 'pdf2', False)
    # Processing the call keyword arguments (line 233)
    int_653174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 43), 'int')
    keyword_653175 = int_653174
    kwargs_653176 = {'decimal': keyword_653175}
    # Getting the type of 'assert_almost_equal' (line 233)
    assert_almost_equal_653171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 233)
    assert_almost_equal_call_result_653177 = invoke(stypy.reporting.localization.Localization(__file__, 233, 4), assert_almost_equal_653171, *[pdf_653172, pdf2_653173], **kwargs_653176)
    
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to log(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'pdf' (line 235)
    pdf_653180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'pdf', False)
    # Processing the call keyword arguments (line 235)
    kwargs_653181 = {}
    # Getting the type of 'np' (line 235)
    np_653178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'np', False)
    # Obtaining the member 'log' of a type (line 235)
    log_653179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 13), np_653178, 'log')
    # Calling log(args, kwargs) (line 235)
    log_call_result_653182 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), log_653179, *[pdf_653180], **kwargs_653181)
    
    # Assigning a type to the variable 'logpdf' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'logpdf', log_call_result_653182)
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to logpdf(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'xs' (line 236)
    xs_653185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 26), 'xs', False)
    # Processing the call keyword arguments (line 236)
    kwargs_653186 = {}
    # Getting the type of 'gkde' (line 236)
    gkde_653183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 14), 'gkde', False)
    # Obtaining the member 'logpdf' of a type (line 236)
    logpdf_653184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 14), gkde_653183, 'logpdf')
    # Calling logpdf(args, kwargs) (line 236)
    logpdf_call_result_653187 = invoke(stypy.reporting.localization.Localization(__file__, 236, 14), logpdf_653184, *[xs_653185], **kwargs_653186)
    
    # Assigning a type to the variable 'logpdf2' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'logpdf2', logpdf_call_result_653187)
    
    # Call to assert_almost_equal(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'logpdf' (line 237)
    logpdf_653189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 24), 'logpdf', False)
    # Getting the type of 'logpdf2' (line 237)
    logpdf2_653190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'logpdf2', False)
    # Processing the call keyword arguments (line 237)
    int_653191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 49), 'int')
    keyword_653192 = int_653191
    kwargs_653193 = {'decimal': keyword_653192}
    # Getting the type of 'assert_almost_equal' (line 237)
    assert_almost_equal_653188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 237)
    assert_almost_equal_call_result_653194 = invoke(stypy.reporting.localization.Localization(__file__, 237, 4), assert_almost_equal_653188, *[logpdf_653189, logpdf2_653190], **kwargs_653193)
    
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to gaussian_kde(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'xs' (line 240)
    xs_653197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 30), 'xs', False)
    # Processing the call keyword arguments (line 240)
    kwargs_653198 = {}
    # Getting the type of 'stats' (line 240)
    stats_653195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'stats', False)
    # Obtaining the member 'gaussian_kde' of a type (line 240)
    gaussian_kde_653196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), stats_653195, 'gaussian_kde')
    # Calling gaussian_kde(args, kwargs) (line 240)
    gaussian_kde_call_result_653199 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), gaussian_kde_653196, *[xs_653197], **kwargs_653198)
    
    # Assigning a type to the variable 'gkde' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'gkde', gaussian_kde_call_result_653199)
    
    # Assigning a Call to a Name (line 241):
    
    # Assigning a Call to a Name (line 241):
    
    # Call to log(...): (line 241)
    # Processing the call arguments (line 241)
    
    # Call to evaluate(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'xn' (line 241)
    xn_653204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'xn', False)
    # Processing the call keyword arguments (line 241)
    kwargs_653205 = {}
    # Getting the type of 'gkde' (line 241)
    gkde_653202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'gkde', False)
    # Obtaining the member 'evaluate' of a type (line 241)
    evaluate_653203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 17), gkde_653202, 'evaluate')
    # Calling evaluate(args, kwargs) (line 241)
    evaluate_call_result_653206 = invoke(stypy.reporting.localization.Localization(__file__, 241, 17), evaluate_653203, *[xn_653204], **kwargs_653205)
    
    # Processing the call keyword arguments (line 241)
    kwargs_653207 = {}
    # Getting the type of 'np' (line 241)
    np_653200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 10), 'np', False)
    # Obtaining the member 'log' of a type (line 241)
    log_653201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 10), np_653200, 'log')
    # Calling log(args, kwargs) (line 241)
    log_call_result_653208 = invoke(stypy.reporting.localization.Localization(__file__, 241, 10), log_653201, *[evaluate_call_result_653206], **kwargs_653207)
    
    # Assigning a type to the variable 'pdf' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'pdf', log_call_result_653208)
    
    # Assigning a Call to a Name (line 242):
    
    # Assigning a Call to a Name (line 242):
    
    # Call to logpdf(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'xn' (line 242)
    xn_653211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 23), 'xn', False)
    # Processing the call keyword arguments (line 242)
    kwargs_653212 = {}
    # Getting the type of 'gkde' (line 242)
    gkde_653209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'gkde', False)
    # Obtaining the member 'logpdf' of a type (line 242)
    logpdf_653210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 11), gkde_653209, 'logpdf')
    # Calling logpdf(args, kwargs) (line 242)
    logpdf_call_result_653213 = invoke(stypy.reporting.localization.Localization(__file__, 242, 11), logpdf_653210, *[xn_653211], **kwargs_653212)
    
    # Assigning a type to the variable 'pdf2' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'pdf2', logpdf_call_result_653213)
    
    # Call to assert_almost_equal(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'pdf' (line 243)
    pdf_653215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'pdf', False)
    # Getting the type of 'pdf2' (line 243)
    pdf2_653216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'pdf2', False)
    # Processing the call keyword arguments (line 243)
    int_653217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 43), 'int')
    keyword_653218 = int_653217
    kwargs_653219 = {'decimal': keyword_653218}
    # Getting the type of 'assert_almost_equal' (line 243)
    assert_almost_equal_653214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 243)
    assert_almost_equal_call_result_653220 = invoke(stypy.reporting.localization.Localization(__file__, 243, 4), assert_almost_equal_653214, *[pdf_653215, pdf2_653216], **kwargs_653219)
    
    
    # ################# End of 'test_pdf_logpdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_pdf_logpdf' in the type store
    # Getting the type of 'stypy_return_type' (line 222)
    stypy_return_type_653221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_653221)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_pdf_logpdf'
    return stypy_return_type_653221

# Assigning a type to the variable 'test_pdf_logpdf' (line 222)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'test_pdf_logpdf', test_pdf_logpdf)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

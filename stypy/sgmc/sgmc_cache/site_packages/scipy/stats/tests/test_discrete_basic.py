
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy.testing as npt
4: import numpy as np
5: from scipy._lib.six import xrange
6: import pytest
7: 
8: from scipy import stats
9: from .common_tests import (check_normalization, check_moment, check_mean_expect,
10:                            check_var_expect, check_skew_expect,
11:                            check_kurt_expect, check_entropy,
12:                            check_private_entropy, check_edge_support,
13:                            check_named_args, check_random_state_property,
14:                            check_pickling, check_rvs_broadcast)
15: from scipy.stats._distr_params import distdiscrete
16: 
17: vals = ([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
18: distdiscrete += [[stats.rv_discrete(values=vals), ()]]
19: 
20: 
21: def cases_test_discrete_basic():
22:     seen = set()
23:     for distname, arg in distdiscrete:
24:         yield distname, arg, distname not in seen
25:         seen.add(distname)
26: 
27: 
28: @pytest.mark.parametrize('distname,arg,first_case', cases_test_discrete_basic())
29: def test_discrete_basic(distname, arg, first_case):
30:     try:
31:         distfn = getattr(stats, distname)
32:     except TypeError:
33:         distfn = distname
34:         distname = 'sample distribution'
35:     np.random.seed(9765456)
36:     rvs = distfn.rvs(size=2000, *arg)
37:     supp = np.unique(rvs)
38:     m, v = distfn.stats(*arg)
39:     check_cdf_ppf(distfn, arg, supp, distname + ' cdf_ppf')
40: 
41:     check_pmf_cdf(distfn, arg, distname)
42:     check_oth(distfn, arg, supp, distname + ' oth')
43:     check_edge_support(distfn, arg)
44: 
45:     alpha = 0.01
46:     check_discrete_chisquare(distfn, arg, rvs, alpha,
47:            distname + ' chisquare')
48: 
49:     if first_case:
50:         locscale_defaults = (0,)
51:         meths = [distfn.pmf, distfn.logpmf, distfn.cdf, distfn.logcdf,
52:                  distfn.logsf]
53:         # make sure arguments are within support
54:         spec_k = {'randint': 11, 'hypergeom': 4, 'bernoulli': 0, }
55:         k = spec_k.get(distname, 1)
56:         check_named_args(distfn, k, arg, locscale_defaults, meths)
57:         if distname != 'sample distribution':
58:             check_scale_docstring(distfn)
59:         check_random_state_property(distfn, arg)
60:         check_pickling(distfn, arg)
61: 
62:         # Entropy
63:         check_entropy(distfn, arg, distname)
64:         if distfn.__class__._entropy != stats.rv_discrete._entropy:
65:             check_private_entropy(distfn, arg, stats.rv_discrete)
66: 
67: 
68: @pytest.mark.parametrize('distname,arg', distdiscrete)
69: def test_moments(distname, arg):
70:     try:
71:         distfn = getattr(stats, distname)
72:     except TypeError:
73:         distfn = distname
74:         distname = 'sample distribution'
75:     m, v, s, k = distfn.stats(*arg, moments='mvsk')
76:     check_normalization(distfn, arg, distname)
77: 
78:     # compare `stats` and `moment` methods
79:     check_moment(distfn, arg, m, v, distname)
80:     check_mean_expect(distfn, arg, m, distname)
81:     check_var_expect(distfn, arg, m, v, distname)
82:     check_skew_expect(distfn, arg, m, v, s, distname)
83:     if distname not in ['zipf']:
84:         check_kurt_expect(distfn, arg, m, v, k, distname)
85: 
86:     # frozen distr moments
87:     check_moment_frozen(distfn, arg, m, 1)
88:     check_moment_frozen(distfn, arg, v+m*m, 2)
89: 
90: 
91: @pytest.mark.parametrize('dist,shape_args', distdiscrete)
92: def test_rvs_broadcast(dist, shape_args):
93:     # If shape_only is True, it means the _rvs method of the
94:     # distribution uses more than one random number to generate a random
95:     # variate.  That means the result of using rvs with broadcasting or
96:     # with a nontrivial size will not necessarily be the same as using the
97:     # numpy.vectorize'd version of rvs(), so we can only compare the shapes
98:     # of the results, not the values.
99:     # Whether or not a distribution is in the following list is an
100:     # implementation detail of the distribution, not a requirement.  If
101:     # the implementation the rvs() method of a distribution changes, this
102:     # test might also have to be changed.
103:     shape_only = dist in ['skellam']
104: 
105:     try:
106:         distfunc = getattr(stats, dist)
107:     except TypeError:
108:         distfunc = dist
109:         dist = 'rv_discrete(values=(%r, %r))' % (dist.xk, dist.pk)
110:     loc = np.zeros(2)
111:     nargs = distfunc.numargs
112:     allargs = []
113:     bshape = []
114:     # Generate shape parameter arguments...
115:     for k in range(nargs):
116:         shp = (k + 3,) + (1,)*(k + 1)
117:         param_val = shape_args[k]
118:         allargs.append(param_val*np.ones(shp, dtype=np.array(param_val).dtype))
119:         bshape.insert(0, shp[0])
120:     allargs.append(loc)
121:     bshape.append(loc.size)
122:     # bshape holds the expected shape when loc, scale, and the shape
123:     # parameters are all broadcast together.
124:     check_rvs_broadcast(distfunc, dist, allargs, bshape, shape_only, [np.int_])
125: 
126: 
127: def check_cdf_ppf(distfn, arg, supp, msg):
128:     # cdf is a step function, and ppf(q) = min{k : cdf(k) >= q, k integer}
129:     npt.assert_array_equal(distfn.ppf(distfn.cdf(supp, *arg), *arg),
130:                            supp, msg + '-roundtrip')
131:     npt.assert_array_equal(distfn.ppf(distfn.cdf(supp, *arg) - 1e-8, *arg),
132:                            supp, msg + '-roundtrip')
133: 
134:     if not hasattr(distfn, 'xk'):
135:         supp1 = supp[supp < distfn.b]
136:         npt.assert_array_equal(distfn.ppf(distfn.cdf(supp1, *arg) + 1e-8, *arg),
137:                                supp1 + distfn.inc, msg + ' ppf-cdf-next')
138:         # -1e-8 could cause an error if pmf < 1e-8
139: 
140: 
141: def check_pmf_cdf(distfn, arg, distname):
142:     if hasattr(distfn, 'xk'):
143:         index = distfn.xk
144:     else:
145:         startind = int(distfn.ppf(0.01, *arg) - 1)
146:         index = list(range(startind, startind + 10))
147:     cdfs = distfn.cdf(index, *arg)
148:     pmfs_cum = distfn.pmf(index, *arg).cumsum()
149: 
150:     atol, rtol = 1e-10, 1e-10
151:     if distname == 'skellam':    # ncx2 accuracy
152:         atol, rtol = 1e-5, 1e-5
153:     npt.assert_allclose(cdfs - cdfs[0], pmfs_cum - pmfs_cum[0],
154:                         atol=atol, rtol=rtol)
155: 
156: 
157: def check_moment_frozen(distfn, arg, m, k):
158:     npt.assert_allclose(distfn(*arg).moment(k), m,
159:                         atol=1e-10, rtol=1e-10)
160: 
161: 
162: def check_oth(distfn, arg, supp, msg):
163:     # checking other methods of distfn
164:     npt.assert_allclose(distfn.sf(supp, *arg), 1. - distfn.cdf(supp, *arg),
165:                         atol=1e-10, rtol=1e-10)
166: 
167:     q = np.linspace(0.01, 0.99, 20)
168:     npt.assert_allclose(distfn.isf(q, *arg), distfn.ppf(1. - q, *arg),
169:                         atol=1e-10, rtol=1e-10)
170: 
171:     median_sf = distfn.isf(0.5, *arg)
172:     npt.assert_(distfn.sf(median_sf - 1, *arg) > 0.5)
173:     npt.assert_(distfn.cdf(median_sf + 1, *arg) > 0.5)
174: 
175: 
176: def check_discrete_chisquare(distfn, arg, rvs, alpha, msg):
177:     '''Perform chisquare test for random sample of a discrete distribution
178: 
179:     Parameters
180:     ----------
181:     distname : string
182:         name of distribution function
183:     arg : sequence
184:         parameters of distribution
185:     alpha : float
186:         significance level, threshold for p-value
187: 
188:     Returns
189:     -------
190:     result : bool
191:         0 if test passes, 1 if test fails
192: 
193:     '''
194:     wsupp = 0.05
195: 
196:     # construct intervals with minimum mass `wsupp`.
197:     # intervals are left-half-open as in a cdf difference
198:     lo = int(max(distfn.a, -1000))
199:     distsupport = xrange(lo, int(min(distfn.b, 1000)) + 1)
200:     last = 0
201:     distsupp = [lo]
202:     distmass = []
203:     for ii in distsupport:
204:         current = distfn.cdf(ii, *arg)
205:         if current - last >= wsupp - 1e-14:
206:             distsupp.append(ii)
207:             distmass.append(current - last)
208:             last = current
209:             if current > (1 - wsupp):
210:                 break
211:     if distsupp[-1] < distfn.b:
212:         distsupp.append(distfn.b)
213:         distmass.append(1 - last)
214:     distsupp = np.array(distsupp)
215:     distmass = np.array(distmass)
216: 
217:     # convert intervals to right-half-open as required by histogram
218:     histsupp = distsupp + 1e-8
219:     histsupp[0] = distfn.a
220: 
221:     # find sample frequencies and perform chisquare test
222:     freq, hsupp = np.histogram(rvs, histsupp)
223:     chis, pval = stats.chisquare(np.array(freq), len(rvs)*distmass)
224: 
225:     npt.assert_(pval > alpha,
226:                 'chisquare - test for %s at arg = %s with pval = %s' %
227:                 (msg, str(arg), str(pval)))
228: 
229: 
230: def check_scale_docstring(distfn):
231:     if distfn.__doc__ is not None:
232:         # Docstrings can be stripped if interpreter is run with -OO
233:         npt.assert_('scale' not in distfn.__doc__)
234: 
235: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy.testing' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_634733 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_634733) is not StypyTypeError):

    if (import_634733 != 'pyd_module'):
        __import__(import_634733)
        sys_modules_634734 = sys.modules[import_634733]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'npt', sys_modules_634734.module_type_store, module_type_store)
    else:
        import numpy.testing as npt

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'npt', numpy.testing, module_type_store)

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_634733)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_634735 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_634735) is not StypyTypeError):

    if (import_634735 != 'pyd_module'):
        __import__(import_634735)
        sys_modules_634736 = sys.modules[import_634735]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_634736.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_634735)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy._lib.six import xrange' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_634737 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.six')

if (type(import_634737) is not StypyTypeError):

    if (import_634737 != 'pyd_module'):
        __import__(import_634737)
        sys_modules_634738 = sys.modules[import_634737]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.six', sys_modules_634738.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_634738, sys_modules_634738.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.six', import_634737)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import pytest' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_634739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_634739) is not StypyTypeError):

    if (import_634739 != 'pyd_module'):
        __import__(import_634739)
        sys_modules_634740 = sys.modules[import_634739]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_634740.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_634739)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy import stats' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_634741 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy')

if (type(import_634741) is not StypyTypeError):

    if (import_634741 != 'pyd_module'):
        __import__(import_634741)
        sys_modules_634742 = sys.modules[import_634741]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy', sys_modules_634742.module_type_store, module_type_store, ['stats'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_634742, sys_modules_634742.module_type_store, module_type_store)
    else:
        from scipy import stats

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy', None, module_type_store, ['stats'], [stats])

else:
    # Assigning a type to the variable 'scipy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy', import_634741)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.stats.tests.common_tests import check_normalization, check_moment, check_mean_expect, check_var_expect, check_skew_expect, check_kurt_expect, check_entropy, check_private_entropy, check_edge_support, check_named_args, check_random_state_property, check_pickling, check_rvs_broadcast' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_634743 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.tests.common_tests')

if (type(import_634743) is not StypyTypeError):

    if (import_634743 != 'pyd_module'):
        __import__(import_634743)
        sys_modules_634744 = sys.modules[import_634743]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.tests.common_tests', sys_modules_634744.module_type_store, module_type_store, ['check_normalization', 'check_moment', 'check_mean_expect', 'check_var_expect', 'check_skew_expect', 'check_kurt_expect', 'check_entropy', 'check_private_entropy', 'check_edge_support', 'check_named_args', 'check_random_state_property', 'check_pickling', 'check_rvs_broadcast'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_634744, sys_modules_634744.module_type_store, module_type_store)
    else:
        from scipy.stats.tests.common_tests import check_normalization, check_moment, check_mean_expect, check_var_expect, check_skew_expect, check_kurt_expect, check_entropy, check_private_entropy, check_edge_support, check_named_args, check_random_state_property, check_pickling, check_rvs_broadcast

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.tests.common_tests', None, module_type_store, ['check_normalization', 'check_moment', 'check_mean_expect', 'check_var_expect', 'check_skew_expect', 'check_kurt_expect', 'check_entropy', 'check_private_entropy', 'check_edge_support', 'check_named_args', 'check_random_state_property', 'check_pickling', 'check_rvs_broadcast'], [check_normalization, check_moment, check_mean_expect, check_var_expect, check_skew_expect, check_kurt_expect, check_entropy, check_private_entropy, check_edge_support, check_named_args, check_random_state_property, check_pickling, check_rvs_broadcast])

else:
    # Assigning a type to the variable 'scipy.stats.tests.common_tests' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.tests.common_tests', import_634743)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.stats._distr_params import distdiscrete' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_634745 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.stats._distr_params')

if (type(import_634745) is not StypyTypeError):

    if (import_634745 != 'pyd_module'):
        __import__(import_634745)
        sys_modules_634746 = sys.modules[import_634745]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.stats._distr_params', sys_modules_634746.module_type_store, module_type_store, ['distdiscrete'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_634746, sys_modules_634746.module_type_store, module_type_store)
    else:
        from scipy.stats._distr_params import distdiscrete

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.stats._distr_params', None, module_type_store, ['distdiscrete'], [distdiscrete])

else:
    # Assigning a type to the variable 'scipy.stats._distr_params' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.stats._distr_params', import_634745)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')


# Assigning a Tuple to a Name (line 17):

# Assigning a Tuple to a Name (line 17):

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_634747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_634748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_634749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 8), list_634748, int_634749)
# Adding element type (line 17)
int_634750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 8), list_634748, int_634750)
# Adding element type (line 17)
int_634751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 8), list_634748, int_634751)
# Adding element type (line 17)
int_634752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 8), list_634748, int_634752)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 8), tuple_634747, list_634748)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_634753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
float_634754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_634753, float_634754)
# Adding element type (line 17)
float_634755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_634753, float_634755)
# Adding element type (line 17)
float_634756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_634753, float_634756)
# Adding element type (line 17)
float_634757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 22), list_634753, float_634757)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 8), tuple_634747, list_634753)

# Assigning a type to the variable 'vals' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'vals', tuple_634747)

# Getting the type of 'distdiscrete' (line 18)
distdiscrete_634758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distdiscrete')

# Obtaining an instance of the builtin type 'list' (line 18)
list_634759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 18)
list_634760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)

# Call to rv_discrete(...): (line 18)
# Processing the call keyword arguments (line 18)
# Getting the type of 'vals' (line 18)
vals_634763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 43), 'vals', False)
keyword_634764 = vals_634763
kwargs_634765 = {'values': keyword_634764}
# Getting the type of 'stats' (line 18)
stats_634761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'stats', False)
# Obtaining the member 'rv_discrete' of a type (line 18)
rv_discrete_634762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 18), stats_634761, 'rv_discrete')
# Calling rv_discrete(args, kwargs) (line 18)
rv_discrete_call_result_634766 = invoke(stypy.reporting.localization.Localization(__file__, 18, 18), rv_discrete_634762, *[], **kwargs_634765)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 17), list_634760, rv_discrete_call_result_634766)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_634767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 17), list_634760, tuple_634767)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 16), list_634759, list_634760)

# Applying the binary operator '+=' (line 18)
result_iadd_634768 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 0), '+=', distdiscrete_634758, list_634759)
# Assigning a type to the variable 'distdiscrete' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distdiscrete', result_iadd_634768)


@norecursion
def cases_test_discrete_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cases_test_discrete_basic'
    module_type_store = module_type_store.open_function_context('cases_test_discrete_basic', 21, 0, False)
    
    # Passed parameters checking function
    cases_test_discrete_basic.stypy_localization = localization
    cases_test_discrete_basic.stypy_type_of_self = None
    cases_test_discrete_basic.stypy_type_store = module_type_store
    cases_test_discrete_basic.stypy_function_name = 'cases_test_discrete_basic'
    cases_test_discrete_basic.stypy_param_names_list = []
    cases_test_discrete_basic.stypy_varargs_param_name = None
    cases_test_discrete_basic.stypy_kwargs_param_name = None
    cases_test_discrete_basic.stypy_call_defaults = defaults
    cases_test_discrete_basic.stypy_call_varargs = varargs
    cases_test_discrete_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cases_test_discrete_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cases_test_discrete_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cases_test_discrete_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to set(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_634770 = {}
    # Getting the type of 'set' (line 22)
    set_634769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'set', False)
    # Calling set(args, kwargs) (line 22)
    set_call_result_634771 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), set_634769, *[], **kwargs_634770)
    
    # Assigning a type to the variable 'seen' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'seen', set_call_result_634771)
    
    # Getting the type of 'distdiscrete' (line 23)
    distdiscrete_634772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'distdiscrete')
    # Testing the type of a for loop iterable (line 23)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 23, 4), distdiscrete_634772)
    # Getting the type of the for loop variable (line 23)
    for_loop_var_634773 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 23, 4), distdiscrete_634772)
    # Assigning a type to the variable 'distname' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'distname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), for_loop_var_634773))
    # Assigning a type to the variable 'arg' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), for_loop_var_634773))
    # SSA begins for a for statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Creating a generator
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_634774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    # Getting the type of 'distname' (line 24)
    distname_634775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'distname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), tuple_634774, distname_634775)
    # Adding element type (line 24)
    # Getting the type of 'arg' (line 24)
    arg_634776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'arg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), tuple_634774, arg_634776)
    # Adding element type (line 24)
    
    # Getting the type of 'distname' (line 24)
    distname_634777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'distname')
    # Getting the type of 'seen' (line 24)
    seen_634778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 45), 'seen')
    # Applying the binary operator 'notin' (line 24)
    result_contains_634779 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 29), 'notin', distname_634777, seen_634778)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), tuple_634774, result_contains_634779)
    
    GeneratorType_634780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 8), GeneratorType_634780, tuple_634774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'stypy_return_type', GeneratorType_634780)
    
    # Call to add(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'distname' (line 25)
    distname_634783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'distname', False)
    # Processing the call keyword arguments (line 25)
    kwargs_634784 = {}
    # Getting the type of 'seen' (line 25)
    seen_634781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'seen', False)
    # Obtaining the member 'add' of a type (line 25)
    add_634782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), seen_634781, 'add')
    # Calling add(args, kwargs) (line 25)
    add_call_result_634785 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), add_634782, *[distname_634783], **kwargs_634784)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'cases_test_discrete_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cases_test_discrete_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_634786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634786)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cases_test_discrete_basic'
    return stypy_return_type_634786

# Assigning a type to the variable 'cases_test_discrete_basic' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'cases_test_discrete_basic', cases_test_discrete_basic)

@norecursion
def test_discrete_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_discrete_basic'
    module_type_store = module_type_store.open_function_context('test_discrete_basic', 28, 0, False)
    
    # Passed parameters checking function
    test_discrete_basic.stypy_localization = localization
    test_discrete_basic.stypy_type_of_self = None
    test_discrete_basic.stypy_type_store = module_type_store
    test_discrete_basic.stypy_function_name = 'test_discrete_basic'
    test_discrete_basic.stypy_param_names_list = ['distname', 'arg', 'first_case']
    test_discrete_basic.stypy_varargs_param_name = None
    test_discrete_basic.stypy_kwargs_param_name = None
    test_discrete_basic.stypy_call_defaults = defaults
    test_discrete_basic.stypy_call_varargs = varargs
    test_discrete_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_discrete_basic', ['distname', 'arg', 'first_case'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_discrete_basic', localization, ['distname', 'arg', 'first_case'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_discrete_basic(...)' code ##################

    
    
    # SSA begins for try-except statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 31):
    
    # Assigning a Call to a Name (line 31):
    
    # Call to getattr(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'stats' (line 31)
    stats_634788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'stats', False)
    # Getting the type of 'distname' (line 31)
    distname_634789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 32), 'distname', False)
    # Processing the call keyword arguments (line 31)
    kwargs_634790 = {}
    # Getting the type of 'getattr' (line 31)
    getattr_634787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'getattr', False)
    # Calling getattr(args, kwargs) (line 31)
    getattr_call_result_634791 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), getattr_634787, *[stats_634788, distname_634789], **kwargs_634790)
    
    # Assigning a type to the variable 'distfn' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'distfn', getattr_call_result_634791)
    # SSA branch for the except part of a try statement (line 30)
    # SSA branch for the except 'TypeError' branch of a try statement (line 30)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 33):
    
    # Assigning a Name to a Name (line 33):
    # Getting the type of 'distname' (line 33)
    distname_634792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'distname')
    # Assigning a type to the variable 'distfn' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'distfn', distname_634792)
    
    # Assigning a Str to a Name (line 34):
    
    # Assigning a Str to a Name (line 34):
    str_634793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'str', 'sample distribution')
    # Assigning a type to the variable 'distname' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'distname', str_634793)
    # SSA join for try-except statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to seed(...): (line 35)
    # Processing the call arguments (line 35)
    int_634797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 19), 'int')
    # Processing the call keyword arguments (line 35)
    kwargs_634798 = {}
    # Getting the type of 'np' (line 35)
    np_634794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 35)
    random_634795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), np_634794, 'random')
    # Obtaining the member 'seed' of a type (line 35)
    seed_634796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), random_634795, 'seed')
    # Calling seed(args, kwargs) (line 35)
    seed_call_result_634799 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), seed_634796, *[int_634797], **kwargs_634798)
    
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to rvs(...): (line 36)
    # Getting the type of 'arg' (line 36)
    arg_634802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'arg', False)
    # Processing the call keyword arguments (line 36)
    int_634803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'int')
    keyword_634804 = int_634803
    kwargs_634805 = {'size': keyword_634804}
    # Getting the type of 'distfn' (line 36)
    distfn_634800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'distfn', False)
    # Obtaining the member 'rvs' of a type (line 36)
    rvs_634801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 10), distfn_634800, 'rvs')
    # Calling rvs(args, kwargs) (line 36)
    rvs_call_result_634806 = invoke(stypy.reporting.localization.Localization(__file__, 36, 10), rvs_634801, *[arg_634802], **kwargs_634805)
    
    # Assigning a type to the variable 'rvs' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'rvs', rvs_call_result_634806)
    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to unique(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'rvs' (line 37)
    rvs_634809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'rvs', False)
    # Processing the call keyword arguments (line 37)
    kwargs_634810 = {}
    # Getting the type of 'np' (line 37)
    np_634807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'np', False)
    # Obtaining the member 'unique' of a type (line 37)
    unique_634808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 11), np_634807, 'unique')
    # Calling unique(args, kwargs) (line 37)
    unique_call_result_634811 = invoke(stypy.reporting.localization.Localization(__file__, 37, 11), unique_634808, *[rvs_634809], **kwargs_634810)
    
    # Assigning a type to the variable 'supp' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'supp', unique_call_result_634811)
    
    # Assigning a Call to a Tuple (line 38):
    
    # Assigning a Subscript to a Name (line 38):
    
    # Obtaining the type of the subscript
    int_634812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'int')
    
    # Call to stats(...): (line 38)
    # Getting the type of 'arg' (line 38)
    arg_634815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'arg', False)
    # Processing the call keyword arguments (line 38)
    kwargs_634816 = {}
    # Getting the type of 'distfn' (line 38)
    distfn_634813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'distfn', False)
    # Obtaining the member 'stats' of a type (line 38)
    stats_634814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), distfn_634813, 'stats')
    # Calling stats(args, kwargs) (line 38)
    stats_call_result_634817 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), stats_634814, *[arg_634815], **kwargs_634816)
    
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___634818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), stats_call_result_634817, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_634819 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), getitem___634818, int_634812)
    
    # Assigning a type to the variable 'tuple_var_assignment_634719' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'tuple_var_assignment_634719', subscript_call_result_634819)
    
    # Assigning a Subscript to a Name (line 38):
    
    # Obtaining the type of the subscript
    int_634820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'int')
    
    # Call to stats(...): (line 38)
    # Getting the type of 'arg' (line 38)
    arg_634823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'arg', False)
    # Processing the call keyword arguments (line 38)
    kwargs_634824 = {}
    # Getting the type of 'distfn' (line 38)
    distfn_634821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'distfn', False)
    # Obtaining the member 'stats' of a type (line 38)
    stats_634822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), distfn_634821, 'stats')
    # Calling stats(args, kwargs) (line 38)
    stats_call_result_634825 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), stats_634822, *[arg_634823], **kwargs_634824)
    
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___634826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), stats_call_result_634825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_634827 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), getitem___634826, int_634820)
    
    # Assigning a type to the variable 'tuple_var_assignment_634720' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'tuple_var_assignment_634720', subscript_call_result_634827)
    
    # Assigning a Name to a Name (line 38):
    # Getting the type of 'tuple_var_assignment_634719' (line 38)
    tuple_var_assignment_634719_634828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'tuple_var_assignment_634719')
    # Assigning a type to the variable 'm' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'm', tuple_var_assignment_634719_634828)
    
    # Assigning a Name to a Name (line 38):
    # Getting the type of 'tuple_var_assignment_634720' (line 38)
    tuple_var_assignment_634720_634829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'tuple_var_assignment_634720')
    # Assigning a type to the variable 'v' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'v', tuple_var_assignment_634720_634829)
    
    # Call to check_cdf_ppf(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'distfn' (line 39)
    distfn_634831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'distfn', False)
    # Getting the type of 'arg' (line 39)
    arg_634832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'arg', False)
    # Getting the type of 'supp' (line 39)
    supp_634833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'supp', False)
    # Getting the type of 'distname' (line 39)
    distname_634834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'distname', False)
    str_634835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 48), 'str', ' cdf_ppf')
    # Applying the binary operator '+' (line 39)
    result_add_634836 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 37), '+', distname_634834, str_634835)
    
    # Processing the call keyword arguments (line 39)
    kwargs_634837 = {}
    # Getting the type of 'check_cdf_ppf' (line 39)
    check_cdf_ppf_634830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'check_cdf_ppf', False)
    # Calling check_cdf_ppf(args, kwargs) (line 39)
    check_cdf_ppf_call_result_634838 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), check_cdf_ppf_634830, *[distfn_634831, arg_634832, supp_634833, result_add_634836], **kwargs_634837)
    
    
    # Call to check_pmf_cdf(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'distfn' (line 41)
    distfn_634840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'distfn', False)
    # Getting the type of 'arg' (line 41)
    arg_634841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'arg', False)
    # Getting the type of 'distname' (line 41)
    distname_634842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'distname', False)
    # Processing the call keyword arguments (line 41)
    kwargs_634843 = {}
    # Getting the type of 'check_pmf_cdf' (line 41)
    check_pmf_cdf_634839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'check_pmf_cdf', False)
    # Calling check_pmf_cdf(args, kwargs) (line 41)
    check_pmf_cdf_call_result_634844 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), check_pmf_cdf_634839, *[distfn_634840, arg_634841, distname_634842], **kwargs_634843)
    
    
    # Call to check_oth(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'distfn' (line 42)
    distfn_634846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'distfn', False)
    # Getting the type of 'arg' (line 42)
    arg_634847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'arg', False)
    # Getting the type of 'supp' (line 42)
    supp_634848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'supp', False)
    # Getting the type of 'distname' (line 42)
    distname_634849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'distname', False)
    str_634850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 44), 'str', ' oth')
    # Applying the binary operator '+' (line 42)
    result_add_634851 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 33), '+', distname_634849, str_634850)
    
    # Processing the call keyword arguments (line 42)
    kwargs_634852 = {}
    # Getting the type of 'check_oth' (line 42)
    check_oth_634845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'check_oth', False)
    # Calling check_oth(args, kwargs) (line 42)
    check_oth_call_result_634853 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), check_oth_634845, *[distfn_634846, arg_634847, supp_634848, result_add_634851], **kwargs_634852)
    
    
    # Call to check_edge_support(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'distfn' (line 43)
    distfn_634855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'distfn', False)
    # Getting the type of 'arg' (line 43)
    arg_634856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'arg', False)
    # Processing the call keyword arguments (line 43)
    kwargs_634857 = {}
    # Getting the type of 'check_edge_support' (line 43)
    check_edge_support_634854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'check_edge_support', False)
    # Calling check_edge_support(args, kwargs) (line 43)
    check_edge_support_call_result_634858 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), check_edge_support_634854, *[distfn_634855, arg_634856], **kwargs_634857)
    
    
    # Assigning a Num to a Name (line 45):
    
    # Assigning a Num to a Name (line 45):
    float_634859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'float')
    # Assigning a type to the variable 'alpha' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'alpha', float_634859)
    
    # Call to check_discrete_chisquare(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'distfn' (line 46)
    distfn_634861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'distfn', False)
    # Getting the type of 'arg' (line 46)
    arg_634862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'arg', False)
    # Getting the type of 'rvs' (line 46)
    rvs_634863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'rvs', False)
    # Getting the type of 'alpha' (line 46)
    alpha_634864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 47), 'alpha', False)
    # Getting the type of 'distname' (line 47)
    distname_634865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'distname', False)
    str_634866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'str', ' chisquare')
    # Applying the binary operator '+' (line 47)
    result_add_634867 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '+', distname_634865, str_634866)
    
    # Processing the call keyword arguments (line 46)
    kwargs_634868 = {}
    # Getting the type of 'check_discrete_chisquare' (line 46)
    check_discrete_chisquare_634860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'check_discrete_chisquare', False)
    # Calling check_discrete_chisquare(args, kwargs) (line 46)
    check_discrete_chisquare_call_result_634869 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), check_discrete_chisquare_634860, *[distfn_634861, arg_634862, rvs_634863, alpha_634864, result_add_634867], **kwargs_634868)
    
    
    # Getting the type of 'first_case' (line 49)
    first_case_634870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 7), 'first_case')
    # Testing the type of an if condition (line 49)
    if_condition_634871 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 4), first_case_634870)
    # Assigning a type to the variable 'if_condition_634871' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'if_condition_634871', if_condition_634871)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 50):
    
    # Assigning a Tuple to a Name (line 50):
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_634872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    int_634873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), tuple_634872, int_634873)
    
    # Assigning a type to the variable 'locscale_defaults' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'locscale_defaults', tuple_634872)
    
    # Assigning a List to a Name (line 51):
    
    # Assigning a List to a Name (line 51):
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_634874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    # Getting the type of 'distfn' (line 51)
    distfn_634875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'distfn')
    # Obtaining the member 'pmf' of a type (line 51)
    pmf_634876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), distfn_634875, 'pmf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), list_634874, pmf_634876)
    # Adding element type (line 51)
    # Getting the type of 'distfn' (line 51)
    distfn_634877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'distfn')
    # Obtaining the member 'logpmf' of a type (line 51)
    logpmf_634878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), distfn_634877, 'logpmf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), list_634874, logpmf_634878)
    # Adding element type (line 51)
    # Getting the type of 'distfn' (line 51)
    distfn_634879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 44), 'distfn')
    # Obtaining the member 'cdf' of a type (line 51)
    cdf_634880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 44), distfn_634879, 'cdf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), list_634874, cdf_634880)
    # Adding element type (line 51)
    # Getting the type of 'distfn' (line 51)
    distfn_634881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 56), 'distfn')
    # Obtaining the member 'logcdf' of a type (line 51)
    logcdf_634882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 56), distfn_634881, 'logcdf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), list_634874, logcdf_634882)
    # Adding element type (line 51)
    # Getting the type of 'distfn' (line 52)
    distfn_634883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'distfn')
    # Obtaining the member 'logsf' of a type (line 52)
    logsf_634884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 17), distfn_634883, 'logsf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), list_634874, logsf_634884)
    
    # Assigning a type to the variable 'meths' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'meths', list_634874)
    
    # Assigning a Dict to a Name (line 54):
    
    # Assigning a Dict to a Name (line 54):
    
    # Obtaining an instance of the builtin type 'dict' (line 54)
    dict_634885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 54)
    # Adding element type (key, value) (line 54)
    str_634886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'str', 'randint')
    int_634887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 17), dict_634885, (str_634886, int_634887))
    # Adding element type (key, value) (line 54)
    str_634888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 33), 'str', 'hypergeom')
    int_634889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 46), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 17), dict_634885, (str_634888, int_634889))
    # Adding element type (key, value) (line 54)
    str_634890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 49), 'str', 'bernoulli')
    int_634891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 62), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 17), dict_634885, (str_634890, int_634891))
    
    # Assigning a type to the variable 'spec_k' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'spec_k', dict_634885)
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to get(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'distname' (line 55)
    distname_634894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'distname', False)
    int_634895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 33), 'int')
    # Processing the call keyword arguments (line 55)
    kwargs_634896 = {}
    # Getting the type of 'spec_k' (line 55)
    spec_k_634892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'spec_k', False)
    # Obtaining the member 'get' of a type (line 55)
    get_634893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 12), spec_k_634892, 'get')
    # Calling get(args, kwargs) (line 55)
    get_call_result_634897 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), get_634893, *[distname_634894, int_634895], **kwargs_634896)
    
    # Assigning a type to the variable 'k' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'k', get_call_result_634897)
    
    # Call to check_named_args(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'distfn' (line 56)
    distfn_634899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'distfn', False)
    # Getting the type of 'k' (line 56)
    k_634900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'k', False)
    # Getting the type of 'arg' (line 56)
    arg_634901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'arg', False)
    # Getting the type of 'locscale_defaults' (line 56)
    locscale_defaults_634902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'locscale_defaults', False)
    # Getting the type of 'meths' (line 56)
    meths_634903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 60), 'meths', False)
    # Processing the call keyword arguments (line 56)
    kwargs_634904 = {}
    # Getting the type of 'check_named_args' (line 56)
    check_named_args_634898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'check_named_args', False)
    # Calling check_named_args(args, kwargs) (line 56)
    check_named_args_call_result_634905 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), check_named_args_634898, *[distfn_634899, k_634900, arg_634901, locscale_defaults_634902, meths_634903], **kwargs_634904)
    
    
    
    # Getting the type of 'distname' (line 57)
    distname_634906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'distname')
    str_634907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'str', 'sample distribution')
    # Applying the binary operator '!=' (line 57)
    result_ne_634908 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 11), '!=', distname_634906, str_634907)
    
    # Testing the type of an if condition (line 57)
    if_condition_634909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), result_ne_634908)
    # Assigning a type to the variable 'if_condition_634909' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_634909', if_condition_634909)
    # SSA begins for if statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to check_scale_docstring(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'distfn' (line 58)
    distfn_634911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'distfn', False)
    # Processing the call keyword arguments (line 58)
    kwargs_634912 = {}
    # Getting the type of 'check_scale_docstring' (line 58)
    check_scale_docstring_634910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'check_scale_docstring', False)
    # Calling check_scale_docstring(args, kwargs) (line 58)
    check_scale_docstring_call_result_634913 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), check_scale_docstring_634910, *[distfn_634911], **kwargs_634912)
    
    # SSA join for if statement (line 57)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_random_state_property(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'distfn' (line 59)
    distfn_634915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'distfn', False)
    # Getting the type of 'arg' (line 59)
    arg_634916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), 'arg', False)
    # Processing the call keyword arguments (line 59)
    kwargs_634917 = {}
    # Getting the type of 'check_random_state_property' (line 59)
    check_random_state_property_634914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'check_random_state_property', False)
    # Calling check_random_state_property(args, kwargs) (line 59)
    check_random_state_property_call_result_634918 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), check_random_state_property_634914, *[distfn_634915, arg_634916], **kwargs_634917)
    
    
    # Call to check_pickling(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'distfn' (line 60)
    distfn_634920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'distfn', False)
    # Getting the type of 'arg' (line 60)
    arg_634921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'arg', False)
    # Processing the call keyword arguments (line 60)
    kwargs_634922 = {}
    # Getting the type of 'check_pickling' (line 60)
    check_pickling_634919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'check_pickling', False)
    # Calling check_pickling(args, kwargs) (line 60)
    check_pickling_call_result_634923 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), check_pickling_634919, *[distfn_634920, arg_634921], **kwargs_634922)
    
    
    # Call to check_entropy(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'distfn' (line 63)
    distfn_634925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'distfn', False)
    # Getting the type of 'arg' (line 63)
    arg_634926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'arg', False)
    # Getting the type of 'distname' (line 63)
    distname_634927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 35), 'distname', False)
    # Processing the call keyword arguments (line 63)
    kwargs_634928 = {}
    # Getting the type of 'check_entropy' (line 63)
    check_entropy_634924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'check_entropy', False)
    # Calling check_entropy(args, kwargs) (line 63)
    check_entropy_call_result_634929 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), check_entropy_634924, *[distfn_634925, arg_634926, distname_634927], **kwargs_634928)
    
    
    
    # Getting the type of 'distfn' (line 64)
    distfn_634930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'distfn')
    # Obtaining the member '__class__' of a type (line 64)
    class___634931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), distfn_634930, '__class__')
    # Obtaining the member '_entropy' of a type (line 64)
    _entropy_634932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), class___634931, '_entropy')
    # Getting the type of 'stats' (line 64)
    stats_634933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'stats')
    # Obtaining the member 'rv_discrete' of a type (line 64)
    rv_discrete_634934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 40), stats_634933, 'rv_discrete')
    # Obtaining the member '_entropy' of a type (line 64)
    _entropy_634935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 40), rv_discrete_634934, '_entropy')
    # Applying the binary operator '!=' (line 64)
    result_ne_634936 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '!=', _entropy_634932, _entropy_634935)
    
    # Testing the type of an if condition (line 64)
    if_condition_634937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_ne_634936)
    # Assigning a type to the variable 'if_condition_634937' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_634937', if_condition_634937)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to check_private_entropy(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'distfn' (line 65)
    distfn_634939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'distfn', False)
    # Getting the type of 'arg' (line 65)
    arg_634940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'arg', False)
    # Getting the type of 'stats' (line 65)
    stats_634941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 47), 'stats', False)
    # Obtaining the member 'rv_discrete' of a type (line 65)
    rv_discrete_634942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 47), stats_634941, 'rv_discrete')
    # Processing the call keyword arguments (line 65)
    kwargs_634943 = {}
    # Getting the type of 'check_private_entropy' (line 65)
    check_private_entropy_634938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'check_private_entropy', False)
    # Calling check_private_entropy(args, kwargs) (line 65)
    check_private_entropy_call_result_634944 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), check_private_entropy_634938, *[distfn_634939, arg_634940, rv_discrete_634942], **kwargs_634943)
    
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_discrete_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_discrete_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_634945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634945)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_discrete_basic'
    return stypy_return_type_634945

# Assigning a type to the variable 'test_discrete_basic' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'test_discrete_basic', test_discrete_basic)

@norecursion
def test_moments(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_moments'
    module_type_store = module_type_store.open_function_context('test_moments', 68, 0, False)
    
    # Passed parameters checking function
    test_moments.stypy_localization = localization
    test_moments.stypy_type_of_self = None
    test_moments.stypy_type_store = module_type_store
    test_moments.stypy_function_name = 'test_moments'
    test_moments.stypy_param_names_list = ['distname', 'arg']
    test_moments.stypy_varargs_param_name = None
    test_moments.stypy_kwargs_param_name = None
    test_moments.stypy_call_defaults = defaults
    test_moments.stypy_call_varargs = varargs
    test_moments.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_moments', ['distname', 'arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_moments', localization, ['distname', 'arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_moments(...)' code ##################

    
    
    # SSA begins for try-except statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to getattr(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'stats' (line 71)
    stats_634947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'stats', False)
    # Getting the type of 'distname' (line 71)
    distname_634948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'distname', False)
    # Processing the call keyword arguments (line 71)
    kwargs_634949 = {}
    # Getting the type of 'getattr' (line 71)
    getattr_634946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'getattr', False)
    # Calling getattr(args, kwargs) (line 71)
    getattr_call_result_634950 = invoke(stypy.reporting.localization.Localization(__file__, 71, 17), getattr_634946, *[stats_634947, distname_634948], **kwargs_634949)
    
    # Assigning a type to the variable 'distfn' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'distfn', getattr_call_result_634950)
    # SSA branch for the except part of a try statement (line 70)
    # SSA branch for the except 'TypeError' branch of a try statement (line 70)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 73):
    
    # Assigning a Name to a Name (line 73):
    # Getting the type of 'distname' (line 73)
    distname_634951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'distname')
    # Assigning a type to the variable 'distfn' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'distfn', distname_634951)
    
    # Assigning a Str to a Name (line 74):
    
    # Assigning a Str to a Name (line 74):
    str_634952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'str', 'sample distribution')
    # Assigning a type to the variable 'distname' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'distname', str_634952)
    # SSA join for try-except statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 75):
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_634953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'int')
    
    # Call to stats(...): (line 75)
    # Getting the type of 'arg' (line 75)
    arg_634956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'arg', False)
    # Processing the call keyword arguments (line 75)
    str_634957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 44), 'str', 'mvsk')
    keyword_634958 = str_634957
    kwargs_634959 = {'moments': keyword_634958}
    # Getting the type of 'distfn' (line 75)
    distfn_634954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'distfn', False)
    # Obtaining the member 'stats' of a type (line 75)
    stats_634955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 17), distfn_634954, 'stats')
    # Calling stats(args, kwargs) (line 75)
    stats_call_result_634960 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), stats_634955, *[arg_634956], **kwargs_634959)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___634961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), stats_call_result_634960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_634962 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___634961, int_634953)
    
    # Assigning a type to the variable 'tuple_var_assignment_634721' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_634721', subscript_call_result_634962)
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_634963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'int')
    
    # Call to stats(...): (line 75)
    # Getting the type of 'arg' (line 75)
    arg_634966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'arg', False)
    # Processing the call keyword arguments (line 75)
    str_634967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 44), 'str', 'mvsk')
    keyword_634968 = str_634967
    kwargs_634969 = {'moments': keyword_634968}
    # Getting the type of 'distfn' (line 75)
    distfn_634964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'distfn', False)
    # Obtaining the member 'stats' of a type (line 75)
    stats_634965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 17), distfn_634964, 'stats')
    # Calling stats(args, kwargs) (line 75)
    stats_call_result_634970 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), stats_634965, *[arg_634966], **kwargs_634969)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___634971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), stats_call_result_634970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_634972 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___634971, int_634963)
    
    # Assigning a type to the variable 'tuple_var_assignment_634722' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_634722', subscript_call_result_634972)
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_634973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'int')
    
    # Call to stats(...): (line 75)
    # Getting the type of 'arg' (line 75)
    arg_634976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'arg', False)
    # Processing the call keyword arguments (line 75)
    str_634977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 44), 'str', 'mvsk')
    keyword_634978 = str_634977
    kwargs_634979 = {'moments': keyword_634978}
    # Getting the type of 'distfn' (line 75)
    distfn_634974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'distfn', False)
    # Obtaining the member 'stats' of a type (line 75)
    stats_634975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 17), distfn_634974, 'stats')
    # Calling stats(args, kwargs) (line 75)
    stats_call_result_634980 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), stats_634975, *[arg_634976], **kwargs_634979)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___634981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), stats_call_result_634980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_634982 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___634981, int_634973)
    
    # Assigning a type to the variable 'tuple_var_assignment_634723' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_634723', subscript_call_result_634982)
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    int_634983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'int')
    
    # Call to stats(...): (line 75)
    # Getting the type of 'arg' (line 75)
    arg_634986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'arg', False)
    # Processing the call keyword arguments (line 75)
    str_634987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 44), 'str', 'mvsk')
    keyword_634988 = str_634987
    kwargs_634989 = {'moments': keyword_634988}
    # Getting the type of 'distfn' (line 75)
    distfn_634984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'distfn', False)
    # Obtaining the member 'stats' of a type (line 75)
    stats_634985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 17), distfn_634984, 'stats')
    # Calling stats(args, kwargs) (line 75)
    stats_call_result_634990 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), stats_634985, *[arg_634986], **kwargs_634989)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___634991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), stats_call_result_634990, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_634992 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), getitem___634991, int_634983)
    
    # Assigning a type to the variable 'tuple_var_assignment_634724' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_634724', subscript_call_result_634992)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_634721' (line 75)
    tuple_var_assignment_634721_634993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_634721')
    # Assigning a type to the variable 'm' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'm', tuple_var_assignment_634721_634993)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_634722' (line 75)
    tuple_var_assignment_634722_634994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_634722')
    # Assigning a type to the variable 'v' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), 'v', tuple_var_assignment_634722_634994)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_634723' (line 75)
    tuple_var_assignment_634723_634995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_634723')
    # Assigning a type to the variable 's' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 10), 's', tuple_var_assignment_634723_634995)
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'tuple_var_assignment_634724' (line 75)
    tuple_var_assignment_634724_634996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'tuple_var_assignment_634724')
    # Assigning a type to the variable 'k' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'k', tuple_var_assignment_634724_634996)
    
    # Call to check_normalization(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'distfn' (line 76)
    distfn_634998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'distfn', False)
    # Getting the type of 'arg' (line 76)
    arg_634999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'arg', False)
    # Getting the type of 'distname' (line 76)
    distname_635000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 37), 'distname', False)
    # Processing the call keyword arguments (line 76)
    kwargs_635001 = {}
    # Getting the type of 'check_normalization' (line 76)
    check_normalization_634997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'check_normalization', False)
    # Calling check_normalization(args, kwargs) (line 76)
    check_normalization_call_result_635002 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), check_normalization_634997, *[distfn_634998, arg_634999, distname_635000], **kwargs_635001)
    
    
    # Call to check_moment(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'distfn' (line 79)
    distfn_635004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'distfn', False)
    # Getting the type of 'arg' (line 79)
    arg_635005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'arg', False)
    # Getting the type of 'm' (line 79)
    m_635006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'm', False)
    # Getting the type of 'v' (line 79)
    v_635007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'v', False)
    # Getting the type of 'distname' (line 79)
    distname_635008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 36), 'distname', False)
    # Processing the call keyword arguments (line 79)
    kwargs_635009 = {}
    # Getting the type of 'check_moment' (line 79)
    check_moment_635003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'check_moment', False)
    # Calling check_moment(args, kwargs) (line 79)
    check_moment_call_result_635010 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), check_moment_635003, *[distfn_635004, arg_635005, m_635006, v_635007, distname_635008], **kwargs_635009)
    
    
    # Call to check_mean_expect(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'distfn' (line 80)
    distfn_635012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'distfn', False)
    # Getting the type of 'arg' (line 80)
    arg_635013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'arg', False)
    # Getting the type of 'm' (line 80)
    m_635014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'm', False)
    # Getting the type of 'distname' (line 80)
    distname_635015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 38), 'distname', False)
    # Processing the call keyword arguments (line 80)
    kwargs_635016 = {}
    # Getting the type of 'check_mean_expect' (line 80)
    check_mean_expect_635011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'check_mean_expect', False)
    # Calling check_mean_expect(args, kwargs) (line 80)
    check_mean_expect_call_result_635017 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), check_mean_expect_635011, *[distfn_635012, arg_635013, m_635014, distname_635015], **kwargs_635016)
    
    
    # Call to check_var_expect(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'distfn' (line 81)
    distfn_635019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'distfn', False)
    # Getting the type of 'arg' (line 81)
    arg_635020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'arg', False)
    # Getting the type of 'm' (line 81)
    m_635021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 34), 'm', False)
    # Getting the type of 'v' (line 81)
    v_635022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 37), 'v', False)
    # Getting the type of 'distname' (line 81)
    distname_635023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'distname', False)
    # Processing the call keyword arguments (line 81)
    kwargs_635024 = {}
    # Getting the type of 'check_var_expect' (line 81)
    check_var_expect_635018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'check_var_expect', False)
    # Calling check_var_expect(args, kwargs) (line 81)
    check_var_expect_call_result_635025 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), check_var_expect_635018, *[distfn_635019, arg_635020, m_635021, v_635022, distname_635023], **kwargs_635024)
    
    
    # Call to check_skew_expect(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'distfn' (line 82)
    distfn_635027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'distfn', False)
    # Getting the type of 'arg' (line 82)
    arg_635028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'arg', False)
    # Getting the type of 'm' (line 82)
    m_635029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 35), 'm', False)
    # Getting the type of 'v' (line 82)
    v_635030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'v', False)
    # Getting the type of 's' (line 82)
    s_635031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 's', False)
    # Getting the type of 'distname' (line 82)
    distname_635032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 44), 'distname', False)
    # Processing the call keyword arguments (line 82)
    kwargs_635033 = {}
    # Getting the type of 'check_skew_expect' (line 82)
    check_skew_expect_635026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'check_skew_expect', False)
    # Calling check_skew_expect(args, kwargs) (line 82)
    check_skew_expect_call_result_635034 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), check_skew_expect_635026, *[distfn_635027, arg_635028, m_635029, v_635030, s_635031, distname_635032], **kwargs_635033)
    
    
    
    # Getting the type of 'distname' (line 83)
    distname_635035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 7), 'distname')
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_635036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    str_635037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'str', 'zipf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 23), list_635036, str_635037)
    
    # Applying the binary operator 'notin' (line 83)
    result_contains_635038 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 7), 'notin', distname_635035, list_635036)
    
    # Testing the type of an if condition (line 83)
    if_condition_635039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 4), result_contains_635038)
    # Assigning a type to the variable 'if_condition_635039' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'if_condition_635039', if_condition_635039)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to check_kurt_expect(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'distfn' (line 84)
    distfn_635041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'distfn', False)
    # Getting the type of 'arg' (line 84)
    arg_635042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), 'arg', False)
    # Getting the type of 'm' (line 84)
    m_635043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 39), 'm', False)
    # Getting the type of 'v' (line 84)
    v_635044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'v', False)
    # Getting the type of 'k' (line 84)
    k_635045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 45), 'k', False)
    # Getting the type of 'distname' (line 84)
    distname_635046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 48), 'distname', False)
    # Processing the call keyword arguments (line 84)
    kwargs_635047 = {}
    # Getting the type of 'check_kurt_expect' (line 84)
    check_kurt_expect_635040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'check_kurt_expect', False)
    # Calling check_kurt_expect(args, kwargs) (line 84)
    check_kurt_expect_call_result_635048 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), check_kurt_expect_635040, *[distfn_635041, arg_635042, m_635043, v_635044, k_635045, distname_635046], **kwargs_635047)
    
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to check_moment_frozen(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'distfn' (line 87)
    distfn_635050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'distfn', False)
    # Getting the type of 'arg' (line 87)
    arg_635051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'arg', False)
    # Getting the type of 'm' (line 87)
    m_635052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'm', False)
    int_635053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 40), 'int')
    # Processing the call keyword arguments (line 87)
    kwargs_635054 = {}
    # Getting the type of 'check_moment_frozen' (line 87)
    check_moment_frozen_635049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'check_moment_frozen', False)
    # Calling check_moment_frozen(args, kwargs) (line 87)
    check_moment_frozen_call_result_635055 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), check_moment_frozen_635049, *[distfn_635050, arg_635051, m_635052, int_635053], **kwargs_635054)
    
    
    # Call to check_moment_frozen(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'distfn' (line 88)
    distfn_635057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'distfn', False)
    # Getting the type of 'arg' (line 88)
    arg_635058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'arg', False)
    # Getting the type of 'v' (line 88)
    v_635059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 37), 'v', False)
    # Getting the type of 'm' (line 88)
    m_635060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 39), 'm', False)
    # Getting the type of 'm' (line 88)
    m_635061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 41), 'm', False)
    # Applying the binary operator '*' (line 88)
    result_mul_635062 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 39), '*', m_635060, m_635061)
    
    # Applying the binary operator '+' (line 88)
    result_add_635063 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 37), '+', v_635059, result_mul_635062)
    
    int_635064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 44), 'int')
    # Processing the call keyword arguments (line 88)
    kwargs_635065 = {}
    # Getting the type of 'check_moment_frozen' (line 88)
    check_moment_frozen_635056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'check_moment_frozen', False)
    # Calling check_moment_frozen(args, kwargs) (line 88)
    check_moment_frozen_call_result_635066 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), check_moment_frozen_635056, *[distfn_635057, arg_635058, result_add_635063, int_635064], **kwargs_635065)
    
    
    # ################# End of 'test_moments(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_moments' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_635067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635067)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_moments'
    return stypy_return_type_635067

# Assigning a type to the variable 'test_moments' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'test_moments', test_moments)

@norecursion
def test_rvs_broadcast(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rvs_broadcast'
    module_type_store = module_type_store.open_function_context('test_rvs_broadcast', 91, 0, False)
    
    # Passed parameters checking function
    test_rvs_broadcast.stypy_localization = localization
    test_rvs_broadcast.stypy_type_of_self = None
    test_rvs_broadcast.stypy_type_store = module_type_store
    test_rvs_broadcast.stypy_function_name = 'test_rvs_broadcast'
    test_rvs_broadcast.stypy_param_names_list = ['dist', 'shape_args']
    test_rvs_broadcast.stypy_varargs_param_name = None
    test_rvs_broadcast.stypy_kwargs_param_name = None
    test_rvs_broadcast.stypy_call_defaults = defaults
    test_rvs_broadcast.stypy_call_varargs = varargs
    test_rvs_broadcast.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_rvs_broadcast', ['dist', 'shape_args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_rvs_broadcast', localization, ['dist', 'shape_args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_rvs_broadcast(...)' code ##################

    
    # Assigning a Compare to a Name (line 103):
    
    # Assigning a Compare to a Name (line 103):
    
    # Getting the type of 'dist' (line 103)
    dist_635068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'dist')
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_635069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    str_635070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'str', 'skellam')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 25), list_635069, str_635070)
    
    # Applying the binary operator 'in' (line 103)
    result_contains_635071 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 17), 'in', dist_635068, list_635069)
    
    # Assigning a type to the variable 'shape_only' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'shape_only', result_contains_635071)
    
    
    # SSA begins for try-except statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to getattr(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'stats' (line 106)
    stats_635073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'stats', False)
    # Getting the type of 'dist' (line 106)
    dist_635074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'dist', False)
    # Processing the call keyword arguments (line 106)
    kwargs_635075 = {}
    # Getting the type of 'getattr' (line 106)
    getattr_635072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'getattr', False)
    # Calling getattr(args, kwargs) (line 106)
    getattr_call_result_635076 = invoke(stypy.reporting.localization.Localization(__file__, 106, 19), getattr_635072, *[stats_635073, dist_635074], **kwargs_635075)
    
    # Assigning a type to the variable 'distfunc' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'distfunc', getattr_call_result_635076)
    # SSA branch for the except part of a try statement (line 105)
    # SSA branch for the except 'TypeError' branch of a try statement (line 105)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 108):
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'dist' (line 108)
    dist_635077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'dist')
    # Assigning a type to the variable 'distfunc' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'distfunc', dist_635077)
    
    # Assigning a BinOp to a Name (line 109):
    
    # Assigning a BinOp to a Name (line 109):
    str_635078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 15), 'str', 'rv_discrete(values=(%r, %r))')
    
    # Obtaining an instance of the builtin type 'tuple' (line 109)
    tuple_635079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 109)
    # Adding element type (line 109)
    # Getting the type of 'dist' (line 109)
    dist_635080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'dist')
    # Obtaining the member 'xk' of a type (line 109)
    xk_635081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 49), dist_635080, 'xk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 49), tuple_635079, xk_635081)
    # Adding element type (line 109)
    # Getting the type of 'dist' (line 109)
    dist_635082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 58), 'dist')
    # Obtaining the member 'pk' of a type (line 109)
    pk_635083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 58), dist_635082, 'pk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 49), tuple_635079, pk_635083)
    
    # Applying the binary operator '%' (line 109)
    result_mod_635084 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 15), '%', str_635078, tuple_635079)
    
    # Assigning a type to the variable 'dist' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'dist', result_mod_635084)
    # SSA join for try-except statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to zeros(...): (line 110)
    # Processing the call arguments (line 110)
    int_635087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 19), 'int')
    # Processing the call keyword arguments (line 110)
    kwargs_635088 = {}
    # Getting the type of 'np' (line 110)
    np_635085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 110)
    zeros_635086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 10), np_635085, 'zeros')
    # Calling zeros(args, kwargs) (line 110)
    zeros_call_result_635089 = invoke(stypy.reporting.localization.Localization(__file__, 110, 10), zeros_635086, *[int_635087], **kwargs_635088)
    
    # Assigning a type to the variable 'loc' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'loc', zeros_call_result_635089)
    
    # Assigning a Attribute to a Name (line 111):
    
    # Assigning a Attribute to a Name (line 111):
    # Getting the type of 'distfunc' (line 111)
    distfunc_635090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'distfunc')
    # Obtaining the member 'numargs' of a type (line 111)
    numargs_635091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), distfunc_635090, 'numargs')
    # Assigning a type to the variable 'nargs' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'nargs', numargs_635091)
    
    # Assigning a List to a Name (line 112):
    
    # Assigning a List to a Name (line 112):
    
    # Obtaining an instance of the builtin type 'list' (line 112)
    list_635092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 112)
    
    # Assigning a type to the variable 'allargs' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'allargs', list_635092)
    
    # Assigning a List to a Name (line 113):
    
    # Assigning a List to a Name (line 113):
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_635093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    
    # Assigning a type to the variable 'bshape' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'bshape', list_635093)
    
    
    # Call to range(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'nargs' (line 115)
    nargs_635095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'nargs', False)
    # Processing the call keyword arguments (line 115)
    kwargs_635096 = {}
    # Getting the type of 'range' (line 115)
    range_635094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'range', False)
    # Calling range(args, kwargs) (line 115)
    range_call_result_635097 = invoke(stypy.reporting.localization.Localization(__file__, 115, 13), range_635094, *[nargs_635095], **kwargs_635096)
    
    # Testing the type of a for loop iterable (line 115)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 4), range_call_result_635097)
    # Getting the type of the for loop variable (line 115)
    for_loop_var_635098 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 4), range_call_result_635097)
    # Assigning a type to the variable 'k' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'k', for_loop_var_635098)
    # SSA begins for a for statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 116):
    
    # Assigning a BinOp to a Name (line 116):
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_635099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    # Getting the type of 'k' (line 116)
    k_635100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'k')
    int_635101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'int')
    # Applying the binary operator '+' (line 116)
    result_add_635102 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), '+', k_635100, int_635101)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 15), tuple_635099, result_add_635102)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_635103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    int_635104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 26), tuple_635103, int_635104)
    
    # Getting the type of 'k' (line 116)
    k_635105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'k')
    int_635106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 35), 'int')
    # Applying the binary operator '+' (line 116)
    result_add_635107 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 31), '+', k_635105, int_635106)
    
    # Applying the binary operator '*' (line 116)
    result_mul_635108 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 25), '*', tuple_635103, result_add_635107)
    
    # Applying the binary operator '+' (line 116)
    result_add_635109 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 14), '+', tuple_635099, result_mul_635108)
    
    # Assigning a type to the variable 'shp' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'shp', result_add_635109)
    
    # Assigning a Subscript to a Name (line 117):
    
    # Assigning a Subscript to a Name (line 117):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 117)
    k_635110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 31), 'k')
    # Getting the type of 'shape_args' (line 117)
    shape_args_635111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'shape_args')
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___635112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 20), shape_args_635111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_635113 = invoke(stypy.reporting.localization.Localization(__file__, 117, 20), getitem___635112, k_635110)
    
    # Assigning a type to the variable 'param_val' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'param_val', subscript_call_result_635113)
    
    # Call to append(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'param_val' (line 118)
    param_val_635116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'param_val', False)
    
    # Call to ones(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'shp' (line 118)
    shp_635119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 41), 'shp', False)
    # Processing the call keyword arguments (line 118)
    
    # Call to array(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'param_val' (line 118)
    param_val_635122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 61), 'param_val', False)
    # Processing the call keyword arguments (line 118)
    kwargs_635123 = {}
    # Getting the type of 'np' (line 118)
    np_635120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 52), 'np', False)
    # Obtaining the member 'array' of a type (line 118)
    array_635121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 52), np_635120, 'array')
    # Calling array(args, kwargs) (line 118)
    array_call_result_635124 = invoke(stypy.reporting.localization.Localization(__file__, 118, 52), array_635121, *[param_val_635122], **kwargs_635123)
    
    # Obtaining the member 'dtype' of a type (line 118)
    dtype_635125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 52), array_call_result_635124, 'dtype')
    keyword_635126 = dtype_635125
    kwargs_635127 = {'dtype': keyword_635126}
    # Getting the type of 'np' (line 118)
    np_635117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'np', False)
    # Obtaining the member 'ones' of a type (line 118)
    ones_635118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 33), np_635117, 'ones')
    # Calling ones(args, kwargs) (line 118)
    ones_call_result_635128 = invoke(stypy.reporting.localization.Localization(__file__, 118, 33), ones_635118, *[shp_635119], **kwargs_635127)
    
    # Applying the binary operator '*' (line 118)
    result_mul_635129 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 23), '*', param_val_635116, ones_call_result_635128)
    
    # Processing the call keyword arguments (line 118)
    kwargs_635130 = {}
    # Getting the type of 'allargs' (line 118)
    allargs_635114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'allargs', False)
    # Obtaining the member 'append' of a type (line 118)
    append_635115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), allargs_635114, 'append')
    # Calling append(args, kwargs) (line 118)
    append_call_result_635131 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), append_635115, *[result_mul_635129], **kwargs_635130)
    
    
    # Call to insert(...): (line 119)
    # Processing the call arguments (line 119)
    int_635134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 22), 'int')
    
    # Obtaining the type of the subscript
    int_635135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 29), 'int')
    # Getting the type of 'shp' (line 119)
    shp_635136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'shp', False)
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___635137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 25), shp_635136, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_635138 = invoke(stypy.reporting.localization.Localization(__file__, 119, 25), getitem___635137, int_635135)
    
    # Processing the call keyword arguments (line 119)
    kwargs_635139 = {}
    # Getting the type of 'bshape' (line 119)
    bshape_635132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'bshape', False)
    # Obtaining the member 'insert' of a type (line 119)
    insert_635133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), bshape_635132, 'insert')
    # Calling insert(args, kwargs) (line 119)
    insert_call_result_635140 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), insert_635133, *[int_635134, subscript_call_result_635138], **kwargs_635139)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'loc' (line 120)
    loc_635143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'loc', False)
    # Processing the call keyword arguments (line 120)
    kwargs_635144 = {}
    # Getting the type of 'allargs' (line 120)
    allargs_635141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'allargs', False)
    # Obtaining the member 'append' of a type (line 120)
    append_635142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 4), allargs_635141, 'append')
    # Calling append(args, kwargs) (line 120)
    append_call_result_635145 = invoke(stypy.reporting.localization.Localization(__file__, 120, 4), append_635142, *[loc_635143], **kwargs_635144)
    
    
    # Call to append(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'loc' (line 121)
    loc_635148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'loc', False)
    # Obtaining the member 'size' of a type (line 121)
    size_635149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 18), loc_635148, 'size')
    # Processing the call keyword arguments (line 121)
    kwargs_635150 = {}
    # Getting the type of 'bshape' (line 121)
    bshape_635146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'bshape', False)
    # Obtaining the member 'append' of a type (line 121)
    append_635147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), bshape_635146, 'append')
    # Calling append(args, kwargs) (line 121)
    append_call_result_635151 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), append_635147, *[size_635149], **kwargs_635150)
    
    
    # Call to check_rvs_broadcast(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'distfunc' (line 124)
    distfunc_635153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'distfunc', False)
    # Getting the type of 'dist' (line 124)
    dist_635154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'dist', False)
    # Getting the type of 'allargs' (line 124)
    allargs_635155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'allargs', False)
    # Getting the type of 'bshape' (line 124)
    bshape_635156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 49), 'bshape', False)
    # Getting the type of 'shape_only' (line 124)
    shape_only_635157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 57), 'shape_only', False)
    
    # Obtaining an instance of the builtin type 'list' (line 124)
    list_635158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 69), 'list')
    # Adding type elements to the builtin type 'list' instance (line 124)
    # Adding element type (line 124)
    # Getting the type of 'np' (line 124)
    np_635159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 70), 'np', False)
    # Obtaining the member 'int_' of a type (line 124)
    int__635160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 70), np_635159, 'int_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 69), list_635158, int__635160)
    
    # Processing the call keyword arguments (line 124)
    kwargs_635161 = {}
    # Getting the type of 'check_rvs_broadcast' (line 124)
    check_rvs_broadcast_635152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'check_rvs_broadcast', False)
    # Calling check_rvs_broadcast(args, kwargs) (line 124)
    check_rvs_broadcast_call_result_635162 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), check_rvs_broadcast_635152, *[distfunc_635153, dist_635154, allargs_635155, bshape_635156, shape_only_635157, list_635158], **kwargs_635161)
    
    
    # ################# End of 'test_rvs_broadcast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rvs_broadcast' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_635163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635163)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rvs_broadcast'
    return stypy_return_type_635163

# Assigning a type to the variable 'test_rvs_broadcast' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'test_rvs_broadcast', test_rvs_broadcast)

@norecursion
def check_cdf_ppf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_cdf_ppf'
    module_type_store = module_type_store.open_function_context('check_cdf_ppf', 127, 0, False)
    
    # Passed parameters checking function
    check_cdf_ppf.stypy_localization = localization
    check_cdf_ppf.stypy_type_of_self = None
    check_cdf_ppf.stypy_type_store = module_type_store
    check_cdf_ppf.stypy_function_name = 'check_cdf_ppf'
    check_cdf_ppf.stypy_param_names_list = ['distfn', 'arg', 'supp', 'msg']
    check_cdf_ppf.stypy_varargs_param_name = None
    check_cdf_ppf.stypy_kwargs_param_name = None
    check_cdf_ppf.stypy_call_defaults = defaults
    check_cdf_ppf.stypy_call_varargs = varargs
    check_cdf_ppf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_cdf_ppf', ['distfn', 'arg', 'supp', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_cdf_ppf', localization, ['distfn', 'arg', 'supp', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_cdf_ppf(...)' code ##################

    
    # Call to assert_array_equal(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Call to ppf(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Call to cdf(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'supp' (line 129)
    supp_635170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'supp', False)
    # Getting the type of 'arg' (line 129)
    arg_635171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 56), 'arg', False)
    # Processing the call keyword arguments (line 129)
    kwargs_635172 = {}
    # Getting the type of 'distfn' (line 129)
    distfn_635168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 129)
    cdf_635169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 38), distfn_635168, 'cdf')
    # Calling cdf(args, kwargs) (line 129)
    cdf_call_result_635173 = invoke(stypy.reporting.localization.Localization(__file__, 129, 38), cdf_635169, *[supp_635170, arg_635171], **kwargs_635172)
    
    # Getting the type of 'arg' (line 129)
    arg_635174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'arg', False)
    # Processing the call keyword arguments (line 129)
    kwargs_635175 = {}
    # Getting the type of 'distfn' (line 129)
    distfn_635166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 129)
    ppf_635167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 27), distfn_635166, 'ppf')
    # Calling ppf(args, kwargs) (line 129)
    ppf_call_result_635176 = invoke(stypy.reporting.localization.Localization(__file__, 129, 27), ppf_635167, *[cdf_call_result_635173, arg_635174], **kwargs_635175)
    
    # Getting the type of 'supp' (line 130)
    supp_635177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'supp', False)
    # Getting the type of 'msg' (line 130)
    msg_635178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'msg', False)
    str_635179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 39), 'str', '-roundtrip')
    # Applying the binary operator '+' (line 130)
    result_add_635180 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 33), '+', msg_635178, str_635179)
    
    # Processing the call keyword arguments (line 129)
    kwargs_635181 = {}
    # Getting the type of 'npt' (line 129)
    npt_635164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'npt', False)
    # Obtaining the member 'assert_array_equal' of a type (line 129)
    assert_array_equal_635165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 4), npt_635164, 'assert_array_equal')
    # Calling assert_array_equal(args, kwargs) (line 129)
    assert_array_equal_call_result_635182 = invoke(stypy.reporting.localization.Localization(__file__, 129, 4), assert_array_equal_635165, *[ppf_call_result_635176, supp_635177, result_add_635180], **kwargs_635181)
    
    
    # Call to assert_array_equal(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Call to ppf(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Call to cdf(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'supp' (line 131)
    supp_635189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 49), 'supp', False)
    # Getting the type of 'arg' (line 131)
    arg_635190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 56), 'arg', False)
    # Processing the call keyword arguments (line 131)
    kwargs_635191 = {}
    # Getting the type of 'distfn' (line 131)
    distfn_635187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 38), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 131)
    cdf_635188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 38), distfn_635187, 'cdf')
    # Calling cdf(args, kwargs) (line 131)
    cdf_call_result_635192 = invoke(stypy.reporting.localization.Localization(__file__, 131, 38), cdf_635188, *[supp_635189, arg_635190], **kwargs_635191)
    
    float_635193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 63), 'float')
    # Applying the binary operator '-' (line 131)
    result_sub_635194 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 38), '-', cdf_call_result_635192, float_635193)
    
    # Getting the type of 'arg' (line 131)
    arg_635195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 70), 'arg', False)
    # Processing the call keyword arguments (line 131)
    kwargs_635196 = {}
    # Getting the type of 'distfn' (line 131)
    distfn_635185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 131)
    ppf_635186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 27), distfn_635185, 'ppf')
    # Calling ppf(args, kwargs) (line 131)
    ppf_call_result_635197 = invoke(stypy.reporting.localization.Localization(__file__, 131, 27), ppf_635186, *[result_sub_635194, arg_635195], **kwargs_635196)
    
    # Getting the type of 'supp' (line 132)
    supp_635198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 27), 'supp', False)
    # Getting the type of 'msg' (line 132)
    msg_635199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'msg', False)
    str_635200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 39), 'str', '-roundtrip')
    # Applying the binary operator '+' (line 132)
    result_add_635201 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 33), '+', msg_635199, str_635200)
    
    # Processing the call keyword arguments (line 131)
    kwargs_635202 = {}
    # Getting the type of 'npt' (line 131)
    npt_635183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'npt', False)
    # Obtaining the member 'assert_array_equal' of a type (line 131)
    assert_array_equal_635184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), npt_635183, 'assert_array_equal')
    # Calling assert_array_equal(args, kwargs) (line 131)
    assert_array_equal_call_result_635203 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), assert_array_equal_635184, *[ppf_call_result_635197, supp_635198, result_add_635201], **kwargs_635202)
    
    
    # Type idiom detected: calculating its left and rigth part (line 134)
    str_635204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 27), 'str', 'xk')
    # Getting the type of 'distfn' (line 134)
    distfn_635205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 19), 'distfn')
    
    (may_be_635206, more_types_in_union_635207) = may_not_provide_member(str_635204, distfn_635205)

    if may_be_635206:

        if more_types_in_union_635207:
            # Runtime conditional SSA (line 134)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'distfn' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'distfn', remove_member_provider_from_union(distfn_635205, 'xk'))
        
        # Assigning a Subscript to a Name (line 135):
        
        # Assigning a Subscript to a Name (line 135):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'supp' (line 135)
        supp_635208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'supp')
        # Getting the type of 'distfn' (line 135)
        distfn_635209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'distfn')
        # Obtaining the member 'b' of a type (line 135)
        b_635210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 28), distfn_635209, 'b')
        # Applying the binary operator '<' (line 135)
        result_lt_635211 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 21), '<', supp_635208, b_635210)
        
        # Getting the type of 'supp' (line 135)
        supp_635212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'supp')
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___635213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), supp_635212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_635214 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), getitem___635213, result_lt_635211)
        
        # Assigning a type to the variable 'supp1' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'supp1', subscript_call_result_635214)
        
        # Call to assert_array_equal(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to ppf(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to cdf(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'supp1' (line 136)
        supp1_635221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 53), 'supp1', False)
        # Getting the type of 'arg' (line 136)
        arg_635222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 61), 'arg', False)
        # Processing the call keyword arguments (line 136)
        kwargs_635223 = {}
        # Getting the type of 'distfn' (line 136)
        distfn_635219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 42), 'distfn', False)
        # Obtaining the member 'cdf' of a type (line 136)
        cdf_635220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 42), distfn_635219, 'cdf')
        # Calling cdf(args, kwargs) (line 136)
        cdf_call_result_635224 = invoke(stypy.reporting.localization.Localization(__file__, 136, 42), cdf_635220, *[supp1_635221, arg_635222], **kwargs_635223)
        
        float_635225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 68), 'float')
        # Applying the binary operator '+' (line 136)
        result_add_635226 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 42), '+', cdf_call_result_635224, float_635225)
        
        # Getting the type of 'arg' (line 136)
        arg_635227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 75), 'arg', False)
        # Processing the call keyword arguments (line 136)
        kwargs_635228 = {}
        # Getting the type of 'distfn' (line 136)
        distfn_635217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'distfn', False)
        # Obtaining the member 'ppf' of a type (line 136)
        ppf_635218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 31), distfn_635217, 'ppf')
        # Calling ppf(args, kwargs) (line 136)
        ppf_call_result_635229 = invoke(stypy.reporting.localization.Localization(__file__, 136, 31), ppf_635218, *[result_add_635226, arg_635227], **kwargs_635228)
        
        # Getting the type of 'supp1' (line 137)
        supp1_635230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'supp1', False)
        # Getting the type of 'distfn' (line 137)
        distfn_635231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'distfn', False)
        # Obtaining the member 'inc' of a type (line 137)
        inc_635232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 39), distfn_635231, 'inc')
        # Applying the binary operator '+' (line 137)
        result_add_635233 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 31), '+', supp1_635230, inc_635232)
        
        # Getting the type of 'msg' (line 137)
        msg_635234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 51), 'msg', False)
        str_635235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 57), 'str', ' ppf-cdf-next')
        # Applying the binary operator '+' (line 137)
        result_add_635236 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 51), '+', msg_635234, str_635235)
        
        # Processing the call keyword arguments (line 136)
        kwargs_635237 = {}
        # Getting the type of 'npt' (line 136)
        npt_635215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'npt', False)
        # Obtaining the member 'assert_array_equal' of a type (line 136)
        assert_array_equal_635216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), npt_635215, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 136)
        assert_array_equal_call_result_635238 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), assert_array_equal_635216, *[ppf_call_result_635229, result_add_635233, result_add_635236], **kwargs_635237)
        

        if more_types_in_union_635207:
            # SSA join for if statement (line 134)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'check_cdf_ppf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_cdf_ppf' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_635239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635239)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_cdf_ppf'
    return stypy_return_type_635239

# Assigning a type to the variable 'check_cdf_ppf' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'check_cdf_ppf', check_cdf_ppf)

@norecursion
def check_pmf_cdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_pmf_cdf'
    module_type_store = module_type_store.open_function_context('check_pmf_cdf', 141, 0, False)
    
    # Passed parameters checking function
    check_pmf_cdf.stypy_localization = localization
    check_pmf_cdf.stypy_type_of_self = None
    check_pmf_cdf.stypy_type_store = module_type_store
    check_pmf_cdf.stypy_function_name = 'check_pmf_cdf'
    check_pmf_cdf.stypy_param_names_list = ['distfn', 'arg', 'distname']
    check_pmf_cdf.stypy_varargs_param_name = None
    check_pmf_cdf.stypy_kwargs_param_name = None
    check_pmf_cdf.stypy_call_defaults = defaults
    check_pmf_cdf.stypy_call_varargs = varargs
    check_pmf_cdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_pmf_cdf', ['distfn', 'arg', 'distname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_pmf_cdf', localization, ['distfn', 'arg', 'distname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_pmf_cdf(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 142)
    str_635240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 23), 'str', 'xk')
    # Getting the type of 'distfn' (line 142)
    distfn_635241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'distfn')
    
    (may_be_635242, more_types_in_union_635243) = may_provide_member(str_635240, distfn_635241)

    if may_be_635242:

        if more_types_in_union_635243:
            # Runtime conditional SSA (line 142)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'distfn' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'distfn', remove_not_member_provider_from_union(distfn_635241, 'xk'))
        
        # Assigning a Attribute to a Name (line 143):
        
        # Assigning a Attribute to a Name (line 143):
        # Getting the type of 'distfn' (line 143)
        distfn_635244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'distfn')
        # Obtaining the member 'xk' of a type (line 143)
        xk_635245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), distfn_635244, 'xk')
        # Assigning a type to the variable 'index' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'index', xk_635245)

        if more_types_in_union_635243:
            # Runtime conditional SSA for else branch (line 142)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_635242) or more_types_in_union_635243):
        # Assigning a type to the variable 'distfn' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'distfn', remove_member_provider_from_union(distfn_635241, 'xk'))
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to int(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Call to ppf(...): (line 145)
        # Processing the call arguments (line 145)
        float_635249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 34), 'float')
        # Getting the type of 'arg' (line 145)
        arg_635250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 41), 'arg', False)
        # Processing the call keyword arguments (line 145)
        kwargs_635251 = {}
        # Getting the type of 'distfn' (line 145)
        distfn_635247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 23), 'distfn', False)
        # Obtaining the member 'ppf' of a type (line 145)
        ppf_635248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 23), distfn_635247, 'ppf')
        # Calling ppf(args, kwargs) (line 145)
        ppf_call_result_635252 = invoke(stypy.reporting.localization.Localization(__file__, 145, 23), ppf_635248, *[float_635249, arg_635250], **kwargs_635251)
        
        int_635253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 48), 'int')
        # Applying the binary operator '-' (line 145)
        result_sub_635254 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 23), '-', ppf_call_result_635252, int_635253)
        
        # Processing the call keyword arguments (line 145)
        kwargs_635255 = {}
        # Getting the type of 'int' (line 145)
        int_635246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'int', False)
        # Calling int(args, kwargs) (line 145)
        int_call_result_635256 = invoke(stypy.reporting.localization.Localization(__file__, 145, 19), int_635246, *[result_sub_635254], **kwargs_635255)
        
        # Assigning a type to the variable 'startind' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'startind', int_call_result_635256)
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to list(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to range(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'startind' (line 146)
        startind_635259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'startind', False)
        # Getting the type of 'startind' (line 146)
        startind_635260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'startind', False)
        int_635261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 48), 'int')
        # Applying the binary operator '+' (line 146)
        result_add_635262 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 37), '+', startind_635260, int_635261)
        
        # Processing the call keyword arguments (line 146)
        kwargs_635263 = {}
        # Getting the type of 'range' (line 146)
        range_635258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 21), 'range', False)
        # Calling range(args, kwargs) (line 146)
        range_call_result_635264 = invoke(stypy.reporting.localization.Localization(__file__, 146, 21), range_635258, *[startind_635259, result_add_635262], **kwargs_635263)
        
        # Processing the call keyword arguments (line 146)
        kwargs_635265 = {}
        # Getting the type of 'list' (line 146)
        list_635257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'list', False)
        # Calling list(args, kwargs) (line 146)
        list_call_result_635266 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), list_635257, *[range_call_result_635264], **kwargs_635265)
        
        # Assigning a type to the variable 'index' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'index', list_call_result_635266)

        if (may_be_635242 and more_types_in_union_635243):
            # SSA join for if statement (line 142)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to cdf(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'index' (line 147)
    index_635269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'index', False)
    # Getting the type of 'arg' (line 147)
    arg_635270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'arg', False)
    # Processing the call keyword arguments (line 147)
    kwargs_635271 = {}
    # Getting the type of 'distfn' (line 147)
    distfn_635267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 147)
    cdf_635268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 11), distfn_635267, 'cdf')
    # Calling cdf(args, kwargs) (line 147)
    cdf_call_result_635272 = invoke(stypy.reporting.localization.Localization(__file__, 147, 11), cdf_635268, *[index_635269, arg_635270], **kwargs_635271)
    
    # Assigning a type to the variable 'cdfs' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'cdfs', cdf_call_result_635272)
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to cumsum(...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_635280 = {}
    
    # Call to pmf(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'index' (line 148)
    index_635275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'index', False)
    # Getting the type of 'arg' (line 148)
    arg_635276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'arg', False)
    # Processing the call keyword arguments (line 148)
    kwargs_635277 = {}
    # Getting the type of 'distfn' (line 148)
    distfn_635273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'distfn', False)
    # Obtaining the member 'pmf' of a type (line 148)
    pmf_635274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 15), distfn_635273, 'pmf')
    # Calling pmf(args, kwargs) (line 148)
    pmf_call_result_635278 = invoke(stypy.reporting.localization.Localization(__file__, 148, 15), pmf_635274, *[index_635275, arg_635276], **kwargs_635277)
    
    # Obtaining the member 'cumsum' of a type (line 148)
    cumsum_635279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 15), pmf_call_result_635278, 'cumsum')
    # Calling cumsum(args, kwargs) (line 148)
    cumsum_call_result_635281 = invoke(stypy.reporting.localization.Localization(__file__, 148, 15), cumsum_635279, *[], **kwargs_635280)
    
    # Assigning a type to the variable 'pmfs_cum' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'pmfs_cum', cumsum_call_result_635281)
    
    # Assigning a Tuple to a Tuple (line 150):
    
    # Assigning a Num to a Name (line 150):
    float_635282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 17), 'float')
    # Assigning a type to the variable 'tuple_assignment_634725' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_634725', float_635282)
    
    # Assigning a Num to a Name (line 150):
    float_635283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 24), 'float')
    # Assigning a type to the variable 'tuple_assignment_634726' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_634726', float_635283)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_assignment_634725' (line 150)
    tuple_assignment_634725_635284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_634725')
    # Assigning a type to the variable 'atol' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'atol', tuple_assignment_634725_635284)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_assignment_634726' (line 150)
    tuple_assignment_634726_635285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_634726')
    # Assigning a type to the variable 'rtol' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 10), 'rtol', tuple_assignment_634726_635285)
    
    
    # Getting the type of 'distname' (line 151)
    distname_635286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'distname')
    str_635287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 19), 'str', 'skellam')
    # Applying the binary operator '==' (line 151)
    result_eq_635288 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 7), '==', distname_635286, str_635287)
    
    # Testing the type of an if condition (line 151)
    if_condition_635289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), result_eq_635288)
    # Assigning a type to the variable 'if_condition_635289' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_635289', if_condition_635289)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 152):
    
    # Assigning a Num to a Name (line 152):
    float_635290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 21), 'float')
    # Assigning a type to the variable 'tuple_assignment_634727' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_assignment_634727', float_635290)
    
    # Assigning a Num to a Name (line 152):
    float_635291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 27), 'float')
    # Assigning a type to the variable 'tuple_assignment_634728' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_assignment_634728', float_635291)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_assignment_634727' (line 152)
    tuple_assignment_634727_635292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_assignment_634727')
    # Assigning a type to the variable 'atol' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'atol', tuple_assignment_634727_635292)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_assignment_634728' (line 152)
    tuple_assignment_634728_635293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_assignment_634728')
    # Assigning a type to the variable 'rtol' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'rtol', tuple_assignment_634728_635293)
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_allclose(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'cdfs' (line 153)
    cdfs_635296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'cdfs', False)
    
    # Obtaining the type of the subscript
    int_635297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 36), 'int')
    # Getting the type of 'cdfs' (line 153)
    cdfs_635298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'cdfs', False)
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___635299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 31), cdfs_635298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_635300 = invoke(stypy.reporting.localization.Localization(__file__, 153, 31), getitem___635299, int_635297)
    
    # Applying the binary operator '-' (line 153)
    result_sub_635301 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 24), '-', cdfs_635296, subscript_call_result_635300)
    
    # Getting the type of 'pmfs_cum' (line 153)
    pmfs_cum_635302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 40), 'pmfs_cum', False)
    
    # Obtaining the type of the subscript
    int_635303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 60), 'int')
    # Getting the type of 'pmfs_cum' (line 153)
    pmfs_cum_635304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 51), 'pmfs_cum', False)
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___635305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 51), pmfs_cum_635304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_635306 = invoke(stypy.reporting.localization.Localization(__file__, 153, 51), getitem___635305, int_635303)
    
    # Applying the binary operator '-' (line 153)
    result_sub_635307 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 40), '-', pmfs_cum_635302, subscript_call_result_635306)
    
    # Processing the call keyword arguments (line 153)
    # Getting the type of 'atol' (line 154)
    atol_635308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'atol', False)
    keyword_635309 = atol_635308
    # Getting the type of 'rtol' (line 154)
    rtol_635310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 40), 'rtol', False)
    keyword_635311 = rtol_635310
    kwargs_635312 = {'rtol': keyword_635311, 'atol': keyword_635309}
    # Getting the type of 'npt' (line 153)
    npt_635294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 153)
    assert_allclose_635295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), npt_635294, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 153)
    assert_allclose_call_result_635313 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), assert_allclose_635295, *[result_sub_635301, result_sub_635307], **kwargs_635312)
    
    
    # ################# End of 'check_pmf_cdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_pmf_cdf' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_635314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635314)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_pmf_cdf'
    return stypy_return_type_635314

# Assigning a type to the variable 'check_pmf_cdf' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'check_pmf_cdf', check_pmf_cdf)

@norecursion
def check_moment_frozen(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_moment_frozen'
    module_type_store = module_type_store.open_function_context('check_moment_frozen', 157, 0, False)
    
    # Passed parameters checking function
    check_moment_frozen.stypy_localization = localization
    check_moment_frozen.stypy_type_of_self = None
    check_moment_frozen.stypy_type_store = module_type_store
    check_moment_frozen.stypy_function_name = 'check_moment_frozen'
    check_moment_frozen.stypy_param_names_list = ['distfn', 'arg', 'm', 'k']
    check_moment_frozen.stypy_varargs_param_name = None
    check_moment_frozen.stypy_kwargs_param_name = None
    check_moment_frozen.stypy_call_defaults = defaults
    check_moment_frozen.stypy_call_varargs = varargs
    check_moment_frozen.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_moment_frozen', ['distfn', 'arg', 'm', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_moment_frozen', localization, ['distfn', 'arg', 'm', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_moment_frozen(...)' code ##################

    
    # Call to assert_allclose(...): (line 158)
    # Processing the call arguments (line 158)
    
    # Call to moment(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'k' (line 158)
    k_635322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 44), 'k', False)
    # Processing the call keyword arguments (line 158)
    kwargs_635323 = {}
    
    # Call to distfn(...): (line 158)
    # Getting the type of 'arg' (line 158)
    arg_635318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'arg', False)
    # Processing the call keyword arguments (line 158)
    kwargs_635319 = {}
    # Getting the type of 'distfn' (line 158)
    distfn_635317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'distfn', False)
    # Calling distfn(args, kwargs) (line 158)
    distfn_call_result_635320 = invoke(stypy.reporting.localization.Localization(__file__, 158, 24), distfn_635317, *[arg_635318], **kwargs_635319)
    
    # Obtaining the member 'moment' of a type (line 158)
    moment_635321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 24), distfn_call_result_635320, 'moment')
    # Calling moment(args, kwargs) (line 158)
    moment_call_result_635324 = invoke(stypy.reporting.localization.Localization(__file__, 158, 24), moment_635321, *[k_635322], **kwargs_635323)
    
    # Getting the type of 'm' (line 158)
    m_635325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 48), 'm', False)
    # Processing the call keyword arguments (line 158)
    float_635326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 29), 'float')
    keyword_635327 = float_635326
    float_635328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 41), 'float')
    keyword_635329 = float_635328
    kwargs_635330 = {'rtol': keyword_635329, 'atol': keyword_635327}
    # Getting the type of 'npt' (line 158)
    npt_635315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 158)
    assert_allclose_635316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 4), npt_635315, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 158)
    assert_allclose_call_result_635331 = invoke(stypy.reporting.localization.Localization(__file__, 158, 4), assert_allclose_635316, *[moment_call_result_635324, m_635325], **kwargs_635330)
    
    
    # ################# End of 'check_moment_frozen(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_moment_frozen' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_635332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_moment_frozen'
    return stypy_return_type_635332

# Assigning a type to the variable 'check_moment_frozen' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'check_moment_frozen', check_moment_frozen)

@norecursion
def check_oth(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_oth'
    module_type_store = module_type_store.open_function_context('check_oth', 162, 0, False)
    
    # Passed parameters checking function
    check_oth.stypy_localization = localization
    check_oth.stypy_type_of_self = None
    check_oth.stypy_type_store = module_type_store
    check_oth.stypy_function_name = 'check_oth'
    check_oth.stypy_param_names_list = ['distfn', 'arg', 'supp', 'msg']
    check_oth.stypy_varargs_param_name = None
    check_oth.stypy_kwargs_param_name = None
    check_oth.stypy_call_defaults = defaults
    check_oth.stypy_call_varargs = varargs
    check_oth.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_oth', ['distfn', 'arg', 'supp', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_oth', localization, ['distfn', 'arg', 'supp', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_oth(...)' code ##################

    
    # Call to assert_allclose(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Call to sf(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'supp' (line 164)
    supp_635337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'supp', False)
    # Getting the type of 'arg' (line 164)
    arg_635338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 41), 'arg', False)
    # Processing the call keyword arguments (line 164)
    kwargs_635339 = {}
    # Getting the type of 'distfn' (line 164)
    distfn_635335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'distfn', False)
    # Obtaining the member 'sf' of a type (line 164)
    sf_635336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 24), distfn_635335, 'sf')
    # Calling sf(args, kwargs) (line 164)
    sf_call_result_635340 = invoke(stypy.reporting.localization.Localization(__file__, 164, 24), sf_635336, *[supp_635337, arg_635338], **kwargs_635339)
    
    float_635341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 47), 'float')
    
    # Call to cdf(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'supp' (line 164)
    supp_635344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 63), 'supp', False)
    # Getting the type of 'arg' (line 164)
    arg_635345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 70), 'arg', False)
    # Processing the call keyword arguments (line 164)
    kwargs_635346 = {}
    # Getting the type of 'distfn' (line 164)
    distfn_635342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 52), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 164)
    cdf_635343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 52), distfn_635342, 'cdf')
    # Calling cdf(args, kwargs) (line 164)
    cdf_call_result_635347 = invoke(stypy.reporting.localization.Localization(__file__, 164, 52), cdf_635343, *[supp_635344, arg_635345], **kwargs_635346)
    
    # Applying the binary operator '-' (line 164)
    result_sub_635348 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 47), '-', float_635341, cdf_call_result_635347)
    
    # Processing the call keyword arguments (line 164)
    float_635349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 29), 'float')
    keyword_635350 = float_635349
    float_635351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 41), 'float')
    keyword_635352 = float_635351
    kwargs_635353 = {'rtol': keyword_635352, 'atol': keyword_635350}
    # Getting the type of 'npt' (line 164)
    npt_635333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 164)
    assert_allclose_635334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 4), npt_635333, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 164)
    assert_allclose_call_result_635354 = invoke(stypy.reporting.localization.Localization(__file__, 164, 4), assert_allclose_635334, *[sf_call_result_635340, result_sub_635348], **kwargs_635353)
    
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to linspace(...): (line 167)
    # Processing the call arguments (line 167)
    float_635357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'float')
    float_635358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 26), 'float')
    int_635359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 32), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_635360 = {}
    # Getting the type of 'np' (line 167)
    np_635355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 167)
    linspace_635356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), np_635355, 'linspace')
    # Calling linspace(args, kwargs) (line 167)
    linspace_call_result_635361 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), linspace_635356, *[float_635357, float_635358, int_635359], **kwargs_635360)
    
    # Assigning a type to the variable 'q' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'q', linspace_call_result_635361)
    
    # Call to assert_allclose(...): (line 168)
    # Processing the call arguments (line 168)
    
    # Call to isf(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'q' (line 168)
    q_635366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'q', False)
    # Getting the type of 'arg' (line 168)
    arg_635367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'arg', False)
    # Processing the call keyword arguments (line 168)
    kwargs_635368 = {}
    # Getting the type of 'distfn' (line 168)
    distfn_635364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'distfn', False)
    # Obtaining the member 'isf' of a type (line 168)
    isf_635365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 24), distfn_635364, 'isf')
    # Calling isf(args, kwargs) (line 168)
    isf_call_result_635369 = invoke(stypy.reporting.localization.Localization(__file__, 168, 24), isf_635365, *[q_635366, arg_635367], **kwargs_635368)
    
    
    # Call to ppf(...): (line 168)
    # Processing the call arguments (line 168)
    float_635372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 56), 'float')
    # Getting the type of 'q' (line 168)
    q_635373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 61), 'q', False)
    # Applying the binary operator '-' (line 168)
    result_sub_635374 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 56), '-', float_635372, q_635373)
    
    # Getting the type of 'arg' (line 168)
    arg_635375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 65), 'arg', False)
    # Processing the call keyword arguments (line 168)
    kwargs_635376 = {}
    # Getting the type of 'distfn' (line 168)
    distfn_635370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 45), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 168)
    ppf_635371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 45), distfn_635370, 'ppf')
    # Calling ppf(args, kwargs) (line 168)
    ppf_call_result_635377 = invoke(stypy.reporting.localization.Localization(__file__, 168, 45), ppf_635371, *[result_sub_635374, arg_635375], **kwargs_635376)
    
    # Processing the call keyword arguments (line 168)
    float_635378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 29), 'float')
    keyword_635379 = float_635378
    float_635380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 41), 'float')
    keyword_635381 = float_635380
    kwargs_635382 = {'rtol': keyword_635381, 'atol': keyword_635379}
    # Getting the type of 'npt' (line 168)
    npt_635362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 168)
    assert_allclose_635363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), npt_635362, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 168)
    assert_allclose_call_result_635383 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), assert_allclose_635363, *[isf_call_result_635369, ppf_call_result_635377], **kwargs_635382)
    
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to isf(...): (line 171)
    # Processing the call arguments (line 171)
    float_635386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 27), 'float')
    # Getting the type of 'arg' (line 171)
    arg_635387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'arg', False)
    # Processing the call keyword arguments (line 171)
    kwargs_635388 = {}
    # Getting the type of 'distfn' (line 171)
    distfn_635384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'distfn', False)
    # Obtaining the member 'isf' of a type (line 171)
    isf_635385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), distfn_635384, 'isf')
    # Calling isf(args, kwargs) (line 171)
    isf_call_result_635389 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), isf_635385, *[float_635386, arg_635387], **kwargs_635388)
    
    # Assigning a type to the variable 'median_sf' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'median_sf', isf_call_result_635389)
    
    # Call to assert_(...): (line 172)
    # Processing the call arguments (line 172)
    
    
    # Call to sf(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'median_sf' (line 172)
    median_sf_635394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 26), 'median_sf', False)
    int_635395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 38), 'int')
    # Applying the binary operator '-' (line 172)
    result_sub_635396 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 26), '-', median_sf_635394, int_635395)
    
    # Getting the type of 'arg' (line 172)
    arg_635397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 42), 'arg', False)
    # Processing the call keyword arguments (line 172)
    kwargs_635398 = {}
    # Getting the type of 'distfn' (line 172)
    distfn_635392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'distfn', False)
    # Obtaining the member 'sf' of a type (line 172)
    sf_635393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), distfn_635392, 'sf')
    # Calling sf(args, kwargs) (line 172)
    sf_call_result_635399 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), sf_635393, *[result_sub_635396, arg_635397], **kwargs_635398)
    
    float_635400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 49), 'float')
    # Applying the binary operator '>' (line 172)
    result_gt_635401 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 16), '>', sf_call_result_635399, float_635400)
    
    # Processing the call keyword arguments (line 172)
    kwargs_635402 = {}
    # Getting the type of 'npt' (line 172)
    npt_635390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 172)
    assert__635391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 4), npt_635390, 'assert_')
    # Calling assert_(args, kwargs) (line 172)
    assert__call_result_635403 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), assert__635391, *[result_gt_635401], **kwargs_635402)
    
    
    # Call to assert_(...): (line 173)
    # Processing the call arguments (line 173)
    
    
    # Call to cdf(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'median_sf' (line 173)
    median_sf_635408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'median_sf', False)
    int_635409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 39), 'int')
    # Applying the binary operator '+' (line 173)
    result_add_635410 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 27), '+', median_sf_635408, int_635409)
    
    # Getting the type of 'arg' (line 173)
    arg_635411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 43), 'arg', False)
    # Processing the call keyword arguments (line 173)
    kwargs_635412 = {}
    # Getting the type of 'distfn' (line 173)
    distfn_635406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 173)
    cdf_635407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), distfn_635406, 'cdf')
    # Calling cdf(args, kwargs) (line 173)
    cdf_call_result_635413 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), cdf_635407, *[result_add_635410, arg_635411], **kwargs_635412)
    
    float_635414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 50), 'float')
    # Applying the binary operator '>' (line 173)
    result_gt_635415 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '>', cdf_call_result_635413, float_635414)
    
    # Processing the call keyword arguments (line 173)
    kwargs_635416 = {}
    # Getting the type of 'npt' (line 173)
    npt_635404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 173)
    assert__635405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), npt_635404, 'assert_')
    # Calling assert_(args, kwargs) (line 173)
    assert__call_result_635417 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), assert__635405, *[result_gt_635415], **kwargs_635416)
    
    
    # ################# End of 'check_oth(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_oth' in the type store
    # Getting the type of 'stypy_return_type' (line 162)
    stypy_return_type_635418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635418)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_oth'
    return stypy_return_type_635418

# Assigning a type to the variable 'check_oth' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'check_oth', check_oth)

@norecursion
def check_discrete_chisquare(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_discrete_chisquare'
    module_type_store = module_type_store.open_function_context('check_discrete_chisquare', 176, 0, False)
    
    # Passed parameters checking function
    check_discrete_chisquare.stypy_localization = localization
    check_discrete_chisquare.stypy_type_of_self = None
    check_discrete_chisquare.stypy_type_store = module_type_store
    check_discrete_chisquare.stypy_function_name = 'check_discrete_chisquare'
    check_discrete_chisquare.stypy_param_names_list = ['distfn', 'arg', 'rvs', 'alpha', 'msg']
    check_discrete_chisquare.stypy_varargs_param_name = None
    check_discrete_chisquare.stypy_kwargs_param_name = None
    check_discrete_chisquare.stypy_call_defaults = defaults
    check_discrete_chisquare.stypy_call_varargs = varargs
    check_discrete_chisquare.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_discrete_chisquare', ['distfn', 'arg', 'rvs', 'alpha', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_discrete_chisquare', localization, ['distfn', 'arg', 'rvs', 'alpha', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_discrete_chisquare(...)' code ##################

    str_635419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', 'Perform chisquare test for random sample of a discrete distribution\n\n    Parameters\n    ----------\n    distname : string\n        name of distribution function\n    arg : sequence\n        parameters of distribution\n    alpha : float\n        significance level, threshold for p-value\n\n    Returns\n    -------\n    result : bool\n        0 if test passes, 1 if test fails\n\n    ')
    
    # Assigning a Num to a Name (line 194):
    
    # Assigning a Num to a Name (line 194):
    float_635420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 12), 'float')
    # Assigning a type to the variable 'wsupp' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'wsupp', float_635420)
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to int(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Call to max(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'distfn' (line 198)
    distfn_635423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'distfn', False)
    # Obtaining the member 'a' of a type (line 198)
    a_635424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 17), distfn_635423, 'a')
    int_635425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 27), 'int')
    # Processing the call keyword arguments (line 198)
    kwargs_635426 = {}
    # Getting the type of 'max' (line 198)
    max_635422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 13), 'max', False)
    # Calling max(args, kwargs) (line 198)
    max_call_result_635427 = invoke(stypy.reporting.localization.Localization(__file__, 198, 13), max_635422, *[a_635424, int_635425], **kwargs_635426)
    
    # Processing the call keyword arguments (line 198)
    kwargs_635428 = {}
    # Getting the type of 'int' (line 198)
    int_635421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 9), 'int', False)
    # Calling int(args, kwargs) (line 198)
    int_call_result_635429 = invoke(stypy.reporting.localization.Localization(__file__, 198, 9), int_635421, *[max_call_result_635427], **kwargs_635428)
    
    # Assigning a type to the variable 'lo' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'lo', int_call_result_635429)
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to xrange(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'lo' (line 199)
    lo_635431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'lo', False)
    
    # Call to int(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Call to min(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'distfn' (line 199)
    distfn_635434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 'distfn', False)
    # Obtaining the member 'b' of a type (line 199)
    b_635435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 37), distfn_635434, 'b')
    int_635436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 47), 'int')
    # Processing the call keyword arguments (line 199)
    kwargs_635437 = {}
    # Getting the type of 'min' (line 199)
    min_635433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'min', False)
    # Calling min(args, kwargs) (line 199)
    min_call_result_635438 = invoke(stypy.reporting.localization.Localization(__file__, 199, 33), min_635433, *[b_635435, int_635436], **kwargs_635437)
    
    # Processing the call keyword arguments (line 199)
    kwargs_635439 = {}
    # Getting the type of 'int' (line 199)
    int_635432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'int', False)
    # Calling int(args, kwargs) (line 199)
    int_call_result_635440 = invoke(stypy.reporting.localization.Localization(__file__, 199, 29), int_635432, *[min_call_result_635438], **kwargs_635439)
    
    int_635441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 56), 'int')
    # Applying the binary operator '+' (line 199)
    result_add_635442 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 29), '+', int_call_result_635440, int_635441)
    
    # Processing the call keyword arguments (line 199)
    kwargs_635443 = {}
    # Getting the type of 'xrange' (line 199)
    xrange_635430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'xrange', False)
    # Calling xrange(args, kwargs) (line 199)
    xrange_call_result_635444 = invoke(stypy.reporting.localization.Localization(__file__, 199, 18), xrange_635430, *[lo_635431, result_add_635442], **kwargs_635443)
    
    # Assigning a type to the variable 'distsupport' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'distsupport', xrange_call_result_635444)
    
    # Assigning a Num to a Name (line 200):
    
    # Assigning a Num to a Name (line 200):
    int_635445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 11), 'int')
    # Assigning a type to the variable 'last' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'last', int_635445)
    
    # Assigning a List to a Name (line 201):
    
    # Assigning a List to a Name (line 201):
    
    # Obtaining an instance of the builtin type 'list' (line 201)
    list_635446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 201)
    # Adding element type (line 201)
    # Getting the type of 'lo' (line 201)
    lo_635447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'lo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 15), list_635446, lo_635447)
    
    # Assigning a type to the variable 'distsupp' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'distsupp', list_635446)
    
    # Assigning a List to a Name (line 202):
    
    # Assigning a List to a Name (line 202):
    
    # Obtaining an instance of the builtin type 'list' (line 202)
    list_635448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 202)
    
    # Assigning a type to the variable 'distmass' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'distmass', list_635448)
    
    # Getting the type of 'distsupport' (line 203)
    distsupport_635449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'distsupport')
    # Testing the type of a for loop iterable (line 203)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 203, 4), distsupport_635449)
    # Getting the type of the for loop variable (line 203)
    for_loop_var_635450 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 203, 4), distsupport_635449)
    # Assigning a type to the variable 'ii' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'ii', for_loop_var_635450)
    # SSA begins for a for statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to cdf(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'ii' (line 204)
    ii_635453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'ii', False)
    # Getting the type of 'arg' (line 204)
    arg_635454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 34), 'arg', False)
    # Processing the call keyword arguments (line 204)
    kwargs_635455 = {}
    # Getting the type of 'distfn' (line 204)
    distfn_635451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 204)
    cdf_635452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), distfn_635451, 'cdf')
    # Calling cdf(args, kwargs) (line 204)
    cdf_call_result_635456 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), cdf_635452, *[ii_635453, arg_635454], **kwargs_635455)
    
    # Assigning a type to the variable 'current' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'current', cdf_call_result_635456)
    
    
    # Getting the type of 'current' (line 205)
    current_635457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'current')
    # Getting the type of 'last' (line 205)
    last_635458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'last')
    # Applying the binary operator '-' (line 205)
    result_sub_635459 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), '-', current_635457, last_635458)
    
    # Getting the type of 'wsupp' (line 205)
    wsupp_635460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'wsupp')
    float_635461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 37), 'float')
    # Applying the binary operator '-' (line 205)
    result_sub_635462 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 29), '-', wsupp_635460, float_635461)
    
    # Applying the binary operator '>=' (line 205)
    result_ge_635463 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), '>=', result_sub_635459, result_sub_635462)
    
    # Testing the type of an if condition (line 205)
    if_condition_635464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_ge_635463)
    # Assigning a type to the variable 'if_condition_635464' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_635464', if_condition_635464)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'ii' (line 206)
    ii_635467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 28), 'ii', False)
    # Processing the call keyword arguments (line 206)
    kwargs_635468 = {}
    # Getting the type of 'distsupp' (line 206)
    distsupp_635465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'distsupp', False)
    # Obtaining the member 'append' of a type (line 206)
    append_635466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), distsupp_635465, 'append')
    # Calling append(args, kwargs) (line 206)
    append_call_result_635469 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), append_635466, *[ii_635467], **kwargs_635468)
    
    
    # Call to append(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'current' (line 207)
    current_635472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 28), 'current', False)
    # Getting the type of 'last' (line 207)
    last_635473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 'last', False)
    # Applying the binary operator '-' (line 207)
    result_sub_635474 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 28), '-', current_635472, last_635473)
    
    # Processing the call keyword arguments (line 207)
    kwargs_635475 = {}
    # Getting the type of 'distmass' (line 207)
    distmass_635470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'distmass', False)
    # Obtaining the member 'append' of a type (line 207)
    append_635471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), distmass_635470, 'append')
    # Calling append(args, kwargs) (line 207)
    append_call_result_635476 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), append_635471, *[result_sub_635474], **kwargs_635475)
    
    
    # Assigning a Name to a Name (line 208):
    
    # Assigning a Name to a Name (line 208):
    # Getting the type of 'current' (line 208)
    current_635477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 'current')
    # Assigning a type to the variable 'last' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'last', current_635477)
    
    
    # Getting the type of 'current' (line 209)
    current_635478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'current')
    int_635479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 26), 'int')
    # Getting the type of 'wsupp' (line 209)
    wsupp_635480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'wsupp')
    # Applying the binary operator '-' (line 209)
    result_sub_635481 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 26), '-', int_635479, wsupp_635480)
    
    # Applying the binary operator '>' (line 209)
    result_gt_635482 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 15), '>', current_635478, result_sub_635481)
    
    # Testing the type of an if condition (line 209)
    if_condition_635483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 12), result_gt_635482)
    # Assigning a type to the variable 'if_condition_635483' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'if_condition_635483', if_condition_635483)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_635484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 16), 'int')
    # Getting the type of 'distsupp' (line 211)
    distsupp_635485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 7), 'distsupp')
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___635486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 7), distsupp_635485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_635487 = invoke(stypy.reporting.localization.Localization(__file__, 211, 7), getitem___635486, int_635484)
    
    # Getting the type of 'distfn' (line 211)
    distfn_635488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'distfn')
    # Obtaining the member 'b' of a type (line 211)
    b_635489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 22), distfn_635488, 'b')
    # Applying the binary operator '<' (line 211)
    result_lt_635490 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 7), '<', subscript_call_result_635487, b_635489)
    
    # Testing the type of an if condition (line 211)
    if_condition_635491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 4), result_lt_635490)
    # Assigning a type to the variable 'if_condition_635491' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'if_condition_635491', if_condition_635491)
    # SSA begins for if statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'distfn' (line 212)
    distfn_635494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'distfn', False)
    # Obtaining the member 'b' of a type (line 212)
    b_635495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 24), distfn_635494, 'b')
    # Processing the call keyword arguments (line 212)
    kwargs_635496 = {}
    # Getting the type of 'distsupp' (line 212)
    distsupp_635492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'distsupp', False)
    # Obtaining the member 'append' of a type (line 212)
    append_635493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), distsupp_635492, 'append')
    # Calling append(args, kwargs) (line 212)
    append_call_result_635497 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), append_635493, *[b_635495], **kwargs_635496)
    
    
    # Call to append(...): (line 213)
    # Processing the call arguments (line 213)
    int_635500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 24), 'int')
    # Getting the type of 'last' (line 213)
    last_635501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 28), 'last', False)
    # Applying the binary operator '-' (line 213)
    result_sub_635502 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 24), '-', int_635500, last_635501)
    
    # Processing the call keyword arguments (line 213)
    kwargs_635503 = {}
    # Getting the type of 'distmass' (line 213)
    distmass_635498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'distmass', False)
    # Obtaining the member 'append' of a type (line 213)
    append_635499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), distmass_635498, 'append')
    # Calling append(args, kwargs) (line 213)
    append_call_result_635504 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), append_635499, *[result_sub_635502], **kwargs_635503)
    
    # SSA join for if statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to array(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'distsupp' (line 214)
    distsupp_635507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'distsupp', False)
    # Processing the call keyword arguments (line 214)
    kwargs_635508 = {}
    # Getting the type of 'np' (line 214)
    np_635505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 214)
    array_635506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 15), np_635505, 'array')
    # Calling array(args, kwargs) (line 214)
    array_call_result_635509 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), array_635506, *[distsupp_635507], **kwargs_635508)
    
    # Assigning a type to the variable 'distsupp' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'distsupp', array_call_result_635509)
    
    # Assigning a Call to a Name (line 215):
    
    # Assigning a Call to a Name (line 215):
    
    # Call to array(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'distmass' (line 215)
    distmass_635512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'distmass', False)
    # Processing the call keyword arguments (line 215)
    kwargs_635513 = {}
    # Getting the type of 'np' (line 215)
    np_635510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 215)
    array_635511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), np_635510, 'array')
    # Calling array(args, kwargs) (line 215)
    array_call_result_635514 = invoke(stypy.reporting.localization.Localization(__file__, 215, 15), array_635511, *[distmass_635512], **kwargs_635513)
    
    # Assigning a type to the variable 'distmass' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'distmass', array_call_result_635514)
    
    # Assigning a BinOp to a Name (line 218):
    
    # Assigning a BinOp to a Name (line 218):
    # Getting the type of 'distsupp' (line 218)
    distsupp_635515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'distsupp')
    float_635516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 26), 'float')
    # Applying the binary operator '+' (line 218)
    result_add_635517 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 15), '+', distsupp_635515, float_635516)
    
    # Assigning a type to the variable 'histsupp' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'histsupp', result_add_635517)
    
    # Assigning a Attribute to a Subscript (line 219):
    
    # Assigning a Attribute to a Subscript (line 219):
    # Getting the type of 'distfn' (line 219)
    distfn_635518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 18), 'distfn')
    # Obtaining the member 'a' of a type (line 219)
    a_635519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 18), distfn_635518, 'a')
    # Getting the type of 'histsupp' (line 219)
    histsupp_635520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'histsupp')
    int_635521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 13), 'int')
    # Storing an element on a container (line 219)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 4), histsupp_635520, (int_635521, a_635519))
    
    # Assigning a Call to a Tuple (line 222):
    
    # Assigning a Subscript to a Name (line 222):
    
    # Obtaining the type of the subscript
    int_635522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 4), 'int')
    
    # Call to histogram(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'rvs' (line 222)
    rvs_635525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'rvs', False)
    # Getting the type of 'histsupp' (line 222)
    histsupp_635526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 36), 'histsupp', False)
    # Processing the call keyword arguments (line 222)
    kwargs_635527 = {}
    # Getting the type of 'np' (line 222)
    np_635523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'np', False)
    # Obtaining the member 'histogram' of a type (line 222)
    histogram_635524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 18), np_635523, 'histogram')
    # Calling histogram(args, kwargs) (line 222)
    histogram_call_result_635528 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), histogram_635524, *[rvs_635525, histsupp_635526], **kwargs_635527)
    
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___635529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 4), histogram_call_result_635528, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_635530 = invoke(stypy.reporting.localization.Localization(__file__, 222, 4), getitem___635529, int_635522)
    
    # Assigning a type to the variable 'tuple_var_assignment_634729' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_634729', subscript_call_result_635530)
    
    # Assigning a Subscript to a Name (line 222):
    
    # Obtaining the type of the subscript
    int_635531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 4), 'int')
    
    # Call to histogram(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'rvs' (line 222)
    rvs_635534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'rvs', False)
    # Getting the type of 'histsupp' (line 222)
    histsupp_635535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 36), 'histsupp', False)
    # Processing the call keyword arguments (line 222)
    kwargs_635536 = {}
    # Getting the type of 'np' (line 222)
    np_635532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'np', False)
    # Obtaining the member 'histogram' of a type (line 222)
    histogram_635533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 18), np_635532, 'histogram')
    # Calling histogram(args, kwargs) (line 222)
    histogram_call_result_635537 = invoke(stypy.reporting.localization.Localization(__file__, 222, 18), histogram_635533, *[rvs_635534, histsupp_635535], **kwargs_635536)
    
    # Obtaining the member '__getitem__' of a type (line 222)
    getitem___635538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 4), histogram_call_result_635537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
    subscript_call_result_635539 = invoke(stypy.reporting.localization.Localization(__file__, 222, 4), getitem___635538, int_635531)
    
    # Assigning a type to the variable 'tuple_var_assignment_634730' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_634730', subscript_call_result_635539)
    
    # Assigning a Name to a Name (line 222):
    # Getting the type of 'tuple_var_assignment_634729' (line 222)
    tuple_var_assignment_634729_635540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_634729')
    # Assigning a type to the variable 'freq' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'freq', tuple_var_assignment_634729_635540)
    
    # Assigning a Name to a Name (line 222):
    # Getting the type of 'tuple_var_assignment_634730' (line 222)
    tuple_var_assignment_634730_635541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'tuple_var_assignment_634730')
    # Assigning a type to the variable 'hsupp' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 10), 'hsupp', tuple_var_assignment_634730_635541)
    
    # Assigning a Call to a Tuple (line 223):
    
    # Assigning a Subscript to a Name (line 223):
    
    # Obtaining the type of the subscript
    int_635542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 4), 'int')
    
    # Call to chisquare(...): (line 223)
    # Processing the call arguments (line 223)
    
    # Call to array(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'freq' (line 223)
    freq_635547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'freq', False)
    # Processing the call keyword arguments (line 223)
    kwargs_635548 = {}
    # Getting the type of 'np' (line 223)
    np_635545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'np', False)
    # Obtaining the member 'array' of a type (line 223)
    array_635546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 33), np_635545, 'array')
    # Calling array(args, kwargs) (line 223)
    array_call_result_635549 = invoke(stypy.reporting.localization.Localization(__file__, 223, 33), array_635546, *[freq_635547], **kwargs_635548)
    
    
    # Call to len(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'rvs' (line 223)
    rvs_635551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 53), 'rvs', False)
    # Processing the call keyword arguments (line 223)
    kwargs_635552 = {}
    # Getting the type of 'len' (line 223)
    len_635550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 49), 'len', False)
    # Calling len(args, kwargs) (line 223)
    len_call_result_635553 = invoke(stypy.reporting.localization.Localization(__file__, 223, 49), len_635550, *[rvs_635551], **kwargs_635552)
    
    # Getting the type of 'distmass' (line 223)
    distmass_635554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 58), 'distmass', False)
    # Applying the binary operator '*' (line 223)
    result_mul_635555 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 49), '*', len_call_result_635553, distmass_635554)
    
    # Processing the call keyword arguments (line 223)
    kwargs_635556 = {}
    # Getting the type of 'stats' (line 223)
    stats_635543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'stats', False)
    # Obtaining the member 'chisquare' of a type (line 223)
    chisquare_635544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 17), stats_635543, 'chisquare')
    # Calling chisquare(args, kwargs) (line 223)
    chisquare_call_result_635557 = invoke(stypy.reporting.localization.Localization(__file__, 223, 17), chisquare_635544, *[array_call_result_635549, result_mul_635555], **kwargs_635556)
    
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___635558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 4), chisquare_call_result_635557, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_635559 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), getitem___635558, int_635542)
    
    # Assigning a type to the variable 'tuple_var_assignment_634731' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_var_assignment_634731', subscript_call_result_635559)
    
    # Assigning a Subscript to a Name (line 223):
    
    # Obtaining the type of the subscript
    int_635560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 4), 'int')
    
    # Call to chisquare(...): (line 223)
    # Processing the call arguments (line 223)
    
    # Call to array(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'freq' (line 223)
    freq_635565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'freq', False)
    # Processing the call keyword arguments (line 223)
    kwargs_635566 = {}
    # Getting the type of 'np' (line 223)
    np_635563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 33), 'np', False)
    # Obtaining the member 'array' of a type (line 223)
    array_635564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 33), np_635563, 'array')
    # Calling array(args, kwargs) (line 223)
    array_call_result_635567 = invoke(stypy.reporting.localization.Localization(__file__, 223, 33), array_635564, *[freq_635565], **kwargs_635566)
    
    
    # Call to len(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'rvs' (line 223)
    rvs_635569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 53), 'rvs', False)
    # Processing the call keyword arguments (line 223)
    kwargs_635570 = {}
    # Getting the type of 'len' (line 223)
    len_635568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 49), 'len', False)
    # Calling len(args, kwargs) (line 223)
    len_call_result_635571 = invoke(stypy.reporting.localization.Localization(__file__, 223, 49), len_635568, *[rvs_635569], **kwargs_635570)
    
    # Getting the type of 'distmass' (line 223)
    distmass_635572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 58), 'distmass', False)
    # Applying the binary operator '*' (line 223)
    result_mul_635573 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 49), '*', len_call_result_635571, distmass_635572)
    
    # Processing the call keyword arguments (line 223)
    kwargs_635574 = {}
    # Getting the type of 'stats' (line 223)
    stats_635561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'stats', False)
    # Obtaining the member 'chisquare' of a type (line 223)
    chisquare_635562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 17), stats_635561, 'chisquare')
    # Calling chisquare(args, kwargs) (line 223)
    chisquare_call_result_635575 = invoke(stypy.reporting.localization.Localization(__file__, 223, 17), chisquare_635562, *[array_call_result_635567, result_mul_635573], **kwargs_635574)
    
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___635576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 4), chisquare_call_result_635575, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_635577 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), getitem___635576, int_635560)
    
    # Assigning a type to the variable 'tuple_var_assignment_634732' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_var_assignment_634732', subscript_call_result_635577)
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'tuple_var_assignment_634731' (line 223)
    tuple_var_assignment_634731_635578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_var_assignment_634731')
    # Assigning a type to the variable 'chis' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'chis', tuple_var_assignment_634731_635578)
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'tuple_var_assignment_634732' (line 223)
    tuple_var_assignment_634732_635579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_var_assignment_634732')
    # Assigning a type to the variable 'pval' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 10), 'pval', tuple_var_assignment_634732_635579)
    
    # Call to assert_(...): (line 225)
    # Processing the call arguments (line 225)
    
    # Getting the type of 'pval' (line 225)
    pval_635582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'pval', False)
    # Getting the type of 'alpha' (line 225)
    alpha_635583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'alpha', False)
    # Applying the binary operator '>' (line 225)
    result_gt_635584 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 16), '>', pval_635582, alpha_635583)
    
    str_635585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 16), 'str', 'chisquare - test for %s at arg = %s with pval = %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 227)
    tuple_635586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 227)
    # Adding element type (line 227)
    # Getting the type of 'msg' (line 227)
    msg_635587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 17), 'msg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 17), tuple_635586, msg_635587)
    # Adding element type (line 227)
    
    # Call to str(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'arg' (line 227)
    arg_635589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'arg', False)
    # Processing the call keyword arguments (line 227)
    kwargs_635590 = {}
    # Getting the type of 'str' (line 227)
    str_635588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'str', False)
    # Calling str(args, kwargs) (line 227)
    str_call_result_635591 = invoke(stypy.reporting.localization.Localization(__file__, 227, 22), str_635588, *[arg_635589], **kwargs_635590)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 17), tuple_635586, str_call_result_635591)
    # Adding element type (line 227)
    
    # Call to str(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'pval' (line 227)
    pval_635593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 36), 'pval', False)
    # Processing the call keyword arguments (line 227)
    kwargs_635594 = {}
    # Getting the type of 'str' (line 227)
    str_635592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 32), 'str', False)
    # Calling str(args, kwargs) (line 227)
    str_call_result_635595 = invoke(stypy.reporting.localization.Localization(__file__, 227, 32), str_635592, *[pval_635593], **kwargs_635594)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 17), tuple_635586, str_call_result_635595)
    
    # Applying the binary operator '%' (line 226)
    result_mod_635596 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 16), '%', str_635585, tuple_635586)
    
    # Processing the call keyword arguments (line 225)
    kwargs_635597 = {}
    # Getting the type of 'npt' (line 225)
    npt_635580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 225)
    assert__635581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 4), npt_635580, 'assert_')
    # Calling assert_(args, kwargs) (line 225)
    assert__call_result_635598 = invoke(stypy.reporting.localization.Localization(__file__, 225, 4), assert__635581, *[result_gt_635584, result_mod_635596], **kwargs_635597)
    
    
    # ################# End of 'check_discrete_chisquare(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_discrete_chisquare' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_635599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635599)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_discrete_chisquare'
    return stypy_return_type_635599

# Assigning a type to the variable 'check_discrete_chisquare' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'check_discrete_chisquare', check_discrete_chisquare)

@norecursion
def check_scale_docstring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_scale_docstring'
    module_type_store = module_type_store.open_function_context('check_scale_docstring', 230, 0, False)
    
    # Passed parameters checking function
    check_scale_docstring.stypy_localization = localization
    check_scale_docstring.stypy_type_of_self = None
    check_scale_docstring.stypy_type_store = module_type_store
    check_scale_docstring.stypy_function_name = 'check_scale_docstring'
    check_scale_docstring.stypy_param_names_list = ['distfn']
    check_scale_docstring.stypy_varargs_param_name = None
    check_scale_docstring.stypy_kwargs_param_name = None
    check_scale_docstring.stypy_call_defaults = defaults
    check_scale_docstring.stypy_call_varargs = varargs
    check_scale_docstring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_scale_docstring', ['distfn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_scale_docstring', localization, ['distfn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_scale_docstring(...)' code ##################

    
    
    # Getting the type of 'distfn' (line 231)
    distfn_635600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 7), 'distfn')
    # Obtaining the member '__doc__' of a type (line 231)
    doc___635601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 7), distfn_635600, '__doc__')
    # Getting the type of 'None' (line 231)
    None_635602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 29), 'None')
    # Applying the binary operator 'isnot' (line 231)
    result_is_not_635603 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 7), 'isnot', doc___635601, None_635602)
    
    # Testing the type of an if condition (line 231)
    if_condition_635604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 4), result_is_not_635603)
    # Assigning a type to the variable 'if_condition_635604' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'if_condition_635604', if_condition_635604)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_(...): (line 233)
    # Processing the call arguments (line 233)
    
    str_635607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'str', 'scale')
    # Getting the type of 'distfn' (line 233)
    distfn_635608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 35), 'distfn', False)
    # Obtaining the member '__doc__' of a type (line 233)
    doc___635609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 35), distfn_635608, '__doc__')
    # Applying the binary operator 'notin' (line 233)
    result_contains_635610 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 20), 'notin', str_635607, doc___635609)
    
    # Processing the call keyword arguments (line 233)
    kwargs_635611 = {}
    # Getting the type of 'npt' (line 233)
    npt_635605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 233)
    assert__635606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), npt_635605, 'assert_')
    # Calling assert_(args, kwargs) (line 233)
    assert__call_result_635612 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assert__635606, *[result_contains_635610], **kwargs_635611)
    
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_scale_docstring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_scale_docstring' in the type store
    # Getting the type of 'stypy_return_type' (line 230)
    stypy_return_type_635613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635613)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_scale_docstring'
    return stypy_return_type_635613

# Assigning a type to the variable 'check_scale_docstring' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'check_scale_docstring', check_scale_docstring)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: import numpy.testing as npt
5: import pytest
6: from pytest import raises as assert_raises
7: from scipy._lib._numpy_compat import suppress_warnings
8: from scipy.integrate import IntegrationWarning
9: 
10: from scipy import stats
11: from scipy.special import betainc
12: from. common_tests import (check_normalization, check_moment, check_mean_expect,
13:                            check_var_expect, check_skew_expect,
14:                            check_kurt_expect, check_entropy,
15:                            check_private_entropy,
16:                            check_edge_support, check_named_args,
17:                            check_random_state_property,
18:                            check_meth_dtype, check_ppf_dtype, check_cmplx_deriv,
19:                            check_pickling, check_rvs_broadcast)
20: from scipy.stats._distr_params import distcont
21: 
22: '''
23: Test all continuous distributions.
24: 
25: Parameters were chosen for those distributions that pass the
26: Kolmogorov-Smirnov test.  This provides safe parameters for each
27: distributions so that we can perform further testing of class methods.
28: 
29: These tests currently check only/mostly for serious errors and exceptions,
30: not for numerically exact results.
31: '''
32: 
33: # Note that you need to add new distributions you want tested
34: # to _distr_params
35: 
36: DECIMAL = 5  # specify the precision of the tests  # increased from 0 to 5
37: 
38: # Last four of these fail all around. Need to be checked
39: distcont_extra = [
40:     ['betaprime', (100, 86)],
41:     ['fatiguelife', (5,)],
42:     ['mielke', (4.6420495492121487, 0.59707419545516938)],
43:     ['invweibull', (0.58847112119264788,)],
44:     # burr: sample mean test fails still for c<1
45:     ['burr', (0.94839838075366045, 4.3820284068855795)],
46:     # genextreme: sample mean test, sf-logsf test fail
47:     ['genextreme', (3.3184017469423535,)],
48: ]
49: 
50: 
51: distslow = ['rdist', 'gausshyper', 'recipinvgauss', 'ksone', 'genexpon',
52:             'vonmises', 'vonmises_line', 'mielke', 'semicircular',
53:             'cosine', 'invweibull', 'powerlognorm', 'johnsonsu', 'kstwobign']
54: # distslow are sorted by speed (very slow to slow)
55: 
56: 
57: # These distributions fail the complex derivative test below.
58: # Here 'fail' mean produce wrong results and/or raise exceptions, depending
59: # on the implementation details of corresponding special functions.
60: # cf https://github.com/scipy/scipy/pull/4979 for a discussion.
61: fails_cmplx = set(['beta', 'betaprime', 'chi', 'chi2', 'dgamma', 'dweibull',
62:                    'erlang', 'f', 'gamma', 'gausshyper', 'gengamma',
63:                    'gennorm', 'genpareto', 'halfgennorm', 'invgamma',
64:                    'ksone', 'kstwobign', 'levy_l', 'loggamma', 'logistic',
65:                    'maxwell', 'nakagami', 'ncf', 'nct', 'ncx2',
66:                    'pearson3', 'rice', 't', 'skewnorm', 'tukeylambda',
67:                    'vonmises', 'vonmises_line', 'rv_histogram_instance'])
68: 
69: _h = np.histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6,
70:                    6, 6, 6, 7, 7, 7, 8, 8, 9], bins=8)
71: histogram_test_instance = stats.rv_histogram(_h)
72: 
73: 
74: def cases_test_cont_basic():
75:     for distname, arg in distcont[:] + [(histogram_test_instance, tuple())]:
76:         if distname == 'levy_stable':
77:             continue
78:         elif distname in distslow:
79:             yield pytest.param(distname, arg, marks=pytest.mark.slow)
80:         else:
81:             yield distname, arg
82: 
83: 
84: @pytest.mark.parametrize('distname,arg', cases_test_cont_basic())
85: def test_cont_basic(distname, arg):
86:     # this test skips slow distributions
87: 
88:     if distname == 'truncnorm':
89:         pytest.xfail(reason=distname)
90: 
91:     try:
92:         distfn = getattr(stats, distname)
93:     except TypeError:
94:         distfn = distname
95:         distname = 'rv_histogram_instance'
96:     np.random.seed(765456)
97:     sn = 500
98:     with suppress_warnings() as sup:
99:         # frechet_l and frechet_r are deprecated, so all their
100:         # methods generate DeprecationWarnings.
101:         sup.filter(category=DeprecationWarning, message=".*frechet_")
102:         rvs = distfn.rvs(size=sn, *arg)
103:         sm = rvs.mean()
104:         sv = rvs.var()
105:         m, v = distfn.stats(*arg)
106: 
107:         check_sample_meanvar_(distfn, arg, m, v, sm, sv, sn, distname + 'sample mean test')
108:         check_cdf_ppf(distfn, arg, distname)
109:         check_sf_isf(distfn, arg, distname)
110:         check_pdf(distfn, arg, distname)
111:         check_pdf_logpdf(distfn, arg, distname)
112:         check_cdf_logcdf(distfn, arg, distname)
113:         check_sf_logsf(distfn, arg, distname)
114: 
115:         alpha = 0.01
116:         if distname == 'rv_histogram_instance':
117:             check_distribution_rvs(distfn.cdf, arg, alpha, rvs)
118:         else:
119:             check_distribution_rvs(distname, arg, alpha, rvs)
120: 
121:         locscale_defaults = (0, 1)
122:         meths = [distfn.pdf, distfn.logpdf, distfn.cdf, distfn.logcdf,
123:                  distfn.logsf]
124:         # make sure arguments are within support
125:         spec_x = {'frechet_l': -0.5, 'weibull_max': -0.5, 'levy_l': -0.5,
126:                   'pareto': 1.5, 'tukeylambda': 0.3,
127:                   'rv_histogram_instance': 5.0}
128:         x = spec_x.get(distname, 0.5)
129:         if distname == 'invweibull':
130:             arg = (1,)
131:         elif distname == 'ksone':
132:             arg = (3,)
133:         check_named_args(distfn, x, arg, locscale_defaults, meths)
134:         check_random_state_property(distfn, arg)
135:         check_pickling(distfn, arg)
136: 
137:         # Entropy
138:         if distname not in ['ksone', 'kstwobign']:
139:             check_entropy(distfn, arg, distname)
140: 
141:         if distfn.numargs == 0:
142:             check_vecentropy(distfn, arg)
143: 
144:         if (distfn.__class__._entropy != stats.rv_continuous._entropy
145:                 and distname != 'vonmises'):
146:             check_private_entropy(distfn, arg, stats.rv_continuous)
147: 
148:         check_edge_support(distfn, arg)
149: 
150:         check_meth_dtype(distfn, arg, meths)
151:         check_ppf_dtype(distfn, arg)
152: 
153:         if distname not in fails_cmplx:
154:             check_cmplx_deriv(distfn, arg)
155: 
156:         if distname != 'truncnorm':
157:             check_ppf_private(distfn, arg, distname)
158: 
159: 
160: def test_levy_stable_random_state_property():
161:     # levy_stable only implements rvs(), so it is skipped in the
162:     # main loop in test_cont_basic(). Here we apply just the test
163:     # check_random_state_property to levy_stable.
164:     check_random_state_property(stats.levy_stable, (0.5, 0.1))
165: 
166: 
167: def cases_test_moments():
168:     fail_normalization = set(['vonmises', 'ksone'])
169:     fail_higher = set(['vonmises', 'ksone', 'ncf'])
170: 
171:     for distname, arg in distcont[:] + [(histogram_test_instance, tuple())]:
172:         if distname == 'levy_stable':
173:             continue
174: 
175:         cond1 = distname not in fail_normalization
176:         cond2 = distname not in fail_higher
177: 
178:         yield distname, arg, cond1, cond2, False
179: 
180:         if not cond1 or not cond2:
181:             # Run the distributions that have issues twice, once skipping the
182:             # not_ok parts, once with the not_ok parts but marked as knownfail
183:             yield pytest.param(distname, arg, True, True, True,
184:                                marks=pytest.mark.xfail)
185: 
186: 
187: @pytest.mark.slow
188: @pytest.mark.parametrize('distname,arg,normalization_ok,higher_ok,is_xfailing',
189:                          cases_test_moments())
190: def test_moments(distname, arg, normalization_ok, higher_ok, is_xfailing):
191:     try:
192:         distfn = getattr(stats, distname)
193:     except TypeError:
194:         distfn = distname
195:         distname = 'rv_histogram_instance'
196: 
197:     with suppress_warnings() as sup:
198:         sup.filter(IntegrationWarning,
199:                    "The integral is probably divergent, or slowly convergent.")
200:         sup.filter(category=DeprecationWarning, message=".*frechet_")
201:         if is_xfailing:
202:             sup.filter(IntegrationWarning)
203: 
204:         m, v, s, k = distfn.stats(*arg, moments='mvsk')
205: 
206:         if normalization_ok:
207:             check_normalization(distfn, arg, distname)
208: 
209:         if higher_ok:
210:             check_mean_expect(distfn, arg, m, distname)
211:             check_skew_expect(distfn, arg, m, v, s, distname)
212:             check_var_expect(distfn, arg, m, v, distname)
213:             check_kurt_expect(distfn, arg, m, v, k, distname)
214: 
215:         check_loc_scale(distfn, arg, m, v, distname)
216:         check_moment(distfn, arg, m, v, distname)
217: 
218: 
219: @pytest.mark.parametrize('dist,shape_args', distcont)
220: def test_rvs_broadcast(dist, shape_args):
221:     if dist in ['gausshyper', 'genexpon']:
222:         pytest.skip("too slow")
223: 
224:     # If shape_only is True, it means the _rvs method of the
225:     # distribution uses more than one random number to generate a random
226:     # variate.  That means the result of using rvs with broadcasting or
227:     # with a nontrivial size will not necessarily be the same as using the
228:     # numpy.vectorize'd version of rvs(), so we can only compare the shapes
229:     # of the results, not the values.
230:     # Whether or not a distribution is in the following list is an
231:     # implementation detail of the distribution, not a requirement.  If
232:     # the implementation the rvs() method of a distribution changes, this
233:     # test might also have to be changed.
234:     shape_only = dist in ['betaprime', 'dgamma', 'exponnorm',
235:                           'nct', 'dweibull', 'rice', 'levy_stable',
236:                           'skewnorm']
237: 
238:     distfunc = getattr(stats, dist)
239:     loc = np.zeros(2)
240:     scale = np.ones((3, 1))
241:     nargs = distfunc.numargs
242:     allargs = []
243:     bshape = [3, 2]
244:     # Generate shape parameter arguments...
245:     for k in range(nargs):
246:         shp = (k + 4,) + (1,)*(k + 2)
247:         allargs.append(shape_args[k]*np.ones(shp))
248:         bshape.insert(0, k + 4)
249:     allargs.extend([loc, scale])
250:     # bshape holds the expected shape when loc, scale, and the shape
251:     # parameters are all broadcast together.
252: 
253:     check_rvs_broadcast(distfunc, dist, allargs, bshape, shape_only, 'd')
254: 
255: 
256: def test_rvs_gh2069_regression():
257:     # Regression tests for gh-2069.  In scipy 0.17 and earlier,
258:     # these tests would fail.
259:     #
260:     # A typical example of the broken behavior:
261:     # >>> norm.rvs(loc=np.zeros(5), scale=np.ones(5))
262:     # array([-2.49613705, -2.49613705, -2.49613705, -2.49613705, -2.49613705])
263:     np.random.seed(123)
264:     vals = stats.norm.rvs(loc=np.zeros(5), scale=1)
265:     d = np.diff(vals)
266:     npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
267:     vals = stats.norm.rvs(loc=0, scale=np.ones(5))
268:     d = np.diff(vals)
269:     npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
270:     vals = stats.norm.rvs(loc=np.zeros(5), scale=np.ones(5))
271:     d = np.diff(vals)
272:     npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
273:     vals = stats.norm.rvs(loc=np.array([[0], [0]]), scale=np.ones(5))
274:     d = np.diff(vals.ravel())
275:     npt.assert_(np.all(d != 0), "All the values are equal, but they shouldn't be!")
276: 
277:     assert_raises(ValueError, stats.norm.rvs, [[0, 0], [0, 0]],
278:                   [[1, 1], [1, 1]], 1)
279:     assert_raises(ValueError, stats.gamma.rvs, [2, 3, 4, 5], 0, 1, (2, 2))
280:     assert_raises(ValueError, stats.gamma.rvs, [1, 1, 1, 1], [0, 0, 0, 0],
281:                      [[1], [2]], (4,))
282: 
283: 
284: def check_sample_meanvar_(distfn, arg, m, v, sm, sv, sn, msg):
285:     # this did not work, skipped silently by nose
286:     if np.isfinite(m):
287:         check_sample_mean(sm, sv, sn, m)
288:     if np.isfinite(v):
289:         check_sample_var(sv, sn, v)
290: 
291: 
292: def check_sample_mean(sm, v, n, popmean):
293:     # from stats.stats.ttest_1samp(a, popmean):
294:     # Calculates the t-obtained for the independent samples T-test on ONE group
295:     # of scores a, given a population mean.
296:     #
297:     # Returns: t-value, two-tailed prob
298:     df = n-1
299:     svar = ((n-1)*v) / float(df)    # looks redundant
300:     t = (sm-popmean) / np.sqrt(svar*(1.0/n))
301:     prob = betainc(0.5*df, 0.5, df/(df + t*t))
302: 
303:     # return t,prob
304:     npt.assert_(prob > 0.01, 'mean fail, t,prob = %f, %f, m, sm=%f,%f' %
305:                 (t, prob, popmean, sm))
306: 
307: 
308: def check_sample_var(sv, n, popvar):
309:     # two-sided chisquare test for sample variance equal to
310:     # hypothesized variance
311:     df = n-1
312:     chi2 = (n-1)*popvar/float(popvar)
313:     pval = stats.distributions.chi2.sf(chi2, df) * 2
314:     npt.assert_(pval > 0.01, 'var fail, t, pval = %f, %f, v, sv=%f, %f' %
315:                 (chi2, pval, popvar, sv))
316: 
317: 
318: def check_cdf_ppf(distfn, arg, msg):
319:     values = [0.001, 0.5, 0.999]
320:     npt.assert_almost_equal(distfn.cdf(distfn.ppf(values, *arg), *arg),
321:                             values, decimal=DECIMAL, err_msg=msg +
322:                             ' - cdf-ppf roundtrip')
323: 
324: 
325: def check_sf_isf(distfn, arg, msg):
326:     npt.assert_almost_equal(distfn.sf(distfn.isf([0.1, 0.5, 0.9], *arg), *arg),
327:                             [0.1, 0.5, 0.9], decimal=DECIMAL, err_msg=msg +
328:                             ' - sf-isf roundtrip')
329:     npt.assert_almost_equal(distfn.cdf([0.1, 0.9], *arg),
330:                             1.0 - distfn.sf([0.1, 0.9], *arg),
331:                             decimal=DECIMAL, err_msg=msg +
332:                             ' - cdf-sf relationship')
333: 
334: 
335: def check_pdf(distfn, arg, msg):
336:     # compares pdf at median with numerical derivative of cdf
337:     median = distfn.ppf(0.5, *arg)
338:     eps = 1e-6
339:     pdfv = distfn.pdf(median, *arg)
340:     if (pdfv < 1e-4) or (pdfv > 1e4):
341:         # avoid checking a case where pdf is close to zero or
342:         # huge (singularity)
343:         median = median + 0.1
344:         pdfv = distfn.pdf(median, *arg)
345:     cdfdiff = (distfn.cdf(median + eps, *arg) -
346:                distfn.cdf(median - eps, *arg))/eps/2.0
347:     # replace with better diff and better test (more points),
348:     # actually, this works pretty well
349:     msg += ' - cdf-pdf relationship'
350:     npt.assert_almost_equal(pdfv, cdfdiff, decimal=DECIMAL, err_msg=msg)
351: 
352: 
353: def check_pdf_logpdf(distfn, args, msg):
354:     # compares pdf at several points with the log of the pdf
355:     points = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
356:     vals = distfn.ppf(points, *args)
357:     pdf = distfn.pdf(vals, *args)
358:     logpdf = distfn.logpdf(vals, *args)
359:     pdf = pdf[pdf != 0]
360:     logpdf = logpdf[np.isfinite(logpdf)]
361:     msg += " - logpdf-log(pdf) relationship"
362:     npt.assert_almost_equal(np.log(pdf), logpdf, decimal=7, err_msg=msg)
363: 
364: 
365: def check_sf_logsf(distfn, args, msg):
366:     # compares sf at several points with the log of the sf
367:     points = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
368:     vals = distfn.ppf(points, *args)
369:     sf = distfn.sf(vals, *args)
370:     logsf = distfn.logsf(vals, *args)
371:     sf = sf[sf != 0]
372:     logsf = logsf[np.isfinite(logsf)]
373:     msg += " - logsf-log(sf) relationship"
374:     npt.assert_almost_equal(np.log(sf), logsf, decimal=7, err_msg=msg)
375: 
376: 
377: def check_cdf_logcdf(distfn, args, msg):
378:     # compares cdf at several points with the log of the cdf
379:     points = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
380:     vals = distfn.ppf(points, *args)
381:     cdf = distfn.cdf(vals, *args)
382:     logcdf = distfn.logcdf(vals, *args)
383:     cdf = cdf[cdf != 0]
384:     logcdf = logcdf[np.isfinite(logcdf)]
385:     msg += " - logcdf-log(cdf) relationship"
386:     npt.assert_almost_equal(np.log(cdf), logcdf, decimal=7, err_msg=msg)
387: 
388: 
389: def check_distribution_rvs(dist, args, alpha, rvs):
390:     # test from scipy.stats.tests
391:     # this version reuses existing random variables
392:     D, pval = stats.kstest(rvs, dist, args=args, N=1000)
393:     if (pval < alpha):
394:         D, pval = stats.kstest(dist, '', args=args, N=1000)
395:         npt.assert_(pval > alpha, "D = " + str(D) + "; pval = " + str(pval) +
396:                     "; alpha = " + str(alpha) + "\nargs = " + str(args))
397: 
398: 
399: def check_vecentropy(distfn, args):
400:     npt.assert_equal(distfn.vecentropy(*args), distfn._entropy(*args))
401: 
402: 
403: def check_loc_scale(distfn, arg, m, v, msg):
404:     loc, scale = 10.0, 10.0
405:     mt, vt = distfn.stats(loc=loc, scale=scale, *arg)
406:     npt.assert_allclose(m*scale + loc, mt)
407:     npt.assert_allclose(v*scale*scale, vt)
408: 
409: 
410: def check_ppf_private(distfn, arg, msg):
411:     # fails by design for truncnorm self.nb not defined
412:     ppfs = distfn._ppf(np.array([0.1, 0.5, 0.9]), *arg)
413:     npt.assert_(not np.any(np.isnan(ppfs)), msg + 'ppf private is nan')
414: 
415: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633099 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_633099) is not StypyTypeError):

    if (import_633099 != 'pyd_module'):
        __import__(import_633099)
        sys_modules_633100 = sys.modules[import_633099]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_633100.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_633099)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy.testing' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633101 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_633101) is not StypyTypeError):

    if (import_633101 != 'pyd_module'):
        __import__(import_633101)
        sys_modules_633102 = sys.modules[import_633101]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'npt', sys_modules_633102.module_type_store, module_type_store)
    else:
        import numpy.testing as npt

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'npt', numpy.testing, module_type_store)

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_633101)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import pytest' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633103 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest')

if (type(import_633103) is not StypyTypeError):

    if (import_633103 != 'pyd_module'):
        __import__(import_633103)
        sys_modules_633104 = sys.modules[import_633103]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', sys_modules_633104.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pytest', import_633103)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from pytest import assert_raises' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633105 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_633105) is not StypyTypeError):

    if (import_633105 != 'pyd_module'):
        __import__(import_633105)
        sys_modules_633106 = sys.modules[import_633105]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_633106.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_633106, sys_modules_633106.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_633105)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633107 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat')

if (type(import_633107) is not StypyTypeError):

    if (import_633107 != 'pyd_module'):
        __import__(import_633107)
        sys_modules_633108 = sys.modules[import_633107]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', sys_modules_633108.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_633108, sys_modules_633108.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', import_633107)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.integrate import IntegrationWarning' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633109 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.integrate')

if (type(import_633109) is not StypyTypeError):

    if (import_633109 != 'pyd_module'):
        __import__(import_633109)
        sys_modules_633110 = sys.modules[import_633109]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.integrate', sys_modules_633110.module_type_store, module_type_store, ['IntegrationWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_633110, sys_modules_633110.module_type_store, module_type_store)
    else:
        from scipy.integrate import IntegrationWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.integrate', None, module_type_store, ['IntegrationWarning'], [IntegrationWarning])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.integrate', import_633109)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy import stats' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633111 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy')

if (type(import_633111) is not StypyTypeError):

    if (import_633111 != 'pyd_module'):
        __import__(import_633111)
        sys_modules_633112 = sys.modules[import_633111]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy', sys_modules_633112.module_type_store, module_type_store, ['stats'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_633112, sys_modules_633112.module_type_store, module_type_store)
    else:
        from scipy import stats

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy', None, module_type_store, ['stats'], [stats])

else:
    # Assigning a type to the variable 'scipy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy', import_633111)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.special import betainc' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633113 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special')

if (type(import_633113) is not StypyTypeError):

    if (import_633113 != 'pyd_module'):
        __import__(import_633113)
        sys_modules_633114 = sys.modules[import_633113]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', sys_modules_633114.module_type_store, module_type_store, ['betainc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_633114, sys_modules_633114.module_type_store, module_type_store)
    else:
        from scipy.special import betainc

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', None, module_type_store, ['betainc'], [betainc])

else:
    # Assigning a type to the variable 'scipy.special' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', import_633113)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.stats.tests.common_tests import check_normalization, check_moment, check_mean_expect, check_var_expect, check_skew_expect, check_kurt_expect, check_entropy, check_private_entropy, check_edge_support, check_named_args, check_random_state_property, check_meth_dtype, check_ppf_dtype, check_cmplx_deriv, check_pickling, check_rvs_broadcast' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633115 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.stats.tests.common_tests')

if (type(import_633115) is not StypyTypeError):

    if (import_633115 != 'pyd_module'):
        __import__(import_633115)
        sys_modules_633116 = sys.modules[import_633115]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.stats.tests.common_tests', sys_modules_633116.module_type_store, module_type_store, ['check_normalization', 'check_moment', 'check_mean_expect', 'check_var_expect', 'check_skew_expect', 'check_kurt_expect', 'check_entropy', 'check_private_entropy', 'check_edge_support', 'check_named_args', 'check_random_state_property', 'check_meth_dtype', 'check_ppf_dtype', 'check_cmplx_deriv', 'check_pickling', 'check_rvs_broadcast'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_633116, sys_modules_633116.module_type_store, module_type_store)
    else:
        from scipy.stats.tests.common_tests import check_normalization, check_moment, check_mean_expect, check_var_expect, check_skew_expect, check_kurt_expect, check_entropy, check_private_entropy, check_edge_support, check_named_args, check_random_state_property, check_meth_dtype, check_ppf_dtype, check_cmplx_deriv, check_pickling, check_rvs_broadcast

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.stats.tests.common_tests', None, module_type_store, ['check_normalization', 'check_moment', 'check_mean_expect', 'check_var_expect', 'check_skew_expect', 'check_kurt_expect', 'check_entropy', 'check_private_entropy', 'check_edge_support', 'check_named_args', 'check_random_state_property', 'check_meth_dtype', 'check_ppf_dtype', 'check_cmplx_deriv', 'check_pickling', 'check_rvs_broadcast'], [check_normalization, check_moment, check_mean_expect, check_var_expect, check_skew_expect, check_kurt_expect, check_entropy, check_private_entropy, check_edge_support, check_named_args, check_random_state_property, check_meth_dtype, check_ppf_dtype, check_cmplx_deriv, check_pickling, check_rvs_broadcast])

else:
    # Assigning a type to the variable 'scipy.stats.tests.common_tests' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.stats.tests.common_tests', import_633115)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.stats._distr_params import distcont' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_633117 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.stats._distr_params')

if (type(import_633117) is not StypyTypeError):

    if (import_633117 != 'pyd_module'):
        __import__(import_633117)
        sys_modules_633118 = sys.modules[import_633117]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.stats._distr_params', sys_modules_633118.module_type_store, module_type_store, ['distcont'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_633118, sys_modules_633118.module_type_store, module_type_store)
    else:
        from scipy.stats._distr_params import distcont

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.stats._distr_params', None, module_type_store, ['distcont'], [distcont])

else:
    # Assigning a type to the variable 'scipy.stats._distr_params' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.stats._distr_params', import_633117)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

str_633119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\nTest all continuous distributions.\n\nParameters were chosen for those distributions that pass the\nKolmogorov-Smirnov test.  This provides safe parameters for each\ndistributions so that we can perform further testing of class methods.\n\nThese tests currently check only/mostly for serious errors and exceptions,\nnot for numerically exact results.\n')

# Assigning a Num to a Name (line 36):

# Assigning a Num to a Name (line 36):
int_633120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 10), 'int')
# Assigning a type to the variable 'DECIMAL' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'DECIMAL', int_633120)

# Assigning a List to a Name (line 39):

# Assigning a List to a Name (line 39):

# Obtaining an instance of the builtin type 'list' (line 39)
list_633121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)

# Obtaining an instance of the builtin type 'list' (line 40)
list_633122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
str_633123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 5), 'str', 'betaprime')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), list_633122, str_633123)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_633124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
int_633125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), tuple_633124, int_633125)
# Adding element type (line 40)
int_633126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), tuple_633124, int_633126)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), list_633122, tuple_633124)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_633121, list_633122)
# Adding element type (line 39)

# Obtaining an instance of the builtin type 'list' (line 41)
list_633127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
str_633128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 5), 'str', 'fatiguelife')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 4), list_633127, str_633128)
# Adding element type (line 41)

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_633129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
int_633130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 21), tuple_633129, int_633130)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 4), list_633127, tuple_633129)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_633121, list_633127)
# Adding element type (line 39)

# Obtaining an instance of the builtin type 'list' (line 42)
list_633131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)
str_633132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 5), 'str', 'mielke')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 4), list_633131, str_633132)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_633133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
float_633134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 16), tuple_633133, float_633134)
# Adding element type (line 42)
float_633135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 16), tuple_633133, float_633135)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 4), list_633131, tuple_633133)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_633121, list_633131)
# Adding element type (line 39)

# Obtaining an instance of the builtin type 'list' (line 43)
list_633136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
str_633137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 5), 'str', 'invweibull')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), list_633136, str_633137)
# Adding element type (line 43)

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_633138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
float_633139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), tuple_633138, float_633139)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), list_633136, tuple_633138)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_633121, list_633136)
# Adding element type (line 39)

# Obtaining an instance of the builtin type 'list' (line 45)
list_633140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 45)
# Adding element type (line 45)
str_633141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 5), 'str', 'burr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 4), list_633140, str_633141)
# Adding element type (line 45)

# Obtaining an instance of the builtin type 'tuple' (line 45)
tuple_633142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 45)
# Adding element type (line 45)
float_633143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 14), tuple_633142, float_633143)
# Adding element type (line 45)
float_633144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 14), tuple_633142, float_633144)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 4), list_633140, tuple_633142)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_633121, list_633140)
# Adding element type (line 39)

# Obtaining an instance of the builtin type 'list' (line 47)
list_633145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 47)
# Adding element type (line 47)
str_633146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 5), 'str', 'genextreme')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 4), list_633145, str_633146)
# Adding element type (line 47)

# Obtaining an instance of the builtin type 'tuple' (line 47)
tuple_633147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 47)
# Adding element type (line 47)
float_633148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 20), tuple_633147, float_633148)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 4), list_633145, tuple_633147)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_633121, list_633145)

# Assigning a type to the variable 'distcont_extra' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'distcont_extra', list_633121)

# Assigning a List to a Name (line 51):

# Assigning a List to a Name (line 51):

# Obtaining an instance of the builtin type 'list' (line 51)
list_633149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 51)
# Adding element type (line 51)
str_633150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'str', 'rdist')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633150)
# Adding element type (line 51)
str_633151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'str', 'gausshyper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633151)
# Adding element type (line 51)
str_633152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 35), 'str', 'recipinvgauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633152)
# Adding element type (line 51)
str_633153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 52), 'str', 'ksone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633153)
# Adding element type (line 51)
str_633154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 61), 'str', 'genexpon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633154)
# Adding element type (line 51)
str_633155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'str', 'vonmises')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633155)
# Adding element type (line 51)
str_633156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 24), 'str', 'vonmises_line')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633156)
# Adding element type (line 51)
str_633157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 41), 'str', 'mielke')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633157)
# Adding element type (line 51)
str_633158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 51), 'str', 'semicircular')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633158)
# Adding element type (line 51)
str_633159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 12), 'str', 'cosine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633159)
# Adding element type (line 51)
str_633160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 22), 'str', 'invweibull')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633160)
# Adding element type (line 51)
str_633161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'str', 'powerlognorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633161)
# Adding element type (line 51)
str_633162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 52), 'str', 'johnsonsu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633162)
# Adding element type (line 51)
str_633163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 65), 'str', 'kstwobign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_633149, str_633163)

# Assigning a type to the variable 'distslow' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'distslow', list_633149)

# Assigning a Call to a Name (line 61):

# Assigning a Call to a Name (line 61):

# Call to set(...): (line 61)
# Processing the call arguments (line 61)

# Obtaining an instance of the builtin type 'list' (line 61)
list_633165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
str_633166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'str', 'beta')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633166)
# Adding element type (line 61)
str_633167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'str', 'betaprime')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633167)
# Adding element type (line 61)
str_633168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'str', 'chi')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633168)
# Adding element type (line 61)
str_633169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 47), 'str', 'chi2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633169)
# Adding element type (line 61)
str_633170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 55), 'str', 'dgamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633170)
# Adding element type (line 61)
str_633171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 65), 'str', 'dweibull')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633171)
# Adding element type (line 61)
str_633172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'str', 'erlang')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633172)
# Adding element type (line 61)
str_633173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633173)
# Adding element type (line 61)
str_633174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 34), 'str', 'gamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633174)
# Adding element type (line 61)
str_633175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'str', 'gausshyper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633175)
# Adding element type (line 61)
str_633176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 57), 'str', 'gengamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633176)
# Adding element type (line 61)
str_633177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'str', 'gennorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633177)
# Adding element type (line 61)
str_633178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 30), 'str', 'genpareto')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633178)
# Adding element type (line 61)
str_633179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 43), 'str', 'halfgennorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633179)
# Adding element type (line 61)
str_633180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 58), 'str', 'invgamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633180)
# Adding element type (line 61)
str_633181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'str', 'ksone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633181)
# Adding element type (line 61)
str_633182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'str', 'kstwobign')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633182)
# Adding element type (line 61)
str_633183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 41), 'str', 'levy_l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633183)
# Adding element type (line 61)
str_633184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 51), 'str', 'loggamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633184)
# Adding element type (line 61)
str_633185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 63), 'str', 'logistic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633185)
# Adding element type (line 61)
str_633186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'str', 'maxwell')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633186)
# Adding element type (line 61)
str_633187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'str', 'nakagami')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633187)
# Adding element type (line 61)
str_633188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'str', 'ncf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633188)
# Adding element type (line 61)
str_633189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 49), 'str', 'nct')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633189)
# Adding element type (line 61)
str_633190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 56), 'str', 'ncx2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633190)
# Adding element type (line 61)
str_633191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'str', 'pearson3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633191)
# Adding element type (line 61)
str_633192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'str', 'rice')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633192)
# Adding element type (line 61)
str_633193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 39), 'str', 't')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633193)
# Adding element type (line 61)
str_633194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'str', 'skewnorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633194)
# Adding element type (line 61)
str_633195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 56), 'str', 'tukeylambda')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633195)
# Adding element type (line 61)
str_633196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 19), 'str', 'vonmises')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633196)
# Adding element type (line 61)
str_633197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'str', 'vonmises_line')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633197)
# Adding element type (line 61)
str_633198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'str', 'rv_histogram_instance')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 18), list_633165, str_633198)

# Processing the call keyword arguments (line 61)
kwargs_633199 = {}
# Getting the type of 'set' (line 61)
set_633164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'set', False)
# Calling set(args, kwargs) (line 61)
set_call_result_633200 = invoke(stypy.reporting.localization.Localization(__file__, 61, 14), set_633164, *[list_633165], **kwargs_633199)

# Assigning a type to the variable 'fails_cmplx' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'fails_cmplx', set_call_result_633200)

# Assigning a Call to a Name (line 69):

# Assigning a Call to a Name (line 69):

# Call to histogram(...): (line 69)
# Processing the call arguments (line 69)

# Obtaining an instance of the builtin type 'list' (line 69)
list_633203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 69)
# Adding element type (line 69)
int_633204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633204)
# Adding element type (line 69)
int_633205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633205)
# Adding element type (line 69)
int_633206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633206)
# Adding element type (line 69)
int_633207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633207)
# Adding element type (line 69)
int_633208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633208)
# Adding element type (line 69)
int_633209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633209)
# Adding element type (line 69)
int_633210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633210)
# Adding element type (line 69)
int_633211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633211)
# Adding element type (line 69)
int_633212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633212)
# Adding element type (line 69)
int_633213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633213)
# Adding element type (line 69)
int_633214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633214)
# Adding element type (line 69)
int_633215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633215)
# Adding element type (line 69)
int_633216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633216)
# Adding element type (line 69)
int_633217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633217)
# Adding element type (line 69)
int_633218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633218)
# Adding element type (line 69)
int_633219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633219)
# Adding element type (line 69)
int_633220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633220)
# Adding element type (line 69)
int_633221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633221)
# Adding element type (line 69)
int_633222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633222)
# Adding element type (line 69)
int_633223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633223)
# Adding element type (line 69)
int_633224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633224)
# Adding element type (line 69)
int_633225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633225)
# Adding element type (line 69)
int_633226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633226)
# Adding element type (line 69)
int_633227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633227)
# Adding element type (line 69)
int_633228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 18), list_633203, int_633228)

# Processing the call keyword arguments (line 69)
int_633229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 52), 'int')
keyword_633230 = int_633229
kwargs_633231 = {'bins': keyword_633230}
# Getting the type of 'np' (line 69)
np_633201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 5), 'np', False)
# Obtaining the member 'histogram' of a type (line 69)
histogram_633202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 5), np_633201, 'histogram')
# Calling histogram(args, kwargs) (line 69)
histogram_call_result_633232 = invoke(stypy.reporting.localization.Localization(__file__, 69, 5), histogram_633202, *[list_633203], **kwargs_633231)

# Assigning a type to the variable '_h' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), '_h', histogram_call_result_633232)

# Assigning a Call to a Name (line 71):

# Assigning a Call to a Name (line 71):

# Call to rv_histogram(...): (line 71)
# Processing the call arguments (line 71)
# Getting the type of '_h' (line 71)
_h_633235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 45), '_h', False)
# Processing the call keyword arguments (line 71)
kwargs_633236 = {}
# Getting the type of 'stats' (line 71)
stats_633233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'stats', False)
# Obtaining the member 'rv_histogram' of a type (line 71)
rv_histogram_633234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), stats_633233, 'rv_histogram')
# Calling rv_histogram(args, kwargs) (line 71)
rv_histogram_call_result_633237 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), rv_histogram_633234, *[_h_633235], **kwargs_633236)

# Assigning a type to the variable 'histogram_test_instance' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'histogram_test_instance', rv_histogram_call_result_633237)

@norecursion
def cases_test_cont_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cases_test_cont_basic'
    module_type_store = module_type_store.open_function_context('cases_test_cont_basic', 74, 0, False)
    
    # Passed parameters checking function
    cases_test_cont_basic.stypy_localization = localization
    cases_test_cont_basic.stypy_type_of_self = None
    cases_test_cont_basic.stypy_type_store = module_type_store
    cases_test_cont_basic.stypy_function_name = 'cases_test_cont_basic'
    cases_test_cont_basic.stypy_param_names_list = []
    cases_test_cont_basic.stypy_varargs_param_name = None
    cases_test_cont_basic.stypy_kwargs_param_name = None
    cases_test_cont_basic.stypy_call_defaults = defaults
    cases_test_cont_basic.stypy_call_varargs = varargs
    cases_test_cont_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cases_test_cont_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cases_test_cont_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cases_test_cont_basic(...)' code ##################

    
    
    # Obtaining the type of the subscript
    slice_633238 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 75, 25), None, None, None)
    # Getting the type of 'distcont' (line 75)
    distcont_633239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'distcont')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___633240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), distcont_633239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_633241 = invoke(stypy.reporting.localization.Localization(__file__, 75, 25), getitem___633240, slice_633238)
    
    
    # Obtaining an instance of the builtin type 'list' (line 75)
    list_633242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 75)
    # Adding element type (line 75)
    
    # Obtaining an instance of the builtin type 'tuple' (line 75)
    tuple_633243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'histogram_test_instance' (line 75)
    histogram_test_instance_633244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 41), 'histogram_test_instance')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 41), tuple_633243, histogram_test_instance_633244)
    # Adding element type (line 75)
    
    # Call to tuple(...): (line 75)
    # Processing the call keyword arguments (line 75)
    kwargs_633246 = {}
    # Getting the type of 'tuple' (line 75)
    tuple_633245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 66), 'tuple', False)
    # Calling tuple(args, kwargs) (line 75)
    tuple_call_result_633247 = invoke(stypy.reporting.localization.Localization(__file__, 75, 66), tuple_633245, *[], **kwargs_633246)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 41), tuple_633243, tuple_call_result_633247)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 39), list_633242, tuple_633243)
    
    # Applying the binary operator '+' (line 75)
    result_add_633248 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 25), '+', subscript_call_result_633241, list_633242)
    
    # Testing the type of a for loop iterable (line 75)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_add_633248)
    # Getting the type of the for loop variable (line 75)
    for_loop_var_633249 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 75, 4), result_add_633248)
    # Assigning a type to the variable 'distname' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'distname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 4), for_loop_var_633249))
    # Assigning a type to the variable 'arg' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 4), for_loop_var_633249))
    # SSA begins for a for statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'distname' (line 76)
    distname_633250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'distname')
    str_633251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 23), 'str', 'levy_stable')
    # Applying the binary operator '==' (line 76)
    result_eq_633252 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), '==', distname_633250, str_633251)
    
    # Testing the type of an if condition (line 76)
    if_condition_633253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_eq_633252)
    # Assigning a type to the variable 'if_condition_633253' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_633253', if_condition_633253)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA branch for the else part of an if statement (line 76)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'distname' (line 78)
    distname_633254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'distname')
    # Getting the type of 'distslow' (line 78)
    distslow_633255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'distslow')
    # Applying the binary operator 'in' (line 78)
    result_contains_633256 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 13), 'in', distname_633254, distslow_633255)
    
    # Testing the type of an if condition (line 78)
    if_condition_633257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 13), result_contains_633256)
    # Assigning a type to the variable 'if_condition_633257' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'if_condition_633257', if_condition_633257)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Creating a generator
    
    # Call to param(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'distname' (line 79)
    distname_633260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'distname', False)
    # Getting the type of 'arg' (line 79)
    arg_633261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 41), 'arg', False)
    # Processing the call keyword arguments (line 79)
    # Getting the type of 'pytest' (line 79)
    pytest_633262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 52), 'pytest', False)
    # Obtaining the member 'mark' of a type (line 79)
    mark_633263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 52), pytest_633262, 'mark')
    # Obtaining the member 'slow' of a type (line 79)
    slow_633264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 52), mark_633263, 'slow')
    keyword_633265 = slow_633264
    kwargs_633266 = {'marks': keyword_633265}
    # Getting the type of 'pytest' (line 79)
    pytest_633258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'pytest', False)
    # Obtaining the member 'param' of a type (line 79)
    param_633259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 18), pytest_633258, 'param')
    # Calling param(args, kwargs) (line 79)
    param_call_result_633267 = invoke(stypy.reporting.localization.Localization(__file__, 79, 18), param_633259, *[distname_633260, arg_633261], **kwargs_633266)
    
    GeneratorType_633268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 12), GeneratorType_633268, param_call_result_633267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type', GeneratorType_633268)
    # SSA branch for the else part of an if statement (line 78)
    module_type_store.open_ssa_branch('else')
    # Creating a generator
    
    # Obtaining an instance of the builtin type 'tuple' (line 81)
    tuple_633269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 81)
    # Adding element type (line 81)
    # Getting the type of 'distname' (line 81)
    distname_633270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 18), 'distname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 18), tuple_633269, distname_633270)
    # Adding element type (line 81)
    # Getting the type of 'arg' (line 81)
    arg_633271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'arg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 18), tuple_633269, arg_633271)
    
    GeneratorType_633272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 12), GeneratorType_633272, tuple_633269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', GeneratorType_633272)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'cases_test_cont_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cases_test_cont_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_633273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_633273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cases_test_cont_basic'
    return stypy_return_type_633273

# Assigning a type to the variable 'cases_test_cont_basic' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'cases_test_cont_basic', cases_test_cont_basic)

@norecursion
def test_cont_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_cont_basic'
    module_type_store = module_type_store.open_function_context('test_cont_basic', 84, 0, False)
    
    # Passed parameters checking function
    test_cont_basic.stypy_localization = localization
    test_cont_basic.stypy_type_of_self = None
    test_cont_basic.stypy_type_store = module_type_store
    test_cont_basic.stypy_function_name = 'test_cont_basic'
    test_cont_basic.stypy_param_names_list = ['distname', 'arg']
    test_cont_basic.stypy_varargs_param_name = None
    test_cont_basic.stypy_kwargs_param_name = None
    test_cont_basic.stypy_call_defaults = defaults
    test_cont_basic.stypy_call_varargs = varargs
    test_cont_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_cont_basic', ['distname', 'arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_cont_basic', localization, ['distname', 'arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_cont_basic(...)' code ##################

    
    
    # Getting the type of 'distname' (line 88)
    distname_633274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 7), 'distname')
    str_633275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 19), 'str', 'truncnorm')
    # Applying the binary operator '==' (line 88)
    result_eq_633276 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 7), '==', distname_633274, str_633275)
    
    # Testing the type of an if condition (line 88)
    if_condition_633277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 4), result_eq_633276)
    # Assigning a type to the variable 'if_condition_633277' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'if_condition_633277', if_condition_633277)
    # SSA begins for if statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to xfail(...): (line 89)
    # Processing the call keyword arguments (line 89)
    # Getting the type of 'distname' (line 89)
    distname_633280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'distname', False)
    keyword_633281 = distname_633280
    kwargs_633282 = {'reason': keyword_633281}
    # Getting the type of 'pytest' (line 89)
    pytest_633278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'pytest', False)
    # Obtaining the member 'xfail' of a type (line 89)
    xfail_633279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), pytest_633278, 'xfail')
    # Calling xfail(args, kwargs) (line 89)
    xfail_call_result_633283 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), xfail_633279, *[], **kwargs_633282)
    
    # SSA join for if statement (line 88)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to getattr(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'stats' (line 92)
    stats_633285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'stats', False)
    # Getting the type of 'distname' (line 92)
    distname_633286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'distname', False)
    # Processing the call keyword arguments (line 92)
    kwargs_633287 = {}
    # Getting the type of 'getattr' (line 92)
    getattr_633284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'getattr', False)
    # Calling getattr(args, kwargs) (line 92)
    getattr_call_result_633288 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), getattr_633284, *[stats_633285, distname_633286], **kwargs_633287)
    
    # Assigning a type to the variable 'distfn' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'distfn', getattr_call_result_633288)
    # SSA branch for the except part of a try statement (line 91)
    # SSA branch for the except 'TypeError' branch of a try statement (line 91)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 94):
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'distname' (line 94)
    distname_633289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'distname')
    # Assigning a type to the variable 'distfn' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'distfn', distname_633289)
    
    # Assigning a Str to a Name (line 95):
    
    # Assigning a Str to a Name (line 95):
    str_633290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'str', 'rv_histogram_instance')
    # Assigning a type to the variable 'distname' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'distname', str_633290)
    # SSA join for try-except statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to seed(...): (line 96)
    # Processing the call arguments (line 96)
    int_633294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'int')
    # Processing the call keyword arguments (line 96)
    kwargs_633295 = {}
    # Getting the type of 'np' (line 96)
    np_633291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 96)
    random_633292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 4), np_633291, 'random')
    # Obtaining the member 'seed' of a type (line 96)
    seed_633293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 4), random_633292, 'seed')
    # Calling seed(args, kwargs) (line 96)
    seed_call_result_633296 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), seed_633293, *[int_633294], **kwargs_633295)
    
    
    # Assigning a Num to a Name (line 97):
    
    # Assigning a Num to a Name (line 97):
    int_633297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 9), 'int')
    # Assigning a type to the variable 'sn' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'sn', int_633297)
    
    # Call to suppress_warnings(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_633299 = {}
    # Getting the type of 'suppress_warnings' (line 98)
    suppress_warnings_633298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 98)
    suppress_warnings_call_result_633300 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), suppress_warnings_633298, *[], **kwargs_633299)
    
    with_633301 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 98, 9), suppress_warnings_call_result_633300, 'with parameter', '__enter__', '__exit__')

    if with_633301:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 98)
        enter___633302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 9), suppress_warnings_call_result_633300, '__enter__')
        with_enter_633303 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), enter___633302)
        # Assigning a type to the variable 'sup' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'sup', with_enter_633303)
        
        # Call to filter(...): (line 101)
        # Processing the call keyword arguments (line 101)
        # Getting the type of 'DeprecationWarning' (line 101)
        DeprecationWarning_633306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'DeprecationWarning', False)
        keyword_633307 = DeprecationWarning_633306
        str_633308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 56), 'str', '.*frechet_')
        keyword_633309 = str_633308
        kwargs_633310 = {'category': keyword_633307, 'message': keyword_633309}
        # Getting the type of 'sup' (line 101)
        sup_633304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 101)
        filter_633305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), sup_633304, 'filter')
        # Calling filter(args, kwargs) (line 101)
        filter_call_result_633311 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), filter_633305, *[], **kwargs_633310)
        
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to rvs(...): (line 102)
        # Getting the type of 'arg' (line 102)
        arg_633314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'arg', False)
        # Processing the call keyword arguments (line 102)
        # Getting the type of 'sn' (line 102)
        sn_633315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'sn', False)
        keyword_633316 = sn_633315
        kwargs_633317 = {'size': keyword_633316}
        # Getting the type of 'distfn' (line 102)
        distfn_633312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'distfn', False)
        # Obtaining the member 'rvs' of a type (line 102)
        rvs_633313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 14), distfn_633312, 'rvs')
        # Calling rvs(args, kwargs) (line 102)
        rvs_call_result_633318 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), rvs_633313, *[arg_633314], **kwargs_633317)
        
        # Assigning a type to the variable 'rvs' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'rvs', rvs_call_result_633318)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to mean(...): (line 103)
        # Processing the call keyword arguments (line 103)
        kwargs_633321 = {}
        # Getting the type of 'rvs' (line 103)
        rvs_633319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'rvs', False)
        # Obtaining the member 'mean' of a type (line 103)
        mean_633320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 13), rvs_633319, 'mean')
        # Calling mean(args, kwargs) (line 103)
        mean_call_result_633322 = invoke(stypy.reporting.localization.Localization(__file__, 103, 13), mean_633320, *[], **kwargs_633321)
        
        # Assigning a type to the variable 'sm' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'sm', mean_call_result_633322)
        
        # Assigning a Call to a Name (line 104):
        
        # Assigning a Call to a Name (line 104):
        
        # Call to var(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_633325 = {}
        # Getting the type of 'rvs' (line 104)
        rvs_633323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'rvs', False)
        # Obtaining the member 'var' of a type (line 104)
        var_633324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), rvs_633323, 'var')
        # Calling var(args, kwargs) (line 104)
        var_call_result_633326 = invoke(stypy.reporting.localization.Localization(__file__, 104, 13), var_633324, *[], **kwargs_633325)
        
        # Assigning a type to the variable 'sv' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'sv', var_call_result_633326)
        
        # Assigning a Call to a Tuple (line 105):
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_633327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'int')
        
        # Call to stats(...): (line 105)
        # Getting the type of 'arg' (line 105)
        arg_633330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'arg', False)
        # Processing the call keyword arguments (line 105)
        kwargs_633331 = {}
        # Getting the type of 'distfn' (line 105)
        distfn_633328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'distfn', False)
        # Obtaining the member 'stats' of a type (line 105)
        stats_633329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), distfn_633328, 'stats')
        # Calling stats(args, kwargs) (line 105)
        stats_call_result_633332 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), stats_633329, *[arg_633330], **kwargs_633331)
        
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___633333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), stats_call_result_633332, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_633334 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), getitem___633333, int_633327)
        
        # Assigning a type to the variable 'tuple_var_assignment_633085' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_633085', subscript_call_result_633334)
        
        # Assigning a Subscript to a Name (line 105):
        
        # Obtaining the type of the subscript
        int_633335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'int')
        
        # Call to stats(...): (line 105)
        # Getting the type of 'arg' (line 105)
        arg_633338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'arg', False)
        # Processing the call keyword arguments (line 105)
        kwargs_633339 = {}
        # Getting the type of 'distfn' (line 105)
        distfn_633336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'distfn', False)
        # Obtaining the member 'stats' of a type (line 105)
        stats_633337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), distfn_633336, 'stats')
        # Calling stats(args, kwargs) (line 105)
        stats_call_result_633340 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), stats_633337, *[arg_633338], **kwargs_633339)
        
        # Obtaining the member '__getitem__' of a type (line 105)
        getitem___633341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), stats_call_result_633340, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 105)
        subscript_call_result_633342 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), getitem___633341, int_633335)
        
        # Assigning a type to the variable 'tuple_var_assignment_633086' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_633086', subscript_call_result_633342)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_633085' (line 105)
        tuple_var_assignment_633085_633343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_633085')
        # Assigning a type to the variable 'm' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'm', tuple_var_assignment_633085_633343)
        
        # Assigning a Name to a Name (line 105):
        # Getting the type of 'tuple_var_assignment_633086' (line 105)
        tuple_var_assignment_633086_633344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tuple_var_assignment_633086')
        # Assigning a type to the variable 'v' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'v', tuple_var_assignment_633086_633344)
        
        # Call to check_sample_meanvar_(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'distfn' (line 107)
        distfn_633346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'distfn', False)
        # Getting the type of 'arg' (line 107)
        arg_633347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 38), 'arg', False)
        # Getting the type of 'm' (line 107)
        m_633348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 43), 'm', False)
        # Getting the type of 'v' (line 107)
        v_633349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 46), 'v', False)
        # Getting the type of 'sm' (line 107)
        sm_633350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 49), 'sm', False)
        # Getting the type of 'sv' (line 107)
        sv_633351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 53), 'sv', False)
        # Getting the type of 'sn' (line 107)
        sn_633352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 57), 'sn', False)
        # Getting the type of 'distname' (line 107)
        distname_633353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 61), 'distname', False)
        str_633354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 72), 'str', 'sample mean test')
        # Applying the binary operator '+' (line 107)
        result_add_633355 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 61), '+', distname_633353, str_633354)
        
        # Processing the call keyword arguments (line 107)
        kwargs_633356 = {}
        # Getting the type of 'check_sample_meanvar_' (line 107)
        check_sample_meanvar__633345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'check_sample_meanvar_', False)
        # Calling check_sample_meanvar_(args, kwargs) (line 107)
        check_sample_meanvar__call_result_633357 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), check_sample_meanvar__633345, *[distfn_633346, arg_633347, m_633348, v_633349, sm_633350, sv_633351, sn_633352, result_add_633355], **kwargs_633356)
        
        
        # Call to check_cdf_ppf(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'distfn' (line 108)
        distfn_633359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'distfn', False)
        # Getting the type of 'arg' (line 108)
        arg_633360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'arg', False)
        # Getting the type of 'distname' (line 108)
        distname_633361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'distname', False)
        # Processing the call keyword arguments (line 108)
        kwargs_633362 = {}
        # Getting the type of 'check_cdf_ppf' (line 108)
        check_cdf_ppf_633358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'check_cdf_ppf', False)
        # Calling check_cdf_ppf(args, kwargs) (line 108)
        check_cdf_ppf_call_result_633363 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), check_cdf_ppf_633358, *[distfn_633359, arg_633360, distname_633361], **kwargs_633362)
        
        
        # Call to check_sf_isf(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'distfn' (line 109)
        distfn_633365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'distfn', False)
        # Getting the type of 'arg' (line 109)
        arg_633366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 29), 'arg', False)
        # Getting the type of 'distname' (line 109)
        distname_633367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'distname', False)
        # Processing the call keyword arguments (line 109)
        kwargs_633368 = {}
        # Getting the type of 'check_sf_isf' (line 109)
        check_sf_isf_633364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'check_sf_isf', False)
        # Calling check_sf_isf(args, kwargs) (line 109)
        check_sf_isf_call_result_633369 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), check_sf_isf_633364, *[distfn_633365, arg_633366, distname_633367], **kwargs_633368)
        
        
        # Call to check_pdf(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'distfn' (line 110)
        distfn_633371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'distfn', False)
        # Getting the type of 'arg' (line 110)
        arg_633372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'arg', False)
        # Getting the type of 'distname' (line 110)
        distname_633373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 31), 'distname', False)
        # Processing the call keyword arguments (line 110)
        kwargs_633374 = {}
        # Getting the type of 'check_pdf' (line 110)
        check_pdf_633370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'check_pdf', False)
        # Calling check_pdf(args, kwargs) (line 110)
        check_pdf_call_result_633375 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), check_pdf_633370, *[distfn_633371, arg_633372, distname_633373], **kwargs_633374)
        
        
        # Call to check_pdf_logpdf(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'distfn' (line 111)
        distfn_633377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'distfn', False)
        # Getting the type of 'arg' (line 111)
        arg_633378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'arg', False)
        # Getting the type of 'distname' (line 111)
        distname_633379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'distname', False)
        # Processing the call keyword arguments (line 111)
        kwargs_633380 = {}
        # Getting the type of 'check_pdf_logpdf' (line 111)
        check_pdf_logpdf_633376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'check_pdf_logpdf', False)
        # Calling check_pdf_logpdf(args, kwargs) (line 111)
        check_pdf_logpdf_call_result_633381 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), check_pdf_logpdf_633376, *[distfn_633377, arg_633378, distname_633379], **kwargs_633380)
        
        
        # Call to check_cdf_logcdf(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'distfn' (line 112)
        distfn_633383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'distfn', False)
        # Getting the type of 'arg' (line 112)
        arg_633384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 33), 'arg', False)
        # Getting the type of 'distname' (line 112)
        distname_633385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'distname', False)
        # Processing the call keyword arguments (line 112)
        kwargs_633386 = {}
        # Getting the type of 'check_cdf_logcdf' (line 112)
        check_cdf_logcdf_633382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'check_cdf_logcdf', False)
        # Calling check_cdf_logcdf(args, kwargs) (line 112)
        check_cdf_logcdf_call_result_633387 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), check_cdf_logcdf_633382, *[distfn_633383, arg_633384, distname_633385], **kwargs_633386)
        
        
        # Call to check_sf_logsf(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'distfn' (line 113)
        distfn_633389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'distfn', False)
        # Getting the type of 'arg' (line 113)
        arg_633390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'arg', False)
        # Getting the type of 'distname' (line 113)
        distname_633391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 36), 'distname', False)
        # Processing the call keyword arguments (line 113)
        kwargs_633392 = {}
        # Getting the type of 'check_sf_logsf' (line 113)
        check_sf_logsf_633388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'check_sf_logsf', False)
        # Calling check_sf_logsf(args, kwargs) (line 113)
        check_sf_logsf_call_result_633393 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), check_sf_logsf_633388, *[distfn_633389, arg_633390, distname_633391], **kwargs_633392)
        
        
        # Assigning a Num to a Name (line 115):
        
        # Assigning a Num to a Name (line 115):
        float_633394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 16), 'float')
        # Assigning a type to the variable 'alpha' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'alpha', float_633394)
        
        
        # Getting the type of 'distname' (line 116)
        distname_633395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'distname')
        str_633396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'str', 'rv_histogram_instance')
        # Applying the binary operator '==' (line 116)
        result_eq_633397 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 11), '==', distname_633395, str_633396)
        
        # Testing the type of an if condition (line 116)
        if_condition_633398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), result_eq_633397)
        # Assigning a type to the variable 'if_condition_633398' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_633398', if_condition_633398)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_distribution_rvs(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'distfn' (line 117)
        distfn_633400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 35), 'distfn', False)
        # Obtaining the member 'cdf' of a type (line 117)
        cdf_633401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 35), distfn_633400, 'cdf')
        # Getting the type of 'arg' (line 117)
        arg_633402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 47), 'arg', False)
        # Getting the type of 'alpha' (line 117)
        alpha_633403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 52), 'alpha', False)
        # Getting the type of 'rvs' (line 117)
        rvs_633404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 59), 'rvs', False)
        # Processing the call keyword arguments (line 117)
        kwargs_633405 = {}
        # Getting the type of 'check_distribution_rvs' (line 117)
        check_distribution_rvs_633399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'check_distribution_rvs', False)
        # Calling check_distribution_rvs(args, kwargs) (line 117)
        check_distribution_rvs_call_result_633406 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), check_distribution_rvs_633399, *[cdf_633401, arg_633402, alpha_633403, rvs_633404], **kwargs_633405)
        
        # SSA branch for the else part of an if statement (line 116)
        module_type_store.open_ssa_branch('else')
        
        # Call to check_distribution_rvs(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'distname' (line 119)
        distname_633408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'distname', False)
        # Getting the type of 'arg' (line 119)
        arg_633409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 45), 'arg', False)
        # Getting the type of 'alpha' (line 119)
        alpha_633410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 50), 'alpha', False)
        # Getting the type of 'rvs' (line 119)
        rvs_633411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 57), 'rvs', False)
        # Processing the call keyword arguments (line 119)
        kwargs_633412 = {}
        # Getting the type of 'check_distribution_rvs' (line 119)
        check_distribution_rvs_633407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'check_distribution_rvs', False)
        # Calling check_distribution_rvs(args, kwargs) (line 119)
        check_distribution_rvs_call_result_633413 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), check_distribution_rvs_633407, *[distname_633408, arg_633409, alpha_633410, rvs_633411], **kwargs_633412)
        
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 121):
        
        # Assigning a Tuple to a Name (line 121):
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_633414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        int_633415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 29), tuple_633414, int_633415)
        # Adding element type (line 121)
        int_633416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 29), tuple_633414, int_633416)
        
        # Assigning a type to the variable 'locscale_defaults' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'locscale_defaults', tuple_633414)
        
        # Assigning a List to a Name (line 122):
        
        # Assigning a List to a Name (line 122):
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_633417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        # Getting the type of 'distfn' (line 122)
        distfn_633418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'distfn')
        # Obtaining the member 'pdf' of a type (line 122)
        pdf_633419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), distfn_633418, 'pdf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), list_633417, pdf_633419)
        # Adding element type (line 122)
        # Getting the type of 'distfn' (line 122)
        distfn_633420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'distfn')
        # Obtaining the member 'logpdf' of a type (line 122)
        logpdf_633421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 29), distfn_633420, 'logpdf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), list_633417, logpdf_633421)
        # Adding element type (line 122)
        # Getting the type of 'distfn' (line 122)
        distfn_633422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 44), 'distfn')
        # Obtaining the member 'cdf' of a type (line 122)
        cdf_633423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 44), distfn_633422, 'cdf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), list_633417, cdf_633423)
        # Adding element type (line 122)
        # Getting the type of 'distfn' (line 122)
        distfn_633424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 56), 'distfn')
        # Obtaining the member 'logcdf' of a type (line 122)
        logcdf_633425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 56), distfn_633424, 'logcdf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), list_633417, logcdf_633425)
        # Adding element type (line 122)
        # Getting the type of 'distfn' (line 123)
        distfn_633426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'distfn')
        # Obtaining the member 'logsf' of a type (line 123)
        logsf_633427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 17), distfn_633426, 'logsf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), list_633417, logsf_633427)
        
        # Assigning a type to the variable 'meths' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'meths', list_633417)
        
        # Assigning a Dict to a Name (line 125):
        
        # Assigning a Dict to a Name (line 125):
        
        # Obtaining an instance of the builtin type 'dict' (line 125)
        dict_633428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 17), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 125)
        # Adding element type (key, value) (line 125)
        str_633429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'str', 'frechet_l')
        float_633430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 31), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 17), dict_633428, (str_633429, float_633430))
        # Adding element type (key, value) (line 125)
        str_633431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 37), 'str', 'weibull_max')
        float_633432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 52), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 17), dict_633428, (str_633431, float_633432))
        # Adding element type (key, value) (line 125)
        str_633433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 58), 'str', 'levy_l')
        float_633434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 68), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 17), dict_633428, (str_633433, float_633434))
        # Adding element type (key, value) (line 125)
        str_633435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 18), 'str', 'pareto')
        float_633436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 17), dict_633428, (str_633435, float_633436))
        # Adding element type (key, value) (line 125)
        str_633437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 33), 'str', 'tukeylambda')
        float_633438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 48), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 17), dict_633428, (str_633437, float_633438))
        # Adding element type (key, value) (line 125)
        str_633439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'str', 'rv_histogram_instance')
        float_633440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 43), 'float')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 17), dict_633428, (str_633439, float_633440))
        
        # Assigning a type to the variable 'spec_x' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'spec_x', dict_633428)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to get(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'distname' (line 128)
        distname_633443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'distname', False)
        float_633444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 33), 'float')
        # Processing the call keyword arguments (line 128)
        kwargs_633445 = {}
        # Getting the type of 'spec_x' (line 128)
        spec_x_633441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'spec_x', False)
        # Obtaining the member 'get' of a type (line 128)
        get_633442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), spec_x_633441, 'get')
        # Calling get(args, kwargs) (line 128)
        get_call_result_633446 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), get_633442, *[distname_633443, float_633444], **kwargs_633445)
        
        # Assigning a type to the variable 'x' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'x', get_call_result_633446)
        
        
        # Getting the type of 'distname' (line 129)
        distname_633447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'distname')
        str_633448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 23), 'str', 'invweibull')
        # Applying the binary operator '==' (line 129)
        result_eq_633449 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 11), '==', distname_633447, str_633448)
        
        # Testing the type of an if condition (line 129)
        if_condition_633450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), result_eq_633449)
        # Assigning a type to the variable 'if_condition_633450' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_633450', if_condition_633450)
        # SSA begins for if statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 130):
        
        # Assigning a Tuple to a Name (line 130):
        
        # Obtaining an instance of the builtin type 'tuple' (line 130)
        tuple_633451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 130)
        # Adding element type (line 130)
        int_633452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 19), tuple_633451, int_633452)
        
        # Assigning a type to the variable 'arg' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'arg', tuple_633451)
        # SSA branch for the else part of an if statement (line 129)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'distname' (line 131)
        distname_633453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'distname')
        str_633454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 25), 'str', 'ksone')
        # Applying the binary operator '==' (line 131)
        result_eq_633455 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 13), '==', distname_633453, str_633454)
        
        # Testing the type of an if condition (line 131)
        if_condition_633456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 13), result_eq_633455)
        # Assigning a type to the variable 'if_condition_633456' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'if_condition_633456', if_condition_633456)
        # SSA begins for if statement (line 131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 132):
        
        # Assigning a Tuple to a Name (line 132):
        
        # Obtaining an instance of the builtin type 'tuple' (line 132)
        tuple_633457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 132)
        # Adding element type (line 132)
        int_633458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 19), tuple_633457, int_633458)
        
        # Assigning a type to the variable 'arg' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'arg', tuple_633457)
        # SSA join for if statement (line 131)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 129)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to check_named_args(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'distfn' (line 133)
        distfn_633460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'distfn', False)
        # Getting the type of 'x' (line 133)
        x_633461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 33), 'x', False)
        # Getting the type of 'arg' (line 133)
        arg_633462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'arg', False)
        # Getting the type of 'locscale_defaults' (line 133)
        locscale_defaults_633463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), 'locscale_defaults', False)
        # Getting the type of 'meths' (line 133)
        meths_633464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 60), 'meths', False)
        # Processing the call keyword arguments (line 133)
        kwargs_633465 = {}
        # Getting the type of 'check_named_args' (line 133)
        check_named_args_633459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'check_named_args', False)
        # Calling check_named_args(args, kwargs) (line 133)
        check_named_args_call_result_633466 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), check_named_args_633459, *[distfn_633460, x_633461, arg_633462, locscale_defaults_633463, meths_633464], **kwargs_633465)
        
        
        # Call to check_random_state_property(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'distfn' (line 134)
        distfn_633468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'distfn', False)
        # Getting the type of 'arg' (line 134)
        arg_633469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 44), 'arg', False)
        # Processing the call keyword arguments (line 134)
        kwargs_633470 = {}
        # Getting the type of 'check_random_state_property' (line 134)
        check_random_state_property_633467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'check_random_state_property', False)
        # Calling check_random_state_property(args, kwargs) (line 134)
        check_random_state_property_call_result_633471 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), check_random_state_property_633467, *[distfn_633468, arg_633469], **kwargs_633470)
        
        
        # Call to check_pickling(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'distfn' (line 135)
        distfn_633473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'distfn', False)
        # Getting the type of 'arg' (line 135)
        arg_633474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'arg', False)
        # Processing the call keyword arguments (line 135)
        kwargs_633475 = {}
        # Getting the type of 'check_pickling' (line 135)
        check_pickling_633472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'check_pickling', False)
        # Calling check_pickling(args, kwargs) (line 135)
        check_pickling_call_result_633476 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), check_pickling_633472, *[distfn_633473, arg_633474], **kwargs_633475)
        
        
        
        # Getting the type of 'distname' (line 138)
        distname_633477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'distname')
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_633478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        str_633479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 28), 'str', 'ksone')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 27), list_633478, str_633479)
        # Adding element type (line 138)
        str_633480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 37), 'str', 'kstwobign')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 27), list_633478, str_633480)
        
        # Applying the binary operator 'notin' (line 138)
        result_contains_633481 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 11), 'notin', distname_633477, list_633478)
        
        # Testing the type of an if condition (line 138)
        if_condition_633482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 8), result_contains_633481)
        # Assigning a type to the variable 'if_condition_633482' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'if_condition_633482', if_condition_633482)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_entropy(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'distfn' (line 139)
        distfn_633484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'distfn', False)
        # Getting the type of 'arg' (line 139)
        arg_633485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'arg', False)
        # Getting the type of 'distname' (line 139)
        distname_633486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 39), 'distname', False)
        # Processing the call keyword arguments (line 139)
        kwargs_633487 = {}
        # Getting the type of 'check_entropy' (line 139)
        check_entropy_633483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'check_entropy', False)
        # Calling check_entropy(args, kwargs) (line 139)
        check_entropy_call_result_633488 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), check_entropy_633483, *[distfn_633484, arg_633485, distname_633486], **kwargs_633487)
        
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'distfn' (line 141)
        distfn_633489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'distfn')
        # Obtaining the member 'numargs' of a type (line 141)
        numargs_633490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 11), distfn_633489, 'numargs')
        int_633491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'int')
        # Applying the binary operator '==' (line 141)
        result_eq_633492 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), '==', numargs_633490, int_633491)
        
        # Testing the type of an if condition (line 141)
        if_condition_633493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_eq_633492)
        # Assigning a type to the variable 'if_condition_633493' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_633493', if_condition_633493)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_vecentropy(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'distfn' (line 142)
        distfn_633495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'distfn', False)
        # Getting the type of 'arg' (line 142)
        arg_633496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), 'arg', False)
        # Processing the call keyword arguments (line 142)
        kwargs_633497 = {}
        # Getting the type of 'check_vecentropy' (line 142)
        check_vecentropy_633494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'check_vecentropy', False)
        # Calling check_vecentropy(args, kwargs) (line 142)
        check_vecentropy_call_result_633498 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), check_vecentropy_633494, *[distfn_633495, arg_633496], **kwargs_633497)
        
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'distfn' (line 144)
        distfn_633499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'distfn')
        # Obtaining the member '__class__' of a type (line 144)
        class___633500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), distfn_633499, '__class__')
        # Obtaining the member '_entropy' of a type (line 144)
        _entropy_633501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), class___633500, '_entropy')
        # Getting the type of 'stats' (line 144)
        stats_633502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'stats')
        # Obtaining the member 'rv_continuous' of a type (line 144)
        rv_continuous_633503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 41), stats_633502, 'rv_continuous')
        # Obtaining the member '_entropy' of a type (line 144)
        _entropy_633504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 41), rv_continuous_633503, '_entropy')
        # Applying the binary operator '!=' (line 144)
        result_ne_633505 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), '!=', _entropy_633501, _entropy_633504)
        
        
        # Getting the type of 'distname' (line 145)
        distname_633506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'distname')
        str_633507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 32), 'str', 'vonmises')
        # Applying the binary operator '!=' (line 145)
        result_ne_633508 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 20), '!=', distname_633506, str_633507)
        
        # Applying the binary operator 'and' (line 144)
        result_and_keyword_633509 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), 'and', result_ne_633505, result_ne_633508)
        
        # Testing the type of an if condition (line 144)
        if_condition_633510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), result_and_keyword_633509)
        # Assigning a type to the variable 'if_condition_633510' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_633510', if_condition_633510)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_private_entropy(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'distfn' (line 146)
        distfn_633512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'distfn', False)
        # Getting the type of 'arg' (line 146)
        arg_633513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'arg', False)
        # Getting the type of 'stats' (line 146)
        stats_633514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 47), 'stats', False)
        # Obtaining the member 'rv_continuous' of a type (line 146)
        rv_continuous_633515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 47), stats_633514, 'rv_continuous')
        # Processing the call keyword arguments (line 146)
        kwargs_633516 = {}
        # Getting the type of 'check_private_entropy' (line 146)
        check_private_entropy_633511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'check_private_entropy', False)
        # Calling check_private_entropy(args, kwargs) (line 146)
        check_private_entropy_call_result_633517 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), check_private_entropy_633511, *[distfn_633512, arg_633513, rv_continuous_633515], **kwargs_633516)
        
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to check_edge_support(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'distfn' (line 148)
        distfn_633519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'distfn', False)
        # Getting the type of 'arg' (line 148)
        arg_633520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'arg', False)
        # Processing the call keyword arguments (line 148)
        kwargs_633521 = {}
        # Getting the type of 'check_edge_support' (line 148)
        check_edge_support_633518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'check_edge_support', False)
        # Calling check_edge_support(args, kwargs) (line 148)
        check_edge_support_call_result_633522 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), check_edge_support_633518, *[distfn_633519, arg_633520], **kwargs_633521)
        
        
        # Call to check_meth_dtype(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'distfn' (line 150)
        distfn_633524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 25), 'distfn', False)
        # Getting the type of 'arg' (line 150)
        arg_633525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 33), 'arg', False)
        # Getting the type of 'meths' (line 150)
        meths_633526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 38), 'meths', False)
        # Processing the call keyword arguments (line 150)
        kwargs_633527 = {}
        # Getting the type of 'check_meth_dtype' (line 150)
        check_meth_dtype_633523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'check_meth_dtype', False)
        # Calling check_meth_dtype(args, kwargs) (line 150)
        check_meth_dtype_call_result_633528 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), check_meth_dtype_633523, *[distfn_633524, arg_633525, meths_633526], **kwargs_633527)
        
        
        # Call to check_ppf_dtype(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'distfn' (line 151)
        distfn_633530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'distfn', False)
        # Getting the type of 'arg' (line 151)
        arg_633531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 32), 'arg', False)
        # Processing the call keyword arguments (line 151)
        kwargs_633532 = {}
        # Getting the type of 'check_ppf_dtype' (line 151)
        check_ppf_dtype_633529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'check_ppf_dtype', False)
        # Calling check_ppf_dtype(args, kwargs) (line 151)
        check_ppf_dtype_call_result_633533 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), check_ppf_dtype_633529, *[distfn_633530, arg_633531], **kwargs_633532)
        
        
        
        # Getting the type of 'distname' (line 153)
        distname_633534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'distname')
        # Getting the type of 'fails_cmplx' (line 153)
        fails_cmplx_633535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 27), 'fails_cmplx')
        # Applying the binary operator 'notin' (line 153)
        result_contains_633536 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), 'notin', distname_633534, fails_cmplx_633535)
        
        # Testing the type of an if condition (line 153)
        if_condition_633537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), result_contains_633536)
        # Assigning a type to the variable 'if_condition_633537' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_633537', if_condition_633537)
        # SSA begins for if statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_cmplx_deriv(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'distfn' (line 154)
        distfn_633539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 30), 'distfn', False)
        # Getting the type of 'arg' (line 154)
        arg_633540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 38), 'arg', False)
        # Processing the call keyword arguments (line 154)
        kwargs_633541 = {}
        # Getting the type of 'check_cmplx_deriv' (line 154)
        check_cmplx_deriv_633538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'check_cmplx_deriv', False)
        # Calling check_cmplx_deriv(args, kwargs) (line 154)
        check_cmplx_deriv_call_result_633542 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), check_cmplx_deriv_633538, *[distfn_633539, arg_633540], **kwargs_633541)
        
        # SSA join for if statement (line 153)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'distname' (line 156)
        distname_633543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'distname')
        str_633544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 23), 'str', 'truncnorm')
        # Applying the binary operator '!=' (line 156)
        result_ne_633545 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), '!=', distname_633543, str_633544)
        
        # Testing the type of an if condition (line 156)
        if_condition_633546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_ne_633545)
        # Assigning a type to the variable 'if_condition_633546' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_633546', if_condition_633546)
        # SSA begins for if statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_ppf_private(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'distfn' (line 157)
        distfn_633548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'distfn', False)
        # Getting the type of 'arg' (line 157)
        arg_633549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 38), 'arg', False)
        # Getting the type of 'distname' (line 157)
        distname_633550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'distname', False)
        # Processing the call keyword arguments (line 157)
        kwargs_633551 = {}
        # Getting the type of 'check_ppf_private' (line 157)
        check_ppf_private_633547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'check_ppf_private', False)
        # Calling check_ppf_private(args, kwargs) (line 157)
        check_ppf_private_call_result_633552 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), check_ppf_private_633547, *[distfn_633548, arg_633549, distname_633550], **kwargs_633551)
        
        # SSA join for if statement (line 156)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 98)
        exit___633553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 9), suppress_warnings_call_result_633300, '__exit__')
        with_exit_633554 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), exit___633553, None, None, None)

    
    # ################# End of 'test_cont_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_cont_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_633555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_633555)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_cont_basic'
    return stypy_return_type_633555

# Assigning a type to the variable 'test_cont_basic' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'test_cont_basic', test_cont_basic)

@norecursion
def test_levy_stable_random_state_property(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_levy_stable_random_state_property'
    module_type_store = module_type_store.open_function_context('test_levy_stable_random_state_property', 160, 0, False)
    
    # Passed parameters checking function
    test_levy_stable_random_state_property.stypy_localization = localization
    test_levy_stable_random_state_property.stypy_type_of_self = None
    test_levy_stable_random_state_property.stypy_type_store = module_type_store
    test_levy_stable_random_state_property.stypy_function_name = 'test_levy_stable_random_state_property'
    test_levy_stable_random_state_property.stypy_param_names_list = []
    test_levy_stable_random_state_property.stypy_varargs_param_name = None
    test_levy_stable_random_state_property.stypy_kwargs_param_name = None
    test_levy_stable_random_state_property.stypy_call_defaults = defaults
    test_levy_stable_random_state_property.stypy_call_varargs = varargs
    test_levy_stable_random_state_property.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_levy_stable_random_state_property', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_levy_stable_random_state_property', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_levy_stable_random_state_property(...)' code ##################

    
    # Call to check_random_state_property(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'stats' (line 164)
    stats_633557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 32), 'stats', False)
    # Obtaining the member 'levy_stable' of a type (line 164)
    levy_stable_633558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 32), stats_633557, 'levy_stable')
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_633559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    float_633560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 52), tuple_633559, float_633560)
    # Adding element type (line 164)
    float_633561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 57), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 52), tuple_633559, float_633561)
    
    # Processing the call keyword arguments (line 164)
    kwargs_633562 = {}
    # Getting the type of 'check_random_state_property' (line 164)
    check_random_state_property_633556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'check_random_state_property', False)
    # Calling check_random_state_property(args, kwargs) (line 164)
    check_random_state_property_call_result_633563 = invoke(stypy.reporting.localization.Localization(__file__, 164, 4), check_random_state_property_633556, *[levy_stable_633558, tuple_633559], **kwargs_633562)
    
    
    # ################# End of 'test_levy_stable_random_state_property(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_levy_stable_random_state_property' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_633564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_633564)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_levy_stable_random_state_property'
    return stypy_return_type_633564

# Assigning a type to the variable 'test_levy_stable_random_state_property' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'test_levy_stable_random_state_property', test_levy_stable_random_state_property)

@norecursion
def cases_test_moments(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cases_test_moments'
    module_type_store = module_type_store.open_function_context('cases_test_moments', 167, 0, False)
    
    # Passed parameters checking function
    cases_test_moments.stypy_localization = localization
    cases_test_moments.stypy_type_of_self = None
    cases_test_moments.stypy_type_store = module_type_store
    cases_test_moments.stypy_function_name = 'cases_test_moments'
    cases_test_moments.stypy_param_names_list = []
    cases_test_moments.stypy_varargs_param_name = None
    cases_test_moments.stypy_kwargs_param_name = None
    cases_test_moments.stypy_call_defaults = defaults
    cases_test_moments.stypy_call_varargs = varargs
    cases_test_moments.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cases_test_moments', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cases_test_moments', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cases_test_moments(...)' code ##################

    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to set(...): (line 168)
    # Processing the call arguments (line 168)
    
    # Obtaining an instance of the builtin type 'list' (line 168)
    list_633566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 168)
    # Adding element type (line 168)
    str_633567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 30), 'str', 'vonmises')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 29), list_633566, str_633567)
    # Adding element type (line 168)
    str_633568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 42), 'str', 'ksone')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 29), list_633566, str_633568)
    
    # Processing the call keyword arguments (line 168)
    kwargs_633569 = {}
    # Getting the type of 'set' (line 168)
    set_633565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'set', False)
    # Calling set(args, kwargs) (line 168)
    set_call_result_633570 = invoke(stypy.reporting.localization.Localization(__file__, 168, 25), set_633565, *[list_633566], **kwargs_633569)
    
    # Assigning a type to the variable 'fail_normalization' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'fail_normalization', set_call_result_633570)
    
    # Assigning a Call to a Name (line 169):
    
    # Assigning a Call to a Name (line 169):
    
    # Call to set(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Obtaining an instance of the builtin type 'list' (line 169)
    list_633572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 169)
    # Adding element type (line 169)
    str_633573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 23), 'str', 'vonmises')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), list_633572, str_633573)
    # Adding element type (line 169)
    str_633574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 35), 'str', 'ksone')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), list_633572, str_633574)
    # Adding element type (line 169)
    str_633575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 44), 'str', 'ncf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 22), list_633572, str_633575)
    
    # Processing the call keyword arguments (line 169)
    kwargs_633576 = {}
    # Getting the type of 'set' (line 169)
    set_633571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 18), 'set', False)
    # Calling set(args, kwargs) (line 169)
    set_call_result_633577 = invoke(stypy.reporting.localization.Localization(__file__, 169, 18), set_633571, *[list_633572], **kwargs_633576)
    
    # Assigning a type to the variable 'fail_higher' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'fail_higher', set_call_result_633577)
    
    
    # Obtaining the type of the subscript
    slice_633578 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 171, 25), None, None, None)
    # Getting the type of 'distcont' (line 171)
    distcont_633579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'distcont')
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___633580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 25), distcont_633579, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_633581 = invoke(stypy.reporting.localization.Localization(__file__, 171, 25), getitem___633580, slice_633578)
    
    
    # Obtaining an instance of the builtin type 'list' (line 171)
    list_633582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 171)
    # Adding element type (line 171)
    
    # Obtaining an instance of the builtin type 'tuple' (line 171)
    tuple_633583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 171)
    # Adding element type (line 171)
    # Getting the type of 'histogram_test_instance' (line 171)
    histogram_test_instance_633584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 41), 'histogram_test_instance')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 41), tuple_633583, histogram_test_instance_633584)
    # Adding element type (line 171)
    
    # Call to tuple(...): (line 171)
    # Processing the call keyword arguments (line 171)
    kwargs_633586 = {}
    # Getting the type of 'tuple' (line 171)
    tuple_633585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 66), 'tuple', False)
    # Calling tuple(args, kwargs) (line 171)
    tuple_call_result_633587 = invoke(stypy.reporting.localization.Localization(__file__, 171, 66), tuple_633585, *[], **kwargs_633586)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 41), tuple_633583, tuple_call_result_633587)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 39), list_633582, tuple_633583)
    
    # Applying the binary operator '+' (line 171)
    result_add_633588 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 25), '+', subscript_call_result_633581, list_633582)
    
    # Testing the type of a for loop iterable (line 171)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 4), result_add_633588)
    # Getting the type of the for loop variable (line 171)
    for_loop_var_633589 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 4), result_add_633588)
    # Assigning a type to the variable 'distname' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'distname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 4), for_loop_var_633589))
    # Assigning a type to the variable 'arg' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 4), for_loop_var_633589))
    # SSA begins for a for statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'distname' (line 172)
    distname_633590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'distname')
    str_633591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 23), 'str', 'levy_stable')
    # Applying the binary operator '==' (line 172)
    result_eq_633592 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 11), '==', distname_633590, str_633591)
    
    # Testing the type of an if condition (line 172)
    if_condition_633593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), result_eq_633592)
    # Assigning a type to the variable 'if_condition_633593' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_633593', if_condition_633593)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Compare to a Name (line 175):
    
    # Assigning a Compare to a Name (line 175):
    
    # Getting the type of 'distname' (line 175)
    distname_633594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'distname')
    # Getting the type of 'fail_normalization' (line 175)
    fail_normalization_633595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 32), 'fail_normalization')
    # Applying the binary operator 'notin' (line 175)
    result_contains_633596 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 16), 'notin', distname_633594, fail_normalization_633595)
    
    # Assigning a type to the variable 'cond1' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'cond1', result_contains_633596)
    
    # Assigning a Compare to a Name (line 176):
    
    # Assigning a Compare to a Name (line 176):
    
    # Getting the type of 'distname' (line 176)
    distname_633597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'distname')
    # Getting the type of 'fail_higher' (line 176)
    fail_higher_633598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'fail_higher')
    # Applying the binary operator 'notin' (line 176)
    result_contains_633599 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 16), 'notin', distname_633597, fail_higher_633598)
    
    # Assigning a type to the variable 'cond2' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'cond2', result_contains_633599)
    # Creating a generator
    
    # Obtaining an instance of the builtin type 'tuple' (line 178)
    tuple_633600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 178)
    # Adding element type (line 178)
    # Getting the type of 'distname' (line 178)
    distname_633601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 14), 'distname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 14), tuple_633600, distname_633601)
    # Adding element type (line 178)
    # Getting the type of 'arg' (line 178)
    arg_633602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'arg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 14), tuple_633600, arg_633602)
    # Adding element type (line 178)
    # Getting the type of 'cond1' (line 178)
    cond1_633603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 29), 'cond1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 14), tuple_633600, cond1_633603)
    # Adding element type (line 178)
    # Getting the type of 'cond2' (line 178)
    cond2_633604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 36), 'cond2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 14), tuple_633600, cond2_633604)
    # Adding element type (line 178)
    # Getting the type of 'False' (line 178)
    False_633605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 43), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 14), tuple_633600, False_633605)
    
    GeneratorType_633606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 8), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 8), GeneratorType_633606, tuple_633600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', GeneratorType_633606)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'cond1' (line 180)
    cond1_633607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'cond1')
    # Applying the 'not' unary operator (line 180)
    result_not__633608 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 11), 'not', cond1_633607)
    
    
    # Getting the type of 'cond2' (line 180)
    cond2_633609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'cond2')
    # Applying the 'not' unary operator (line 180)
    result_not__633610 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 24), 'not', cond2_633609)
    
    # Applying the binary operator 'or' (line 180)
    result_or_keyword_633611 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 11), 'or', result_not__633608, result_not__633610)
    
    # Testing the type of an if condition (line 180)
    if_condition_633612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 8), result_or_keyword_633611)
    # Assigning a type to the variable 'if_condition_633612' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'if_condition_633612', if_condition_633612)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Creating a generator
    
    # Call to param(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'distname' (line 183)
    distname_633615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'distname', False)
    # Getting the type of 'arg' (line 183)
    arg_633616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 41), 'arg', False)
    # Getting the type of 'True' (line 183)
    True_633617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 46), 'True', False)
    # Getting the type of 'True' (line 183)
    True_633618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 52), 'True', False)
    # Getting the type of 'True' (line 183)
    True_633619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 58), 'True', False)
    # Processing the call keyword arguments (line 183)
    # Getting the type of 'pytest' (line 184)
    pytest_633620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'pytest', False)
    # Obtaining the member 'mark' of a type (line 184)
    mark_633621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 37), pytest_633620, 'mark')
    # Obtaining the member 'xfail' of a type (line 184)
    xfail_633622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 37), mark_633621, 'xfail')
    keyword_633623 = xfail_633622
    kwargs_633624 = {'marks': keyword_633623}
    # Getting the type of 'pytest' (line 183)
    pytest_633613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'pytest', False)
    # Obtaining the member 'param' of a type (line 183)
    param_633614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 18), pytest_633613, 'param')
    # Calling param(args, kwargs) (line 183)
    param_call_result_633625 = invoke(stypy.reporting.localization.Localization(__file__, 183, 18), param_633614, *[distname_633615, arg_633616, True_633617, True_633618, True_633619], **kwargs_633624)
    
    GeneratorType_633626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 12), GeneratorType_633626, param_call_result_633625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'stypy_return_type', GeneratorType_633626)
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'cases_test_moments(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cases_test_moments' in the type store
    # Getting the type of 'stypy_return_type' (line 167)
    stypy_return_type_633627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_633627)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cases_test_moments'
    return stypy_return_type_633627

# Assigning a type to the variable 'cases_test_moments' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'cases_test_moments', cases_test_moments)

@norecursion
def test_moments(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_moments'
    module_type_store = module_type_store.open_function_context('test_moments', 187, 0, False)
    
    # Passed parameters checking function
    test_moments.stypy_localization = localization
    test_moments.stypy_type_of_self = None
    test_moments.stypy_type_store = module_type_store
    test_moments.stypy_function_name = 'test_moments'
    test_moments.stypy_param_names_list = ['distname', 'arg', 'normalization_ok', 'higher_ok', 'is_xfailing']
    test_moments.stypy_varargs_param_name = None
    test_moments.stypy_kwargs_param_name = None
    test_moments.stypy_call_defaults = defaults
    test_moments.stypy_call_varargs = varargs
    test_moments.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_moments', ['distname', 'arg', 'normalization_ok', 'higher_ok', 'is_xfailing'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_moments', localization, ['distname', 'arg', 'normalization_ok', 'higher_ok', 'is_xfailing'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_moments(...)' code ##################

    
    
    # SSA begins for try-except statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to getattr(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'stats' (line 192)
    stats_633629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 'stats', False)
    # Getting the type of 'distname' (line 192)
    distname_633630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 32), 'distname', False)
    # Processing the call keyword arguments (line 192)
    kwargs_633631 = {}
    # Getting the type of 'getattr' (line 192)
    getattr_633628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'getattr', False)
    # Calling getattr(args, kwargs) (line 192)
    getattr_call_result_633632 = invoke(stypy.reporting.localization.Localization(__file__, 192, 17), getattr_633628, *[stats_633629, distname_633630], **kwargs_633631)
    
    # Assigning a type to the variable 'distfn' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'distfn', getattr_call_result_633632)
    # SSA branch for the except part of a try statement (line 191)
    # SSA branch for the except 'TypeError' branch of a try statement (line 191)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 194):
    
    # Assigning a Name to a Name (line 194):
    # Getting the type of 'distname' (line 194)
    distname_633633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'distname')
    # Assigning a type to the variable 'distfn' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'distfn', distname_633633)
    
    # Assigning a Str to a Name (line 195):
    
    # Assigning a Str to a Name (line 195):
    str_633634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 19), 'str', 'rv_histogram_instance')
    # Assigning a type to the variable 'distname' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'distname', str_633634)
    # SSA join for try-except statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to suppress_warnings(...): (line 197)
    # Processing the call keyword arguments (line 197)
    kwargs_633636 = {}
    # Getting the type of 'suppress_warnings' (line 197)
    suppress_warnings_633635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 197)
    suppress_warnings_call_result_633637 = invoke(stypy.reporting.localization.Localization(__file__, 197, 9), suppress_warnings_633635, *[], **kwargs_633636)
    
    with_633638 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 197, 9), suppress_warnings_call_result_633637, 'with parameter', '__enter__', '__exit__')

    if with_633638:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 197)
        enter___633639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 9), suppress_warnings_call_result_633637, '__enter__')
        with_enter_633640 = invoke(stypy.reporting.localization.Localization(__file__, 197, 9), enter___633639)
        # Assigning a type to the variable 'sup' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'sup', with_enter_633640)
        
        # Call to filter(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'IntegrationWarning' (line 198)
        IntegrationWarning_633643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'IntegrationWarning', False)
        str_633644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 19), 'str', 'The integral is probably divergent, or slowly convergent.')
        # Processing the call keyword arguments (line 198)
        kwargs_633645 = {}
        # Getting the type of 'sup' (line 198)
        sup_633641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 198)
        filter_633642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), sup_633641, 'filter')
        # Calling filter(args, kwargs) (line 198)
        filter_call_result_633646 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), filter_633642, *[IntegrationWarning_633643, str_633644], **kwargs_633645)
        
        
        # Call to filter(...): (line 200)
        # Processing the call keyword arguments (line 200)
        # Getting the type of 'DeprecationWarning' (line 200)
        DeprecationWarning_633649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'DeprecationWarning', False)
        keyword_633650 = DeprecationWarning_633649
        str_633651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 56), 'str', '.*frechet_')
        keyword_633652 = str_633651
        kwargs_633653 = {'category': keyword_633650, 'message': keyword_633652}
        # Getting the type of 'sup' (line 200)
        sup_633647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 200)
        filter_633648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), sup_633647, 'filter')
        # Calling filter(args, kwargs) (line 200)
        filter_call_result_633654 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), filter_633648, *[], **kwargs_633653)
        
        
        # Getting the type of 'is_xfailing' (line 201)
        is_xfailing_633655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'is_xfailing')
        # Testing the type of an if condition (line 201)
        if_condition_633656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), is_xfailing_633655)
        # Assigning a type to the variable 'if_condition_633656' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_633656', if_condition_633656)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to filter(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'IntegrationWarning' (line 202)
        IntegrationWarning_633659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'IntegrationWarning', False)
        # Processing the call keyword arguments (line 202)
        kwargs_633660 = {}
        # Getting the type of 'sup' (line 202)
        sup_633657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'sup', False)
        # Obtaining the member 'filter' of a type (line 202)
        filter_633658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), sup_633657, 'filter')
        # Calling filter(args, kwargs) (line 202)
        filter_call_result_633661 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), filter_633658, *[IntegrationWarning_633659], **kwargs_633660)
        
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 204):
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        int_633662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'int')
        
        # Call to stats(...): (line 204)
        # Getting the type of 'arg' (line 204)
        arg_633665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'arg', False)
        # Processing the call keyword arguments (line 204)
        str_633666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 48), 'str', 'mvsk')
        keyword_633667 = str_633666
        kwargs_633668 = {'moments': keyword_633667}
        # Getting the type of 'distfn' (line 204)
        distfn_633663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'distfn', False)
        # Obtaining the member 'stats' of a type (line 204)
        stats_633664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), distfn_633663, 'stats')
        # Calling stats(args, kwargs) (line 204)
        stats_call_result_633669 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), stats_633664, *[arg_633665], **kwargs_633668)
        
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___633670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), stats_call_result_633669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_633671 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), getitem___633670, int_633662)
        
        # Assigning a type to the variable 'tuple_var_assignment_633087' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_633087', subscript_call_result_633671)
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        int_633672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'int')
        
        # Call to stats(...): (line 204)
        # Getting the type of 'arg' (line 204)
        arg_633675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'arg', False)
        # Processing the call keyword arguments (line 204)
        str_633676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 48), 'str', 'mvsk')
        keyword_633677 = str_633676
        kwargs_633678 = {'moments': keyword_633677}
        # Getting the type of 'distfn' (line 204)
        distfn_633673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'distfn', False)
        # Obtaining the member 'stats' of a type (line 204)
        stats_633674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), distfn_633673, 'stats')
        # Calling stats(args, kwargs) (line 204)
        stats_call_result_633679 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), stats_633674, *[arg_633675], **kwargs_633678)
        
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___633680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), stats_call_result_633679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_633681 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), getitem___633680, int_633672)
        
        # Assigning a type to the variable 'tuple_var_assignment_633088' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_633088', subscript_call_result_633681)
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        int_633682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'int')
        
        # Call to stats(...): (line 204)
        # Getting the type of 'arg' (line 204)
        arg_633685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'arg', False)
        # Processing the call keyword arguments (line 204)
        str_633686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 48), 'str', 'mvsk')
        keyword_633687 = str_633686
        kwargs_633688 = {'moments': keyword_633687}
        # Getting the type of 'distfn' (line 204)
        distfn_633683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'distfn', False)
        # Obtaining the member 'stats' of a type (line 204)
        stats_633684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), distfn_633683, 'stats')
        # Calling stats(args, kwargs) (line 204)
        stats_call_result_633689 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), stats_633684, *[arg_633685], **kwargs_633688)
        
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___633690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), stats_call_result_633689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_633691 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), getitem___633690, int_633682)
        
        # Assigning a type to the variable 'tuple_var_assignment_633089' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_633089', subscript_call_result_633691)
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        int_633692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'int')
        
        # Call to stats(...): (line 204)
        # Getting the type of 'arg' (line 204)
        arg_633695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'arg', False)
        # Processing the call keyword arguments (line 204)
        str_633696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 48), 'str', 'mvsk')
        keyword_633697 = str_633696
        kwargs_633698 = {'moments': keyword_633697}
        # Getting the type of 'distfn' (line 204)
        distfn_633693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'distfn', False)
        # Obtaining the member 'stats' of a type (line 204)
        stats_633694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), distfn_633693, 'stats')
        # Calling stats(args, kwargs) (line 204)
        stats_call_result_633699 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), stats_633694, *[arg_633695], **kwargs_633698)
        
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___633700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), stats_call_result_633699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_633701 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), getitem___633700, int_633692)
        
        # Assigning a type to the variable 'tuple_var_assignment_633090' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_633090', subscript_call_result_633701)
        
        # Assigning a Name to a Name (line 204):
        # Getting the type of 'tuple_var_assignment_633087' (line 204)
        tuple_var_assignment_633087_633702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_633087')
        # Assigning a type to the variable 'm' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'm', tuple_var_assignment_633087_633702)
        
        # Assigning a Name to a Name (line 204):
        # Getting the type of 'tuple_var_assignment_633088' (line 204)
        tuple_var_assignment_633088_633703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_633088')
        # Assigning a type to the variable 'v' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'v', tuple_var_assignment_633088_633703)
        
        # Assigning a Name to a Name (line 204):
        # Getting the type of 'tuple_var_assignment_633089' (line 204)
        tuple_var_assignment_633089_633704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_633089')
        # Assigning a type to the variable 's' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 14), 's', tuple_var_assignment_633089_633704)
        
        # Assigning a Name to a Name (line 204):
        # Getting the type of 'tuple_var_assignment_633090' (line 204)
        tuple_var_assignment_633090_633705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_633090')
        # Assigning a type to the variable 'k' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 17), 'k', tuple_var_assignment_633090_633705)
        
        # Getting the type of 'normalization_ok' (line 206)
        normalization_ok_633706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'normalization_ok')
        # Testing the type of an if condition (line 206)
        if_condition_633707 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), normalization_ok_633706)
        # Assigning a type to the variable 'if_condition_633707' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'if_condition_633707', if_condition_633707)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_normalization(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'distfn' (line 207)
        distfn_633709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 32), 'distfn', False)
        # Getting the type of 'arg' (line 207)
        arg_633710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 40), 'arg', False)
        # Getting the type of 'distname' (line 207)
        distname_633711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 45), 'distname', False)
        # Processing the call keyword arguments (line 207)
        kwargs_633712 = {}
        # Getting the type of 'check_normalization' (line 207)
        check_normalization_633708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'check_normalization', False)
        # Calling check_normalization(args, kwargs) (line 207)
        check_normalization_call_result_633713 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), check_normalization_633708, *[distfn_633709, arg_633710, distname_633711], **kwargs_633712)
        
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'higher_ok' (line 209)
        higher_ok_633714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'higher_ok')
        # Testing the type of an if condition (line 209)
        if_condition_633715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 8), higher_ok_633714)
        # Assigning a type to the variable 'if_condition_633715' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'if_condition_633715', if_condition_633715)
        # SSA begins for if statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_mean_expect(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'distfn' (line 210)
        distfn_633717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'distfn', False)
        # Getting the type of 'arg' (line 210)
        arg_633718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 38), 'arg', False)
        # Getting the type of 'm' (line 210)
        m_633719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 43), 'm', False)
        # Getting the type of 'distname' (line 210)
        distname_633720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 46), 'distname', False)
        # Processing the call keyword arguments (line 210)
        kwargs_633721 = {}
        # Getting the type of 'check_mean_expect' (line 210)
        check_mean_expect_633716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'check_mean_expect', False)
        # Calling check_mean_expect(args, kwargs) (line 210)
        check_mean_expect_call_result_633722 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), check_mean_expect_633716, *[distfn_633717, arg_633718, m_633719, distname_633720], **kwargs_633721)
        
        
        # Call to check_skew_expect(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'distfn' (line 211)
        distfn_633724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'distfn', False)
        # Getting the type of 'arg' (line 211)
        arg_633725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'arg', False)
        # Getting the type of 'm' (line 211)
        m_633726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 43), 'm', False)
        # Getting the type of 'v' (line 211)
        v_633727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 46), 'v', False)
        # Getting the type of 's' (line 211)
        s_633728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 49), 's', False)
        # Getting the type of 'distname' (line 211)
        distname_633729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 52), 'distname', False)
        # Processing the call keyword arguments (line 211)
        kwargs_633730 = {}
        # Getting the type of 'check_skew_expect' (line 211)
        check_skew_expect_633723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'check_skew_expect', False)
        # Calling check_skew_expect(args, kwargs) (line 211)
        check_skew_expect_call_result_633731 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), check_skew_expect_633723, *[distfn_633724, arg_633725, m_633726, v_633727, s_633728, distname_633729], **kwargs_633730)
        
        
        # Call to check_var_expect(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'distfn' (line 212)
        distfn_633733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'distfn', False)
        # Getting the type of 'arg' (line 212)
        arg_633734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'arg', False)
        # Getting the type of 'm' (line 212)
        m_633735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 42), 'm', False)
        # Getting the type of 'v' (line 212)
        v_633736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 45), 'v', False)
        # Getting the type of 'distname' (line 212)
        distname_633737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 48), 'distname', False)
        # Processing the call keyword arguments (line 212)
        kwargs_633738 = {}
        # Getting the type of 'check_var_expect' (line 212)
        check_var_expect_633732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'check_var_expect', False)
        # Calling check_var_expect(args, kwargs) (line 212)
        check_var_expect_call_result_633739 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), check_var_expect_633732, *[distfn_633733, arg_633734, m_633735, v_633736, distname_633737], **kwargs_633738)
        
        
        # Call to check_kurt_expect(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'distfn' (line 213)
        distfn_633741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'distfn', False)
        # Getting the type of 'arg' (line 213)
        arg_633742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 38), 'arg', False)
        # Getting the type of 'm' (line 213)
        m_633743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'm', False)
        # Getting the type of 'v' (line 213)
        v_633744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 46), 'v', False)
        # Getting the type of 'k' (line 213)
        k_633745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 49), 'k', False)
        # Getting the type of 'distname' (line 213)
        distname_633746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 52), 'distname', False)
        # Processing the call keyword arguments (line 213)
        kwargs_633747 = {}
        # Getting the type of 'check_kurt_expect' (line 213)
        check_kurt_expect_633740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'check_kurt_expect', False)
        # Calling check_kurt_expect(args, kwargs) (line 213)
        check_kurt_expect_call_result_633748 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), check_kurt_expect_633740, *[distfn_633741, arg_633742, m_633743, v_633744, k_633745, distname_633746], **kwargs_633747)
        
        # SSA join for if statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to check_loc_scale(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'distfn' (line 215)
        distfn_633750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'distfn', False)
        # Getting the type of 'arg' (line 215)
        arg_633751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 32), 'arg', False)
        # Getting the type of 'm' (line 215)
        m_633752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 37), 'm', False)
        # Getting the type of 'v' (line 215)
        v_633753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 40), 'v', False)
        # Getting the type of 'distname' (line 215)
        distname_633754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 43), 'distname', False)
        # Processing the call keyword arguments (line 215)
        kwargs_633755 = {}
        # Getting the type of 'check_loc_scale' (line 215)
        check_loc_scale_633749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'check_loc_scale', False)
        # Calling check_loc_scale(args, kwargs) (line 215)
        check_loc_scale_call_result_633756 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), check_loc_scale_633749, *[distfn_633750, arg_633751, m_633752, v_633753, distname_633754], **kwargs_633755)
        
        
        # Call to check_moment(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'distfn' (line 216)
        distfn_633758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'distfn', False)
        # Getting the type of 'arg' (line 216)
        arg_633759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 29), 'arg', False)
        # Getting the type of 'm' (line 216)
        m_633760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'm', False)
        # Getting the type of 'v' (line 216)
        v_633761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 37), 'v', False)
        # Getting the type of 'distname' (line 216)
        distname_633762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 40), 'distname', False)
        # Processing the call keyword arguments (line 216)
        kwargs_633763 = {}
        # Getting the type of 'check_moment' (line 216)
        check_moment_633757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'check_moment', False)
        # Calling check_moment(args, kwargs) (line 216)
        check_moment_call_result_633764 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), check_moment_633757, *[distfn_633758, arg_633759, m_633760, v_633761, distname_633762], **kwargs_633763)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 197)
        exit___633765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 9), suppress_warnings_call_result_633637, '__exit__')
        with_exit_633766 = invoke(stypy.reporting.localization.Localization(__file__, 197, 9), exit___633765, None, None, None)

    
    # ################# End of 'test_moments(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_moments' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_633767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_633767)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_moments'
    return stypy_return_type_633767

# Assigning a type to the variable 'test_moments' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'test_moments', test_moments)

@norecursion
def test_rvs_broadcast(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rvs_broadcast'
    module_type_store = module_type_store.open_function_context('test_rvs_broadcast', 219, 0, False)
    
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

    
    
    # Getting the type of 'dist' (line 221)
    dist_633768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 7), 'dist')
    
    # Obtaining an instance of the builtin type 'list' (line 221)
    list_633769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 221)
    # Adding element type (line 221)
    str_633770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 16), 'str', 'gausshyper')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 15), list_633769, str_633770)
    # Adding element type (line 221)
    str_633771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 30), 'str', 'genexpon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 15), list_633769, str_633771)
    
    # Applying the binary operator 'in' (line 221)
    result_contains_633772 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 7), 'in', dist_633768, list_633769)
    
    # Testing the type of an if condition (line 221)
    if_condition_633773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 4), result_contains_633772)
    # Assigning a type to the variable 'if_condition_633773' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'if_condition_633773', if_condition_633773)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to skip(...): (line 222)
    # Processing the call arguments (line 222)
    str_633776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 20), 'str', 'too slow')
    # Processing the call keyword arguments (line 222)
    kwargs_633777 = {}
    # Getting the type of 'pytest' (line 222)
    pytest_633774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'pytest', False)
    # Obtaining the member 'skip' of a type (line 222)
    skip_633775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), pytest_633774, 'skip')
    # Calling skip(args, kwargs) (line 222)
    skip_call_result_633778 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), skip_633775, *[str_633776], **kwargs_633777)
    
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Compare to a Name (line 234):
    
    # Assigning a Compare to a Name (line 234):
    
    # Getting the type of 'dist' (line 234)
    dist_633779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'dist')
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_633780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    str_633781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 26), 'str', 'betaprime')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 25), list_633780, str_633781)
    # Adding element type (line 234)
    str_633782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 39), 'str', 'dgamma')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 25), list_633780, str_633782)
    # Adding element type (line 234)
    str_633783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 49), 'str', 'exponnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 25), list_633780, str_633783)
    # Adding element type (line 234)
    str_633784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'str', 'nct')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 25), list_633780, str_633784)
    # Adding element type (line 234)
    str_633785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 33), 'str', 'dweibull')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 25), list_633780, str_633785)
    # Adding element type (line 234)
    str_633786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 45), 'str', 'rice')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 25), list_633780, str_633786)
    # Adding element type (line 234)
    str_633787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 53), 'str', 'levy_stable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 25), list_633780, str_633787)
    # Adding element type (line 234)
    str_633788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 26), 'str', 'skewnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 25), list_633780, str_633788)
    
    # Applying the binary operator 'in' (line 234)
    result_contains_633789 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 17), 'in', dist_633779, list_633780)
    
    # Assigning a type to the variable 'shape_only' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'shape_only', result_contains_633789)
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to getattr(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'stats' (line 238)
    stats_633791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'stats', False)
    # Getting the type of 'dist' (line 238)
    dist_633792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 30), 'dist', False)
    # Processing the call keyword arguments (line 238)
    kwargs_633793 = {}
    # Getting the type of 'getattr' (line 238)
    getattr_633790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 238)
    getattr_call_result_633794 = invoke(stypy.reporting.localization.Localization(__file__, 238, 15), getattr_633790, *[stats_633791, dist_633792], **kwargs_633793)
    
    # Assigning a type to the variable 'distfunc' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'distfunc', getattr_call_result_633794)
    
    # Assigning a Call to a Name (line 239):
    
    # Assigning a Call to a Name (line 239):
    
    # Call to zeros(...): (line 239)
    # Processing the call arguments (line 239)
    int_633797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'int')
    # Processing the call keyword arguments (line 239)
    kwargs_633798 = {}
    # Getting the type of 'np' (line 239)
    np_633795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 239)
    zeros_633796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 10), np_633795, 'zeros')
    # Calling zeros(args, kwargs) (line 239)
    zeros_call_result_633799 = invoke(stypy.reporting.localization.Localization(__file__, 239, 10), zeros_633796, *[int_633797], **kwargs_633798)
    
    # Assigning a type to the variable 'loc' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'loc', zeros_call_result_633799)
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to ones(...): (line 240)
    # Processing the call arguments (line 240)
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_633802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    int_633803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 21), tuple_633802, int_633803)
    # Adding element type (line 240)
    int_633804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 21), tuple_633802, int_633804)
    
    # Processing the call keyword arguments (line 240)
    kwargs_633805 = {}
    # Getting the type of 'np' (line 240)
    np_633800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'np', False)
    # Obtaining the member 'ones' of a type (line 240)
    ones_633801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), np_633800, 'ones')
    # Calling ones(args, kwargs) (line 240)
    ones_call_result_633806 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), ones_633801, *[tuple_633802], **kwargs_633805)
    
    # Assigning a type to the variable 'scale' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'scale', ones_call_result_633806)
    
    # Assigning a Attribute to a Name (line 241):
    
    # Assigning a Attribute to a Name (line 241):
    # Getting the type of 'distfunc' (line 241)
    distfunc_633807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'distfunc')
    # Obtaining the member 'numargs' of a type (line 241)
    numargs_633808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), distfunc_633807, 'numargs')
    # Assigning a type to the variable 'nargs' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'nargs', numargs_633808)
    
    # Assigning a List to a Name (line 242):
    
    # Assigning a List to a Name (line 242):
    
    # Obtaining an instance of the builtin type 'list' (line 242)
    list_633809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 242)
    
    # Assigning a type to the variable 'allargs' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'allargs', list_633809)
    
    # Assigning a List to a Name (line 243):
    
    # Assigning a List to a Name (line 243):
    
    # Obtaining an instance of the builtin type 'list' (line 243)
    list_633810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 243)
    # Adding element type (line 243)
    int_633811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 13), list_633810, int_633811)
    # Adding element type (line 243)
    int_633812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 13), list_633810, int_633812)
    
    # Assigning a type to the variable 'bshape' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'bshape', list_633810)
    
    
    # Call to range(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'nargs' (line 245)
    nargs_633814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 19), 'nargs', False)
    # Processing the call keyword arguments (line 245)
    kwargs_633815 = {}
    # Getting the type of 'range' (line 245)
    range_633813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'range', False)
    # Calling range(args, kwargs) (line 245)
    range_call_result_633816 = invoke(stypy.reporting.localization.Localization(__file__, 245, 13), range_633813, *[nargs_633814], **kwargs_633815)
    
    # Testing the type of a for loop iterable (line 245)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 4), range_call_result_633816)
    # Getting the type of the for loop variable (line 245)
    for_loop_var_633817 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 4), range_call_result_633816)
    # Assigning a type to the variable 'k' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'k', for_loop_var_633817)
    # SSA begins for a for statement (line 245)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 246):
    
    # Assigning a BinOp to a Name (line 246):
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_633818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    # Getting the type of 'k' (line 246)
    k_633819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'k')
    int_633820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 19), 'int')
    # Applying the binary operator '+' (line 246)
    result_add_633821 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 15), '+', k_633819, int_633820)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), tuple_633818, result_add_633821)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_633822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    int_633823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 26), tuple_633822, int_633823)
    
    # Getting the type of 'k' (line 246)
    k_633824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'k')
    int_633825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 35), 'int')
    # Applying the binary operator '+' (line 246)
    result_add_633826 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 31), '+', k_633824, int_633825)
    
    # Applying the binary operator '*' (line 246)
    result_mul_633827 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 25), '*', tuple_633822, result_add_633826)
    
    # Applying the binary operator '+' (line 246)
    result_add_633828 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 14), '+', tuple_633818, result_mul_633827)
    
    # Assigning a type to the variable 'shp' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'shp', result_add_633828)
    
    # Call to append(...): (line 247)
    # Processing the call arguments (line 247)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 247)
    k_633831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 34), 'k', False)
    # Getting the type of 'shape_args' (line 247)
    shape_args_633832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'shape_args', False)
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___633833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 23), shape_args_633832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_633834 = invoke(stypy.reporting.localization.Localization(__file__, 247, 23), getitem___633833, k_633831)
    
    
    # Call to ones(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'shp' (line 247)
    shp_633837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 45), 'shp', False)
    # Processing the call keyword arguments (line 247)
    kwargs_633838 = {}
    # Getting the type of 'np' (line 247)
    np_633835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'np', False)
    # Obtaining the member 'ones' of a type (line 247)
    ones_633836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 37), np_633835, 'ones')
    # Calling ones(args, kwargs) (line 247)
    ones_call_result_633839 = invoke(stypy.reporting.localization.Localization(__file__, 247, 37), ones_633836, *[shp_633837], **kwargs_633838)
    
    # Applying the binary operator '*' (line 247)
    result_mul_633840 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 23), '*', subscript_call_result_633834, ones_call_result_633839)
    
    # Processing the call keyword arguments (line 247)
    kwargs_633841 = {}
    # Getting the type of 'allargs' (line 247)
    allargs_633829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'allargs', False)
    # Obtaining the member 'append' of a type (line 247)
    append_633830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), allargs_633829, 'append')
    # Calling append(args, kwargs) (line 247)
    append_call_result_633842 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), append_633830, *[result_mul_633840], **kwargs_633841)
    
    
    # Call to insert(...): (line 248)
    # Processing the call arguments (line 248)
    int_633845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 22), 'int')
    # Getting the type of 'k' (line 248)
    k_633846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 25), 'k', False)
    int_633847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 29), 'int')
    # Applying the binary operator '+' (line 248)
    result_add_633848 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 25), '+', k_633846, int_633847)
    
    # Processing the call keyword arguments (line 248)
    kwargs_633849 = {}
    # Getting the type of 'bshape' (line 248)
    bshape_633843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'bshape', False)
    # Obtaining the member 'insert' of a type (line 248)
    insert_633844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), bshape_633843, 'insert')
    # Calling insert(args, kwargs) (line 248)
    insert_call_result_633850 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), insert_633844, *[int_633845, result_add_633848], **kwargs_633849)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to extend(...): (line 249)
    # Processing the call arguments (line 249)
    
    # Obtaining an instance of the builtin type 'list' (line 249)
    list_633853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 249)
    # Adding element type (line 249)
    # Getting the type of 'loc' (line 249)
    loc_633854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'loc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 19), list_633853, loc_633854)
    # Adding element type (line 249)
    # Getting the type of 'scale' (line 249)
    scale_633855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'scale', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 19), list_633853, scale_633855)
    
    # Processing the call keyword arguments (line 249)
    kwargs_633856 = {}
    # Getting the type of 'allargs' (line 249)
    allargs_633851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'allargs', False)
    # Obtaining the member 'extend' of a type (line 249)
    extend_633852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 4), allargs_633851, 'extend')
    # Calling extend(args, kwargs) (line 249)
    extend_call_result_633857 = invoke(stypy.reporting.localization.Localization(__file__, 249, 4), extend_633852, *[list_633853], **kwargs_633856)
    
    
    # Call to check_rvs_broadcast(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'distfunc' (line 253)
    distfunc_633859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 24), 'distfunc', False)
    # Getting the type of 'dist' (line 253)
    dist_633860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 34), 'dist', False)
    # Getting the type of 'allargs' (line 253)
    allargs_633861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 40), 'allargs', False)
    # Getting the type of 'bshape' (line 253)
    bshape_633862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 49), 'bshape', False)
    # Getting the type of 'shape_only' (line 253)
    shape_only_633863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 57), 'shape_only', False)
    str_633864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 69), 'str', 'd')
    # Processing the call keyword arguments (line 253)
    kwargs_633865 = {}
    # Getting the type of 'check_rvs_broadcast' (line 253)
    check_rvs_broadcast_633858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'check_rvs_broadcast', False)
    # Calling check_rvs_broadcast(args, kwargs) (line 253)
    check_rvs_broadcast_call_result_633866 = invoke(stypy.reporting.localization.Localization(__file__, 253, 4), check_rvs_broadcast_633858, *[distfunc_633859, dist_633860, allargs_633861, bshape_633862, shape_only_633863, str_633864], **kwargs_633865)
    
    
    # ################# End of 'test_rvs_broadcast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rvs_broadcast' in the type store
    # Getting the type of 'stypy_return_type' (line 219)
    stypy_return_type_633867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_633867)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rvs_broadcast'
    return stypy_return_type_633867

# Assigning a type to the variable 'test_rvs_broadcast' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'test_rvs_broadcast', test_rvs_broadcast)

@norecursion
def test_rvs_gh2069_regression(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_rvs_gh2069_regression'
    module_type_store = module_type_store.open_function_context('test_rvs_gh2069_regression', 256, 0, False)
    
    # Passed parameters checking function
    test_rvs_gh2069_regression.stypy_localization = localization
    test_rvs_gh2069_regression.stypy_type_of_self = None
    test_rvs_gh2069_regression.stypy_type_store = module_type_store
    test_rvs_gh2069_regression.stypy_function_name = 'test_rvs_gh2069_regression'
    test_rvs_gh2069_regression.stypy_param_names_list = []
    test_rvs_gh2069_regression.stypy_varargs_param_name = None
    test_rvs_gh2069_regression.stypy_kwargs_param_name = None
    test_rvs_gh2069_regression.stypy_call_defaults = defaults
    test_rvs_gh2069_regression.stypy_call_varargs = varargs
    test_rvs_gh2069_regression.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_rvs_gh2069_regression', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_rvs_gh2069_regression', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_rvs_gh2069_regression(...)' code ##################

    
    # Call to seed(...): (line 263)
    # Processing the call arguments (line 263)
    int_633871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 19), 'int')
    # Processing the call keyword arguments (line 263)
    kwargs_633872 = {}
    # Getting the type of 'np' (line 263)
    np_633868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 263)
    random_633869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 4), np_633868, 'random')
    # Obtaining the member 'seed' of a type (line 263)
    seed_633870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 4), random_633869, 'seed')
    # Calling seed(args, kwargs) (line 263)
    seed_call_result_633873 = invoke(stypy.reporting.localization.Localization(__file__, 263, 4), seed_633870, *[int_633871], **kwargs_633872)
    
    
    # Assigning a Call to a Name (line 264):
    
    # Assigning a Call to a Name (line 264):
    
    # Call to rvs(...): (line 264)
    # Processing the call keyword arguments (line 264)
    
    # Call to zeros(...): (line 264)
    # Processing the call arguments (line 264)
    int_633879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 39), 'int')
    # Processing the call keyword arguments (line 264)
    kwargs_633880 = {}
    # Getting the type of 'np' (line 264)
    np_633877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'np', False)
    # Obtaining the member 'zeros' of a type (line 264)
    zeros_633878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 30), np_633877, 'zeros')
    # Calling zeros(args, kwargs) (line 264)
    zeros_call_result_633881 = invoke(stypy.reporting.localization.Localization(__file__, 264, 30), zeros_633878, *[int_633879], **kwargs_633880)
    
    keyword_633882 = zeros_call_result_633881
    int_633883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 49), 'int')
    keyword_633884 = int_633883
    kwargs_633885 = {'loc': keyword_633882, 'scale': keyword_633884}
    # Getting the type of 'stats' (line 264)
    stats_633874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'stats', False)
    # Obtaining the member 'norm' of a type (line 264)
    norm_633875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 11), stats_633874, 'norm')
    # Obtaining the member 'rvs' of a type (line 264)
    rvs_633876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 11), norm_633875, 'rvs')
    # Calling rvs(args, kwargs) (line 264)
    rvs_call_result_633886 = invoke(stypy.reporting.localization.Localization(__file__, 264, 11), rvs_633876, *[], **kwargs_633885)
    
    # Assigning a type to the variable 'vals' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'vals', rvs_call_result_633886)
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to diff(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'vals' (line 265)
    vals_633889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'vals', False)
    # Processing the call keyword arguments (line 265)
    kwargs_633890 = {}
    # Getting the type of 'np' (line 265)
    np_633887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 265)
    diff_633888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), np_633887, 'diff')
    # Calling diff(args, kwargs) (line 265)
    diff_call_result_633891 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), diff_633888, *[vals_633889], **kwargs_633890)
    
    # Assigning a type to the variable 'd' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'd', diff_call_result_633891)
    
    # Call to assert_(...): (line 266)
    # Processing the call arguments (line 266)
    
    # Call to all(...): (line 266)
    # Processing the call arguments (line 266)
    
    # Getting the type of 'd' (line 266)
    d_633896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 23), 'd', False)
    int_633897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 28), 'int')
    # Applying the binary operator '!=' (line 266)
    result_ne_633898 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 23), '!=', d_633896, int_633897)
    
    # Processing the call keyword arguments (line 266)
    kwargs_633899 = {}
    # Getting the type of 'np' (line 266)
    np_633894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 266)
    all_633895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 16), np_633894, 'all')
    # Calling all(args, kwargs) (line 266)
    all_call_result_633900 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), all_633895, *[result_ne_633898], **kwargs_633899)
    
    str_633901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 32), 'str', "All the values are equal, but they shouldn't be!")
    # Processing the call keyword arguments (line 266)
    kwargs_633902 = {}
    # Getting the type of 'npt' (line 266)
    npt_633892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 266)
    assert__633893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 4), npt_633892, 'assert_')
    # Calling assert_(args, kwargs) (line 266)
    assert__call_result_633903 = invoke(stypy.reporting.localization.Localization(__file__, 266, 4), assert__633893, *[all_call_result_633900, str_633901], **kwargs_633902)
    
    
    # Assigning a Call to a Name (line 267):
    
    # Assigning a Call to a Name (line 267):
    
    # Call to rvs(...): (line 267)
    # Processing the call keyword arguments (line 267)
    int_633907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 30), 'int')
    keyword_633908 = int_633907
    
    # Call to ones(...): (line 267)
    # Processing the call arguments (line 267)
    int_633911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 47), 'int')
    # Processing the call keyword arguments (line 267)
    kwargs_633912 = {}
    # Getting the type of 'np' (line 267)
    np_633909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 39), 'np', False)
    # Obtaining the member 'ones' of a type (line 267)
    ones_633910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 39), np_633909, 'ones')
    # Calling ones(args, kwargs) (line 267)
    ones_call_result_633913 = invoke(stypy.reporting.localization.Localization(__file__, 267, 39), ones_633910, *[int_633911], **kwargs_633912)
    
    keyword_633914 = ones_call_result_633913
    kwargs_633915 = {'loc': keyword_633908, 'scale': keyword_633914}
    # Getting the type of 'stats' (line 267)
    stats_633904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'stats', False)
    # Obtaining the member 'norm' of a type (line 267)
    norm_633905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 11), stats_633904, 'norm')
    # Obtaining the member 'rvs' of a type (line 267)
    rvs_633906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 11), norm_633905, 'rvs')
    # Calling rvs(args, kwargs) (line 267)
    rvs_call_result_633916 = invoke(stypy.reporting.localization.Localization(__file__, 267, 11), rvs_633906, *[], **kwargs_633915)
    
    # Assigning a type to the variable 'vals' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'vals', rvs_call_result_633916)
    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 268):
    
    # Call to diff(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'vals' (line 268)
    vals_633919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'vals', False)
    # Processing the call keyword arguments (line 268)
    kwargs_633920 = {}
    # Getting the type of 'np' (line 268)
    np_633917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 268)
    diff_633918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), np_633917, 'diff')
    # Calling diff(args, kwargs) (line 268)
    diff_call_result_633921 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), diff_633918, *[vals_633919], **kwargs_633920)
    
    # Assigning a type to the variable 'd' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'd', diff_call_result_633921)
    
    # Call to assert_(...): (line 269)
    # Processing the call arguments (line 269)
    
    # Call to all(...): (line 269)
    # Processing the call arguments (line 269)
    
    # Getting the type of 'd' (line 269)
    d_633926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 23), 'd', False)
    int_633927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 28), 'int')
    # Applying the binary operator '!=' (line 269)
    result_ne_633928 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 23), '!=', d_633926, int_633927)
    
    # Processing the call keyword arguments (line 269)
    kwargs_633929 = {}
    # Getting the type of 'np' (line 269)
    np_633924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 269)
    all_633925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 16), np_633924, 'all')
    # Calling all(args, kwargs) (line 269)
    all_call_result_633930 = invoke(stypy.reporting.localization.Localization(__file__, 269, 16), all_633925, *[result_ne_633928], **kwargs_633929)
    
    str_633931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 32), 'str', "All the values are equal, but they shouldn't be!")
    # Processing the call keyword arguments (line 269)
    kwargs_633932 = {}
    # Getting the type of 'npt' (line 269)
    npt_633922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 269)
    assert__633923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 4), npt_633922, 'assert_')
    # Calling assert_(args, kwargs) (line 269)
    assert__call_result_633933 = invoke(stypy.reporting.localization.Localization(__file__, 269, 4), assert__633923, *[all_call_result_633930, str_633931], **kwargs_633932)
    
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to rvs(...): (line 270)
    # Processing the call keyword arguments (line 270)
    
    # Call to zeros(...): (line 270)
    # Processing the call arguments (line 270)
    int_633939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 39), 'int')
    # Processing the call keyword arguments (line 270)
    kwargs_633940 = {}
    # Getting the type of 'np' (line 270)
    np_633937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 30), 'np', False)
    # Obtaining the member 'zeros' of a type (line 270)
    zeros_633938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 30), np_633937, 'zeros')
    # Calling zeros(args, kwargs) (line 270)
    zeros_call_result_633941 = invoke(stypy.reporting.localization.Localization(__file__, 270, 30), zeros_633938, *[int_633939], **kwargs_633940)
    
    keyword_633942 = zeros_call_result_633941
    
    # Call to ones(...): (line 270)
    # Processing the call arguments (line 270)
    int_633945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 57), 'int')
    # Processing the call keyword arguments (line 270)
    kwargs_633946 = {}
    # Getting the type of 'np' (line 270)
    np_633943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 49), 'np', False)
    # Obtaining the member 'ones' of a type (line 270)
    ones_633944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 49), np_633943, 'ones')
    # Calling ones(args, kwargs) (line 270)
    ones_call_result_633947 = invoke(stypy.reporting.localization.Localization(__file__, 270, 49), ones_633944, *[int_633945], **kwargs_633946)
    
    keyword_633948 = ones_call_result_633947
    kwargs_633949 = {'loc': keyword_633942, 'scale': keyword_633948}
    # Getting the type of 'stats' (line 270)
    stats_633934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'stats', False)
    # Obtaining the member 'norm' of a type (line 270)
    norm_633935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 11), stats_633934, 'norm')
    # Obtaining the member 'rvs' of a type (line 270)
    rvs_633936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 11), norm_633935, 'rvs')
    # Calling rvs(args, kwargs) (line 270)
    rvs_call_result_633950 = invoke(stypy.reporting.localization.Localization(__file__, 270, 11), rvs_633936, *[], **kwargs_633949)
    
    # Assigning a type to the variable 'vals' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'vals', rvs_call_result_633950)
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to diff(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'vals' (line 271)
    vals_633953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'vals', False)
    # Processing the call keyword arguments (line 271)
    kwargs_633954 = {}
    # Getting the type of 'np' (line 271)
    np_633951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 271)
    diff_633952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), np_633951, 'diff')
    # Calling diff(args, kwargs) (line 271)
    diff_call_result_633955 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), diff_633952, *[vals_633953], **kwargs_633954)
    
    # Assigning a type to the variable 'd' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'd', diff_call_result_633955)
    
    # Call to assert_(...): (line 272)
    # Processing the call arguments (line 272)
    
    # Call to all(...): (line 272)
    # Processing the call arguments (line 272)
    
    # Getting the type of 'd' (line 272)
    d_633960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'd', False)
    int_633961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 28), 'int')
    # Applying the binary operator '!=' (line 272)
    result_ne_633962 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 23), '!=', d_633960, int_633961)
    
    # Processing the call keyword arguments (line 272)
    kwargs_633963 = {}
    # Getting the type of 'np' (line 272)
    np_633958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 272)
    all_633959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), np_633958, 'all')
    # Calling all(args, kwargs) (line 272)
    all_call_result_633964 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), all_633959, *[result_ne_633962], **kwargs_633963)
    
    str_633965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 32), 'str', "All the values are equal, but they shouldn't be!")
    # Processing the call keyword arguments (line 272)
    kwargs_633966 = {}
    # Getting the type of 'npt' (line 272)
    npt_633956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 272)
    assert__633957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 4), npt_633956, 'assert_')
    # Calling assert_(args, kwargs) (line 272)
    assert__call_result_633967 = invoke(stypy.reporting.localization.Localization(__file__, 272, 4), assert__633957, *[all_call_result_633964, str_633965], **kwargs_633966)
    
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to rvs(...): (line 273)
    # Processing the call keyword arguments (line 273)
    
    # Call to array(...): (line 273)
    # Processing the call arguments (line 273)
    
    # Obtaining an instance of the builtin type 'list' (line 273)
    list_633973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 273)
    # Adding element type (line 273)
    
    # Obtaining an instance of the builtin type 'list' (line 273)
    list_633974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 273)
    # Adding element type (line 273)
    int_633975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 40), list_633974, int_633975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 39), list_633973, list_633974)
    # Adding element type (line 273)
    
    # Obtaining an instance of the builtin type 'list' (line 273)
    list_633976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 273)
    # Adding element type (line 273)
    int_633977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 45), list_633976, int_633977)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 39), list_633973, list_633976)
    
    # Processing the call keyword arguments (line 273)
    kwargs_633978 = {}
    # Getting the type of 'np' (line 273)
    np_633971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'np', False)
    # Obtaining the member 'array' of a type (line 273)
    array_633972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 30), np_633971, 'array')
    # Calling array(args, kwargs) (line 273)
    array_call_result_633979 = invoke(stypy.reporting.localization.Localization(__file__, 273, 30), array_633972, *[list_633973], **kwargs_633978)
    
    keyword_633980 = array_call_result_633979
    
    # Call to ones(...): (line 273)
    # Processing the call arguments (line 273)
    int_633983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 66), 'int')
    # Processing the call keyword arguments (line 273)
    kwargs_633984 = {}
    # Getting the type of 'np' (line 273)
    np_633981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 58), 'np', False)
    # Obtaining the member 'ones' of a type (line 273)
    ones_633982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 58), np_633981, 'ones')
    # Calling ones(args, kwargs) (line 273)
    ones_call_result_633985 = invoke(stypy.reporting.localization.Localization(__file__, 273, 58), ones_633982, *[int_633983], **kwargs_633984)
    
    keyword_633986 = ones_call_result_633985
    kwargs_633987 = {'loc': keyword_633980, 'scale': keyword_633986}
    # Getting the type of 'stats' (line 273)
    stats_633968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'stats', False)
    # Obtaining the member 'norm' of a type (line 273)
    norm_633969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 11), stats_633968, 'norm')
    # Obtaining the member 'rvs' of a type (line 273)
    rvs_633970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 11), norm_633969, 'rvs')
    # Calling rvs(args, kwargs) (line 273)
    rvs_call_result_633988 = invoke(stypy.reporting.localization.Localization(__file__, 273, 11), rvs_633970, *[], **kwargs_633987)
    
    # Assigning a type to the variable 'vals' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'vals', rvs_call_result_633988)
    
    # Assigning a Call to a Name (line 274):
    
    # Assigning a Call to a Name (line 274):
    
    # Call to diff(...): (line 274)
    # Processing the call arguments (line 274)
    
    # Call to ravel(...): (line 274)
    # Processing the call keyword arguments (line 274)
    kwargs_633993 = {}
    # Getting the type of 'vals' (line 274)
    vals_633991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'vals', False)
    # Obtaining the member 'ravel' of a type (line 274)
    ravel_633992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), vals_633991, 'ravel')
    # Calling ravel(args, kwargs) (line 274)
    ravel_call_result_633994 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), ravel_633992, *[], **kwargs_633993)
    
    # Processing the call keyword arguments (line 274)
    kwargs_633995 = {}
    # Getting the type of 'np' (line 274)
    np_633989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 274)
    diff_633990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), np_633989, 'diff')
    # Calling diff(args, kwargs) (line 274)
    diff_call_result_633996 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), diff_633990, *[ravel_call_result_633994], **kwargs_633995)
    
    # Assigning a type to the variable 'd' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'd', diff_call_result_633996)
    
    # Call to assert_(...): (line 275)
    # Processing the call arguments (line 275)
    
    # Call to all(...): (line 275)
    # Processing the call arguments (line 275)
    
    # Getting the type of 'd' (line 275)
    d_634001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'd', False)
    int_634002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 28), 'int')
    # Applying the binary operator '!=' (line 275)
    result_ne_634003 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 23), '!=', d_634001, int_634002)
    
    # Processing the call keyword arguments (line 275)
    kwargs_634004 = {}
    # Getting the type of 'np' (line 275)
    np_633999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 275)
    all_634000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), np_633999, 'all')
    # Calling all(args, kwargs) (line 275)
    all_call_result_634005 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), all_634000, *[result_ne_634003], **kwargs_634004)
    
    str_634006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 32), 'str', "All the values are equal, but they shouldn't be!")
    # Processing the call keyword arguments (line 275)
    kwargs_634007 = {}
    # Getting the type of 'npt' (line 275)
    npt_633997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 275)
    assert__633998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 4), npt_633997, 'assert_')
    # Calling assert_(args, kwargs) (line 275)
    assert__call_result_634008 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), assert__633998, *[all_call_result_634005, str_634006], **kwargs_634007)
    
    
    # Call to assert_raises(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'ValueError' (line 277)
    ValueError_634010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'ValueError', False)
    # Getting the type of 'stats' (line 277)
    stats_634011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 30), 'stats', False)
    # Obtaining the member 'norm' of a type (line 277)
    norm_634012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 30), stats_634011, 'norm')
    # Obtaining the member 'rvs' of a type (line 277)
    rvs_634013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 30), norm_634012, 'rvs')
    
    # Obtaining an instance of the builtin type 'list' (line 277)
    list_634014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 277)
    # Adding element type (line 277)
    
    # Obtaining an instance of the builtin type 'list' (line 277)
    list_634015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 277)
    # Adding element type (line 277)
    int_634016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 47), list_634015, int_634016)
    # Adding element type (line 277)
    int_634017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 47), list_634015, int_634017)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 46), list_634014, list_634015)
    # Adding element type (line 277)
    
    # Obtaining an instance of the builtin type 'list' (line 277)
    list_634018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 277)
    # Adding element type (line 277)
    int_634019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 56), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 55), list_634018, int_634019)
    # Adding element type (line 277)
    int_634020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 55), list_634018, int_634020)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 46), list_634014, list_634018)
    
    
    # Obtaining an instance of the builtin type 'list' (line 278)
    list_634021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 278)
    # Adding element type (line 278)
    
    # Obtaining an instance of the builtin type 'list' (line 278)
    list_634022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 278)
    # Adding element type (line 278)
    int_634023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 19), list_634022, int_634023)
    # Adding element type (line 278)
    int_634024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 19), list_634022, int_634024)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 18), list_634021, list_634022)
    # Adding element type (line 278)
    
    # Obtaining an instance of the builtin type 'list' (line 278)
    list_634025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 278)
    # Adding element type (line 278)
    int_634026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 27), list_634025, int_634026)
    # Adding element type (line 278)
    int_634027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 27), list_634025, int_634027)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 18), list_634021, list_634025)
    
    int_634028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 36), 'int')
    # Processing the call keyword arguments (line 277)
    kwargs_634029 = {}
    # Getting the type of 'assert_raises' (line 277)
    assert_raises_634009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 277)
    assert_raises_call_result_634030 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), assert_raises_634009, *[ValueError_634010, rvs_634013, list_634014, list_634021, int_634028], **kwargs_634029)
    
    
    # Call to assert_raises(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'ValueError' (line 279)
    ValueError_634032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 18), 'ValueError', False)
    # Getting the type of 'stats' (line 279)
    stats_634033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 30), 'stats', False)
    # Obtaining the member 'gamma' of a type (line 279)
    gamma_634034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 30), stats_634033, 'gamma')
    # Obtaining the member 'rvs' of a type (line 279)
    rvs_634035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 30), gamma_634034, 'rvs')
    
    # Obtaining an instance of the builtin type 'list' (line 279)
    list_634036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 279)
    # Adding element type (line 279)
    int_634037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 47), list_634036, int_634037)
    # Adding element type (line 279)
    int_634038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 47), list_634036, int_634038)
    # Adding element type (line 279)
    int_634039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 47), list_634036, int_634039)
    # Adding element type (line 279)
    int_634040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 47), list_634036, int_634040)
    
    int_634041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 61), 'int')
    int_634042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 64), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 279)
    tuple_634043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 279)
    # Adding element type (line 279)
    int_634044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 68), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 68), tuple_634043, int_634044)
    # Adding element type (line 279)
    int_634045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 71), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 68), tuple_634043, int_634045)
    
    # Processing the call keyword arguments (line 279)
    kwargs_634046 = {}
    # Getting the type of 'assert_raises' (line 279)
    assert_raises_634031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 279)
    assert_raises_call_result_634047 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), assert_raises_634031, *[ValueError_634032, rvs_634035, list_634036, int_634041, int_634042, tuple_634043], **kwargs_634046)
    
    
    # Call to assert_raises(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'ValueError' (line 280)
    ValueError_634049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 18), 'ValueError', False)
    # Getting the type of 'stats' (line 280)
    stats_634050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 30), 'stats', False)
    # Obtaining the member 'gamma' of a type (line 280)
    gamma_634051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 30), stats_634050, 'gamma')
    # Obtaining the member 'rvs' of a type (line 280)
    rvs_634052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 30), gamma_634051, 'rvs')
    
    # Obtaining an instance of the builtin type 'list' (line 280)
    list_634053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 280)
    # Adding element type (line 280)
    int_634054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 47), list_634053, int_634054)
    # Adding element type (line 280)
    int_634055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 47), list_634053, int_634055)
    # Adding element type (line 280)
    int_634056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 47), list_634053, int_634056)
    # Adding element type (line 280)
    int_634057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 47), list_634053, int_634057)
    
    
    # Obtaining an instance of the builtin type 'list' (line 280)
    list_634058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 280)
    # Adding element type (line 280)
    int_634059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 61), list_634058, int_634059)
    # Adding element type (line 280)
    int_634060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 61), list_634058, int_634060)
    # Adding element type (line 280)
    int_634061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 68), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 61), list_634058, int_634061)
    # Adding element type (line 280)
    int_634062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 71), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 61), list_634058, int_634062)
    
    
    # Obtaining an instance of the builtin type 'list' (line 281)
    list_634063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 281)
    # Adding element type (line 281)
    
    # Obtaining an instance of the builtin type 'list' (line 281)
    list_634064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 281)
    # Adding element type (line 281)
    int_634065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 22), list_634064, int_634065)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 21), list_634063, list_634064)
    # Adding element type (line 281)
    
    # Obtaining an instance of the builtin type 'list' (line 281)
    list_634066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 281)
    # Adding element type (line 281)
    int_634067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 27), list_634066, int_634067)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 21), list_634063, list_634066)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 281)
    tuple_634068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 281)
    # Adding element type (line 281)
    int_634069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 34), tuple_634068, int_634069)
    
    # Processing the call keyword arguments (line 280)
    kwargs_634070 = {}
    # Getting the type of 'assert_raises' (line 280)
    assert_raises_634048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 280)
    assert_raises_call_result_634071 = invoke(stypy.reporting.localization.Localization(__file__, 280, 4), assert_raises_634048, *[ValueError_634049, rvs_634052, list_634053, list_634058, list_634063, tuple_634068], **kwargs_634070)
    
    
    # ################# End of 'test_rvs_gh2069_regression(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_rvs_gh2069_regression' in the type store
    # Getting the type of 'stypy_return_type' (line 256)
    stypy_return_type_634072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634072)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_rvs_gh2069_regression'
    return stypy_return_type_634072

# Assigning a type to the variable 'test_rvs_gh2069_regression' (line 256)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'test_rvs_gh2069_regression', test_rvs_gh2069_regression)

@norecursion
def check_sample_meanvar_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_sample_meanvar_'
    module_type_store = module_type_store.open_function_context('check_sample_meanvar_', 284, 0, False)
    
    # Passed parameters checking function
    check_sample_meanvar_.stypy_localization = localization
    check_sample_meanvar_.stypy_type_of_self = None
    check_sample_meanvar_.stypy_type_store = module_type_store
    check_sample_meanvar_.stypy_function_name = 'check_sample_meanvar_'
    check_sample_meanvar_.stypy_param_names_list = ['distfn', 'arg', 'm', 'v', 'sm', 'sv', 'sn', 'msg']
    check_sample_meanvar_.stypy_varargs_param_name = None
    check_sample_meanvar_.stypy_kwargs_param_name = None
    check_sample_meanvar_.stypy_call_defaults = defaults
    check_sample_meanvar_.stypy_call_varargs = varargs
    check_sample_meanvar_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_sample_meanvar_', ['distfn', 'arg', 'm', 'v', 'sm', 'sv', 'sn', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_sample_meanvar_', localization, ['distfn', 'arg', 'm', 'v', 'sm', 'sv', 'sn', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_sample_meanvar_(...)' code ##################

    
    
    # Call to isfinite(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'm' (line 286)
    m_634075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'm', False)
    # Processing the call keyword arguments (line 286)
    kwargs_634076 = {}
    # Getting the type of 'np' (line 286)
    np_634073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 7), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 286)
    isfinite_634074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 7), np_634073, 'isfinite')
    # Calling isfinite(args, kwargs) (line 286)
    isfinite_call_result_634077 = invoke(stypy.reporting.localization.Localization(__file__, 286, 7), isfinite_634074, *[m_634075], **kwargs_634076)
    
    # Testing the type of an if condition (line 286)
    if_condition_634078 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 4), isfinite_call_result_634077)
    # Assigning a type to the variable 'if_condition_634078' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'if_condition_634078', if_condition_634078)
    # SSA begins for if statement (line 286)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to check_sample_mean(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'sm' (line 287)
    sm_634080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'sm', False)
    # Getting the type of 'sv' (line 287)
    sv_634081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 30), 'sv', False)
    # Getting the type of 'sn' (line 287)
    sn_634082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 34), 'sn', False)
    # Getting the type of 'm' (line 287)
    m_634083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 38), 'm', False)
    # Processing the call keyword arguments (line 287)
    kwargs_634084 = {}
    # Getting the type of 'check_sample_mean' (line 287)
    check_sample_mean_634079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'check_sample_mean', False)
    # Calling check_sample_mean(args, kwargs) (line 287)
    check_sample_mean_call_result_634085 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), check_sample_mean_634079, *[sm_634080, sv_634081, sn_634082, m_634083], **kwargs_634084)
    
    # SSA join for if statement (line 286)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isfinite(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'v' (line 288)
    v_634088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'v', False)
    # Processing the call keyword arguments (line 288)
    kwargs_634089 = {}
    # Getting the type of 'np' (line 288)
    np_634086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 7), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 288)
    isfinite_634087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 7), np_634086, 'isfinite')
    # Calling isfinite(args, kwargs) (line 288)
    isfinite_call_result_634090 = invoke(stypy.reporting.localization.Localization(__file__, 288, 7), isfinite_634087, *[v_634088], **kwargs_634089)
    
    # Testing the type of an if condition (line 288)
    if_condition_634091 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 4), isfinite_call_result_634090)
    # Assigning a type to the variable 'if_condition_634091' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'if_condition_634091', if_condition_634091)
    # SSA begins for if statement (line 288)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to check_sample_var(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'sv' (line 289)
    sv_634093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 25), 'sv', False)
    # Getting the type of 'sn' (line 289)
    sn_634094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 29), 'sn', False)
    # Getting the type of 'v' (line 289)
    v_634095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 33), 'v', False)
    # Processing the call keyword arguments (line 289)
    kwargs_634096 = {}
    # Getting the type of 'check_sample_var' (line 289)
    check_sample_var_634092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'check_sample_var', False)
    # Calling check_sample_var(args, kwargs) (line 289)
    check_sample_var_call_result_634097 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), check_sample_var_634092, *[sv_634093, sn_634094, v_634095], **kwargs_634096)
    
    # SSA join for if statement (line 288)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_sample_meanvar_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_sample_meanvar_' in the type store
    # Getting the type of 'stypy_return_type' (line 284)
    stypy_return_type_634098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634098)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_sample_meanvar_'
    return stypy_return_type_634098

# Assigning a type to the variable 'check_sample_meanvar_' (line 284)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'check_sample_meanvar_', check_sample_meanvar_)

@norecursion
def check_sample_mean(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_sample_mean'
    module_type_store = module_type_store.open_function_context('check_sample_mean', 292, 0, False)
    
    # Passed parameters checking function
    check_sample_mean.stypy_localization = localization
    check_sample_mean.stypy_type_of_self = None
    check_sample_mean.stypy_type_store = module_type_store
    check_sample_mean.stypy_function_name = 'check_sample_mean'
    check_sample_mean.stypy_param_names_list = ['sm', 'v', 'n', 'popmean']
    check_sample_mean.stypy_varargs_param_name = None
    check_sample_mean.stypy_kwargs_param_name = None
    check_sample_mean.stypy_call_defaults = defaults
    check_sample_mean.stypy_call_varargs = varargs
    check_sample_mean.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_sample_mean', ['sm', 'v', 'n', 'popmean'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_sample_mean', localization, ['sm', 'v', 'n', 'popmean'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_sample_mean(...)' code ##################

    
    # Assigning a BinOp to a Name (line 298):
    
    # Assigning a BinOp to a Name (line 298):
    # Getting the type of 'n' (line 298)
    n_634099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 9), 'n')
    int_634100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 11), 'int')
    # Applying the binary operator '-' (line 298)
    result_sub_634101 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 9), '-', n_634099, int_634100)
    
    # Assigning a type to the variable 'df' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'df', result_sub_634101)
    
    # Assigning a BinOp to a Name (line 299):
    
    # Assigning a BinOp to a Name (line 299):
    # Getting the type of 'n' (line 299)
    n_634102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 13), 'n')
    int_634103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 15), 'int')
    # Applying the binary operator '-' (line 299)
    result_sub_634104 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 13), '-', n_634102, int_634103)
    
    # Getting the type of 'v' (line 299)
    v_634105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'v')
    # Applying the binary operator '*' (line 299)
    result_mul_634106 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 12), '*', result_sub_634104, v_634105)
    
    
    # Call to float(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'df' (line 299)
    df_634108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'df', False)
    # Processing the call keyword arguments (line 299)
    kwargs_634109 = {}
    # Getting the type of 'float' (line 299)
    float_634107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 23), 'float', False)
    # Calling float(args, kwargs) (line 299)
    float_call_result_634110 = invoke(stypy.reporting.localization.Localization(__file__, 299, 23), float_634107, *[df_634108], **kwargs_634109)
    
    # Applying the binary operator 'div' (line 299)
    result_div_634111 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 11), 'div', result_mul_634106, float_call_result_634110)
    
    # Assigning a type to the variable 'svar' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'svar', result_div_634111)
    
    # Assigning a BinOp to a Name (line 300):
    
    # Assigning a BinOp to a Name (line 300):
    # Getting the type of 'sm' (line 300)
    sm_634112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 9), 'sm')
    # Getting the type of 'popmean' (line 300)
    popmean_634113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'popmean')
    # Applying the binary operator '-' (line 300)
    result_sub_634114 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 9), '-', sm_634112, popmean_634113)
    
    
    # Call to sqrt(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'svar' (line 300)
    svar_634117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 31), 'svar', False)
    float_634118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 37), 'float')
    # Getting the type of 'n' (line 300)
    n_634119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 41), 'n', False)
    # Applying the binary operator 'div' (line 300)
    result_div_634120 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 37), 'div', float_634118, n_634119)
    
    # Applying the binary operator '*' (line 300)
    result_mul_634121 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 31), '*', svar_634117, result_div_634120)
    
    # Processing the call keyword arguments (line 300)
    kwargs_634122 = {}
    # Getting the type of 'np' (line 300)
    np_634115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 300)
    sqrt_634116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 23), np_634115, 'sqrt')
    # Calling sqrt(args, kwargs) (line 300)
    sqrt_call_result_634123 = invoke(stypy.reporting.localization.Localization(__file__, 300, 23), sqrt_634116, *[result_mul_634121], **kwargs_634122)
    
    # Applying the binary operator 'div' (line 300)
    result_div_634124 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 8), 'div', result_sub_634114, sqrt_call_result_634123)
    
    # Assigning a type to the variable 't' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 't', result_div_634124)
    
    # Assigning a Call to a Name (line 301):
    
    # Assigning a Call to a Name (line 301):
    
    # Call to betainc(...): (line 301)
    # Processing the call arguments (line 301)
    float_634126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 19), 'float')
    # Getting the type of 'df' (line 301)
    df_634127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'df', False)
    # Applying the binary operator '*' (line 301)
    result_mul_634128 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 19), '*', float_634126, df_634127)
    
    float_634129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 27), 'float')
    # Getting the type of 'df' (line 301)
    df_634130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 32), 'df', False)
    # Getting the type of 'df' (line 301)
    df_634131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'df', False)
    # Getting the type of 't' (line 301)
    t_634132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 41), 't', False)
    # Getting the type of 't' (line 301)
    t_634133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 43), 't', False)
    # Applying the binary operator '*' (line 301)
    result_mul_634134 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 41), '*', t_634132, t_634133)
    
    # Applying the binary operator '+' (line 301)
    result_add_634135 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 36), '+', df_634131, result_mul_634134)
    
    # Applying the binary operator 'div' (line 301)
    result_div_634136 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 32), 'div', df_634130, result_add_634135)
    
    # Processing the call keyword arguments (line 301)
    kwargs_634137 = {}
    # Getting the type of 'betainc' (line 301)
    betainc_634125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'betainc', False)
    # Calling betainc(args, kwargs) (line 301)
    betainc_call_result_634138 = invoke(stypy.reporting.localization.Localization(__file__, 301, 11), betainc_634125, *[result_mul_634128, float_634129, result_div_634136], **kwargs_634137)
    
    # Assigning a type to the variable 'prob' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'prob', betainc_call_result_634138)
    
    # Call to assert_(...): (line 304)
    # Processing the call arguments (line 304)
    
    # Getting the type of 'prob' (line 304)
    prob_634141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'prob', False)
    float_634142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 23), 'float')
    # Applying the binary operator '>' (line 304)
    result_gt_634143 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 16), '>', prob_634141, float_634142)
    
    str_634144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 29), 'str', 'mean fail, t,prob = %f, %f, m, sm=%f,%f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 305)
    tuple_634145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 305)
    # Adding element type (line 305)
    # Getting the type of 't' (line 305)
    t_634146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 17), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 17), tuple_634145, t_634146)
    # Adding element type (line 305)
    # Getting the type of 'prob' (line 305)
    prob_634147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'prob', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 17), tuple_634145, prob_634147)
    # Adding element type (line 305)
    # Getting the type of 'popmean' (line 305)
    popmean_634148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 26), 'popmean', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 17), tuple_634145, popmean_634148)
    # Adding element type (line 305)
    # Getting the type of 'sm' (line 305)
    sm_634149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 35), 'sm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 17), tuple_634145, sm_634149)
    
    # Applying the binary operator '%' (line 304)
    result_mod_634150 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 29), '%', str_634144, tuple_634145)
    
    # Processing the call keyword arguments (line 304)
    kwargs_634151 = {}
    # Getting the type of 'npt' (line 304)
    npt_634139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 304)
    assert__634140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 4), npt_634139, 'assert_')
    # Calling assert_(args, kwargs) (line 304)
    assert__call_result_634152 = invoke(stypy.reporting.localization.Localization(__file__, 304, 4), assert__634140, *[result_gt_634143, result_mod_634150], **kwargs_634151)
    
    
    # ################# End of 'check_sample_mean(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_sample_mean' in the type store
    # Getting the type of 'stypy_return_type' (line 292)
    stypy_return_type_634153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634153)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_sample_mean'
    return stypy_return_type_634153

# Assigning a type to the variable 'check_sample_mean' (line 292)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 0), 'check_sample_mean', check_sample_mean)

@norecursion
def check_sample_var(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_sample_var'
    module_type_store = module_type_store.open_function_context('check_sample_var', 308, 0, False)
    
    # Passed parameters checking function
    check_sample_var.stypy_localization = localization
    check_sample_var.stypy_type_of_self = None
    check_sample_var.stypy_type_store = module_type_store
    check_sample_var.stypy_function_name = 'check_sample_var'
    check_sample_var.stypy_param_names_list = ['sv', 'n', 'popvar']
    check_sample_var.stypy_varargs_param_name = None
    check_sample_var.stypy_kwargs_param_name = None
    check_sample_var.stypy_call_defaults = defaults
    check_sample_var.stypy_call_varargs = varargs
    check_sample_var.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_sample_var', ['sv', 'n', 'popvar'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_sample_var', localization, ['sv', 'n', 'popvar'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_sample_var(...)' code ##################

    
    # Assigning a BinOp to a Name (line 311):
    
    # Assigning a BinOp to a Name (line 311):
    # Getting the type of 'n' (line 311)
    n_634154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 9), 'n')
    int_634155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 11), 'int')
    # Applying the binary operator '-' (line 311)
    result_sub_634156 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 9), '-', n_634154, int_634155)
    
    # Assigning a type to the variable 'df' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'df', result_sub_634156)
    
    # Assigning a BinOp to a Name (line 312):
    
    # Assigning a BinOp to a Name (line 312):
    # Getting the type of 'n' (line 312)
    n_634157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'n')
    int_634158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 14), 'int')
    # Applying the binary operator '-' (line 312)
    result_sub_634159 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 12), '-', n_634157, int_634158)
    
    # Getting the type of 'popvar' (line 312)
    popvar_634160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), 'popvar')
    # Applying the binary operator '*' (line 312)
    result_mul_634161 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 11), '*', result_sub_634159, popvar_634160)
    
    
    # Call to float(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'popvar' (line 312)
    popvar_634163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 30), 'popvar', False)
    # Processing the call keyword arguments (line 312)
    kwargs_634164 = {}
    # Getting the type of 'float' (line 312)
    float_634162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 24), 'float', False)
    # Calling float(args, kwargs) (line 312)
    float_call_result_634165 = invoke(stypy.reporting.localization.Localization(__file__, 312, 24), float_634162, *[popvar_634163], **kwargs_634164)
    
    # Applying the binary operator 'div' (line 312)
    result_div_634166 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 23), 'div', result_mul_634161, float_call_result_634165)
    
    # Assigning a type to the variable 'chi2' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'chi2', result_div_634166)
    
    # Assigning a BinOp to a Name (line 313):
    
    # Assigning a BinOp to a Name (line 313):
    
    # Call to sf(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'chi2' (line 313)
    chi2_634171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 39), 'chi2', False)
    # Getting the type of 'df' (line 313)
    df_634172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 45), 'df', False)
    # Processing the call keyword arguments (line 313)
    kwargs_634173 = {}
    # Getting the type of 'stats' (line 313)
    stats_634167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 11), 'stats', False)
    # Obtaining the member 'distributions' of a type (line 313)
    distributions_634168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 11), stats_634167, 'distributions')
    # Obtaining the member 'chi2' of a type (line 313)
    chi2_634169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 11), distributions_634168, 'chi2')
    # Obtaining the member 'sf' of a type (line 313)
    sf_634170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 11), chi2_634169, 'sf')
    # Calling sf(args, kwargs) (line 313)
    sf_call_result_634174 = invoke(stypy.reporting.localization.Localization(__file__, 313, 11), sf_634170, *[chi2_634171, df_634172], **kwargs_634173)
    
    int_634175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 51), 'int')
    # Applying the binary operator '*' (line 313)
    result_mul_634176 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 11), '*', sf_call_result_634174, int_634175)
    
    # Assigning a type to the variable 'pval' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'pval', result_mul_634176)
    
    # Call to assert_(...): (line 314)
    # Processing the call arguments (line 314)
    
    # Getting the type of 'pval' (line 314)
    pval_634179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'pval', False)
    float_634180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 23), 'float')
    # Applying the binary operator '>' (line 314)
    result_gt_634181 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 16), '>', pval_634179, float_634180)
    
    str_634182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 29), 'str', 'var fail, t, pval = %f, %f, v, sv=%f, %f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 315)
    tuple_634183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 315)
    # Adding element type (line 315)
    # Getting the type of 'chi2' (line 315)
    chi2_634184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 17), 'chi2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 17), tuple_634183, chi2_634184)
    # Adding element type (line 315)
    # Getting the type of 'pval' (line 315)
    pval_634185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'pval', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 17), tuple_634183, pval_634185)
    # Adding element type (line 315)
    # Getting the type of 'popvar' (line 315)
    popvar_634186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 29), 'popvar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 17), tuple_634183, popvar_634186)
    # Adding element type (line 315)
    # Getting the type of 'sv' (line 315)
    sv_634187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 37), 'sv', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 17), tuple_634183, sv_634187)
    
    # Applying the binary operator '%' (line 314)
    result_mod_634188 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 29), '%', str_634182, tuple_634183)
    
    # Processing the call keyword arguments (line 314)
    kwargs_634189 = {}
    # Getting the type of 'npt' (line 314)
    npt_634177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 314)
    assert__634178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 4), npt_634177, 'assert_')
    # Calling assert_(args, kwargs) (line 314)
    assert__call_result_634190 = invoke(stypy.reporting.localization.Localization(__file__, 314, 4), assert__634178, *[result_gt_634181, result_mod_634188], **kwargs_634189)
    
    
    # ################# End of 'check_sample_var(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_sample_var' in the type store
    # Getting the type of 'stypy_return_type' (line 308)
    stypy_return_type_634191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634191)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_sample_var'
    return stypy_return_type_634191

# Assigning a type to the variable 'check_sample_var' (line 308)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 0), 'check_sample_var', check_sample_var)

@norecursion
def check_cdf_ppf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_cdf_ppf'
    module_type_store = module_type_store.open_function_context('check_cdf_ppf', 318, 0, False)
    
    # Passed parameters checking function
    check_cdf_ppf.stypy_localization = localization
    check_cdf_ppf.stypy_type_of_self = None
    check_cdf_ppf.stypy_type_store = module_type_store
    check_cdf_ppf.stypy_function_name = 'check_cdf_ppf'
    check_cdf_ppf.stypy_param_names_list = ['distfn', 'arg', 'msg']
    check_cdf_ppf.stypy_varargs_param_name = None
    check_cdf_ppf.stypy_kwargs_param_name = None
    check_cdf_ppf.stypy_call_defaults = defaults
    check_cdf_ppf.stypy_call_varargs = varargs
    check_cdf_ppf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_cdf_ppf', ['distfn', 'arg', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_cdf_ppf', localization, ['distfn', 'arg', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_cdf_ppf(...)' code ##################

    
    # Assigning a List to a Name (line 319):
    
    # Assigning a List to a Name (line 319):
    
    # Obtaining an instance of the builtin type 'list' (line 319)
    list_634192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 319)
    # Adding element type (line 319)
    float_634193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 14), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 13), list_634192, float_634193)
    # Adding element type (line 319)
    float_634194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 13), list_634192, float_634194)
    # Adding element type (line 319)
    float_634195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 13), list_634192, float_634195)
    
    # Assigning a type to the variable 'values' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'values', list_634192)
    
    # Call to assert_almost_equal(...): (line 320)
    # Processing the call arguments (line 320)
    
    # Call to cdf(...): (line 320)
    # Processing the call arguments (line 320)
    
    # Call to ppf(...): (line 320)
    # Processing the call arguments (line 320)
    # Getting the type of 'values' (line 320)
    values_634202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 50), 'values', False)
    # Getting the type of 'arg' (line 320)
    arg_634203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 59), 'arg', False)
    # Processing the call keyword arguments (line 320)
    kwargs_634204 = {}
    # Getting the type of 'distfn' (line 320)
    distfn_634200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 39), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 320)
    ppf_634201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 39), distfn_634200, 'ppf')
    # Calling ppf(args, kwargs) (line 320)
    ppf_call_result_634205 = invoke(stypy.reporting.localization.Localization(__file__, 320, 39), ppf_634201, *[values_634202, arg_634203], **kwargs_634204)
    
    # Getting the type of 'arg' (line 320)
    arg_634206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 66), 'arg', False)
    # Processing the call keyword arguments (line 320)
    kwargs_634207 = {}
    # Getting the type of 'distfn' (line 320)
    distfn_634198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 28), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 320)
    cdf_634199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 28), distfn_634198, 'cdf')
    # Calling cdf(args, kwargs) (line 320)
    cdf_call_result_634208 = invoke(stypy.reporting.localization.Localization(__file__, 320, 28), cdf_634199, *[ppf_call_result_634205, arg_634206], **kwargs_634207)
    
    # Getting the type of 'values' (line 321)
    values_634209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 28), 'values', False)
    # Processing the call keyword arguments (line 320)
    # Getting the type of 'DECIMAL' (line 321)
    DECIMAL_634210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 44), 'DECIMAL', False)
    keyword_634211 = DECIMAL_634210
    # Getting the type of 'msg' (line 321)
    msg_634212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 61), 'msg', False)
    str_634213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 28), 'str', ' - cdf-ppf roundtrip')
    # Applying the binary operator '+' (line 321)
    result_add_634214 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 61), '+', msg_634212, str_634213)
    
    keyword_634215 = result_add_634214
    kwargs_634216 = {'decimal': keyword_634211, 'err_msg': keyword_634215}
    # Getting the type of 'npt' (line 320)
    npt_634196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 320)
    assert_almost_equal_634197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 4), npt_634196, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 320)
    assert_almost_equal_call_result_634217 = invoke(stypy.reporting.localization.Localization(__file__, 320, 4), assert_almost_equal_634197, *[cdf_call_result_634208, values_634209], **kwargs_634216)
    
    
    # ################# End of 'check_cdf_ppf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_cdf_ppf' in the type store
    # Getting the type of 'stypy_return_type' (line 318)
    stypy_return_type_634218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634218)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_cdf_ppf'
    return stypy_return_type_634218

# Assigning a type to the variable 'check_cdf_ppf' (line 318)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'check_cdf_ppf', check_cdf_ppf)

@norecursion
def check_sf_isf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_sf_isf'
    module_type_store = module_type_store.open_function_context('check_sf_isf', 325, 0, False)
    
    # Passed parameters checking function
    check_sf_isf.stypy_localization = localization
    check_sf_isf.stypy_type_of_self = None
    check_sf_isf.stypy_type_store = module_type_store
    check_sf_isf.stypy_function_name = 'check_sf_isf'
    check_sf_isf.stypy_param_names_list = ['distfn', 'arg', 'msg']
    check_sf_isf.stypy_varargs_param_name = None
    check_sf_isf.stypy_kwargs_param_name = None
    check_sf_isf.stypy_call_defaults = defaults
    check_sf_isf.stypy_call_varargs = varargs
    check_sf_isf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_sf_isf', ['distfn', 'arg', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_sf_isf', localization, ['distfn', 'arg', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_sf_isf(...)' code ##################

    
    # Call to assert_almost_equal(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Call to sf(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Call to isf(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Obtaining an instance of the builtin type 'list' (line 326)
    list_634225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 326)
    # Adding element type (line 326)
    float_634226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 50), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 49), list_634225, float_634226)
    # Adding element type (line 326)
    float_634227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 55), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 49), list_634225, float_634227)
    # Adding element type (line 326)
    float_634228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 60), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 49), list_634225, float_634228)
    
    # Getting the type of 'arg' (line 326)
    arg_634229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 67), 'arg', False)
    # Processing the call keyword arguments (line 326)
    kwargs_634230 = {}
    # Getting the type of 'distfn' (line 326)
    distfn_634223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 38), 'distfn', False)
    # Obtaining the member 'isf' of a type (line 326)
    isf_634224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 38), distfn_634223, 'isf')
    # Calling isf(args, kwargs) (line 326)
    isf_call_result_634231 = invoke(stypy.reporting.localization.Localization(__file__, 326, 38), isf_634224, *[list_634225, arg_634229], **kwargs_634230)
    
    # Getting the type of 'arg' (line 326)
    arg_634232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 74), 'arg', False)
    # Processing the call keyword arguments (line 326)
    kwargs_634233 = {}
    # Getting the type of 'distfn' (line 326)
    distfn_634221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 28), 'distfn', False)
    # Obtaining the member 'sf' of a type (line 326)
    sf_634222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 28), distfn_634221, 'sf')
    # Calling sf(args, kwargs) (line 326)
    sf_call_result_634234 = invoke(stypy.reporting.localization.Localization(__file__, 326, 28), sf_634222, *[isf_call_result_634231, arg_634232], **kwargs_634233)
    
    
    # Obtaining an instance of the builtin type 'list' (line 327)
    list_634235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 327)
    # Adding element type (line 327)
    float_634236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 28), list_634235, float_634236)
    # Adding element type (line 327)
    float_634237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 28), list_634235, float_634237)
    # Adding element type (line 327)
    float_634238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 28), list_634235, float_634238)
    
    # Processing the call keyword arguments (line 326)
    # Getting the type of 'DECIMAL' (line 327)
    DECIMAL_634239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 53), 'DECIMAL', False)
    keyword_634240 = DECIMAL_634239
    # Getting the type of 'msg' (line 327)
    msg_634241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 70), 'msg', False)
    str_634242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 28), 'str', ' - sf-isf roundtrip')
    # Applying the binary operator '+' (line 327)
    result_add_634243 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 70), '+', msg_634241, str_634242)
    
    keyword_634244 = result_add_634243
    kwargs_634245 = {'decimal': keyword_634240, 'err_msg': keyword_634244}
    # Getting the type of 'npt' (line 326)
    npt_634219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 326)
    assert_almost_equal_634220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 4), npt_634219, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 326)
    assert_almost_equal_call_result_634246 = invoke(stypy.reporting.localization.Localization(__file__, 326, 4), assert_almost_equal_634220, *[sf_call_result_634234, list_634235], **kwargs_634245)
    
    
    # Call to assert_almost_equal(...): (line 329)
    # Processing the call arguments (line 329)
    
    # Call to cdf(...): (line 329)
    # Processing the call arguments (line 329)
    
    # Obtaining an instance of the builtin type 'list' (line 329)
    list_634251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 329)
    # Adding element type (line 329)
    float_634252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 39), list_634251, float_634252)
    # Adding element type (line 329)
    float_634253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 39), list_634251, float_634253)
    
    # Getting the type of 'arg' (line 329)
    arg_634254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 52), 'arg', False)
    # Processing the call keyword arguments (line 329)
    kwargs_634255 = {}
    # Getting the type of 'distfn' (line 329)
    distfn_634249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 28), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 329)
    cdf_634250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 28), distfn_634249, 'cdf')
    # Calling cdf(args, kwargs) (line 329)
    cdf_call_result_634256 = invoke(stypy.reporting.localization.Localization(__file__, 329, 28), cdf_634250, *[list_634251, arg_634254], **kwargs_634255)
    
    float_634257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 28), 'float')
    
    # Call to sf(...): (line 330)
    # Processing the call arguments (line 330)
    
    # Obtaining an instance of the builtin type 'list' (line 330)
    list_634260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 330)
    # Adding element type (line 330)
    float_634261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 44), list_634260, float_634261)
    # Adding element type (line 330)
    float_634262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 50), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 44), list_634260, float_634262)
    
    # Getting the type of 'arg' (line 330)
    arg_634263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 57), 'arg', False)
    # Processing the call keyword arguments (line 330)
    kwargs_634264 = {}
    # Getting the type of 'distfn' (line 330)
    distfn_634258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 34), 'distfn', False)
    # Obtaining the member 'sf' of a type (line 330)
    sf_634259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 34), distfn_634258, 'sf')
    # Calling sf(args, kwargs) (line 330)
    sf_call_result_634265 = invoke(stypy.reporting.localization.Localization(__file__, 330, 34), sf_634259, *[list_634260, arg_634263], **kwargs_634264)
    
    # Applying the binary operator '-' (line 330)
    result_sub_634266 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 28), '-', float_634257, sf_call_result_634265)
    
    # Processing the call keyword arguments (line 329)
    # Getting the type of 'DECIMAL' (line 331)
    DECIMAL_634267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'DECIMAL', False)
    keyword_634268 = DECIMAL_634267
    # Getting the type of 'msg' (line 331)
    msg_634269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 53), 'msg', False)
    str_634270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 28), 'str', ' - cdf-sf relationship')
    # Applying the binary operator '+' (line 331)
    result_add_634271 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 53), '+', msg_634269, str_634270)
    
    keyword_634272 = result_add_634271
    kwargs_634273 = {'decimal': keyword_634268, 'err_msg': keyword_634272}
    # Getting the type of 'npt' (line 329)
    npt_634247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 329)
    assert_almost_equal_634248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 4), npt_634247, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 329)
    assert_almost_equal_call_result_634274 = invoke(stypy.reporting.localization.Localization(__file__, 329, 4), assert_almost_equal_634248, *[cdf_call_result_634256, result_sub_634266], **kwargs_634273)
    
    
    # ################# End of 'check_sf_isf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_sf_isf' in the type store
    # Getting the type of 'stypy_return_type' (line 325)
    stypy_return_type_634275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634275)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_sf_isf'
    return stypy_return_type_634275

# Assigning a type to the variable 'check_sf_isf' (line 325)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), 'check_sf_isf', check_sf_isf)

@norecursion
def check_pdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_pdf'
    module_type_store = module_type_store.open_function_context('check_pdf', 335, 0, False)
    
    # Passed parameters checking function
    check_pdf.stypy_localization = localization
    check_pdf.stypy_type_of_self = None
    check_pdf.stypy_type_store = module_type_store
    check_pdf.stypy_function_name = 'check_pdf'
    check_pdf.stypy_param_names_list = ['distfn', 'arg', 'msg']
    check_pdf.stypy_varargs_param_name = None
    check_pdf.stypy_kwargs_param_name = None
    check_pdf.stypy_call_defaults = defaults
    check_pdf.stypy_call_varargs = varargs
    check_pdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_pdf', ['distfn', 'arg', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_pdf', localization, ['distfn', 'arg', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_pdf(...)' code ##################

    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to ppf(...): (line 337)
    # Processing the call arguments (line 337)
    float_634278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 24), 'float')
    # Getting the type of 'arg' (line 337)
    arg_634279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), 'arg', False)
    # Processing the call keyword arguments (line 337)
    kwargs_634280 = {}
    # Getting the type of 'distfn' (line 337)
    distfn_634276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 13), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 337)
    ppf_634277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 13), distfn_634276, 'ppf')
    # Calling ppf(args, kwargs) (line 337)
    ppf_call_result_634281 = invoke(stypy.reporting.localization.Localization(__file__, 337, 13), ppf_634277, *[float_634278, arg_634279], **kwargs_634280)
    
    # Assigning a type to the variable 'median' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'median', ppf_call_result_634281)
    
    # Assigning a Num to a Name (line 338):
    
    # Assigning a Num to a Name (line 338):
    float_634282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 10), 'float')
    # Assigning a type to the variable 'eps' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'eps', float_634282)
    
    # Assigning a Call to a Name (line 339):
    
    # Assigning a Call to a Name (line 339):
    
    # Call to pdf(...): (line 339)
    # Processing the call arguments (line 339)
    # Getting the type of 'median' (line 339)
    median_634285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 22), 'median', False)
    # Getting the type of 'arg' (line 339)
    arg_634286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'arg', False)
    # Processing the call keyword arguments (line 339)
    kwargs_634287 = {}
    # Getting the type of 'distfn' (line 339)
    distfn_634283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 11), 'distfn', False)
    # Obtaining the member 'pdf' of a type (line 339)
    pdf_634284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 11), distfn_634283, 'pdf')
    # Calling pdf(args, kwargs) (line 339)
    pdf_call_result_634288 = invoke(stypy.reporting.localization.Localization(__file__, 339, 11), pdf_634284, *[median_634285, arg_634286], **kwargs_634287)
    
    # Assigning a type to the variable 'pdfv' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'pdfv', pdf_call_result_634288)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'pdfv' (line 340)
    pdfv_634289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'pdfv')
    float_634290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 15), 'float')
    # Applying the binary operator '<' (line 340)
    result_lt_634291 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 8), '<', pdfv_634289, float_634290)
    
    
    # Getting the type of 'pdfv' (line 340)
    pdfv_634292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 25), 'pdfv')
    float_634293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 32), 'float')
    # Applying the binary operator '>' (line 340)
    result_gt_634294 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 25), '>', pdfv_634292, float_634293)
    
    # Applying the binary operator 'or' (line 340)
    result_or_keyword_634295 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 7), 'or', result_lt_634291, result_gt_634294)
    
    # Testing the type of an if condition (line 340)
    if_condition_634296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 4), result_or_keyword_634295)
    # Assigning a type to the variable 'if_condition_634296' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'if_condition_634296', if_condition_634296)
    # SSA begins for if statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 343):
    
    # Assigning a BinOp to a Name (line 343):
    # Getting the type of 'median' (line 343)
    median_634297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 17), 'median')
    float_634298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 26), 'float')
    # Applying the binary operator '+' (line 343)
    result_add_634299 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 17), '+', median_634297, float_634298)
    
    # Assigning a type to the variable 'median' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'median', result_add_634299)
    
    # Assigning a Call to a Name (line 344):
    
    # Assigning a Call to a Name (line 344):
    
    # Call to pdf(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'median' (line 344)
    median_634302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'median', False)
    # Getting the type of 'arg' (line 344)
    arg_634303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 35), 'arg', False)
    # Processing the call keyword arguments (line 344)
    kwargs_634304 = {}
    # Getting the type of 'distfn' (line 344)
    distfn_634300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 15), 'distfn', False)
    # Obtaining the member 'pdf' of a type (line 344)
    pdf_634301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 15), distfn_634300, 'pdf')
    # Calling pdf(args, kwargs) (line 344)
    pdf_call_result_634305 = invoke(stypy.reporting.localization.Localization(__file__, 344, 15), pdf_634301, *[median_634302, arg_634303], **kwargs_634304)
    
    # Assigning a type to the variable 'pdfv' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'pdfv', pdf_call_result_634305)
    # SSA join for if statement (line 340)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 345):
    
    # Assigning a BinOp to a Name (line 345):
    
    # Call to cdf(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'median' (line 345)
    median_634308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 26), 'median', False)
    # Getting the type of 'eps' (line 345)
    eps_634309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 35), 'eps', False)
    # Applying the binary operator '+' (line 345)
    result_add_634310 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 26), '+', median_634308, eps_634309)
    
    # Getting the type of 'arg' (line 345)
    arg_634311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 41), 'arg', False)
    # Processing the call keyword arguments (line 345)
    kwargs_634312 = {}
    # Getting the type of 'distfn' (line 345)
    distfn_634306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 15), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 345)
    cdf_634307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 15), distfn_634306, 'cdf')
    # Calling cdf(args, kwargs) (line 345)
    cdf_call_result_634313 = invoke(stypy.reporting.localization.Localization(__file__, 345, 15), cdf_634307, *[result_add_634310, arg_634311], **kwargs_634312)
    
    
    # Call to cdf(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'median' (line 346)
    median_634316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 26), 'median', False)
    # Getting the type of 'eps' (line 346)
    eps_634317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'eps', False)
    # Applying the binary operator '-' (line 346)
    result_sub_634318 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 26), '-', median_634316, eps_634317)
    
    # Getting the type of 'arg' (line 346)
    arg_634319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 41), 'arg', False)
    # Processing the call keyword arguments (line 346)
    kwargs_634320 = {}
    # Getting the type of 'distfn' (line 346)
    distfn_634314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 346)
    cdf_634315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 15), distfn_634314, 'cdf')
    # Calling cdf(args, kwargs) (line 346)
    cdf_call_result_634321 = invoke(stypy.reporting.localization.Localization(__file__, 346, 15), cdf_634315, *[result_sub_634318, arg_634319], **kwargs_634320)
    
    # Applying the binary operator '-' (line 345)
    result_sub_634322 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 15), '-', cdf_call_result_634313, cdf_call_result_634321)
    
    # Getting the type of 'eps' (line 346)
    eps_634323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 47), 'eps')
    # Applying the binary operator 'div' (line 345)
    result_div_634324 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 14), 'div', result_sub_634322, eps_634323)
    
    float_634325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 51), 'float')
    # Applying the binary operator 'div' (line 346)
    result_div_634326 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 50), 'div', result_div_634324, float_634325)
    
    # Assigning a type to the variable 'cdfdiff' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'cdfdiff', result_div_634326)
    
    # Getting the type of 'msg' (line 349)
    msg_634327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'msg')
    str_634328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 11), 'str', ' - cdf-pdf relationship')
    # Applying the binary operator '+=' (line 349)
    result_iadd_634329 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 4), '+=', msg_634327, str_634328)
    # Assigning a type to the variable 'msg' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'msg', result_iadd_634329)
    
    
    # Call to assert_almost_equal(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'pdfv' (line 350)
    pdfv_634332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'pdfv', False)
    # Getting the type of 'cdfdiff' (line 350)
    cdfdiff_634333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 34), 'cdfdiff', False)
    # Processing the call keyword arguments (line 350)
    # Getting the type of 'DECIMAL' (line 350)
    DECIMAL_634334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 51), 'DECIMAL', False)
    keyword_634335 = DECIMAL_634334
    # Getting the type of 'msg' (line 350)
    msg_634336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 68), 'msg', False)
    keyword_634337 = msg_634336
    kwargs_634338 = {'decimal': keyword_634335, 'err_msg': keyword_634337}
    # Getting the type of 'npt' (line 350)
    npt_634330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 350)
    assert_almost_equal_634331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 4), npt_634330, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 350)
    assert_almost_equal_call_result_634339 = invoke(stypy.reporting.localization.Localization(__file__, 350, 4), assert_almost_equal_634331, *[pdfv_634332, cdfdiff_634333], **kwargs_634338)
    
    
    # ################# End of 'check_pdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_pdf' in the type store
    # Getting the type of 'stypy_return_type' (line 335)
    stypy_return_type_634340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634340)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_pdf'
    return stypy_return_type_634340

# Assigning a type to the variable 'check_pdf' (line 335)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 0), 'check_pdf', check_pdf)

@norecursion
def check_pdf_logpdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_pdf_logpdf'
    module_type_store = module_type_store.open_function_context('check_pdf_logpdf', 353, 0, False)
    
    # Passed parameters checking function
    check_pdf_logpdf.stypy_localization = localization
    check_pdf_logpdf.stypy_type_of_self = None
    check_pdf_logpdf.stypy_type_store = module_type_store
    check_pdf_logpdf.stypy_function_name = 'check_pdf_logpdf'
    check_pdf_logpdf.stypy_param_names_list = ['distfn', 'args', 'msg']
    check_pdf_logpdf.stypy_varargs_param_name = None
    check_pdf_logpdf.stypy_kwargs_param_name = None
    check_pdf_logpdf.stypy_call_defaults = defaults
    check_pdf_logpdf.stypy_call_varargs = varargs
    check_pdf_logpdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_pdf_logpdf', ['distfn', 'args', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_pdf_logpdf', localization, ['distfn', 'args', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_pdf_logpdf(...)' code ##################

    
    # Assigning a Call to a Name (line 355):
    
    # Assigning a Call to a Name (line 355):
    
    # Call to array(...): (line 355)
    # Processing the call arguments (line 355)
    
    # Obtaining an instance of the builtin type 'list' (line 355)
    list_634343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 355)
    # Adding element type (line 355)
    float_634344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_634343, float_634344)
    # Adding element type (line 355)
    float_634345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_634343, float_634345)
    # Adding element type (line 355)
    float_634346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_634343, float_634346)
    # Adding element type (line 355)
    float_634347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_634343, float_634347)
    # Adding element type (line 355)
    float_634348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_634343, float_634348)
    # Adding element type (line 355)
    float_634349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_634343, float_634349)
    # Adding element type (line 355)
    float_634350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 53), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 22), list_634343, float_634350)
    
    # Processing the call keyword arguments (line 355)
    kwargs_634351 = {}
    # Getting the type of 'np' (line 355)
    np_634341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 355)
    array_634342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 13), np_634341, 'array')
    # Calling array(args, kwargs) (line 355)
    array_call_result_634352 = invoke(stypy.reporting.localization.Localization(__file__, 355, 13), array_634342, *[list_634343], **kwargs_634351)
    
    # Assigning a type to the variable 'points' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'points', array_call_result_634352)
    
    # Assigning a Call to a Name (line 356):
    
    # Assigning a Call to a Name (line 356):
    
    # Call to ppf(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'points' (line 356)
    points_634355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'points', False)
    # Getting the type of 'args' (line 356)
    args_634356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 31), 'args', False)
    # Processing the call keyword arguments (line 356)
    kwargs_634357 = {}
    # Getting the type of 'distfn' (line 356)
    distfn_634353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 356)
    ppf_634354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 11), distfn_634353, 'ppf')
    # Calling ppf(args, kwargs) (line 356)
    ppf_call_result_634358 = invoke(stypy.reporting.localization.Localization(__file__, 356, 11), ppf_634354, *[points_634355, args_634356], **kwargs_634357)
    
    # Assigning a type to the variable 'vals' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'vals', ppf_call_result_634358)
    
    # Assigning a Call to a Name (line 357):
    
    # Assigning a Call to a Name (line 357):
    
    # Call to pdf(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'vals' (line 357)
    vals_634361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 21), 'vals', False)
    # Getting the type of 'args' (line 357)
    args_634362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 28), 'args', False)
    # Processing the call keyword arguments (line 357)
    kwargs_634363 = {}
    # Getting the type of 'distfn' (line 357)
    distfn_634359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 10), 'distfn', False)
    # Obtaining the member 'pdf' of a type (line 357)
    pdf_634360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 10), distfn_634359, 'pdf')
    # Calling pdf(args, kwargs) (line 357)
    pdf_call_result_634364 = invoke(stypy.reporting.localization.Localization(__file__, 357, 10), pdf_634360, *[vals_634361, args_634362], **kwargs_634363)
    
    # Assigning a type to the variable 'pdf' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'pdf', pdf_call_result_634364)
    
    # Assigning a Call to a Name (line 358):
    
    # Assigning a Call to a Name (line 358):
    
    # Call to logpdf(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'vals' (line 358)
    vals_634367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'vals', False)
    # Getting the type of 'args' (line 358)
    args_634368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'args', False)
    # Processing the call keyword arguments (line 358)
    kwargs_634369 = {}
    # Getting the type of 'distfn' (line 358)
    distfn_634365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 13), 'distfn', False)
    # Obtaining the member 'logpdf' of a type (line 358)
    logpdf_634366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 13), distfn_634365, 'logpdf')
    # Calling logpdf(args, kwargs) (line 358)
    logpdf_call_result_634370 = invoke(stypy.reporting.localization.Localization(__file__, 358, 13), logpdf_634366, *[vals_634367, args_634368], **kwargs_634369)
    
    # Assigning a type to the variable 'logpdf' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'logpdf', logpdf_call_result_634370)
    
    # Assigning a Subscript to a Name (line 359):
    
    # Assigning a Subscript to a Name (line 359):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'pdf' (line 359)
    pdf_634371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 14), 'pdf')
    int_634372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 21), 'int')
    # Applying the binary operator '!=' (line 359)
    result_ne_634373 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 14), '!=', pdf_634371, int_634372)
    
    # Getting the type of 'pdf' (line 359)
    pdf_634374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 10), 'pdf')
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___634375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 10), pdf_634374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_634376 = invoke(stypy.reporting.localization.Localization(__file__, 359, 10), getitem___634375, result_ne_634373)
    
    # Assigning a type to the variable 'pdf' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'pdf', subscript_call_result_634376)
    
    # Assigning a Subscript to a Name (line 360):
    
    # Assigning a Subscript to a Name (line 360):
    
    # Obtaining the type of the subscript
    
    # Call to isfinite(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'logpdf' (line 360)
    logpdf_634379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 32), 'logpdf', False)
    # Processing the call keyword arguments (line 360)
    kwargs_634380 = {}
    # Getting the type of 'np' (line 360)
    np_634377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 360)
    isfinite_634378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 20), np_634377, 'isfinite')
    # Calling isfinite(args, kwargs) (line 360)
    isfinite_call_result_634381 = invoke(stypy.reporting.localization.Localization(__file__, 360, 20), isfinite_634378, *[logpdf_634379], **kwargs_634380)
    
    # Getting the type of 'logpdf' (line 360)
    logpdf_634382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 13), 'logpdf')
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___634383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 13), logpdf_634382, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_634384 = invoke(stypy.reporting.localization.Localization(__file__, 360, 13), getitem___634383, isfinite_call_result_634381)
    
    # Assigning a type to the variable 'logpdf' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'logpdf', subscript_call_result_634384)
    
    # Getting the type of 'msg' (line 361)
    msg_634385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'msg')
    str_634386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 11), 'str', ' - logpdf-log(pdf) relationship')
    # Applying the binary operator '+=' (line 361)
    result_iadd_634387 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 4), '+=', msg_634385, str_634386)
    # Assigning a type to the variable 'msg' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'msg', result_iadd_634387)
    
    
    # Call to assert_almost_equal(...): (line 362)
    # Processing the call arguments (line 362)
    
    # Call to log(...): (line 362)
    # Processing the call arguments (line 362)
    # Getting the type of 'pdf' (line 362)
    pdf_634392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'pdf', False)
    # Processing the call keyword arguments (line 362)
    kwargs_634393 = {}
    # Getting the type of 'np' (line 362)
    np_634390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 28), 'np', False)
    # Obtaining the member 'log' of a type (line 362)
    log_634391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 28), np_634390, 'log')
    # Calling log(args, kwargs) (line 362)
    log_call_result_634394 = invoke(stypy.reporting.localization.Localization(__file__, 362, 28), log_634391, *[pdf_634392], **kwargs_634393)
    
    # Getting the type of 'logpdf' (line 362)
    logpdf_634395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 41), 'logpdf', False)
    # Processing the call keyword arguments (line 362)
    int_634396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 57), 'int')
    keyword_634397 = int_634396
    # Getting the type of 'msg' (line 362)
    msg_634398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 68), 'msg', False)
    keyword_634399 = msg_634398
    kwargs_634400 = {'decimal': keyword_634397, 'err_msg': keyword_634399}
    # Getting the type of 'npt' (line 362)
    npt_634388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 362)
    assert_almost_equal_634389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 4), npt_634388, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 362)
    assert_almost_equal_call_result_634401 = invoke(stypy.reporting.localization.Localization(__file__, 362, 4), assert_almost_equal_634389, *[log_call_result_634394, logpdf_634395], **kwargs_634400)
    
    
    # ################# End of 'check_pdf_logpdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_pdf_logpdf' in the type store
    # Getting the type of 'stypy_return_type' (line 353)
    stypy_return_type_634402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634402)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_pdf_logpdf'
    return stypy_return_type_634402

# Assigning a type to the variable 'check_pdf_logpdf' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'check_pdf_logpdf', check_pdf_logpdf)

@norecursion
def check_sf_logsf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_sf_logsf'
    module_type_store = module_type_store.open_function_context('check_sf_logsf', 365, 0, False)
    
    # Passed parameters checking function
    check_sf_logsf.stypy_localization = localization
    check_sf_logsf.stypy_type_of_self = None
    check_sf_logsf.stypy_type_store = module_type_store
    check_sf_logsf.stypy_function_name = 'check_sf_logsf'
    check_sf_logsf.stypy_param_names_list = ['distfn', 'args', 'msg']
    check_sf_logsf.stypy_varargs_param_name = None
    check_sf_logsf.stypy_kwargs_param_name = None
    check_sf_logsf.stypy_call_defaults = defaults
    check_sf_logsf.stypy_call_varargs = varargs
    check_sf_logsf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_sf_logsf', ['distfn', 'args', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_sf_logsf', localization, ['distfn', 'args', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_sf_logsf(...)' code ##################

    
    # Assigning a Call to a Name (line 367):
    
    # Assigning a Call to a Name (line 367):
    
    # Call to array(...): (line 367)
    # Processing the call arguments (line 367)
    
    # Obtaining an instance of the builtin type 'list' (line 367)
    list_634405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 367)
    # Adding element type (line 367)
    float_634406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_634405, float_634406)
    # Adding element type (line 367)
    float_634407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_634405, float_634407)
    # Adding element type (line 367)
    float_634408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_634405, float_634408)
    # Adding element type (line 367)
    float_634409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_634405, float_634409)
    # Adding element type (line 367)
    float_634410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_634405, float_634410)
    # Adding element type (line 367)
    float_634411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_634405, float_634411)
    # Adding element type (line 367)
    float_634412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 53), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 22), list_634405, float_634412)
    
    # Processing the call keyword arguments (line 367)
    kwargs_634413 = {}
    # Getting the type of 'np' (line 367)
    np_634403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 367)
    array_634404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 13), np_634403, 'array')
    # Calling array(args, kwargs) (line 367)
    array_call_result_634414 = invoke(stypy.reporting.localization.Localization(__file__, 367, 13), array_634404, *[list_634405], **kwargs_634413)
    
    # Assigning a type to the variable 'points' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'points', array_call_result_634414)
    
    # Assigning a Call to a Name (line 368):
    
    # Assigning a Call to a Name (line 368):
    
    # Call to ppf(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'points' (line 368)
    points_634417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 22), 'points', False)
    # Getting the type of 'args' (line 368)
    args_634418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 31), 'args', False)
    # Processing the call keyword arguments (line 368)
    kwargs_634419 = {}
    # Getting the type of 'distfn' (line 368)
    distfn_634415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 368)
    ppf_634416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 11), distfn_634415, 'ppf')
    # Calling ppf(args, kwargs) (line 368)
    ppf_call_result_634420 = invoke(stypy.reporting.localization.Localization(__file__, 368, 11), ppf_634416, *[points_634417, args_634418], **kwargs_634419)
    
    # Assigning a type to the variable 'vals' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'vals', ppf_call_result_634420)
    
    # Assigning a Call to a Name (line 369):
    
    # Assigning a Call to a Name (line 369):
    
    # Call to sf(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'vals' (line 369)
    vals_634423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'vals', False)
    # Getting the type of 'args' (line 369)
    args_634424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 26), 'args', False)
    # Processing the call keyword arguments (line 369)
    kwargs_634425 = {}
    # Getting the type of 'distfn' (line 369)
    distfn_634421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 9), 'distfn', False)
    # Obtaining the member 'sf' of a type (line 369)
    sf_634422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 9), distfn_634421, 'sf')
    # Calling sf(args, kwargs) (line 369)
    sf_call_result_634426 = invoke(stypy.reporting.localization.Localization(__file__, 369, 9), sf_634422, *[vals_634423, args_634424], **kwargs_634425)
    
    # Assigning a type to the variable 'sf' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'sf', sf_call_result_634426)
    
    # Assigning a Call to a Name (line 370):
    
    # Assigning a Call to a Name (line 370):
    
    # Call to logsf(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'vals' (line 370)
    vals_634429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 25), 'vals', False)
    # Getting the type of 'args' (line 370)
    args_634430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 32), 'args', False)
    # Processing the call keyword arguments (line 370)
    kwargs_634431 = {}
    # Getting the type of 'distfn' (line 370)
    distfn_634427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'distfn', False)
    # Obtaining the member 'logsf' of a type (line 370)
    logsf_634428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), distfn_634427, 'logsf')
    # Calling logsf(args, kwargs) (line 370)
    logsf_call_result_634432 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), logsf_634428, *[vals_634429, args_634430], **kwargs_634431)
    
    # Assigning a type to the variable 'logsf' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'logsf', logsf_call_result_634432)
    
    # Assigning a Subscript to a Name (line 371):
    
    # Assigning a Subscript to a Name (line 371):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'sf' (line 371)
    sf_634433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'sf')
    int_634434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 18), 'int')
    # Applying the binary operator '!=' (line 371)
    result_ne_634435 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 12), '!=', sf_634433, int_634434)
    
    # Getting the type of 'sf' (line 371)
    sf_634436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 9), 'sf')
    # Obtaining the member '__getitem__' of a type (line 371)
    getitem___634437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 9), sf_634436, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 371)
    subscript_call_result_634438 = invoke(stypy.reporting.localization.Localization(__file__, 371, 9), getitem___634437, result_ne_634435)
    
    # Assigning a type to the variable 'sf' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'sf', subscript_call_result_634438)
    
    # Assigning a Subscript to a Name (line 372):
    
    # Assigning a Subscript to a Name (line 372):
    
    # Obtaining the type of the subscript
    
    # Call to isfinite(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'logsf' (line 372)
    logsf_634441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 30), 'logsf', False)
    # Processing the call keyword arguments (line 372)
    kwargs_634442 = {}
    # Getting the type of 'np' (line 372)
    np_634439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 18), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 372)
    isfinite_634440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 18), np_634439, 'isfinite')
    # Calling isfinite(args, kwargs) (line 372)
    isfinite_call_result_634443 = invoke(stypy.reporting.localization.Localization(__file__, 372, 18), isfinite_634440, *[logsf_634441], **kwargs_634442)
    
    # Getting the type of 'logsf' (line 372)
    logsf_634444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'logsf')
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___634445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), logsf_634444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 372)
    subscript_call_result_634446 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), getitem___634445, isfinite_call_result_634443)
    
    # Assigning a type to the variable 'logsf' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'logsf', subscript_call_result_634446)
    
    # Getting the type of 'msg' (line 373)
    msg_634447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'msg')
    str_634448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 11), 'str', ' - logsf-log(sf) relationship')
    # Applying the binary operator '+=' (line 373)
    result_iadd_634449 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 4), '+=', msg_634447, str_634448)
    # Assigning a type to the variable 'msg' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'msg', result_iadd_634449)
    
    
    # Call to assert_almost_equal(...): (line 374)
    # Processing the call arguments (line 374)
    
    # Call to log(...): (line 374)
    # Processing the call arguments (line 374)
    # Getting the type of 'sf' (line 374)
    sf_634454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 35), 'sf', False)
    # Processing the call keyword arguments (line 374)
    kwargs_634455 = {}
    # Getting the type of 'np' (line 374)
    np_634452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'np', False)
    # Obtaining the member 'log' of a type (line 374)
    log_634453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 28), np_634452, 'log')
    # Calling log(args, kwargs) (line 374)
    log_call_result_634456 = invoke(stypy.reporting.localization.Localization(__file__, 374, 28), log_634453, *[sf_634454], **kwargs_634455)
    
    # Getting the type of 'logsf' (line 374)
    logsf_634457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 40), 'logsf', False)
    # Processing the call keyword arguments (line 374)
    int_634458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 55), 'int')
    keyword_634459 = int_634458
    # Getting the type of 'msg' (line 374)
    msg_634460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 66), 'msg', False)
    keyword_634461 = msg_634460
    kwargs_634462 = {'decimal': keyword_634459, 'err_msg': keyword_634461}
    # Getting the type of 'npt' (line 374)
    npt_634450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 374)
    assert_almost_equal_634451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 4), npt_634450, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 374)
    assert_almost_equal_call_result_634463 = invoke(stypy.reporting.localization.Localization(__file__, 374, 4), assert_almost_equal_634451, *[log_call_result_634456, logsf_634457], **kwargs_634462)
    
    
    # ################# End of 'check_sf_logsf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_sf_logsf' in the type store
    # Getting the type of 'stypy_return_type' (line 365)
    stypy_return_type_634464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634464)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_sf_logsf'
    return stypy_return_type_634464

# Assigning a type to the variable 'check_sf_logsf' (line 365)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 0), 'check_sf_logsf', check_sf_logsf)

@norecursion
def check_cdf_logcdf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_cdf_logcdf'
    module_type_store = module_type_store.open_function_context('check_cdf_logcdf', 377, 0, False)
    
    # Passed parameters checking function
    check_cdf_logcdf.stypy_localization = localization
    check_cdf_logcdf.stypy_type_of_self = None
    check_cdf_logcdf.stypy_type_store = module_type_store
    check_cdf_logcdf.stypy_function_name = 'check_cdf_logcdf'
    check_cdf_logcdf.stypy_param_names_list = ['distfn', 'args', 'msg']
    check_cdf_logcdf.stypy_varargs_param_name = None
    check_cdf_logcdf.stypy_kwargs_param_name = None
    check_cdf_logcdf.stypy_call_defaults = defaults
    check_cdf_logcdf.stypy_call_varargs = varargs
    check_cdf_logcdf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_cdf_logcdf', ['distfn', 'args', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_cdf_logcdf', localization, ['distfn', 'args', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_cdf_logcdf(...)' code ##################

    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to array(...): (line 379)
    # Processing the call arguments (line 379)
    
    # Obtaining an instance of the builtin type 'list' (line 379)
    list_634467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 379)
    # Adding element type (line 379)
    float_634468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 22), list_634467, float_634468)
    # Adding element type (line 379)
    float_634469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 22), list_634467, float_634469)
    # Adding element type (line 379)
    float_634470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 22), list_634467, float_634470)
    # Adding element type (line 379)
    float_634471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 22), list_634467, float_634471)
    # Adding element type (line 379)
    float_634472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 22), list_634467, float_634472)
    # Adding element type (line 379)
    float_634473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 22), list_634467, float_634473)
    # Adding element type (line 379)
    float_634474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 53), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 22), list_634467, float_634474)
    
    # Processing the call keyword arguments (line 379)
    kwargs_634475 = {}
    # Getting the type of 'np' (line 379)
    np_634465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 379)
    array_634466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 13), np_634465, 'array')
    # Calling array(args, kwargs) (line 379)
    array_call_result_634476 = invoke(stypy.reporting.localization.Localization(__file__, 379, 13), array_634466, *[list_634467], **kwargs_634475)
    
    # Assigning a type to the variable 'points' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'points', array_call_result_634476)
    
    # Assigning a Call to a Name (line 380):
    
    # Assigning a Call to a Name (line 380):
    
    # Call to ppf(...): (line 380)
    # Processing the call arguments (line 380)
    # Getting the type of 'points' (line 380)
    points_634479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 22), 'points', False)
    # Getting the type of 'args' (line 380)
    args_634480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 31), 'args', False)
    # Processing the call keyword arguments (line 380)
    kwargs_634481 = {}
    # Getting the type of 'distfn' (line 380)
    distfn_634477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 11), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 380)
    ppf_634478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 11), distfn_634477, 'ppf')
    # Calling ppf(args, kwargs) (line 380)
    ppf_call_result_634482 = invoke(stypy.reporting.localization.Localization(__file__, 380, 11), ppf_634478, *[points_634479, args_634480], **kwargs_634481)
    
    # Assigning a type to the variable 'vals' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'vals', ppf_call_result_634482)
    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to cdf(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'vals' (line 381)
    vals_634485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'vals', False)
    # Getting the type of 'args' (line 381)
    args_634486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 28), 'args', False)
    # Processing the call keyword arguments (line 381)
    kwargs_634487 = {}
    # Getting the type of 'distfn' (line 381)
    distfn_634483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 10), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 381)
    cdf_634484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 10), distfn_634483, 'cdf')
    # Calling cdf(args, kwargs) (line 381)
    cdf_call_result_634488 = invoke(stypy.reporting.localization.Localization(__file__, 381, 10), cdf_634484, *[vals_634485, args_634486], **kwargs_634487)
    
    # Assigning a type to the variable 'cdf' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'cdf', cdf_call_result_634488)
    
    # Assigning a Call to a Name (line 382):
    
    # Assigning a Call to a Name (line 382):
    
    # Call to logcdf(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'vals' (line 382)
    vals_634491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 27), 'vals', False)
    # Getting the type of 'args' (line 382)
    args_634492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 34), 'args', False)
    # Processing the call keyword arguments (line 382)
    kwargs_634493 = {}
    # Getting the type of 'distfn' (line 382)
    distfn_634489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 13), 'distfn', False)
    # Obtaining the member 'logcdf' of a type (line 382)
    logcdf_634490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 13), distfn_634489, 'logcdf')
    # Calling logcdf(args, kwargs) (line 382)
    logcdf_call_result_634494 = invoke(stypy.reporting.localization.Localization(__file__, 382, 13), logcdf_634490, *[vals_634491, args_634492], **kwargs_634493)
    
    # Assigning a type to the variable 'logcdf' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'logcdf', logcdf_call_result_634494)
    
    # Assigning a Subscript to a Name (line 383):
    
    # Assigning a Subscript to a Name (line 383):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'cdf' (line 383)
    cdf_634495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 14), 'cdf')
    int_634496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 21), 'int')
    # Applying the binary operator '!=' (line 383)
    result_ne_634497 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 14), '!=', cdf_634495, int_634496)
    
    # Getting the type of 'cdf' (line 383)
    cdf_634498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 10), 'cdf')
    # Obtaining the member '__getitem__' of a type (line 383)
    getitem___634499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 10), cdf_634498, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 383)
    subscript_call_result_634500 = invoke(stypy.reporting.localization.Localization(__file__, 383, 10), getitem___634499, result_ne_634497)
    
    # Assigning a type to the variable 'cdf' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'cdf', subscript_call_result_634500)
    
    # Assigning a Subscript to a Name (line 384):
    
    # Assigning a Subscript to a Name (line 384):
    
    # Obtaining the type of the subscript
    
    # Call to isfinite(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'logcdf' (line 384)
    logcdf_634503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 32), 'logcdf', False)
    # Processing the call keyword arguments (line 384)
    kwargs_634504 = {}
    # Getting the type of 'np' (line 384)
    np_634501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 384)
    isfinite_634502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 20), np_634501, 'isfinite')
    # Calling isfinite(args, kwargs) (line 384)
    isfinite_call_result_634505 = invoke(stypy.reporting.localization.Localization(__file__, 384, 20), isfinite_634502, *[logcdf_634503], **kwargs_634504)
    
    # Getting the type of 'logcdf' (line 384)
    logcdf_634506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 13), 'logcdf')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___634507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 13), logcdf_634506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_634508 = invoke(stypy.reporting.localization.Localization(__file__, 384, 13), getitem___634507, isfinite_call_result_634505)
    
    # Assigning a type to the variable 'logcdf' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'logcdf', subscript_call_result_634508)
    
    # Getting the type of 'msg' (line 385)
    msg_634509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'msg')
    str_634510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 11), 'str', ' - logcdf-log(cdf) relationship')
    # Applying the binary operator '+=' (line 385)
    result_iadd_634511 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 4), '+=', msg_634509, str_634510)
    # Assigning a type to the variable 'msg' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'msg', result_iadd_634511)
    
    
    # Call to assert_almost_equal(...): (line 386)
    # Processing the call arguments (line 386)
    
    # Call to log(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'cdf' (line 386)
    cdf_634516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 35), 'cdf', False)
    # Processing the call keyword arguments (line 386)
    kwargs_634517 = {}
    # Getting the type of 'np' (line 386)
    np_634514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 28), 'np', False)
    # Obtaining the member 'log' of a type (line 386)
    log_634515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 28), np_634514, 'log')
    # Calling log(args, kwargs) (line 386)
    log_call_result_634518 = invoke(stypy.reporting.localization.Localization(__file__, 386, 28), log_634515, *[cdf_634516], **kwargs_634517)
    
    # Getting the type of 'logcdf' (line 386)
    logcdf_634519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 41), 'logcdf', False)
    # Processing the call keyword arguments (line 386)
    int_634520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 57), 'int')
    keyword_634521 = int_634520
    # Getting the type of 'msg' (line 386)
    msg_634522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 68), 'msg', False)
    keyword_634523 = msg_634522
    kwargs_634524 = {'decimal': keyword_634521, 'err_msg': keyword_634523}
    # Getting the type of 'npt' (line 386)
    npt_634512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 386)
    assert_almost_equal_634513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 4), npt_634512, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 386)
    assert_almost_equal_call_result_634525 = invoke(stypy.reporting.localization.Localization(__file__, 386, 4), assert_almost_equal_634513, *[log_call_result_634518, logcdf_634519], **kwargs_634524)
    
    
    # ################# End of 'check_cdf_logcdf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_cdf_logcdf' in the type store
    # Getting the type of 'stypy_return_type' (line 377)
    stypy_return_type_634526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634526)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_cdf_logcdf'
    return stypy_return_type_634526

# Assigning a type to the variable 'check_cdf_logcdf' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'check_cdf_logcdf', check_cdf_logcdf)

@norecursion
def check_distribution_rvs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_distribution_rvs'
    module_type_store = module_type_store.open_function_context('check_distribution_rvs', 389, 0, False)
    
    # Passed parameters checking function
    check_distribution_rvs.stypy_localization = localization
    check_distribution_rvs.stypy_type_of_self = None
    check_distribution_rvs.stypy_type_store = module_type_store
    check_distribution_rvs.stypy_function_name = 'check_distribution_rvs'
    check_distribution_rvs.stypy_param_names_list = ['dist', 'args', 'alpha', 'rvs']
    check_distribution_rvs.stypy_varargs_param_name = None
    check_distribution_rvs.stypy_kwargs_param_name = None
    check_distribution_rvs.stypy_call_defaults = defaults
    check_distribution_rvs.stypy_call_varargs = varargs
    check_distribution_rvs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_distribution_rvs', ['dist', 'args', 'alpha', 'rvs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_distribution_rvs', localization, ['dist', 'args', 'alpha', 'rvs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_distribution_rvs(...)' code ##################

    
    # Assigning a Call to a Tuple (line 392):
    
    # Assigning a Subscript to a Name (line 392):
    
    # Obtaining the type of the subscript
    int_634527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 4), 'int')
    
    # Call to kstest(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'rvs' (line 392)
    rvs_634530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 27), 'rvs', False)
    # Getting the type of 'dist' (line 392)
    dist_634531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 32), 'dist', False)
    # Processing the call keyword arguments (line 392)
    # Getting the type of 'args' (line 392)
    args_634532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 43), 'args', False)
    keyword_634533 = args_634532
    int_634534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 51), 'int')
    keyword_634535 = int_634534
    kwargs_634536 = {'args': keyword_634533, 'N': keyword_634535}
    # Getting the type of 'stats' (line 392)
    stats_634528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 14), 'stats', False)
    # Obtaining the member 'kstest' of a type (line 392)
    kstest_634529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 14), stats_634528, 'kstest')
    # Calling kstest(args, kwargs) (line 392)
    kstest_call_result_634537 = invoke(stypy.reporting.localization.Localization(__file__, 392, 14), kstest_634529, *[rvs_634530, dist_634531], **kwargs_634536)
    
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___634538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 4), kstest_call_result_634537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_634539 = invoke(stypy.reporting.localization.Localization(__file__, 392, 4), getitem___634538, int_634527)
    
    # Assigning a type to the variable 'tuple_var_assignment_633091' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'tuple_var_assignment_633091', subscript_call_result_634539)
    
    # Assigning a Subscript to a Name (line 392):
    
    # Obtaining the type of the subscript
    int_634540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 4), 'int')
    
    # Call to kstest(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'rvs' (line 392)
    rvs_634543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 27), 'rvs', False)
    # Getting the type of 'dist' (line 392)
    dist_634544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 32), 'dist', False)
    # Processing the call keyword arguments (line 392)
    # Getting the type of 'args' (line 392)
    args_634545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 43), 'args', False)
    keyword_634546 = args_634545
    int_634547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 51), 'int')
    keyword_634548 = int_634547
    kwargs_634549 = {'args': keyword_634546, 'N': keyword_634548}
    # Getting the type of 'stats' (line 392)
    stats_634541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 14), 'stats', False)
    # Obtaining the member 'kstest' of a type (line 392)
    kstest_634542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 14), stats_634541, 'kstest')
    # Calling kstest(args, kwargs) (line 392)
    kstest_call_result_634550 = invoke(stypy.reporting.localization.Localization(__file__, 392, 14), kstest_634542, *[rvs_634543, dist_634544], **kwargs_634549)
    
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___634551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 4), kstest_call_result_634550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_634552 = invoke(stypy.reporting.localization.Localization(__file__, 392, 4), getitem___634551, int_634540)
    
    # Assigning a type to the variable 'tuple_var_assignment_633092' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'tuple_var_assignment_633092', subscript_call_result_634552)
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'tuple_var_assignment_633091' (line 392)
    tuple_var_assignment_633091_634553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'tuple_var_assignment_633091')
    # Assigning a type to the variable 'D' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'D', tuple_var_assignment_633091_634553)
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'tuple_var_assignment_633092' (line 392)
    tuple_var_assignment_633092_634554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'tuple_var_assignment_633092')
    # Assigning a type to the variable 'pval' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 7), 'pval', tuple_var_assignment_633092_634554)
    
    
    # Getting the type of 'pval' (line 393)
    pval_634555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'pval')
    # Getting the type of 'alpha' (line 393)
    alpha_634556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'alpha')
    # Applying the binary operator '<' (line 393)
    result_lt_634557 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 8), '<', pval_634555, alpha_634556)
    
    # Testing the type of an if condition (line 393)
    if_condition_634558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 4), result_lt_634557)
    # Assigning a type to the variable 'if_condition_634558' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'if_condition_634558', if_condition_634558)
    # SSA begins for if statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 394):
    
    # Assigning a Subscript to a Name (line 394):
    
    # Obtaining the type of the subscript
    int_634559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 8), 'int')
    
    # Call to kstest(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'dist' (line 394)
    dist_634562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 31), 'dist', False)
    str_634563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 37), 'str', '')
    # Processing the call keyword arguments (line 394)
    # Getting the type of 'args' (line 394)
    args_634564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 46), 'args', False)
    keyword_634565 = args_634564
    int_634566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 54), 'int')
    keyword_634567 = int_634566
    kwargs_634568 = {'args': keyword_634565, 'N': keyword_634567}
    # Getting the type of 'stats' (line 394)
    stats_634560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 18), 'stats', False)
    # Obtaining the member 'kstest' of a type (line 394)
    kstest_634561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 18), stats_634560, 'kstest')
    # Calling kstest(args, kwargs) (line 394)
    kstest_call_result_634569 = invoke(stypy.reporting.localization.Localization(__file__, 394, 18), kstest_634561, *[dist_634562, str_634563], **kwargs_634568)
    
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___634570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), kstest_call_result_634569, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_634571 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), getitem___634570, int_634559)
    
    # Assigning a type to the variable 'tuple_var_assignment_633093' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'tuple_var_assignment_633093', subscript_call_result_634571)
    
    # Assigning a Subscript to a Name (line 394):
    
    # Obtaining the type of the subscript
    int_634572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 8), 'int')
    
    # Call to kstest(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'dist' (line 394)
    dist_634575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 31), 'dist', False)
    str_634576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 37), 'str', '')
    # Processing the call keyword arguments (line 394)
    # Getting the type of 'args' (line 394)
    args_634577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 46), 'args', False)
    keyword_634578 = args_634577
    int_634579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 54), 'int')
    keyword_634580 = int_634579
    kwargs_634581 = {'args': keyword_634578, 'N': keyword_634580}
    # Getting the type of 'stats' (line 394)
    stats_634573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 18), 'stats', False)
    # Obtaining the member 'kstest' of a type (line 394)
    kstest_634574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 18), stats_634573, 'kstest')
    # Calling kstest(args, kwargs) (line 394)
    kstest_call_result_634582 = invoke(stypy.reporting.localization.Localization(__file__, 394, 18), kstest_634574, *[dist_634575, str_634576], **kwargs_634581)
    
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___634583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 8), kstest_call_result_634582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_634584 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), getitem___634583, int_634572)
    
    # Assigning a type to the variable 'tuple_var_assignment_633094' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'tuple_var_assignment_633094', subscript_call_result_634584)
    
    # Assigning a Name to a Name (line 394):
    # Getting the type of 'tuple_var_assignment_633093' (line 394)
    tuple_var_assignment_633093_634585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'tuple_var_assignment_633093')
    # Assigning a type to the variable 'D' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'D', tuple_var_assignment_633093_634585)
    
    # Assigning a Name to a Name (line 394):
    # Getting the type of 'tuple_var_assignment_633094' (line 394)
    tuple_var_assignment_633094_634586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'tuple_var_assignment_633094')
    # Assigning a type to the variable 'pval' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 11), 'pval', tuple_var_assignment_633094_634586)
    
    # Call to assert_(...): (line 395)
    # Processing the call arguments (line 395)
    
    # Getting the type of 'pval' (line 395)
    pval_634589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 20), 'pval', False)
    # Getting the type of 'alpha' (line 395)
    alpha_634590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'alpha', False)
    # Applying the binary operator '>' (line 395)
    result_gt_634591 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 20), '>', pval_634589, alpha_634590)
    
    str_634592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 34), 'str', 'D = ')
    
    # Call to str(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'D' (line 395)
    D_634594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 47), 'D', False)
    # Processing the call keyword arguments (line 395)
    kwargs_634595 = {}
    # Getting the type of 'str' (line 395)
    str_634593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 43), 'str', False)
    # Calling str(args, kwargs) (line 395)
    str_call_result_634596 = invoke(stypy.reporting.localization.Localization(__file__, 395, 43), str_634593, *[D_634594], **kwargs_634595)
    
    # Applying the binary operator '+' (line 395)
    result_add_634597 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 34), '+', str_634592, str_call_result_634596)
    
    str_634598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 52), 'str', '; pval = ')
    # Applying the binary operator '+' (line 395)
    result_add_634599 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 50), '+', result_add_634597, str_634598)
    
    
    # Call to str(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'pval' (line 395)
    pval_634601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 70), 'pval', False)
    # Processing the call keyword arguments (line 395)
    kwargs_634602 = {}
    # Getting the type of 'str' (line 395)
    str_634600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 66), 'str', False)
    # Calling str(args, kwargs) (line 395)
    str_call_result_634603 = invoke(stypy.reporting.localization.Localization(__file__, 395, 66), str_634600, *[pval_634601], **kwargs_634602)
    
    # Applying the binary operator '+' (line 395)
    result_add_634604 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 64), '+', result_add_634599, str_call_result_634603)
    
    str_634605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 20), 'str', '; alpha = ')
    # Applying the binary operator '+' (line 395)
    result_add_634606 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 76), '+', result_add_634604, str_634605)
    
    
    # Call to str(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'alpha' (line 396)
    alpha_634608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 39), 'alpha', False)
    # Processing the call keyword arguments (line 396)
    kwargs_634609 = {}
    # Getting the type of 'str' (line 396)
    str_634607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 35), 'str', False)
    # Calling str(args, kwargs) (line 396)
    str_call_result_634610 = invoke(stypy.reporting.localization.Localization(__file__, 396, 35), str_634607, *[alpha_634608], **kwargs_634609)
    
    # Applying the binary operator '+' (line 396)
    result_add_634611 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 33), '+', result_add_634606, str_call_result_634610)
    
    str_634612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 48), 'str', '\nargs = ')
    # Applying the binary operator '+' (line 396)
    result_add_634613 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 46), '+', result_add_634611, str_634612)
    
    
    # Call to str(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'args' (line 396)
    args_634615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 66), 'args', False)
    # Processing the call keyword arguments (line 396)
    kwargs_634616 = {}
    # Getting the type of 'str' (line 396)
    str_634614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 62), 'str', False)
    # Calling str(args, kwargs) (line 396)
    str_call_result_634617 = invoke(stypy.reporting.localization.Localization(__file__, 396, 62), str_634614, *[args_634615], **kwargs_634616)
    
    # Applying the binary operator '+' (line 396)
    result_add_634618 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 60), '+', result_add_634613, str_call_result_634617)
    
    # Processing the call keyword arguments (line 395)
    kwargs_634619 = {}
    # Getting the type of 'npt' (line 395)
    npt_634587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 395)
    assert__634588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), npt_634587, 'assert_')
    # Calling assert_(args, kwargs) (line 395)
    assert__call_result_634620 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), assert__634588, *[result_gt_634591, result_add_634618], **kwargs_634619)
    
    # SSA join for if statement (line 393)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_distribution_rvs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_distribution_rvs' in the type store
    # Getting the type of 'stypy_return_type' (line 389)
    stypy_return_type_634621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634621)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_distribution_rvs'
    return stypy_return_type_634621

# Assigning a type to the variable 'check_distribution_rvs' (line 389)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'check_distribution_rvs', check_distribution_rvs)

@norecursion
def check_vecentropy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_vecentropy'
    module_type_store = module_type_store.open_function_context('check_vecentropy', 399, 0, False)
    
    # Passed parameters checking function
    check_vecentropy.stypy_localization = localization
    check_vecentropy.stypy_type_of_self = None
    check_vecentropy.stypy_type_store = module_type_store
    check_vecentropy.stypy_function_name = 'check_vecentropy'
    check_vecentropy.stypy_param_names_list = ['distfn', 'args']
    check_vecentropy.stypy_varargs_param_name = None
    check_vecentropy.stypy_kwargs_param_name = None
    check_vecentropy.stypy_call_defaults = defaults
    check_vecentropy.stypy_call_varargs = varargs
    check_vecentropy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_vecentropy', ['distfn', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_vecentropy', localization, ['distfn', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_vecentropy(...)' code ##################

    
    # Call to assert_equal(...): (line 400)
    # Processing the call arguments (line 400)
    
    # Call to vecentropy(...): (line 400)
    # Getting the type of 'args' (line 400)
    args_634626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 40), 'args', False)
    # Processing the call keyword arguments (line 400)
    kwargs_634627 = {}
    # Getting the type of 'distfn' (line 400)
    distfn_634624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 'distfn', False)
    # Obtaining the member 'vecentropy' of a type (line 400)
    vecentropy_634625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 21), distfn_634624, 'vecentropy')
    # Calling vecentropy(args, kwargs) (line 400)
    vecentropy_call_result_634628 = invoke(stypy.reporting.localization.Localization(__file__, 400, 21), vecentropy_634625, *[args_634626], **kwargs_634627)
    
    
    # Call to _entropy(...): (line 400)
    # Getting the type of 'args' (line 400)
    args_634631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 64), 'args', False)
    # Processing the call keyword arguments (line 400)
    kwargs_634632 = {}
    # Getting the type of 'distfn' (line 400)
    distfn_634629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 47), 'distfn', False)
    # Obtaining the member '_entropy' of a type (line 400)
    _entropy_634630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 47), distfn_634629, '_entropy')
    # Calling _entropy(args, kwargs) (line 400)
    _entropy_call_result_634633 = invoke(stypy.reporting.localization.Localization(__file__, 400, 47), _entropy_634630, *[args_634631], **kwargs_634632)
    
    # Processing the call keyword arguments (line 400)
    kwargs_634634 = {}
    # Getting the type of 'npt' (line 400)
    npt_634622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 400)
    assert_equal_634623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 4), npt_634622, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 400)
    assert_equal_call_result_634635 = invoke(stypy.reporting.localization.Localization(__file__, 400, 4), assert_equal_634623, *[vecentropy_call_result_634628, _entropy_call_result_634633], **kwargs_634634)
    
    
    # ################# End of 'check_vecentropy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_vecentropy' in the type store
    # Getting the type of 'stypy_return_type' (line 399)
    stypy_return_type_634636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634636)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_vecentropy'
    return stypy_return_type_634636

# Assigning a type to the variable 'check_vecentropy' (line 399)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'check_vecentropy', check_vecentropy)

@norecursion
def check_loc_scale(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_loc_scale'
    module_type_store = module_type_store.open_function_context('check_loc_scale', 403, 0, False)
    
    # Passed parameters checking function
    check_loc_scale.stypy_localization = localization
    check_loc_scale.stypy_type_of_self = None
    check_loc_scale.stypy_type_store = module_type_store
    check_loc_scale.stypy_function_name = 'check_loc_scale'
    check_loc_scale.stypy_param_names_list = ['distfn', 'arg', 'm', 'v', 'msg']
    check_loc_scale.stypy_varargs_param_name = None
    check_loc_scale.stypy_kwargs_param_name = None
    check_loc_scale.stypy_call_defaults = defaults
    check_loc_scale.stypy_call_varargs = varargs
    check_loc_scale.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_loc_scale', ['distfn', 'arg', 'm', 'v', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_loc_scale', localization, ['distfn', 'arg', 'm', 'v', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_loc_scale(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 404):
    
    # Assigning a Num to a Name (line 404):
    float_634637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 17), 'float')
    # Assigning a type to the variable 'tuple_assignment_633095' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_assignment_633095', float_634637)
    
    # Assigning a Num to a Name (line 404):
    float_634638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 23), 'float')
    # Assigning a type to the variable 'tuple_assignment_633096' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_assignment_633096', float_634638)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'tuple_assignment_633095' (line 404)
    tuple_assignment_633095_634639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_assignment_633095')
    # Assigning a type to the variable 'loc' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'loc', tuple_assignment_633095_634639)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'tuple_assignment_633096' (line 404)
    tuple_assignment_633096_634640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'tuple_assignment_633096')
    # Assigning a type to the variable 'scale' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 9), 'scale', tuple_assignment_633096_634640)
    
    # Assigning a Call to a Tuple (line 405):
    
    # Assigning a Subscript to a Name (line 405):
    
    # Obtaining the type of the subscript
    int_634641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 4), 'int')
    
    # Call to stats(...): (line 405)
    # Getting the type of 'arg' (line 405)
    arg_634644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 49), 'arg', False)
    # Processing the call keyword arguments (line 405)
    # Getting the type of 'loc' (line 405)
    loc_634645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 30), 'loc', False)
    keyword_634646 = loc_634645
    # Getting the type of 'scale' (line 405)
    scale_634647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 41), 'scale', False)
    keyword_634648 = scale_634647
    kwargs_634649 = {'loc': keyword_634646, 'scale': keyword_634648}
    # Getting the type of 'distfn' (line 405)
    distfn_634642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'distfn', False)
    # Obtaining the member 'stats' of a type (line 405)
    stats_634643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 13), distfn_634642, 'stats')
    # Calling stats(args, kwargs) (line 405)
    stats_call_result_634650 = invoke(stypy.reporting.localization.Localization(__file__, 405, 13), stats_634643, *[arg_634644], **kwargs_634649)
    
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___634651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 4), stats_call_result_634650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_634652 = invoke(stypy.reporting.localization.Localization(__file__, 405, 4), getitem___634651, int_634641)
    
    # Assigning a type to the variable 'tuple_var_assignment_633097' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'tuple_var_assignment_633097', subscript_call_result_634652)
    
    # Assigning a Subscript to a Name (line 405):
    
    # Obtaining the type of the subscript
    int_634653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 4), 'int')
    
    # Call to stats(...): (line 405)
    # Getting the type of 'arg' (line 405)
    arg_634656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 49), 'arg', False)
    # Processing the call keyword arguments (line 405)
    # Getting the type of 'loc' (line 405)
    loc_634657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 30), 'loc', False)
    keyword_634658 = loc_634657
    # Getting the type of 'scale' (line 405)
    scale_634659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 41), 'scale', False)
    keyword_634660 = scale_634659
    kwargs_634661 = {'loc': keyword_634658, 'scale': keyword_634660}
    # Getting the type of 'distfn' (line 405)
    distfn_634654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'distfn', False)
    # Obtaining the member 'stats' of a type (line 405)
    stats_634655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 13), distfn_634654, 'stats')
    # Calling stats(args, kwargs) (line 405)
    stats_call_result_634662 = invoke(stypy.reporting.localization.Localization(__file__, 405, 13), stats_634655, *[arg_634656], **kwargs_634661)
    
    # Obtaining the member '__getitem__' of a type (line 405)
    getitem___634663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 4), stats_call_result_634662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 405)
    subscript_call_result_634664 = invoke(stypy.reporting.localization.Localization(__file__, 405, 4), getitem___634663, int_634653)
    
    # Assigning a type to the variable 'tuple_var_assignment_633098' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'tuple_var_assignment_633098', subscript_call_result_634664)
    
    # Assigning a Name to a Name (line 405):
    # Getting the type of 'tuple_var_assignment_633097' (line 405)
    tuple_var_assignment_633097_634665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'tuple_var_assignment_633097')
    # Assigning a type to the variable 'mt' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'mt', tuple_var_assignment_633097_634665)
    
    # Assigning a Name to a Name (line 405):
    # Getting the type of 'tuple_var_assignment_633098' (line 405)
    tuple_var_assignment_633098_634666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'tuple_var_assignment_633098')
    # Assigning a type to the variable 'vt' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'vt', tuple_var_assignment_633098_634666)
    
    # Call to assert_allclose(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'm' (line 406)
    m_634669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 24), 'm', False)
    # Getting the type of 'scale' (line 406)
    scale_634670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 26), 'scale', False)
    # Applying the binary operator '*' (line 406)
    result_mul_634671 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 24), '*', m_634669, scale_634670)
    
    # Getting the type of 'loc' (line 406)
    loc_634672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 34), 'loc', False)
    # Applying the binary operator '+' (line 406)
    result_add_634673 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 24), '+', result_mul_634671, loc_634672)
    
    # Getting the type of 'mt' (line 406)
    mt_634674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 39), 'mt', False)
    # Processing the call keyword arguments (line 406)
    kwargs_634675 = {}
    # Getting the type of 'npt' (line 406)
    npt_634667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 406)
    assert_allclose_634668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), npt_634667, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 406)
    assert_allclose_call_result_634676 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), assert_allclose_634668, *[result_add_634673, mt_634674], **kwargs_634675)
    
    
    # Call to assert_allclose(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'v' (line 407)
    v_634679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 24), 'v', False)
    # Getting the type of 'scale' (line 407)
    scale_634680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 26), 'scale', False)
    # Applying the binary operator '*' (line 407)
    result_mul_634681 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 24), '*', v_634679, scale_634680)
    
    # Getting the type of 'scale' (line 407)
    scale_634682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 32), 'scale', False)
    # Applying the binary operator '*' (line 407)
    result_mul_634683 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 31), '*', result_mul_634681, scale_634682)
    
    # Getting the type of 'vt' (line 407)
    vt_634684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 39), 'vt', False)
    # Processing the call keyword arguments (line 407)
    kwargs_634685 = {}
    # Getting the type of 'npt' (line 407)
    npt_634677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 407)
    assert_allclose_634678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 4), npt_634677, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 407)
    assert_allclose_call_result_634686 = invoke(stypy.reporting.localization.Localization(__file__, 407, 4), assert_allclose_634678, *[result_mul_634683, vt_634684], **kwargs_634685)
    
    
    # ################# End of 'check_loc_scale(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_loc_scale' in the type store
    # Getting the type of 'stypy_return_type' (line 403)
    stypy_return_type_634687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634687)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_loc_scale'
    return stypy_return_type_634687

# Assigning a type to the variable 'check_loc_scale' (line 403)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 0), 'check_loc_scale', check_loc_scale)

@norecursion
def check_ppf_private(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_ppf_private'
    module_type_store = module_type_store.open_function_context('check_ppf_private', 410, 0, False)
    
    # Passed parameters checking function
    check_ppf_private.stypy_localization = localization
    check_ppf_private.stypy_type_of_self = None
    check_ppf_private.stypy_type_store = module_type_store
    check_ppf_private.stypy_function_name = 'check_ppf_private'
    check_ppf_private.stypy_param_names_list = ['distfn', 'arg', 'msg']
    check_ppf_private.stypy_varargs_param_name = None
    check_ppf_private.stypy_kwargs_param_name = None
    check_ppf_private.stypy_call_defaults = defaults
    check_ppf_private.stypy_call_varargs = varargs
    check_ppf_private.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_ppf_private', ['distfn', 'arg', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_ppf_private', localization, ['distfn', 'arg', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_ppf_private(...)' code ##################

    
    # Assigning a Call to a Name (line 412):
    
    # Assigning a Call to a Name (line 412):
    
    # Call to _ppf(...): (line 412)
    # Processing the call arguments (line 412)
    
    # Call to array(...): (line 412)
    # Processing the call arguments (line 412)
    
    # Obtaining an instance of the builtin type 'list' (line 412)
    list_634692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 412)
    # Adding element type (line 412)
    float_634693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 32), list_634692, float_634693)
    # Adding element type (line 412)
    float_634694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 32), list_634692, float_634694)
    # Adding element type (line 412)
    float_634695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 32), list_634692, float_634695)
    
    # Processing the call keyword arguments (line 412)
    kwargs_634696 = {}
    # Getting the type of 'np' (line 412)
    np_634690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 23), 'np', False)
    # Obtaining the member 'array' of a type (line 412)
    array_634691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 23), np_634690, 'array')
    # Calling array(args, kwargs) (line 412)
    array_call_result_634697 = invoke(stypy.reporting.localization.Localization(__file__, 412, 23), array_634691, *[list_634692], **kwargs_634696)
    
    # Getting the type of 'arg' (line 412)
    arg_634698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 51), 'arg', False)
    # Processing the call keyword arguments (line 412)
    kwargs_634699 = {}
    # Getting the type of 'distfn' (line 412)
    distfn_634688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 11), 'distfn', False)
    # Obtaining the member '_ppf' of a type (line 412)
    _ppf_634689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 11), distfn_634688, '_ppf')
    # Calling _ppf(args, kwargs) (line 412)
    _ppf_call_result_634700 = invoke(stypy.reporting.localization.Localization(__file__, 412, 11), _ppf_634689, *[array_call_result_634697, arg_634698], **kwargs_634699)
    
    # Assigning a type to the variable 'ppfs' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'ppfs', _ppf_call_result_634700)
    
    # Call to assert_(...): (line 413)
    # Processing the call arguments (line 413)
    
    
    # Call to any(...): (line 413)
    # Processing the call arguments (line 413)
    
    # Call to isnan(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'ppfs' (line 413)
    ppfs_634707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 36), 'ppfs', False)
    # Processing the call keyword arguments (line 413)
    kwargs_634708 = {}
    # Getting the type of 'np' (line 413)
    np_634705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 27), 'np', False)
    # Obtaining the member 'isnan' of a type (line 413)
    isnan_634706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 27), np_634705, 'isnan')
    # Calling isnan(args, kwargs) (line 413)
    isnan_call_result_634709 = invoke(stypy.reporting.localization.Localization(__file__, 413, 27), isnan_634706, *[ppfs_634707], **kwargs_634708)
    
    # Processing the call keyword arguments (line 413)
    kwargs_634710 = {}
    # Getting the type of 'np' (line 413)
    np_634703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 20), 'np', False)
    # Obtaining the member 'any' of a type (line 413)
    any_634704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 20), np_634703, 'any')
    # Calling any(args, kwargs) (line 413)
    any_call_result_634711 = invoke(stypy.reporting.localization.Localization(__file__, 413, 20), any_634704, *[isnan_call_result_634709], **kwargs_634710)
    
    # Applying the 'not' unary operator (line 413)
    result_not__634712 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 16), 'not', any_call_result_634711)
    
    # Getting the type of 'msg' (line 413)
    msg_634713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 44), 'msg', False)
    str_634714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 50), 'str', 'ppf private is nan')
    # Applying the binary operator '+' (line 413)
    result_add_634715 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 44), '+', msg_634713, str_634714)
    
    # Processing the call keyword arguments (line 413)
    kwargs_634716 = {}
    # Getting the type of 'npt' (line 413)
    npt_634701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 413)
    assert__634702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 4), npt_634701, 'assert_')
    # Calling assert_(args, kwargs) (line 413)
    assert__call_result_634717 = invoke(stypy.reporting.localization.Localization(__file__, 413, 4), assert__634702, *[result_not__634712, result_add_634715], **kwargs_634716)
    
    
    # ################# End of 'check_ppf_private(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_ppf_private' in the type store
    # Getting the type of 'stypy_return_type' (line 410)
    stypy_return_type_634718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_634718)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_ppf_private'
    return stypy_return_type_634718

# Assigning a type to the variable 'check_ppf_private' (line 410)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'check_ppf_private', check_ppf_private)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

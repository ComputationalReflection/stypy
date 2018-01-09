
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: 
5: import numpy as np
6: from numpy import arccosh, arcsinh, arctanh
7: from scipy._lib._numpy_compat import suppress_warnings
8: import pytest
9: 
10: from scipy.special import (
11:     lpn, lpmn, lpmv, lqn, lqmn, sph_harm, eval_legendre, eval_hermite,
12:     eval_laguerre, eval_genlaguerre, binom, cbrt, expm1, log1p, zeta,
13:     jn, jv, yn, yv, iv, kv, kn,
14:     gamma, gammaln, gammainc, gammaincc, gammaincinv, gammainccinv, digamma,
15:     beta, betainc, betaincinv, poch,
16:     ellipe, ellipeinc, ellipk, ellipkm1, ellipkinc, ellipj,
17:     erf, erfc, erfinv, erfcinv, exp1, expi, expn,
18:     bdtrik, btdtr, btdtri, btdtria, btdtrib, chndtr, gdtr, gdtrc, gdtrix, gdtrib,
19:     nbdtrik, pdtrik,
20:     mathieu_a, mathieu_b, mathieu_cem, mathieu_sem, mathieu_modcem1,
21:     mathieu_modsem1, mathieu_modcem2, mathieu_modsem2,
22:     ellip_harm, ellip_harm_2, spherical_jn, spherical_yn,
23: )
24: from scipy.integrate import IntegrationWarning
25: 
26: from scipy.special._testutils import FuncData
27: 
28: DATASETS_BOOST = np.load(os.path.join(os.path.dirname(__file__),
29:                                       "data", "boost.npz"))
30: 
31: DATASETS_GSL = np.load(os.path.join(os.path.dirname(__file__),
32:                                     "data", "gsl.npz"))
33: 
34: DATASETS_LOCAL = np.load(os.path.join(os.path.dirname(__file__),
35:                                     "data", "local.npz"))
36: 
37: 
38: def data(func, dataname, *a, **kw):
39:     kw.setdefault('dataname', dataname)
40:     return FuncData(func, DATASETS_BOOST[dataname], *a, **kw)
41: 
42: 
43: def data_gsl(func, dataname, *a, **kw):
44:     kw.setdefault('dataname', dataname)
45:     return FuncData(func, DATASETS_GSL[dataname], *a, **kw)
46: 
47: 
48: def data_local(func, dataname, *a, **kw):
49:     kw.setdefault('dataname', dataname)
50:     return FuncData(func, DATASETS_LOCAL[dataname], *a, **kw)
51: 
52: 
53: def ellipk_(k):
54:     return ellipk(k*k)
55: 
56: 
57: def ellipkinc_(f, k):
58:     return ellipkinc(f, k*k)
59: 
60: 
61: def ellipe_(k):
62:     return ellipe(k*k)
63: 
64: 
65: def ellipeinc_(f, k):
66:     return ellipeinc(f, k*k)
67: 
68: 
69: def ellipj_(k):
70:     return ellipj(k*k)
71: 
72: 
73: def zeta_(x):
74:     return zeta(x, 1.)
75: 
76: 
77: def assoc_legendre_p_boost_(nu, mu, x):
78:     # the boost test data is for integer orders only
79:     return lpmv(mu, nu.astype(int), x)
80: 
81: def legendre_p_via_assoc_(nu, x):
82:     return lpmv(0, nu, x)
83: 
84: def lpn_(n, x):
85:     return lpn(n.astype('l'), x)[0][-1]
86: 
87: def lqn_(n, x):
88:     return lqn(n.astype('l'), x)[0][-1]
89: 
90: def legendre_p_via_lpmn(n, x):
91:     return lpmn(0, n, x)[0][0,-1]
92: 
93: def legendre_q_via_lqmn(n, x):
94:     return lqmn(0, n, x)[0][0,-1]
95: 
96: def mathieu_ce_rad(m, q, x):
97:     return mathieu_cem(m, q, x*180/np.pi)[0]
98: 
99: 
100: def mathieu_se_rad(m, q, x):
101:     return mathieu_sem(m, q, x*180/np.pi)[0]
102: 
103: 
104: def mathieu_mc1_scaled(m, q, x):
105:     # GSL follows a different normalization.
106:     # We follow Abramowitz & Stegun, they apparently something else.
107:     return mathieu_modcem1(m, q, x)[0] * np.sqrt(np.pi/2)
108: 
109: 
110: def mathieu_ms1_scaled(m, q, x):
111:     return mathieu_modsem1(m, q, x)[0] * np.sqrt(np.pi/2)
112: 
113: 
114: def mathieu_mc2_scaled(m, q, x):
115:     return mathieu_modcem2(m, q, x)[0] * np.sqrt(np.pi/2)
116: 
117: 
118: def mathieu_ms2_scaled(m, q, x):
119:     return mathieu_modsem2(m, q, x)[0] * np.sqrt(np.pi/2)
120: 
121: def eval_legendre_ld(n, x):
122:     return eval_legendre(n.astype('l'), x)
123: 
124: def eval_legendre_dd(n, x):
125:     return eval_legendre(n.astype('d'), x)
126: 
127: def eval_hermite_ld(n, x):
128:     return eval_hermite(n.astype('l'), x)
129: 
130: def eval_laguerre_ld(n, x):
131:     return eval_laguerre(n.astype('l'), x)
132: 
133: def eval_laguerre_dd(n, x):
134:     return eval_laguerre(n.astype('d'), x)
135: 
136: def eval_genlaguerre_ldd(n, a, x):
137:     return eval_genlaguerre(n.astype('l'), a, x)
138: 
139: def eval_genlaguerre_ddd(n, a, x):
140:     return eval_genlaguerre(n.astype('d'), a, x)
141: 
142: def bdtrik_comp(y, n, p):
143:     return bdtrik(1-y, n, p)
144: 
145: def btdtri_comp(a, b, p):
146:     return btdtri(a, b, 1-p)
147: 
148: def btdtria_comp(p, b, x):
149:     return btdtria(1-p, b, x)
150: 
151: def btdtrib_comp(a, p, x):
152:     return btdtrib(a, 1-p, x)
153: 
154: def gdtr_(p, x):
155:     return gdtr(1.0, p, x)
156: 
157: def gdtrc_(p, x):
158:     return gdtrc(1.0, p, x)
159: 
160: def gdtrix_(b, p):
161:     return gdtrix(1.0, b, p)
162: 
163: def gdtrix_comp(b, p):
164:     return gdtrix(1.0, b, 1-p)
165: 
166: def gdtrib_(p, x):
167:     return gdtrib(1.0, p, x)
168: 
169: def gdtrib_comp(p, x):
170:     return gdtrib(1.0, 1-p, x)
171: 
172: def nbdtrik_comp(y, n, p):
173:     return nbdtrik(1-y, n, p)
174: 
175: def pdtrik_comp(p, m):
176:     return pdtrik(1-p, m)
177: 
178: def poch_(z, m):
179:     return 1.0 / poch(z, m)
180: 
181: def poch_minus(z, m):
182:     return 1.0 / poch(z, -m)
183: 
184: def spherical_jn_(n, x):
185:     return spherical_jn(n.astype('l'), x)
186: 
187: def spherical_yn_(n, x):
188:     return spherical_yn(n.astype('l'), x)
189: 
190: def sph_harm_(m, n, theta, phi):
191:     y = sph_harm(m, n, theta, phi)
192:     return (y.real, y.imag)
193: 
194: def cexpm1(x, y):
195:     z = expm1(x + 1j*y)
196:     return z.real, z.imag
197: 
198: def clog1p(x, y):
199:     z = log1p(x + 1j*y)
200:     return z.real, z.imag
201: 
202: BOOST_TESTS = [
203:         data(arccosh, 'acosh_data_ipp-acosh_data', 0, 1, rtol=5e-13),
204:         data(arccosh, 'acosh_data_ipp-acosh_data', 0j, 1, rtol=5e-13),
205: 
206:         data(arcsinh, 'asinh_data_ipp-asinh_data', 0, 1, rtol=1e-11),
207:         data(arcsinh, 'asinh_data_ipp-asinh_data', 0j, 1, rtol=1e-11),
208: 
209:         data(arctanh, 'atanh_data_ipp-atanh_data', 0, 1, rtol=1e-11),
210:         data(arctanh, 'atanh_data_ipp-atanh_data', 0j, 1, rtol=1e-11),
211: 
212:         data(assoc_legendre_p_boost_, 'assoc_legendre_p_ipp-assoc_legendre_p', (0,1,2), 3, rtol=1e-11),
213: 
214:         data(legendre_p_via_assoc_, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=1e-11),
215:         data(legendre_p_via_assoc_, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=7e-14),
216:         data(legendre_p_via_lpmn, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=5e-14, vectorized=False),
217:         data(legendre_p_via_lpmn, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=7e-14, vectorized=False),
218:         data(lpn_, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=5e-14, vectorized=False),
219:         data(lpn_, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=3e-13, vectorized=False),
220:         data(eval_legendre_ld, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=6e-14),
221:         data(eval_legendre_ld, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=2e-13),
222:         data(eval_legendre_dd, 'legendre_p_ipp-legendre_p', (0,1), 2, rtol=2e-14),
223:         data(eval_legendre_dd, 'legendre_p_large_ipp-legendre_p_large', (0,1), 2, rtol=2e-13),
224: 
225:         data(lqn_, 'legendre_p_ipp-legendre_p', (0,1), 3, rtol=2e-14, vectorized=False),
226:         data(lqn_, 'legendre_p_large_ipp-legendre_p_large', (0,1), 3, rtol=2e-12, vectorized=False),
227:         data(legendre_q_via_lqmn, 'legendre_p_ipp-legendre_p', (0,1), 3, rtol=2e-14, vectorized=False),
228:         data(legendre_q_via_lqmn, 'legendre_p_large_ipp-legendre_p_large', (0,1), 3, rtol=2e-12, vectorized=False),
229: 
230:         data(beta, 'beta_exp_data_ipp-beta_exp_data', (0,1), 2, rtol=1e-13),
231:         data(beta, 'beta_exp_data_ipp-beta_exp_data', (0,1), 2, rtol=1e-13),
232:         data(beta, 'beta_small_data_ipp-beta_small_data', (0,1), 2),
233:         data(beta, 'beta_med_data_ipp-beta_med_data', (0,1), 2, rtol=5e-13),
234: 
235:         data(betainc, 'ibeta_small_data_ipp-ibeta_small_data', (0,1,2), 5, rtol=6e-15),
236:         data(betainc, 'ibeta_data_ipp-ibeta_data', (0,1,2), 5, rtol=5e-13),
237:         data(betainc, 'ibeta_int_data_ipp-ibeta_int_data', (0,1,2), 5, rtol=2e-14),
238:         data(betainc, 'ibeta_large_data_ipp-ibeta_large_data', (0,1,2), 5, rtol=4e-10),
239: 
240:         data(betaincinv, 'ibeta_inv_data_ipp-ibeta_inv_data', (0,1,2), 3, rtol=1e-5),
241: 
242:         data(btdtr, 'ibeta_small_data_ipp-ibeta_small_data', (0,1,2), 5, rtol=6e-15),
243:         data(btdtr, 'ibeta_data_ipp-ibeta_data', (0,1,2), 5, rtol=4e-13),
244:         data(btdtr, 'ibeta_int_data_ipp-ibeta_int_data', (0,1,2), 5, rtol=2e-14),
245:         data(btdtr, 'ibeta_large_data_ipp-ibeta_large_data', (0,1,2), 5, rtol=4e-10),
246: 
247:         data(btdtri, 'ibeta_inv_data_ipp-ibeta_inv_data', (0,1,2), 3, rtol=1e-5),
248:         data(btdtri_comp, 'ibeta_inv_data_ipp-ibeta_inv_data', (0,1,2), 4, rtol=8e-7),
249: 
250:         data(btdtria, 'ibeta_inva_data_ipp-ibeta_inva_data', (2,0,1), 3, rtol=5e-9),
251:         data(btdtria_comp, 'ibeta_inva_data_ipp-ibeta_inva_data', (2,0,1), 4, rtol=5e-9),
252: 
253:         data(btdtrib, 'ibeta_inva_data_ipp-ibeta_inva_data', (0,2,1), 5, rtol=5e-9),
254:         data(btdtrib_comp, 'ibeta_inva_data_ipp-ibeta_inva_data', (0,2,1), 6, rtol=5e-9),
255: 
256:         data(binom, 'binomial_data_ipp-binomial_data', (0,1), 2, rtol=1e-13),
257:         data(binom, 'binomial_large_data_ipp-binomial_large_data', (0,1), 2, rtol=5e-13),
258: 
259:         data(bdtrik, 'binomial_quantile_ipp-binomial_quantile_data', (2,0,1), 3, rtol=5e-9),
260:         data(bdtrik_comp, 'binomial_quantile_ipp-binomial_quantile_data', (2,0,1), 4, rtol=5e-9),
261: 
262:         data(nbdtrik, 'negative_binomial_quantile_ipp-negative_binomial_quantile_data', (2,0,1), 3, rtol=4e-9),
263:         data(nbdtrik_comp, 'negative_binomial_quantile_ipp-negative_binomial_quantile_data', (2,0,1), 4, rtol=4e-9),
264: 
265:         data(pdtrik, 'poisson_quantile_ipp-poisson_quantile_data', (1,0), 2, rtol=3e-9),
266:         data(pdtrik_comp, 'poisson_quantile_ipp-poisson_quantile_data', (1,0), 3, rtol=4e-9),
267: 
268:         data(cbrt, 'cbrt_data_ipp-cbrt_data', 1, 0),
269: 
270:         data(digamma, 'digamma_data_ipp-digamma_data', 0, 1),
271:         data(digamma, 'digamma_data_ipp-digamma_data', 0j, 1),
272:         data(digamma, 'digamma_neg_data_ipp-digamma_neg_data', 0, 1, rtol=1e-13),
273:         data(digamma, 'digamma_neg_data_ipp-digamma_neg_data', 0j, 1, rtol=1e-13),
274:         data(digamma, 'digamma_root_data_ipp-digamma_root_data', 0, 1, rtol=1e-11),
275:         data(digamma, 'digamma_root_data_ipp-digamma_root_data', 0j, 1, rtol=1e-11),
276:         data(digamma, 'digamma_small_data_ipp-digamma_small_data', 0, 1),
277:         data(digamma, 'digamma_small_data_ipp-digamma_small_data', 0j, 1, rtol=1e-14),
278: 
279:         data(ellipk_, 'ellint_k_data_ipp-ellint_k_data', 0, 1),
280:         data(ellipkinc_, 'ellint_f_data_ipp-ellint_f_data', (0,1), 2, rtol=1e-14),
281:         data(ellipe_, 'ellint_e_data_ipp-ellint_e_data', 0, 1),
282:         data(ellipeinc_, 'ellint_e2_data_ipp-ellint_e2_data', (0,1), 2, rtol=1e-14),
283: 
284:         data(erf, 'erf_data_ipp-erf_data', 0, 1),
285:         data(erf, 'erf_data_ipp-erf_data', 0j, 1, rtol=1e-13),
286:         data(erfc, 'erf_data_ipp-erf_data', 0, 2, rtol=6e-15),
287:         data(erf, 'erf_large_data_ipp-erf_large_data', 0, 1),
288:         data(erf, 'erf_large_data_ipp-erf_large_data', 0j, 1),
289:         data(erfc, 'erf_large_data_ipp-erf_large_data', 0, 2, rtol=4e-14),
290:         data(erf, 'erf_small_data_ipp-erf_small_data', 0, 1),
291:         data(erf, 'erf_small_data_ipp-erf_small_data', 0j, 1, rtol=1e-13),
292:         data(erfc, 'erf_small_data_ipp-erf_small_data', 0, 2),
293: 
294:         data(erfinv, 'erf_inv_data_ipp-erf_inv_data', 0, 1),
295:         data(erfcinv, 'erfc_inv_data_ipp-erfc_inv_data', 0, 1),
296:         data(erfcinv, 'erfc_inv_big_data_ipp-erfc_inv_big_data2', 0, 1),
297: 
298:         data(exp1, 'expint_1_data_ipp-expint_1_data', 1, 2, rtol=1e-13),
299:         data(exp1, 'expint_1_data_ipp-expint_1_data', 1j, 2, rtol=5e-9),
300:         data(expi, 'expinti_data_ipp-expinti_data', 0, 1, rtol=1e-13),
301:         data(expi, 'expinti_data_double_ipp-expinti_data_double', 0, 1, rtol=1e-13),
302: 
303:         data(expn, 'expint_small_data_ipp-expint_small_data', (0,1), 2),
304:         data(expn, 'expint_data_ipp-expint_data', (0,1), 2, rtol=1e-14),
305: 
306:         data(gamma, 'test_gamma_data_ipp-near_0', 0, 1),
307:         data(gamma, 'test_gamma_data_ipp-near_1', 0, 1),
308:         data(gamma, 'test_gamma_data_ipp-near_2', 0, 1),
309:         data(gamma, 'test_gamma_data_ipp-near_m10', 0, 1),
310:         data(gamma, 'test_gamma_data_ipp-near_m55', 0, 1, rtol=7e-12),
311:         data(gamma, 'test_gamma_data_ipp-factorials', 0, 1, rtol=4e-14),
312:         data(gamma, 'test_gamma_data_ipp-near_0', 0j, 1, rtol=2e-9),
313:         data(gamma, 'test_gamma_data_ipp-near_1', 0j, 1, rtol=2e-9),
314:         data(gamma, 'test_gamma_data_ipp-near_2', 0j, 1, rtol=2e-9),
315:         data(gamma, 'test_gamma_data_ipp-near_m10', 0j, 1, rtol=2e-9),
316:         data(gamma, 'test_gamma_data_ipp-near_m55', 0j, 1, rtol=2e-9),
317:         data(gamma, 'test_gamma_data_ipp-factorials', 0j, 1, rtol=2e-13),
318:         data(gammaln, 'test_gamma_data_ipp-near_0', 0, 2, rtol=5e-11),
319:         data(gammaln, 'test_gamma_data_ipp-near_1', 0, 2, rtol=5e-11),
320:         data(gammaln, 'test_gamma_data_ipp-near_2', 0, 2, rtol=2e-10),
321:         data(gammaln, 'test_gamma_data_ipp-near_m10', 0, 2, rtol=5e-11),
322:         data(gammaln, 'test_gamma_data_ipp-near_m55', 0, 2, rtol=5e-11),
323:         data(gammaln, 'test_gamma_data_ipp-factorials', 0, 2),
324: 
325:         data(gammainc, 'igamma_small_data_ipp-igamma_small_data', (0,1), 5, rtol=5e-15),
326:         data(gammainc, 'igamma_med_data_ipp-igamma_med_data', (0,1), 5, rtol=2e-13),
327:         data(gammainc, 'igamma_int_data_ipp-igamma_int_data', (0,1), 5, rtol=2e-13),
328:         data(gammainc, 'igamma_big_data_ipp-igamma_big_data', (0,1), 5, rtol=1e-12),
329: 
330:         data(gdtr_, 'igamma_small_data_ipp-igamma_small_data', (0,1), 5, rtol=1e-13),
331:         data(gdtr_, 'igamma_med_data_ipp-igamma_med_data', (0,1), 5, rtol=2e-13),
332:         data(gdtr_, 'igamma_int_data_ipp-igamma_int_data', (0,1), 5, rtol=2e-13),
333:         data(gdtr_, 'igamma_big_data_ipp-igamma_big_data', (0,1), 5, rtol=2e-9),
334: 
335:         data(gammaincc, 'igamma_small_data_ipp-igamma_small_data', (0,1), 3, rtol=1e-13),
336:         data(gammaincc, 'igamma_med_data_ipp-igamma_med_data', (0,1), 3, rtol=2e-13),
337:         data(gammaincc, 'igamma_int_data_ipp-igamma_int_data', (0,1), 3, rtol=4e-14),
338:         data(gammaincc, 'igamma_big_data_ipp-igamma_big_data', (0,1), 3, rtol=1e-11),
339: 
340:         data(gdtrc_, 'igamma_small_data_ipp-igamma_small_data', (0,1), 3, rtol=1e-13),
341:         data(gdtrc_, 'igamma_med_data_ipp-igamma_med_data', (0,1), 3, rtol=2e-13),
342:         data(gdtrc_, 'igamma_int_data_ipp-igamma_int_data', (0,1), 3, rtol=4e-14),
343:         data(gdtrc_, 'igamma_big_data_ipp-igamma_big_data', (0,1), 3, rtol=1e-11),
344: 
345:         data(gdtrib_, 'igamma_inva_data_ipp-igamma_inva_data', (1,0), 2, rtol=5e-9),
346:         data(gdtrib_comp, 'igamma_inva_data_ipp-igamma_inva_data', (1,0), 3, rtol=5e-9),
347: 
348:         data(poch_, 'tgamma_delta_ratio_data_ipp-tgamma_delta_ratio_data', (0,1), 2, rtol=2e-13),
349:         data(poch_, 'tgamma_delta_ratio_int_ipp-tgamma_delta_ratio_int', (0,1), 2,),
350:         data(poch_, 'tgamma_delta_ratio_int2_ipp-tgamma_delta_ratio_int2', (0,1), 2,),
351:         data(poch_minus, 'tgamma_delta_ratio_data_ipp-tgamma_delta_ratio_data', (0,1), 3, rtol=2e-13),
352:         data(poch_minus, 'tgamma_delta_ratio_int_ipp-tgamma_delta_ratio_int', (0,1), 3),
353:         data(poch_minus, 'tgamma_delta_ratio_int2_ipp-tgamma_delta_ratio_int2', (0,1), 3),
354: 
355: 
356:         data(eval_hermite_ld, 'hermite_ipp-hermite', (0,1), 2, rtol=2e-14),
357:         data(eval_laguerre_ld, 'laguerre2_ipp-laguerre2', (0,1), 2, rtol=7e-12),
358:         data(eval_laguerre_dd, 'laguerre2_ipp-laguerre2', (0,1), 2, knownfailure='hyp2f1 insufficiently accurate.'),
359:         data(eval_genlaguerre_ldd, 'laguerre3_ipp-laguerre3', (0,1,2), 3, rtol=2e-13),
360:         data(eval_genlaguerre_ddd, 'laguerre3_ipp-laguerre3', (0,1,2), 3, knownfailure='hyp2f1 insufficiently accurate.'),
361: 
362:         data(log1p, 'log1p_expm1_data_ipp-log1p_expm1_data', 0, 1),
363:         data(expm1, 'log1p_expm1_data_ipp-log1p_expm1_data', 0, 2),
364: 
365:         data(iv, 'bessel_i_data_ipp-bessel_i_data', (0,1), 2, rtol=1e-12),
366:         data(iv, 'bessel_i_data_ipp-bessel_i_data', (0,1j), 2, rtol=2e-10, atol=1e-306),
367:         data(iv, 'bessel_i_int_data_ipp-bessel_i_int_data', (0,1), 2, rtol=1e-9),
368:         data(iv, 'bessel_i_int_data_ipp-bessel_i_int_data', (0,1j), 2, rtol=2e-10),
369: 
370:         data(jn, 'bessel_j_int_data_ipp-bessel_j_int_data', (0,1), 2, rtol=1e-12),
371:         data(jn, 'bessel_j_int_data_ipp-bessel_j_int_data', (0,1j), 2, rtol=1e-12),
372:         data(jn, 'bessel_j_large_data_ipp-bessel_j_large_data', (0,1), 2, rtol=6e-11),
373:         data(jn, 'bessel_j_large_data_ipp-bessel_j_large_data', (0,1j), 2, rtol=6e-11),
374: 
375:         data(jv, 'bessel_j_int_data_ipp-bessel_j_int_data', (0,1), 2, rtol=1e-12),
376:         data(jv, 'bessel_j_int_data_ipp-bessel_j_int_data', (0,1j), 2, rtol=1e-12),
377:         data(jv, 'bessel_j_data_ipp-bessel_j_data', (0,1), 2, rtol=1e-12),
378:         data(jv, 'bessel_j_data_ipp-bessel_j_data', (0,1j), 2, rtol=1e-12),
379: 
380:         data(kn, 'bessel_k_int_data_ipp-bessel_k_int_data', (0,1), 2, rtol=1e-12),
381: 
382:         data(kv, 'bessel_k_int_data_ipp-bessel_k_int_data', (0,1), 2, rtol=1e-12),
383:         data(kv, 'bessel_k_int_data_ipp-bessel_k_int_data', (0,1j), 2, rtol=1e-12),
384:         data(kv, 'bessel_k_data_ipp-bessel_k_data', (0,1), 2, rtol=1e-12),
385:         data(kv, 'bessel_k_data_ipp-bessel_k_data', (0,1j), 2, rtol=1e-12),
386: 
387:         data(yn, 'bessel_y01_data_ipp-bessel_y01_data', (0,1), 2, rtol=1e-12),
388:         data(yn, 'bessel_yn_data_ipp-bessel_yn_data', (0,1), 2, rtol=1e-12),
389: 
390:         data(yv, 'bessel_yn_data_ipp-bessel_yn_data', (0,1), 2, rtol=1e-12),
391:         data(yv, 'bessel_yn_data_ipp-bessel_yn_data', (0,1j), 2, rtol=1e-12),
392:         data(yv, 'bessel_yv_data_ipp-bessel_yv_data', (0,1), 2, rtol=1e-10),
393:         data(yv, 'bessel_yv_data_ipp-bessel_yv_data', (0,1j), 2, rtol=1e-10),
394: 
395:         data(zeta_, 'zeta_data_ipp-zeta_data', 0, 1, param_filter=(lambda s: s > 1)),
396:         data(zeta_, 'zeta_neg_data_ipp-zeta_neg_data', 0, 1, param_filter=(lambda s: s > 1)),
397:         data(zeta_, 'zeta_1_up_data_ipp-zeta_1_up_data', 0, 1, param_filter=(lambda s: s > 1)),
398:         data(zeta_, 'zeta_1_below_data_ipp-zeta_1_below_data', 0, 1, param_filter=(lambda s: s > 1)),
399: 
400:         data(gammaincinv, 'gamma_inv_small_data_ipp-gamma_inv_small_data', (0,1), 2, rtol=3e-11, knownfailure='gammaincinv bad few small points'),
401:         data(gammaincinv, 'gamma_inv_data_ipp-gamma_inv_data', (0,1), 2, rtol=1e-12),
402:         data(gammaincinv, 'gamma_inv_big_data_ipp-gamma_inv_big_data', (0,1), 2, rtol=1e-11),
403: 
404:         data(gammainccinv, 'gamma_inv_small_data_ipp-gamma_inv_small_data', (0,1), 3, rtol=2e-12),
405:         data(gammainccinv, 'gamma_inv_data_ipp-gamma_inv_data', (0,1), 3, rtol=2e-14),
406:         data(gammainccinv, 'gamma_inv_big_data_ipp-gamma_inv_big_data', (0,1), 3, rtol=3e-12),
407: 
408:         data(gdtrix_, 'gamma_inv_small_data_ipp-gamma_inv_small_data', (0,1), 2, rtol=3e-13, knownfailure='gdtrix unflow some points'),
409:         data(gdtrix_, 'gamma_inv_data_ipp-gamma_inv_data', (0,1), 2, rtol=3e-15),
410:         data(gdtrix_, 'gamma_inv_big_data_ipp-gamma_inv_big_data', (0,1), 2),
411:         data(gdtrix_comp, 'gamma_inv_small_data_ipp-gamma_inv_small_data', (0,1), 2, knownfailure='gdtrix bad some points'),
412:         data(gdtrix_comp, 'gamma_inv_data_ipp-gamma_inv_data', (0,1), 3, rtol=6e-15),
413:         data(gdtrix_comp, 'gamma_inv_big_data_ipp-gamma_inv_big_data', (0,1), 3),
414: 
415:         data(chndtr, 'nccs_ipp-nccs', (2,0,1), 3, rtol=3e-5),
416:         data(chndtr, 'nccs_big_ipp-nccs_big', (2,0,1), 3, rtol=5e-4, knownfailure='chndtr inaccurate some points'),
417: 
418:         data(sph_harm_, 'spherical_harmonic_ipp-spherical_harmonic', (1,0,3,2), (4,5), rtol=5e-11,
419:              param_filter=(lambda p: np.ones(p.shape, '?'),
420:                            lambda p: np.ones(p.shape, '?'),
421:                            lambda p: np.logical_and(p < 2*np.pi, p >= 0),
422:                            lambda p: np.logical_and(p < np.pi, p >= 0))),
423: 
424:         data(spherical_jn_, 'sph_bessel_data_ipp-sph_bessel_data', (0,1), 2, rtol=1e-13),
425:         data(spherical_yn_, 'sph_neumann_data_ipp-sph_neumann_data', (0,1), 2, rtol=8e-15),
426: 
427:         # -- not used yet (function does not exist in scipy):
428:         # 'ellint_pi2_data_ipp-ellint_pi2_data',
429:         # 'ellint_pi3_data_ipp-ellint_pi3_data',
430:         # 'ellint_pi3_large_data_ipp-ellint_pi3_large_data',
431:         # 'ellint_rc_data_ipp-ellint_rc_data',
432:         # 'ellint_rd_data_ipp-ellint_rd_data',
433:         # 'ellint_rf_data_ipp-ellint_rf_data',
434:         # 'ellint_rj_data_ipp-ellint_rj_data',
435:         # 'ncbeta_big_ipp-ncbeta_big',
436:         # 'ncbeta_ipp-ncbeta',
437:         # 'powm1_sqrtp1m1_test_cpp-powm1_data',
438:         # 'powm1_sqrtp1m1_test_cpp-sqrtp1m1_data',
439:         # 'test_gamma_data_ipp-gammap1m1_data',
440:         # 'tgamma_ratio_data_ipp-tgamma_ratio_data',
441: ]
442: 
443: 
444: @pytest.mark.parametrize('test', BOOST_TESTS, ids=repr)
445: def test_boost(test):
446:     _test_factory(test)
447: 
448: 
449: GSL_TESTS = [
450:         data_gsl(mathieu_a, 'mathieu_ab', (0, 1), 2, rtol=1e-13, atol=1e-13),
451:         data_gsl(mathieu_b, 'mathieu_ab', (0, 1), 3, rtol=1e-13, atol=1e-13),
452: 
453:         # Also the GSL output has limited accuracy...
454:         data_gsl(mathieu_ce_rad, 'mathieu_ce_se', (0, 1, 2), 3, rtol=1e-7, atol=1e-13),
455:         data_gsl(mathieu_se_rad, 'mathieu_ce_se', (0, 1, 2), 4, rtol=1e-7, atol=1e-13),
456: 
457:         data_gsl(mathieu_mc1_scaled, 'mathieu_mc_ms', (0, 1, 2), 3, rtol=1e-7, atol=1e-13),
458:         data_gsl(mathieu_ms1_scaled, 'mathieu_mc_ms', (0, 1, 2), 4, rtol=1e-7, atol=1e-13),
459: 
460:         data_gsl(mathieu_mc2_scaled, 'mathieu_mc_ms', (0, 1, 2), 5, rtol=1e-7, atol=1e-13),
461:         data_gsl(mathieu_ms2_scaled, 'mathieu_mc_ms', (0, 1, 2), 6, rtol=1e-7, atol=1e-13),
462: ]
463: 
464: 
465: @pytest.mark.parametrize('test', GSL_TESTS, ids=repr)
466: def test_gsl(test):
467:     _test_factory(test)
468: 
469: 
470: LOCAL_TESTS = [
471:     data_local(ellipkinc, 'ellipkinc_neg_m', (0, 1), 2),
472:     data_local(ellipkm1, 'ellipkm1', 0, 1),
473:     data_local(ellipeinc, 'ellipeinc_neg_m', (0, 1), 2),
474:     data_local(clog1p, 'log1p_expm1_complex', (0,1), (2,3), rtol=1e-14),
475:     data_local(cexpm1, 'log1p_expm1_complex', (0,1), (4,5), rtol=1e-14),
476:     data_local(gammainc, 'gammainc', (0, 1), 2, rtol=1e-12),
477:     data_local(gammaincc, 'gammaincc', (0, 1), 2, rtol=1e-11),
478:     data_local(ellip_harm_2, 'ellip',(0, 1, 2, 3, 4), 6, rtol=1e-10, atol=1e-13),
479:     data_local(ellip_harm, 'ellip',(0, 1, 2, 3, 4), 5, rtol=1e-10, atol=1e-13),
480: ]
481: 
482: 
483: @pytest.mark.parametrize('test', LOCAL_TESTS, ids=repr)
484: def test_local(test):
485:     _test_factory(test)
486: 
487: 
488: def _test_factory(test, dtype=np.double):
489:     '''Boost test'''
490:     with suppress_warnings() as sup:
491:         sup.filter(IntegrationWarning, "The occurrence of roundoff error is detected")
492:         olderr = np.seterr(all='ignore')
493:         try:
494:             test.check(dtype=dtype)
495:         finally:
496:             np.seterr(**olderr)
497: 
498: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_534489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_534489) is not StypyTypeError):

    if (import_534489 != 'pyd_module'):
        __import__(import_534489)
        sys_modules_534490 = sys.modules[import_534489]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_534490.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_534489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import arccosh, arcsinh, arctanh' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_534491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_534491) is not StypyTypeError):

    if (import_534491 != 'pyd_module'):
        __import__(import_534491)
        sys_modules_534492 = sys.modules[import_534491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_534492.module_type_store, module_type_store, ['arccosh', 'arcsinh', 'arctanh'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_534492, sys_modules_534492.module_type_store, module_type_store)
    else:
        from numpy import arccosh, arcsinh, arctanh

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['arccosh', 'arcsinh', 'arctanh'], [arccosh, arcsinh, arctanh])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_534491)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_534493 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat')

if (type(import_534493) is not StypyTypeError):

    if (import_534493 != 'pyd_module'):
        __import__(import_534493)
        sys_modules_534494 = sys.modules[import_534493]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', sys_modules_534494.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_534494, sys_modules_534494.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', import_534493)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import pytest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_534495 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_534495) is not StypyTypeError):

    if (import_534495 != 'pyd_module'):
        __import__(import_534495)
        sys_modules_534496 = sys.modules[import_534495]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_534496.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_534495)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.special import lpn, lpmn, lpmv, lqn, lqmn, sph_harm, eval_legendre, eval_hermite, eval_laguerre, eval_genlaguerre, binom, cbrt, expm1, log1p, zeta, jn, jv, yn, yv, iv, kv, kn, gamma, gammaln, gammainc, gammaincc, gammaincinv, gammainccinv, digamma, beta, betainc, betaincinv, poch, ellipe, ellipeinc, ellipk, ellipkm1, ellipkinc, ellipj, erf, erfc, erfinv, erfcinv, exp1, expi, expn, bdtrik, btdtr, btdtri, btdtria, btdtrib, chndtr, gdtr, gdtrc, gdtrix, gdtrib, nbdtrik, pdtrik, mathieu_a, mathieu_b, mathieu_cem, mathieu_sem, mathieu_modcem1, mathieu_modsem1, mathieu_modcem2, mathieu_modsem2, ellip_harm, ellip_harm_2, spherical_jn, spherical_yn' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_534497 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special')

if (type(import_534497) is not StypyTypeError):

    if (import_534497 != 'pyd_module'):
        __import__(import_534497)
        sys_modules_534498 = sys.modules[import_534497]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', sys_modules_534498.module_type_store, module_type_store, ['lpn', 'lpmn', 'lpmv', 'lqn', 'lqmn', 'sph_harm', 'eval_legendre', 'eval_hermite', 'eval_laguerre', 'eval_genlaguerre', 'binom', 'cbrt', 'expm1', 'log1p', 'zeta', 'jn', 'jv', 'yn', 'yv', 'iv', 'kv', 'kn', 'gamma', 'gammaln', 'gammainc', 'gammaincc', 'gammaincinv', 'gammainccinv', 'digamma', 'beta', 'betainc', 'betaincinv', 'poch', 'ellipe', 'ellipeinc', 'ellipk', 'ellipkm1', 'ellipkinc', 'ellipj', 'erf', 'erfc', 'erfinv', 'erfcinv', 'exp1', 'expi', 'expn', 'bdtrik', 'btdtr', 'btdtri', 'btdtria', 'btdtrib', 'chndtr', 'gdtr', 'gdtrc', 'gdtrix', 'gdtrib', 'nbdtrik', 'pdtrik', 'mathieu_a', 'mathieu_b', 'mathieu_cem', 'mathieu_sem', 'mathieu_modcem1', 'mathieu_modsem1', 'mathieu_modcem2', 'mathieu_modsem2', 'ellip_harm', 'ellip_harm_2', 'spherical_jn', 'spherical_yn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_534498, sys_modules_534498.module_type_store, module_type_store)
    else:
        from scipy.special import lpn, lpmn, lpmv, lqn, lqmn, sph_harm, eval_legendre, eval_hermite, eval_laguerre, eval_genlaguerre, binom, cbrt, expm1, log1p, zeta, jn, jv, yn, yv, iv, kv, kn, gamma, gammaln, gammainc, gammaincc, gammaincinv, gammainccinv, digamma, beta, betainc, betaincinv, poch, ellipe, ellipeinc, ellipk, ellipkm1, ellipkinc, ellipj, erf, erfc, erfinv, erfcinv, exp1, expi, expn, bdtrik, btdtr, btdtri, btdtria, btdtrib, chndtr, gdtr, gdtrc, gdtrix, gdtrib, nbdtrik, pdtrik, mathieu_a, mathieu_b, mathieu_cem, mathieu_sem, mathieu_modcem1, mathieu_modsem1, mathieu_modcem2, mathieu_modsem2, ellip_harm, ellip_harm_2, spherical_jn, spherical_yn

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', None, module_type_store, ['lpn', 'lpmn', 'lpmv', 'lqn', 'lqmn', 'sph_harm', 'eval_legendre', 'eval_hermite', 'eval_laguerre', 'eval_genlaguerre', 'binom', 'cbrt', 'expm1', 'log1p', 'zeta', 'jn', 'jv', 'yn', 'yv', 'iv', 'kv', 'kn', 'gamma', 'gammaln', 'gammainc', 'gammaincc', 'gammaincinv', 'gammainccinv', 'digamma', 'beta', 'betainc', 'betaincinv', 'poch', 'ellipe', 'ellipeinc', 'ellipk', 'ellipkm1', 'ellipkinc', 'ellipj', 'erf', 'erfc', 'erfinv', 'erfcinv', 'exp1', 'expi', 'expn', 'bdtrik', 'btdtr', 'btdtri', 'btdtria', 'btdtrib', 'chndtr', 'gdtr', 'gdtrc', 'gdtrix', 'gdtrib', 'nbdtrik', 'pdtrik', 'mathieu_a', 'mathieu_b', 'mathieu_cem', 'mathieu_sem', 'mathieu_modcem1', 'mathieu_modsem1', 'mathieu_modcem2', 'mathieu_modsem2', 'ellip_harm', 'ellip_harm_2', 'spherical_jn', 'spherical_yn'], [lpn, lpmn, lpmv, lqn, lqmn, sph_harm, eval_legendre, eval_hermite, eval_laguerre, eval_genlaguerre, binom, cbrt, expm1, log1p, zeta, jn, jv, yn, yv, iv, kv, kn, gamma, gammaln, gammainc, gammaincc, gammaincinv, gammainccinv, digamma, beta, betainc, betaincinv, poch, ellipe, ellipeinc, ellipk, ellipkm1, ellipkinc, ellipj, erf, erfc, erfinv, erfcinv, exp1, expi, expn, bdtrik, btdtr, btdtri, btdtria, btdtrib, chndtr, gdtr, gdtrc, gdtrix, gdtrib, nbdtrik, pdtrik, mathieu_a, mathieu_b, mathieu_cem, mathieu_sem, mathieu_modcem1, mathieu_modsem1, mathieu_modcem2, mathieu_modsem2, ellip_harm, ellip_harm_2, spherical_jn, spherical_yn])

else:
    # Assigning a type to the variable 'scipy.special' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', import_534497)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from scipy.integrate import IntegrationWarning' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_534499 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.integrate')

if (type(import_534499) is not StypyTypeError):

    if (import_534499 != 'pyd_module'):
        __import__(import_534499)
        sys_modules_534500 = sys.modules[import_534499]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.integrate', sys_modules_534500.module_type_store, module_type_store, ['IntegrationWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_534500, sys_modules_534500.module_type_store, module_type_store)
    else:
        from scipy.integrate import IntegrationWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.integrate', None, module_type_store, ['IntegrationWarning'], [IntegrationWarning])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'scipy.integrate', import_534499)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy.special._testutils import FuncData' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_534501 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.special._testutils')

if (type(import_534501) is not StypyTypeError):

    if (import_534501 != 'pyd_module'):
        __import__(import_534501)
        sys_modules_534502 = sys.modules[import_534501]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.special._testutils', sys_modules_534502.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_534502, sys_modules_534502.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy.special._testutils', import_534501)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


# Assigning a Call to a Name (line 28):

# Call to load(...): (line 28)
# Processing the call arguments (line 28)

# Call to join(...): (line 28)
# Processing the call arguments (line 28)

# Call to dirname(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of '__file__' (line 28)
file___534511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 54), '__file__', False)
# Processing the call keyword arguments (line 28)
kwargs_534512 = {}
# Getting the type of 'os' (line 28)
os_534508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'os', False)
# Obtaining the member 'path' of a type (line 28)
path_534509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 38), os_534508, 'path')
# Obtaining the member 'dirname' of a type (line 28)
dirname_534510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 38), path_534509, 'dirname')
# Calling dirname(args, kwargs) (line 28)
dirname_call_result_534513 = invoke(stypy.reporting.localization.Localization(__file__, 28, 38), dirname_534510, *[file___534511], **kwargs_534512)

str_534514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 38), 'str', 'data')
str_534515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 46), 'str', 'boost.npz')
# Processing the call keyword arguments (line 28)
kwargs_534516 = {}
# Getting the type of 'os' (line 28)
os_534505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'os', False)
# Obtaining the member 'path' of a type (line 28)
path_534506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), os_534505, 'path')
# Obtaining the member 'join' of a type (line 28)
join_534507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), path_534506, 'join')
# Calling join(args, kwargs) (line 28)
join_call_result_534517 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), join_534507, *[dirname_call_result_534513, str_534514, str_534515], **kwargs_534516)

# Processing the call keyword arguments (line 28)
kwargs_534518 = {}
# Getting the type of 'np' (line 28)
np_534503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'np', False)
# Obtaining the member 'load' of a type (line 28)
load_534504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 17), np_534503, 'load')
# Calling load(args, kwargs) (line 28)
load_call_result_534519 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), load_534504, *[join_call_result_534517], **kwargs_534518)

# Assigning a type to the variable 'DATASETS_BOOST' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'DATASETS_BOOST', load_call_result_534519)

# Assigning a Call to a Name (line 31):

# Call to load(...): (line 31)
# Processing the call arguments (line 31)

# Call to join(...): (line 31)
# Processing the call arguments (line 31)

# Call to dirname(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of '__file__' (line 31)
file___534528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 52), '__file__', False)
# Processing the call keyword arguments (line 31)
kwargs_534529 = {}
# Getting the type of 'os' (line 31)
os_534525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 36), 'os', False)
# Obtaining the member 'path' of a type (line 31)
path_534526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 36), os_534525, 'path')
# Obtaining the member 'dirname' of a type (line 31)
dirname_534527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 36), path_534526, 'dirname')
# Calling dirname(args, kwargs) (line 31)
dirname_call_result_534530 = invoke(stypy.reporting.localization.Localization(__file__, 31, 36), dirname_534527, *[file___534528], **kwargs_534529)

str_534531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'str', 'data')
str_534532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 44), 'str', 'gsl.npz')
# Processing the call keyword arguments (line 31)
kwargs_534533 = {}
# Getting the type of 'os' (line 31)
os_534522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'os', False)
# Obtaining the member 'path' of a type (line 31)
path_534523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 23), os_534522, 'path')
# Obtaining the member 'join' of a type (line 31)
join_534524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 23), path_534523, 'join')
# Calling join(args, kwargs) (line 31)
join_call_result_534534 = invoke(stypy.reporting.localization.Localization(__file__, 31, 23), join_534524, *[dirname_call_result_534530, str_534531, str_534532], **kwargs_534533)

# Processing the call keyword arguments (line 31)
kwargs_534535 = {}
# Getting the type of 'np' (line 31)
np_534520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'np', False)
# Obtaining the member 'load' of a type (line 31)
load_534521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), np_534520, 'load')
# Calling load(args, kwargs) (line 31)
load_call_result_534536 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), load_534521, *[join_call_result_534534], **kwargs_534535)

# Assigning a type to the variable 'DATASETS_GSL' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'DATASETS_GSL', load_call_result_534536)

# Assigning a Call to a Name (line 34):

# Call to load(...): (line 34)
# Processing the call arguments (line 34)

# Call to join(...): (line 34)
# Processing the call arguments (line 34)

# Call to dirname(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of '__file__' (line 34)
file___534545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 54), '__file__', False)
# Processing the call keyword arguments (line 34)
kwargs_534546 = {}
# Getting the type of 'os' (line 34)
os_534542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'os', False)
# Obtaining the member 'path' of a type (line 34)
path_534543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 38), os_534542, 'path')
# Obtaining the member 'dirname' of a type (line 34)
dirname_534544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 38), path_534543, 'dirname')
# Calling dirname(args, kwargs) (line 34)
dirname_call_result_534547 = invoke(stypy.reporting.localization.Localization(__file__, 34, 38), dirname_534544, *[file___534545], **kwargs_534546)

str_534548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'str', 'data')
str_534549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 44), 'str', 'local.npz')
# Processing the call keyword arguments (line 34)
kwargs_534550 = {}
# Getting the type of 'os' (line 34)
os_534539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'os', False)
# Obtaining the member 'path' of a type (line 34)
path_534540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 25), os_534539, 'path')
# Obtaining the member 'join' of a type (line 34)
join_534541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 25), path_534540, 'join')
# Calling join(args, kwargs) (line 34)
join_call_result_534551 = invoke(stypy.reporting.localization.Localization(__file__, 34, 25), join_534541, *[dirname_call_result_534547, str_534548, str_534549], **kwargs_534550)

# Processing the call keyword arguments (line 34)
kwargs_534552 = {}
# Getting the type of 'np' (line 34)
np_534537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'np', False)
# Obtaining the member 'load' of a type (line 34)
load_534538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), np_534537, 'load')
# Calling load(args, kwargs) (line 34)
load_call_result_534553 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), load_534538, *[join_call_result_534551], **kwargs_534552)

# Assigning a type to the variable 'DATASETS_LOCAL' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'DATASETS_LOCAL', load_call_result_534553)

@norecursion
def data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'data'
    module_type_store = module_type_store.open_function_context('data', 38, 0, False)
    
    # Passed parameters checking function
    data.stypy_localization = localization
    data.stypy_type_of_self = None
    data.stypy_type_store = module_type_store
    data.stypy_function_name = 'data'
    data.stypy_param_names_list = ['func', 'dataname']
    data.stypy_varargs_param_name = 'a'
    data.stypy_kwargs_param_name = 'kw'
    data.stypy_call_defaults = defaults
    data.stypy_call_varargs = varargs
    data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'data', ['func', 'dataname'], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'data', localization, ['func', 'dataname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'data(...)' code ##################

    
    # Call to setdefault(...): (line 39)
    # Processing the call arguments (line 39)
    str_534556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'str', 'dataname')
    # Getting the type of 'dataname' (line 39)
    dataname_534557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 30), 'dataname', False)
    # Processing the call keyword arguments (line 39)
    kwargs_534558 = {}
    # Getting the type of 'kw' (line 39)
    kw_534554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'kw', False)
    # Obtaining the member 'setdefault' of a type (line 39)
    setdefault_534555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 4), kw_534554, 'setdefault')
    # Calling setdefault(args, kwargs) (line 39)
    setdefault_call_result_534559 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), setdefault_534555, *[str_534556, dataname_534557], **kwargs_534558)
    
    
    # Call to FuncData(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'func' (line 40)
    func_534561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'func', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'dataname' (line 40)
    dataname_534562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 'dataname', False)
    # Getting the type of 'DATASETS_BOOST' (line 40)
    DATASETS_BOOST_534563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'DATASETS_BOOST', False)
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___534564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 26), DATASETS_BOOST_534563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_534565 = invoke(stypy.reporting.localization.Localization(__file__, 40, 26), getitem___534564, dataname_534562)
    
    # Getting the type of 'a' (line 40)
    a_534566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 53), 'a', False)
    # Processing the call keyword arguments (line 40)
    # Getting the type of 'kw' (line 40)
    kw_534567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 58), 'kw', False)
    kwargs_534568 = {'kw_534567': kw_534567}
    # Getting the type of 'FuncData' (line 40)
    FuncData_534560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 40)
    FuncData_call_result_534569 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), FuncData_534560, *[func_534561, subscript_call_result_534565, a_534566], **kwargs_534568)
    
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type', FuncData_call_result_534569)
    
    # ################# End of 'data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'data' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_534570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534570)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'data'
    return stypy_return_type_534570

# Assigning a type to the variable 'data' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'data', data)

@norecursion
def data_gsl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'data_gsl'
    module_type_store = module_type_store.open_function_context('data_gsl', 43, 0, False)
    
    # Passed parameters checking function
    data_gsl.stypy_localization = localization
    data_gsl.stypy_type_of_self = None
    data_gsl.stypy_type_store = module_type_store
    data_gsl.stypy_function_name = 'data_gsl'
    data_gsl.stypy_param_names_list = ['func', 'dataname']
    data_gsl.stypy_varargs_param_name = 'a'
    data_gsl.stypy_kwargs_param_name = 'kw'
    data_gsl.stypy_call_defaults = defaults
    data_gsl.stypy_call_varargs = varargs
    data_gsl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'data_gsl', ['func', 'dataname'], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'data_gsl', localization, ['func', 'dataname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'data_gsl(...)' code ##################

    
    # Call to setdefault(...): (line 44)
    # Processing the call arguments (line 44)
    str_534573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 18), 'str', 'dataname')
    # Getting the type of 'dataname' (line 44)
    dataname_534574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'dataname', False)
    # Processing the call keyword arguments (line 44)
    kwargs_534575 = {}
    # Getting the type of 'kw' (line 44)
    kw_534571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'kw', False)
    # Obtaining the member 'setdefault' of a type (line 44)
    setdefault_534572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), kw_534571, 'setdefault')
    # Calling setdefault(args, kwargs) (line 44)
    setdefault_call_result_534576 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), setdefault_534572, *[str_534573, dataname_534574], **kwargs_534575)
    
    
    # Call to FuncData(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'func' (line 45)
    func_534578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'func', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'dataname' (line 45)
    dataname_534579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'dataname', False)
    # Getting the type of 'DATASETS_GSL' (line 45)
    DATASETS_GSL_534580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'DATASETS_GSL', False)
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___534581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 26), DATASETS_GSL_534580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_534582 = invoke(stypy.reporting.localization.Localization(__file__, 45, 26), getitem___534581, dataname_534579)
    
    # Getting the type of 'a' (line 45)
    a_534583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 51), 'a', False)
    # Processing the call keyword arguments (line 45)
    # Getting the type of 'kw' (line 45)
    kw_534584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 56), 'kw', False)
    kwargs_534585 = {'kw_534584': kw_534584}
    # Getting the type of 'FuncData' (line 45)
    FuncData_534577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 45)
    FuncData_call_result_534586 = invoke(stypy.reporting.localization.Localization(__file__, 45, 11), FuncData_534577, *[func_534578, subscript_call_result_534582, a_534583], **kwargs_534585)
    
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type', FuncData_call_result_534586)
    
    # ################# End of 'data_gsl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'data_gsl' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_534587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534587)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'data_gsl'
    return stypy_return_type_534587

# Assigning a type to the variable 'data_gsl' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'data_gsl', data_gsl)

@norecursion
def data_local(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'data_local'
    module_type_store = module_type_store.open_function_context('data_local', 48, 0, False)
    
    # Passed parameters checking function
    data_local.stypy_localization = localization
    data_local.stypy_type_of_self = None
    data_local.stypy_type_store = module_type_store
    data_local.stypy_function_name = 'data_local'
    data_local.stypy_param_names_list = ['func', 'dataname']
    data_local.stypy_varargs_param_name = 'a'
    data_local.stypy_kwargs_param_name = 'kw'
    data_local.stypy_call_defaults = defaults
    data_local.stypy_call_varargs = varargs
    data_local.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'data_local', ['func', 'dataname'], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'data_local', localization, ['func', 'dataname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'data_local(...)' code ##################

    
    # Call to setdefault(...): (line 49)
    # Processing the call arguments (line 49)
    str_534590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'str', 'dataname')
    # Getting the type of 'dataname' (line 49)
    dataname_534591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'dataname', False)
    # Processing the call keyword arguments (line 49)
    kwargs_534592 = {}
    # Getting the type of 'kw' (line 49)
    kw_534588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'kw', False)
    # Obtaining the member 'setdefault' of a type (line 49)
    setdefault_534589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), kw_534588, 'setdefault')
    # Calling setdefault(args, kwargs) (line 49)
    setdefault_call_result_534593 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), setdefault_534589, *[str_534590, dataname_534591], **kwargs_534592)
    
    
    # Call to FuncData(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'func' (line 50)
    func_534595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'func', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'dataname' (line 50)
    dataname_534596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), 'dataname', False)
    # Getting the type of 'DATASETS_LOCAL' (line 50)
    DATASETS_LOCAL_534597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'DATASETS_LOCAL', False)
    # Obtaining the member '__getitem__' of a type (line 50)
    getitem___534598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 26), DATASETS_LOCAL_534597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 50)
    subscript_call_result_534599 = invoke(stypy.reporting.localization.Localization(__file__, 50, 26), getitem___534598, dataname_534596)
    
    # Getting the type of 'a' (line 50)
    a_534600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 53), 'a', False)
    # Processing the call keyword arguments (line 50)
    # Getting the type of 'kw' (line 50)
    kw_534601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 58), 'kw', False)
    kwargs_534602 = {'kw_534601': kw_534601}
    # Getting the type of 'FuncData' (line 50)
    FuncData_534594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 50)
    FuncData_call_result_534603 = invoke(stypy.reporting.localization.Localization(__file__, 50, 11), FuncData_534594, *[func_534595, subscript_call_result_534599, a_534600], **kwargs_534602)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', FuncData_call_result_534603)
    
    # ################# End of 'data_local(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'data_local' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_534604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534604)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'data_local'
    return stypy_return_type_534604

# Assigning a type to the variable 'data_local' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'data_local', data_local)

@norecursion
def ellipk_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ellipk_'
    module_type_store = module_type_store.open_function_context('ellipk_', 53, 0, False)
    
    # Passed parameters checking function
    ellipk_.stypy_localization = localization
    ellipk_.stypy_type_of_self = None
    ellipk_.stypy_type_store = module_type_store
    ellipk_.stypy_function_name = 'ellipk_'
    ellipk_.stypy_param_names_list = ['k']
    ellipk_.stypy_varargs_param_name = None
    ellipk_.stypy_kwargs_param_name = None
    ellipk_.stypy_call_defaults = defaults
    ellipk_.stypy_call_varargs = varargs
    ellipk_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ellipk_', ['k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ellipk_', localization, ['k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ellipk_(...)' code ##################

    
    # Call to ellipk(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'k' (line 54)
    k_534606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'k', False)
    # Getting the type of 'k' (line 54)
    k_534607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'k', False)
    # Applying the binary operator '*' (line 54)
    result_mul_534608 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 18), '*', k_534606, k_534607)
    
    # Processing the call keyword arguments (line 54)
    kwargs_534609 = {}
    # Getting the type of 'ellipk' (line 54)
    ellipk_534605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'ellipk', False)
    # Calling ellipk(args, kwargs) (line 54)
    ellipk_call_result_534610 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), ellipk_534605, *[result_mul_534608], **kwargs_534609)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', ellipk_call_result_534610)
    
    # ################# End of 'ellipk_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ellipk_' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_534611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534611)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ellipk_'
    return stypy_return_type_534611

# Assigning a type to the variable 'ellipk_' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'ellipk_', ellipk_)

@norecursion
def ellipkinc_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ellipkinc_'
    module_type_store = module_type_store.open_function_context('ellipkinc_', 57, 0, False)
    
    # Passed parameters checking function
    ellipkinc_.stypy_localization = localization
    ellipkinc_.stypy_type_of_self = None
    ellipkinc_.stypy_type_store = module_type_store
    ellipkinc_.stypy_function_name = 'ellipkinc_'
    ellipkinc_.stypy_param_names_list = ['f', 'k']
    ellipkinc_.stypy_varargs_param_name = None
    ellipkinc_.stypy_kwargs_param_name = None
    ellipkinc_.stypy_call_defaults = defaults
    ellipkinc_.stypy_call_varargs = varargs
    ellipkinc_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ellipkinc_', ['f', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ellipkinc_', localization, ['f', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ellipkinc_(...)' code ##################

    
    # Call to ellipkinc(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'f' (line 58)
    f_534613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'f', False)
    # Getting the type of 'k' (line 58)
    k_534614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'k', False)
    # Getting the type of 'k' (line 58)
    k_534615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'k', False)
    # Applying the binary operator '*' (line 58)
    result_mul_534616 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 24), '*', k_534614, k_534615)
    
    # Processing the call keyword arguments (line 58)
    kwargs_534617 = {}
    # Getting the type of 'ellipkinc' (line 58)
    ellipkinc_534612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'ellipkinc', False)
    # Calling ellipkinc(args, kwargs) (line 58)
    ellipkinc_call_result_534618 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), ellipkinc_534612, *[f_534613, result_mul_534616], **kwargs_534617)
    
    # Assigning a type to the variable 'stypy_return_type' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type', ellipkinc_call_result_534618)
    
    # ################# End of 'ellipkinc_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ellipkinc_' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_534619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534619)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ellipkinc_'
    return stypy_return_type_534619

# Assigning a type to the variable 'ellipkinc_' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'ellipkinc_', ellipkinc_)

@norecursion
def ellipe_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ellipe_'
    module_type_store = module_type_store.open_function_context('ellipe_', 61, 0, False)
    
    # Passed parameters checking function
    ellipe_.stypy_localization = localization
    ellipe_.stypy_type_of_self = None
    ellipe_.stypy_type_store = module_type_store
    ellipe_.stypy_function_name = 'ellipe_'
    ellipe_.stypy_param_names_list = ['k']
    ellipe_.stypy_varargs_param_name = None
    ellipe_.stypy_kwargs_param_name = None
    ellipe_.stypy_call_defaults = defaults
    ellipe_.stypy_call_varargs = varargs
    ellipe_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ellipe_', ['k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ellipe_', localization, ['k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ellipe_(...)' code ##################

    
    # Call to ellipe(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'k' (line 62)
    k_534621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'k', False)
    # Getting the type of 'k' (line 62)
    k_534622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'k', False)
    # Applying the binary operator '*' (line 62)
    result_mul_534623 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 18), '*', k_534621, k_534622)
    
    # Processing the call keyword arguments (line 62)
    kwargs_534624 = {}
    # Getting the type of 'ellipe' (line 62)
    ellipe_534620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'ellipe', False)
    # Calling ellipe(args, kwargs) (line 62)
    ellipe_call_result_534625 = invoke(stypy.reporting.localization.Localization(__file__, 62, 11), ellipe_534620, *[result_mul_534623], **kwargs_534624)
    
    # Assigning a type to the variable 'stypy_return_type' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type', ellipe_call_result_534625)
    
    # ################# End of 'ellipe_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ellipe_' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_534626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534626)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ellipe_'
    return stypy_return_type_534626

# Assigning a type to the variable 'ellipe_' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'ellipe_', ellipe_)

@norecursion
def ellipeinc_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ellipeinc_'
    module_type_store = module_type_store.open_function_context('ellipeinc_', 65, 0, False)
    
    # Passed parameters checking function
    ellipeinc_.stypy_localization = localization
    ellipeinc_.stypy_type_of_self = None
    ellipeinc_.stypy_type_store = module_type_store
    ellipeinc_.stypy_function_name = 'ellipeinc_'
    ellipeinc_.stypy_param_names_list = ['f', 'k']
    ellipeinc_.stypy_varargs_param_name = None
    ellipeinc_.stypy_kwargs_param_name = None
    ellipeinc_.stypy_call_defaults = defaults
    ellipeinc_.stypy_call_varargs = varargs
    ellipeinc_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ellipeinc_', ['f', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ellipeinc_', localization, ['f', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ellipeinc_(...)' code ##################

    
    # Call to ellipeinc(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'f' (line 66)
    f_534628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'f', False)
    # Getting the type of 'k' (line 66)
    k_534629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'k', False)
    # Getting the type of 'k' (line 66)
    k_534630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 26), 'k', False)
    # Applying the binary operator '*' (line 66)
    result_mul_534631 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 24), '*', k_534629, k_534630)
    
    # Processing the call keyword arguments (line 66)
    kwargs_534632 = {}
    # Getting the type of 'ellipeinc' (line 66)
    ellipeinc_534627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'ellipeinc', False)
    # Calling ellipeinc(args, kwargs) (line 66)
    ellipeinc_call_result_534633 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), ellipeinc_534627, *[f_534628, result_mul_534631], **kwargs_534632)
    
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', ellipeinc_call_result_534633)
    
    # ################# End of 'ellipeinc_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ellipeinc_' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_534634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534634)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ellipeinc_'
    return stypy_return_type_534634

# Assigning a type to the variable 'ellipeinc_' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'ellipeinc_', ellipeinc_)

@norecursion
def ellipj_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ellipj_'
    module_type_store = module_type_store.open_function_context('ellipj_', 69, 0, False)
    
    # Passed parameters checking function
    ellipj_.stypy_localization = localization
    ellipj_.stypy_type_of_self = None
    ellipj_.stypy_type_store = module_type_store
    ellipj_.stypy_function_name = 'ellipj_'
    ellipj_.stypy_param_names_list = ['k']
    ellipj_.stypy_varargs_param_name = None
    ellipj_.stypy_kwargs_param_name = None
    ellipj_.stypy_call_defaults = defaults
    ellipj_.stypy_call_varargs = varargs
    ellipj_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ellipj_', ['k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ellipj_', localization, ['k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ellipj_(...)' code ##################

    
    # Call to ellipj(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'k' (line 70)
    k_534636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'k', False)
    # Getting the type of 'k' (line 70)
    k_534637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'k', False)
    # Applying the binary operator '*' (line 70)
    result_mul_534638 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 18), '*', k_534636, k_534637)
    
    # Processing the call keyword arguments (line 70)
    kwargs_534639 = {}
    # Getting the type of 'ellipj' (line 70)
    ellipj_534635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'ellipj', False)
    # Calling ellipj(args, kwargs) (line 70)
    ellipj_call_result_534640 = invoke(stypy.reporting.localization.Localization(__file__, 70, 11), ellipj_534635, *[result_mul_534638], **kwargs_534639)
    
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', ellipj_call_result_534640)
    
    # ################# End of 'ellipj_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ellipj_' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_534641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534641)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ellipj_'
    return stypy_return_type_534641

# Assigning a type to the variable 'ellipj_' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'ellipj_', ellipj_)

@norecursion
def zeta_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'zeta_'
    module_type_store = module_type_store.open_function_context('zeta_', 73, 0, False)
    
    # Passed parameters checking function
    zeta_.stypy_localization = localization
    zeta_.stypy_type_of_self = None
    zeta_.stypy_type_store = module_type_store
    zeta_.stypy_function_name = 'zeta_'
    zeta_.stypy_param_names_list = ['x']
    zeta_.stypy_varargs_param_name = None
    zeta_.stypy_kwargs_param_name = None
    zeta_.stypy_call_defaults = defaults
    zeta_.stypy_call_varargs = varargs
    zeta_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'zeta_', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'zeta_', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'zeta_(...)' code ##################

    
    # Call to zeta(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'x' (line 74)
    x_534643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'x', False)
    float_534644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'float')
    # Processing the call keyword arguments (line 74)
    kwargs_534645 = {}
    # Getting the type of 'zeta' (line 74)
    zeta_534642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'zeta', False)
    # Calling zeta(args, kwargs) (line 74)
    zeta_call_result_534646 = invoke(stypy.reporting.localization.Localization(__file__, 74, 11), zeta_534642, *[x_534643, float_534644], **kwargs_534645)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', zeta_call_result_534646)
    
    # ################# End of 'zeta_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'zeta_' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_534647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534647)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'zeta_'
    return stypy_return_type_534647

# Assigning a type to the variable 'zeta_' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'zeta_', zeta_)

@norecursion
def assoc_legendre_p_boost_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assoc_legendre_p_boost_'
    module_type_store = module_type_store.open_function_context('assoc_legendre_p_boost_', 77, 0, False)
    
    # Passed parameters checking function
    assoc_legendre_p_boost_.stypy_localization = localization
    assoc_legendre_p_boost_.stypy_type_of_self = None
    assoc_legendre_p_boost_.stypy_type_store = module_type_store
    assoc_legendre_p_boost_.stypy_function_name = 'assoc_legendre_p_boost_'
    assoc_legendre_p_boost_.stypy_param_names_list = ['nu', 'mu', 'x']
    assoc_legendre_p_boost_.stypy_varargs_param_name = None
    assoc_legendre_p_boost_.stypy_kwargs_param_name = None
    assoc_legendre_p_boost_.stypy_call_defaults = defaults
    assoc_legendre_p_boost_.stypy_call_varargs = varargs
    assoc_legendre_p_boost_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assoc_legendre_p_boost_', ['nu', 'mu', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assoc_legendre_p_boost_', localization, ['nu', 'mu', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assoc_legendre_p_boost_(...)' code ##################

    
    # Call to lpmv(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'mu' (line 79)
    mu_534649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'mu', False)
    
    # Call to astype(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'int' (line 79)
    int_534652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'int', False)
    # Processing the call keyword arguments (line 79)
    kwargs_534653 = {}
    # Getting the type of 'nu' (line 79)
    nu_534650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'nu', False)
    # Obtaining the member 'astype' of a type (line 79)
    astype_534651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), nu_534650, 'astype')
    # Calling astype(args, kwargs) (line 79)
    astype_call_result_534654 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), astype_534651, *[int_534652], **kwargs_534653)
    
    # Getting the type of 'x' (line 79)
    x_534655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 36), 'x', False)
    # Processing the call keyword arguments (line 79)
    kwargs_534656 = {}
    # Getting the type of 'lpmv' (line 79)
    lpmv_534648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'lpmv', False)
    # Calling lpmv(args, kwargs) (line 79)
    lpmv_call_result_534657 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), lpmv_534648, *[mu_534649, astype_call_result_534654, x_534655], **kwargs_534656)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', lpmv_call_result_534657)
    
    # ################# End of 'assoc_legendre_p_boost_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assoc_legendre_p_boost_' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_534658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534658)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assoc_legendre_p_boost_'
    return stypy_return_type_534658

# Assigning a type to the variable 'assoc_legendre_p_boost_' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'assoc_legendre_p_boost_', assoc_legendre_p_boost_)

@norecursion
def legendre_p_via_assoc_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legendre_p_via_assoc_'
    module_type_store = module_type_store.open_function_context('legendre_p_via_assoc_', 81, 0, False)
    
    # Passed parameters checking function
    legendre_p_via_assoc_.stypy_localization = localization
    legendre_p_via_assoc_.stypy_type_of_self = None
    legendre_p_via_assoc_.stypy_type_store = module_type_store
    legendre_p_via_assoc_.stypy_function_name = 'legendre_p_via_assoc_'
    legendre_p_via_assoc_.stypy_param_names_list = ['nu', 'x']
    legendre_p_via_assoc_.stypy_varargs_param_name = None
    legendre_p_via_assoc_.stypy_kwargs_param_name = None
    legendre_p_via_assoc_.stypy_call_defaults = defaults
    legendre_p_via_assoc_.stypy_call_varargs = varargs
    legendre_p_via_assoc_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legendre_p_via_assoc_', ['nu', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legendre_p_via_assoc_', localization, ['nu', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legendre_p_via_assoc_(...)' code ##################

    
    # Call to lpmv(...): (line 82)
    # Processing the call arguments (line 82)
    int_534660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'int')
    # Getting the type of 'nu' (line 82)
    nu_534661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'nu', False)
    # Getting the type of 'x' (line 82)
    x_534662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'x', False)
    # Processing the call keyword arguments (line 82)
    kwargs_534663 = {}
    # Getting the type of 'lpmv' (line 82)
    lpmv_534659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'lpmv', False)
    # Calling lpmv(args, kwargs) (line 82)
    lpmv_call_result_534664 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), lpmv_534659, *[int_534660, nu_534661, x_534662], **kwargs_534663)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type', lpmv_call_result_534664)
    
    # ################# End of 'legendre_p_via_assoc_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legendre_p_via_assoc_' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_534665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534665)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legendre_p_via_assoc_'
    return stypy_return_type_534665

# Assigning a type to the variable 'legendre_p_via_assoc_' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'legendre_p_via_assoc_', legendre_p_via_assoc_)

@norecursion
def lpn_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lpn_'
    module_type_store = module_type_store.open_function_context('lpn_', 84, 0, False)
    
    # Passed parameters checking function
    lpn_.stypy_localization = localization
    lpn_.stypy_type_of_self = None
    lpn_.stypy_type_store = module_type_store
    lpn_.stypy_function_name = 'lpn_'
    lpn_.stypy_param_names_list = ['n', 'x']
    lpn_.stypy_varargs_param_name = None
    lpn_.stypy_kwargs_param_name = None
    lpn_.stypy_call_defaults = defaults
    lpn_.stypy_call_varargs = varargs
    lpn_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lpn_', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lpn_', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lpn_(...)' code ##################

    
    # Obtaining the type of the subscript
    int_534666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 36), 'int')
    
    # Obtaining the type of the subscript
    int_534667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 33), 'int')
    
    # Call to lpn(...): (line 85)
    # Processing the call arguments (line 85)
    
    # Call to astype(...): (line 85)
    # Processing the call arguments (line 85)
    str_534671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 24), 'str', 'l')
    # Processing the call keyword arguments (line 85)
    kwargs_534672 = {}
    # Getting the type of 'n' (line 85)
    n_534669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'n', False)
    # Obtaining the member 'astype' of a type (line 85)
    astype_534670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), n_534669, 'astype')
    # Calling astype(args, kwargs) (line 85)
    astype_call_result_534673 = invoke(stypy.reporting.localization.Localization(__file__, 85, 15), astype_534670, *[str_534671], **kwargs_534672)
    
    # Getting the type of 'x' (line 85)
    x_534674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), 'x', False)
    # Processing the call keyword arguments (line 85)
    kwargs_534675 = {}
    # Getting the type of 'lpn' (line 85)
    lpn_534668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'lpn', False)
    # Calling lpn(args, kwargs) (line 85)
    lpn_call_result_534676 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), lpn_534668, *[astype_call_result_534673, x_534674], **kwargs_534675)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___534677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 11), lpn_call_result_534676, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_534678 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), getitem___534677, int_534667)
    
    # Obtaining the member '__getitem__' of a type (line 85)
    getitem___534679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 11), subscript_call_result_534678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 85)
    subscript_call_result_534680 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), getitem___534679, int_534666)
    
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type', subscript_call_result_534680)
    
    # ################# End of 'lpn_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lpn_' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_534681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534681)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lpn_'
    return stypy_return_type_534681

# Assigning a type to the variable 'lpn_' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'lpn_', lpn_)

@norecursion
def lqn_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lqn_'
    module_type_store = module_type_store.open_function_context('lqn_', 87, 0, False)
    
    # Passed parameters checking function
    lqn_.stypy_localization = localization
    lqn_.stypy_type_of_self = None
    lqn_.stypy_type_store = module_type_store
    lqn_.stypy_function_name = 'lqn_'
    lqn_.stypy_param_names_list = ['n', 'x']
    lqn_.stypy_varargs_param_name = None
    lqn_.stypy_kwargs_param_name = None
    lqn_.stypy_call_defaults = defaults
    lqn_.stypy_call_varargs = varargs
    lqn_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lqn_', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lqn_', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lqn_(...)' code ##################

    
    # Obtaining the type of the subscript
    int_534682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 36), 'int')
    
    # Obtaining the type of the subscript
    int_534683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 33), 'int')
    
    # Call to lqn(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Call to astype(...): (line 88)
    # Processing the call arguments (line 88)
    str_534687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'str', 'l')
    # Processing the call keyword arguments (line 88)
    kwargs_534688 = {}
    # Getting the type of 'n' (line 88)
    n_534685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'n', False)
    # Obtaining the member 'astype' of a type (line 88)
    astype_534686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), n_534685, 'astype')
    # Calling astype(args, kwargs) (line 88)
    astype_call_result_534689 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), astype_534686, *[str_534687], **kwargs_534688)
    
    # Getting the type of 'x' (line 88)
    x_534690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'x', False)
    # Processing the call keyword arguments (line 88)
    kwargs_534691 = {}
    # Getting the type of 'lqn' (line 88)
    lqn_534684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'lqn', False)
    # Calling lqn(args, kwargs) (line 88)
    lqn_call_result_534692 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), lqn_534684, *[astype_call_result_534689, x_534690], **kwargs_534691)
    
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___534693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 11), lqn_call_result_534692, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_534694 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), getitem___534693, int_534683)
    
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___534695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 11), subscript_call_result_534694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_534696 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), getitem___534695, int_534682)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', subscript_call_result_534696)
    
    # ################# End of 'lqn_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lqn_' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_534697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lqn_'
    return stypy_return_type_534697

# Assigning a type to the variable 'lqn_' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'lqn_', lqn_)

@norecursion
def legendre_p_via_lpmn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legendre_p_via_lpmn'
    module_type_store = module_type_store.open_function_context('legendre_p_via_lpmn', 90, 0, False)
    
    # Passed parameters checking function
    legendre_p_via_lpmn.stypy_localization = localization
    legendre_p_via_lpmn.stypy_type_of_self = None
    legendre_p_via_lpmn.stypy_type_store = module_type_store
    legendre_p_via_lpmn.stypy_function_name = 'legendre_p_via_lpmn'
    legendre_p_via_lpmn.stypy_param_names_list = ['n', 'x']
    legendre_p_via_lpmn.stypy_varargs_param_name = None
    legendre_p_via_lpmn.stypy_kwargs_param_name = None
    legendre_p_via_lpmn.stypy_call_defaults = defaults
    legendre_p_via_lpmn.stypy_call_varargs = varargs
    legendre_p_via_lpmn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legendre_p_via_lpmn', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legendre_p_via_lpmn', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legendre_p_via_lpmn(...)' code ##################

    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 91)
    tuple_534698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 91)
    # Adding element type (line 91)
    int_534699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 28), tuple_534698, int_534699)
    # Adding element type (line 91)
    int_534700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 28), tuple_534698, int_534700)
    
    
    # Obtaining the type of the subscript
    int_534701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'int')
    
    # Call to lpmn(...): (line 91)
    # Processing the call arguments (line 91)
    int_534703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 16), 'int')
    # Getting the type of 'n' (line 91)
    n_534704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'n', False)
    # Getting the type of 'x' (line 91)
    x_534705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'x', False)
    # Processing the call keyword arguments (line 91)
    kwargs_534706 = {}
    # Getting the type of 'lpmn' (line 91)
    lpmn_534702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'lpmn', False)
    # Calling lpmn(args, kwargs) (line 91)
    lpmn_call_result_534707 = invoke(stypy.reporting.localization.Localization(__file__, 91, 11), lpmn_534702, *[int_534703, n_534704, x_534705], **kwargs_534706)
    
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___534708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 11), lpmn_call_result_534707, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_534709 = invoke(stypy.reporting.localization.Localization(__file__, 91, 11), getitem___534708, int_534701)
    
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___534710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 11), subscript_call_result_534709, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_534711 = invoke(stypy.reporting.localization.Localization(__file__, 91, 11), getitem___534710, tuple_534698)
    
    # Assigning a type to the variable 'stypy_return_type' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type', subscript_call_result_534711)
    
    # ################# End of 'legendre_p_via_lpmn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legendre_p_via_lpmn' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_534712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534712)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legendre_p_via_lpmn'
    return stypy_return_type_534712

# Assigning a type to the variable 'legendre_p_via_lpmn' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'legendre_p_via_lpmn', legendre_p_via_lpmn)

@norecursion
def legendre_q_via_lqmn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'legendre_q_via_lqmn'
    module_type_store = module_type_store.open_function_context('legendre_q_via_lqmn', 93, 0, False)
    
    # Passed parameters checking function
    legendre_q_via_lqmn.stypy_localization = localization
    legendre_q_via_lqmn.stypy_type_of_self = None
    legendre_q_via_lqmn.stypy_type_store = module_type_store
    legendre_q_via_lqmn.stypy_function_name = 'legendre_q_via_lqmn'
    legendre_q_via_lqmn.stypy_param_names_list = ['n', 'x']
    legendre_q_via_lqmn.stypy_varargs_param_name = None
    legendre_q_via_lqmn.stypy_kwargs_param_name = None
    legendre_q_via_lqmn.stypy_call_defaults = defaults
    legendre_q_via_lqmn.stypy_call_varargs = varargs
    legendre_q_via_lqmn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'legendre_q_via_lqmn', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'legendre_q_via_lqmn', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'legendre_q_via_lqmn(...)' code ##################

    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 94)
    tuple_534713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 94)
    # Adding element type (line 94)
    int_534714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 28), tuple_534713, int_534714)
    # Adding element type (line 94)
    int_534715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 28), tuple_534713, int_534715)
    
    
    # Obtaining the type of the subscript
    int_534716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'int')
    
    # Call to lqmn(...): (line 94)
    # Processing the call arguments (line 94)
    int_534718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 16), 'int')
    # Getting the type of 'n' (line 94)
    n_534719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'n', False)
    # Getting the type of 'x' (line 94)
    x_534720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), 'x', False)
    # Processing the call keyword arguments (line 94)
    kwargs_534721 = {}
    # Getting the type of 'lqmn' (line 94)
    lqmn_534717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'lqmn', False)
    # Calling lqmn(args, kwargs) (line 94)
    lqmn_call_result_534722 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), lqmn_534717, *[int_534718, n_534719, x_534720], **kwargs_534721)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___534723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), lqmn_call_result_534722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_534724 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), getitem___534723, int_534716)
    
    # Obtaining the member '__getitem__' of a type (line 94)
    getitem___534725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), subscript_call_result_534724, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 94)
    subscript_call_result_534726 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), getitem___534725, tuple_534713)
    
    # Assigning a type to the variable 'stypy_return_type' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type', subscript_call_result_534726)
    
    # ################# End of 'legendre_q_via_lqmn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'legendre_q_via_lqmn' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_534727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534727)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'legendre_q_via_lqmn'
    return stypy_return_type_534727

# Assigning a type to the variable 'legendre_q_via_lqmn' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'legendre_q_via_lqmn', legendre_q_via_lqmn)

@norecursion
def mathieu_ce_rad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mathieu_ce_rad'
    module_type_store = module_type_store.open_function_context('mathieu_ce_rad', 96, 0, False)
    
    # Passed parameters checking function
    mathieu_ce_rad.stypy_localization = localization
    mathieu_ce_rad.stypy_type_of_self = None
    mathieu_ce_rad.stypy_type_store = module_type_store
    mathieu_ce_rad.stypy_function_name = 'mathieu_ce_rad'
    mathieu_ce_rad.stypy_param_names_list = ['m', 'q', 'x']
    mathieu_ce_rad.stypy_varargs_param_name = None
    mathieu_ce_rad.stypy_kwargs_param_name = None
    mathieu_ce_rad.stypy_call_defaults = defaults
    mathieu_ce_rad.stypy_call_varargs = varargs
    mathieu_ce_rad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mathieu_ce_rad', ['m', 'q', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mathieu_ce_rad', localization, ['m', 'q', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mathieu_ce_rad(...)' code ##################

    
    # Obtaining the type of the subscript
    int_534728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 42), 'int')
    
    # Call to mathieu_cem(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'm' (line 97)
    m_534730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'm', False)
    # Getting the type of 'q' (line 97)
    q_534731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'q', False)
    # Getting the type of 'x' (line 97)
    x_534732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'x', False)
    int_534733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 31), 'int')
    # Applying the binary operator '*' (line 97)
    result_mul_534734 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 29), '*', x_534732, int_534733)
    
    # Getting the type of 'np' (line 97)
    np_534735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 35), 'np', False)
    # Obtaining the member 'pi' of a type (line 97)
    pi_534736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 35), np_534735, 'pi')
    # Applying the binary operator 'div' (line 97)
    result_div_534737 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 34), 'div', result_mul_534734, pi_534736)
    
    # Processing the call keyword arguments (line 97)
    kwargs_534738 = {}
    # Getting the type of 'mathieu_cem' (line 97)
    mathieu_cem_534729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'mathieu_cem', False)
    # Calling mathieu_cem(args, kwargs) (line 97)
    mathieu_cem_call_result_534739 = invoke(stypy.reporting.localization.Localization(__file__, 97, 11), mathieu_cem_534729, *[m_534730, q_534731, result_div_534737], **kwargs_534738)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___534740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), mathieu_cem_call_result_534739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_534741 = invoke(stypy.reporting.localization.Localization(__file__, 97, 11), getitem___534740, int_534728)
    
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type', subscript_call_result_534741)
    
    # ################# End of 'mathieu_ce_rad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mathieu_ce_rad' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_534742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534742)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mathieu_ce_rad'
    return stypy_return_type_534742

# Assigning a type to the variable 'mathieu_ce_rad' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'mathieu_ce_rad', mathieu_ce_rad)

@norecursion
def mathieu_se_rad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mathieu_se_rad'
    module_type_store = module_type_store.open_function_context('mathieu_se_rad', 100, 0, False)
    
    # Passed parameters checking function
    mathieu_se_rad.stypy_localization = localization
    mathieu_se_rad.stypy_type_of_self = None
    mathieu_se_rad.stypy_type_store = module_type_store
    mathieu_se_rad.stypy_function_name = 'mathieu_se_rad'
    mathieu_se_rad.stypy_param_names_list = ['m', 'q', 'x']
    mathieu_se_rad.stypy_varargs_param_name = None
    mathieu_se_rad.stypy_kwargs_param_name = None
    mathieu_se_rad.stypy_call_defaults = defaults
    mathieu_se_rad.stypy_call_varargs = varargs
    mathieu_se_rad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mathieu_se_rad', ['m', 'q', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mathieu_se_rad', localization, ['m', 'q', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mathieu_se_rad(...)' code ##################

    
    # Obtaining the type of the subscript
    int_534743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 42), 'int')
    
    # Call to mathieu_sem(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'm' (line 101)
    m_534745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'm', False)
    # Getting the type of 'q' (line 101)
    q_534746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'q', False)
    # Getting the type of 'x' (line 101)
    x_534747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'x', False)
    int_534748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 31), 'int')
    # Applying the binary operator '*' (line 101)
    result_mul_534749 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 29), '*', x_534747, int_534748)
    
    # Getting the type of 'np' (line 101)
    np_534750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'np', False)
    # Obtaining the member 'pi' of a type (line 101)
    pi_534751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 35), np_534750, 'pi')
    # Applying the binary operator 'div' (line 101)
    result_div_534752 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 34), 'div', result_mul_534749, pi_534751)
    
    # Processing the call keyword arguments (line 101)
    kwargs_534753 = {}
    # Getting the type of 'mathieu_sem' (line 101)
    mathieu_sem_534744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'mathieu_sem', False)
    # Calling mathieu_sem(args, kwargs) (line 101)
    mathieu_sem_call_result_534754 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), mathieu_sem_534744, *[m_534745, q_534746, result_div_534752], **kwargs_534753)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___534755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 11), mathieu_sem_call_result_534754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_534756 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), getitem___534755, int_534743)
    
    # Assigning a type to the variable 'stypy_return_type' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type', subscript_call_result_534756)
    
    # ################# End of 'mathieu_se_rad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mathieu_se_rad' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_534757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534757)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mathieu_se_rad'
    return stypy_return_type_534757

# Assigning a type to the variable 'mathieu_se_rad' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'mathieu_se_rad', mathieu_se_rad)

@norecursion
def mathieu_mc1_scaled(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mathieu_mc1_scaled'
    module_type_store = module_type_store.open_function_context('mathieu_mc1_scaled', 104, 0, False)
    
    # Passed parameters checking function
    mathieu_mc1_scaled.stypy_localization = localization
    mathieu_mc1_scaled.stypy_type_of_self = None
    mathieu_mc1_scaled.stypy_type_store = module_type_store
    mathieu_mc1_scaled.stypy_function_name = 'mathieu_mc1_scaled'
    mathieu_mc1_scaled.stypy_param_names_list = ['m', 'q', 'x']
    mathieu_mc1_scaled.stypy_varargs_param_name = None
    mathieu_mc1_scaled.stypy_kwargs_param_name = None
    mathieu_mc1_scaled.stypy_call_defaults = defaults
    mathieu_mc1_scaled.stypy_call_varargs = varargs
    mathieu_mc1_scaled.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mathieu_mc1_scaled', ['m', 'q', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mathieu_mc1_scaled', localization, ['m', 'q', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mathieu_mc1_scaled(...)' code ##################

    
    # Obtaining the type of the subscript
    int_534758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 36), 'int')
    
    # Call to mathieu_modcem1(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'm' (line 107)
    m_534760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'm', False)
    # Getting the type of 'q' (line 107)
    q_534761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'q', False)
    # Getting the type of 'x' (line 107)
    x_534762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'x', False)
    # Processing the call keyword arguments (line 107)
    kwargs_534763 = {}
    # Getting the type of 'mathieu_modcem1' (line 107)
    mathieu_modcem1_534759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'mathieu_modcem1', False)
    # Calling mathieu_modcem1(args, kwargs) (line 107)
    mathieu_modcem1_call_result_534764 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), mathieu_modcem1_534759, *[m_534760, q_534761, x_534762], **kwargs_534763)
    
    # Obtaining the member '__getitem__' of a type (line 107)
    getitem___534765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), mathieu_modcem1_call_result_534764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 107)
    subscript_call_result_534766 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), getitem___534765, int_534758)
    
    
    # Call to sqrt(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'np' (line 107)
    np_534769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 49), 'np', False)
    # Obtaining the member 'pi' of a type (line 107)
    pi_534770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 49), np_534769, 'pi')
    int_534771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 55), 'int')
    # Applying the binary operator 'div' (line 107)
    result_div_534772 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 49), 'div', pi_534770, int_534771)
    
    # Processing the call keyword arguments (line 107)
    kwargs_534773 = {}
    # Getting the type of 'np' (line 107)
    np_534767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 41), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 107)
    sqrt_534768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 41), np_534767, 'sqrt')
    # Calling sqrt(args, kwargs) (line 107)
    sqrt_call_result_534774 = invoke(stypy.reporting.localization.Localization(__file__, 107, 41), sqrt_534768, *[result_div_534772], **kwargs_534773)
    
    # Applying the binary operator '*' (line 107)
    result_mul_534775 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), '*', subscript_call_result_534766, sqrt_call_result_534774)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type', result_mul_534775)
    
    # ################# End of 'mathieu_mc1_scaled(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mathieu_mc1_scaled' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_534776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534776)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mathieu_mc1_scaled'
    return stypy_return_type_534776

# Assigning a type to the variable 'mathieu_mc1_scaled' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'mathieu_mc1_scaled', mathieu_mc1_scaled)

@norecursion
def mathieu_ms1_scaled(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mathieu_ms1_scaled'
    module_type_store = module_type_store.open_function_context('mathieu_ms1_scaled', 110, 0, False)
    
    # Passed parameters checking function
    mathieu_ms1_scaled.stypy_localization = localization
    mathieu_ms1_scaled.stypy_type_of_self = None
    mathieu_ms1_scaled.stypy_type_store = module_type_store
    mathieu_ms1_scaled.stypy_function_name = 'mathieu_ms1_scaled'
    mathieu_ms1_scaled.stypy_param_names_list = ['m', 'q', 'x']
    mathieu_ms1_scaled.stypy_varargs_param_name = None
    mathieu_ms1_scaled.stypy_kwargs_param_name = None
    mathieu_ms1_scaled.stypy_call_defaults = defaults
    mathieu_ms1_scaled.stypy_call_varargs = varargs
    mathieu_ms1_scaled.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mathieu_ms1_scaled', ['m', 'q', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mathieu_ms1_scaled', localization, ['m', 'q', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mathieu_ms1_scaled(...)' code ##################

    
    # Obtaining the type of the subscript
    int_534777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 36), 'int')
    
    # Call to mathieu_modsem1(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'm' (line 111)
    m_534779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'm', False)
    # Getting the type of 'q' (line 111)
    q_534780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'q', False)
    # Getting the type of 'x' (line 111)
    x_534781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'x', False)
    # Processing the call keyword arguments (line 111)
    kwargs_534782 = {}
    # Getting the type of 'mathieu_modsem1' (line 111)
    mathieu_modsem1_534778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'mathieu_modsem1', False)
    # Calling mathieu_modsem1(args, kwargs) (line 111)
    mathieu_modsem1_call_result_534783 = invoke(stypy.reporting.localization.Localization(__file__, 111, 11), mathieu_modsem1_534778, *[m_534779, q_534780, x_534781], **kwargs_534782)
    
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___534784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), mathieu_modsem1_call_result_534783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_534785 = invoke(stypy.reporting.localization.Localization(__file__, 111, 11), getitem___534784, int_534777)
    
    
    # Call to sqrt(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'np' (line 111)
    np_534788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 49), 'np', False)
    # Obtaining the member 'pi' of a type (line 111)
    pi_534789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 49), np_534788, 'pi')
    int_534790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 55), 'int')
    # Applying the binary operator 'div' (line 111)
    result_div_534791 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 49), 'div', pi_534789, int_534790)
    
    # Processing the call keyword arguments (line 111)
    kwargs_534792 = {}
    # Getting the type of 'np' (line 111)
    np_534786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 111)
    sqrt_534787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 41), np_534786, 'sqrt')
    # Calling sqrt(args, kwargs) (line 111)
    sqrt_call_result_534793 = invoke(stypy.reporting.localization.Localization(__file__, 111, 41), sqrt_534787, *[result_div_534791], **kwargs_534792)
    
    # Applying the binary operator '*' (line 111)
    result_mul_534794 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 11), '*', subscript_call_result_534785, sqrt_call_result_534793)
    
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', result_mul_534794)
    
    # ################# End of 'mathieu_ms1_scaled(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mathieu_ms1_scaled' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_534795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534795)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mathieu_ms1_scaled'
    return stypy_return_type_534795

# Assigning a type to the variable 'mathieu_ms1_scaled' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'mathieu_ms1_scaled', mathieu_ms1_scaled)

@norecursion
def mathieu_mc2_scaled(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mathieu_mc2_scaled'
    module_type_store = module_type_store.open_function_context('mathieu_mc2_scaled', 114, 0, False)
    
    # Passed parameters checking function
    mathieu_mc2_scaled.stypy_localization = localization
    mathieu_mc2_scaled.stypy_type_of_self = None
    mathieu_mc2_scaled.stypy_type_store = module_type_store
    mathieu_mc2_scaled.stypy_function_name = 'mathieu_mc2_scaled'
    mathieu_mc2_scaled.stypy_param_names_list = ['m', 'q', 'x']
    mathieu_mc2_scaled.stypy_varargs_param_name = None
    mathieu_mc2_scaled.stypy_kwargs_param_name = None
    mathieu_mc2_scaled.stypy_call_defaults = defaults
    mathieu_mc2_scaled.stypy_call_varargs = varargs
    mathieu_mc2_scaled.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mathieu_mc2_scaled', ['m', 'q', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mathieu_mc2_scaled', localization, ['m', 'q', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mathieu_mc2_scaled(...)' code ##################

    
    # Obtaining the type of the subscript
    int_534796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 36), 'int')
    
    # Call to mathieu_modcem2(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'm' (line 115)
    m_534798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'm', False)
    # Getting the type of 'q' (line 115)
    q_534799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'q', False)
    # Getting the type of 'x' (line 115)
    x_534800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 33), 'x', False)
    # Processing the call keyword arguments (line 115)
    kwargs_534801 = {}
    # Getting the type of 'mathieu_modcem2' (line 115)
    mathieu_modcem2_534797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'mathieu_modcem2', False)
    # Calling mathieu_modcem2(args, kwargs) (line 115)
    mathieu_modcem2_call_result_534802 = invoke(stypy.reporting.localization.Localization(__file__, 115, 11), mathieu_modcem2_534797, *[m_534798, q_534799, x_534800], **kwargs_534801)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___534803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 11), mathieu_modcem2_call_result_534802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_534804 = invoke(stypy.reporting.localization.Localization(__file__, 115, 11), getitem___534803, int_534796)
    
    
    # Call to sqrt(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'np' (line 115)
    np_534807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 49), 'np', False)
    # Obtaining the member 'pi' of a type (line 115)
    pi_534808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 49), np_534807, 'pi')
    int_534809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 55), 'int')
    # Applying the binary operator 'div' (line 115)
    result_div_534810 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 49), 'div', pi_534808, int_534809)
    
    # Processing the call keyword arguments (line 115)
    kwargs_534811 = {}
    # Getting the type of 'np' (line 115)
    np_534805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 115)
    sqrt_534806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 41), np_534805, 'sqrt')
    # Calling sqrt(args, kwargs) (line 115)
    sqrt_call_result_534812 = invoke(stypy.reporting.localization.Localization(__file__, 115, 41), sqrt_534806, *[result_div_534810], **kwargs_534811)
    
    # Applying the binary operator '*' (line 115)
    result_mul_534813 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 11), '*', subscript_call_result_534804, sqrt_call_result_534812)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', result_mul_534813)
    
    # ################# End of 'mathieu_mc2_scaled(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mathieu_mc2_scaled' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_534814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534814)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mathieu_mc2_scaled'
    return stypy_return_type_534814

# Assigning a type to the variable 'mathieu_mc2_scaled' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'mathieu_mc2_scaled', mathieu_mc2_scaled)

@norecursion
def mathieu_ms2_scaled(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mathieu_ms2_scaled'
    module_type_store = module_type_store.open_function_context('mathieu_ms2_scaled', 118, 0, False)
    
    # Passed parameters checking function
    mathieu_ms2_scaled.stypy_localization = localization
    mathieu_ms2_scaled.stypy_type_of_self = None
    mathieu_ms2_scaled.stypy_type_store = module_type_store
    mathieu_ms2_scaled.stypy_function_name = 'mathieu_ms2_scaled'
    mathieu_ms2_scaled.stypy_param_names_list = ['m', 'q', 'x']
    mathieu_ms2_scaled.stypy_varargs_param_name = None
    mathieu_ms2_scaled.stypy_kwargs_param_name = None
    mathieu_ms2_scaled.stypy_call_defaults = defaults
    mathieu_ms2_scaled.stypy_call_varargs = varargs
    mathieu_ms2_scaled.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mathieu_ms2_scaled', ['m', 'q', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mathieu_ms2_scaled', localization, ['m', 'q', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mathieu_ms2_scaled(...)' code ##################

    
    # Obtaining the type of the subscript
    int_534815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 36), 'int')
    
    # Call to mathieu_modsem2(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'm' (line 119)
    m_534817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'm', False)
    # Getting the type of 'q' (line 119)
    q_534818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'q', False)
    # Getting the type of 'x' (line 119)
    x_534819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'x', False)
    # Processing the call keyword arguments (line 119)
    kwargs_534820 = {}
    # Getting the type of 'mathieu_modsem2' (line 119)
    mathieu_modsem2_534816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'mathieu_modsem2', False)
    # Calling mathieu_modsem2(args, kwargs) (line 119)
    mathieu_modsem2_call_result_534821 = invoke(stypy.reporting.localization.Localization(__file__, 119, 11), mathieu_modsem2_534816, *[m_534817, q_534818, x_534819], **kwargs_534820)
    
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___534822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 11), mathieu_modsem2_call_result_534821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_534823 = invoke(stypy.reporting.localization.Localization(__file__, 119, 11), getitem___534822, int_534815)
    
    
    # Call to sqrt(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'np' (line 119)
    np_534826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 49), 'np', False)
    # Obtaining the member 'pi' of a type (line 119)
    pi_534827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 49), np_534826, 'pi')
    int_534828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 55), 'int')
    # Applying the binary operator 'div' (line 119)
    result_div_534829 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 49), 'div', pi_534827, int_534828)
    
    # Processing the call keyword arguments (line 119)
    kwargs_534830 = {}
    # Getting the type of 'np' (line 119)
    np_534824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 41), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 119)
    sqrt_534825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 41), np_534824, 'sqrt')
    # Calling sqrt(args, kwargs) (line 119)
    sqrt_call_result_534831 = invoke(stypy.reporting.localization.Localization(__file__, 119, 41), sqrt_534825, *[result_div_534829], **kwargs_534830)
    
    # Applying the binary operator '*' (line 119)
    result_mul_534832 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '*', subscript_call_result_534823, sqrt_call_result_534831)
    
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type', result_mul_534832)
    
    # ################# End of 'mathieu_ms2_scaled(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mathieu_ms2_scaled' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_534833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534833)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mathieu_ms2_scaled'
    return stypy_return_type_534833

# Assigning a type to the variable 'mathieu_ms2_scaled' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'mathieu_ms2_scaled', mathieu_ms2_scaled)

@norecursion
def eval_legendre_ld(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eval_legendre_ld'
    module_type_store = module_type_store.open_function_context('eval_legendre_ld', 121, 0, False)
    
    # Passed parameters checking function
    eval_legendre_ld.stypy_localization = localization
    eval_legendre_ld.stypy_type_of_self = None
    eval_legendre_ld.stypy_type_store = module_type_store
    eval_legendre_ld.stypy_function_name = 'eval_legendre_ld'
    eval_legendre_ld.stypy_param_names_list = ['n', 'x']
    eval_legendre_ld.stypy_varargs_param_name = None
    eval_legendre_ld.stypy_kwargs_param_name = None
    eval_legendre_ld.stypy_call_defaults = defaults
    eval_legendre_ld.stypy_call_varargs = varargs
    eval_legendre_ld.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eval_legendre_ld', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eval_legendre_ld', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eval_legendre_ld(...)' code ##################

    
    # Call to eval_legendre(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Call to astype(...): (line 122)
    # Processing the call arguments (line 122)
    str_534837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'str', 'l')
    # Processing the call keyword arguments (line 122)
    kwargs_534838 = {}
    # Getting the type of 'n' (line 122)
    n_534835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'n', False)
    # Obtaining the member 'astype' of a type (line 122)
    astype_534836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 25), n_534835, 'astype')
    # Calling astype(args, kwargs) (line 122)
    astype_call_result_534839 = invoke(stypy.reporting.localization.Localization(__file__, 122, 25), astype_534836, *[str_534837], **kwargs_534838)
    
    # Getting the type of 'x' (line 122)
    x_534840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'x', False)
    # Processing the call keyword arguments (line 122)
    kwargs_534841 = {}
    # Getting the type of 'eval_legendre' (line 122)
    eval_legendre_534834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'eval_legendre', False)
    # Calling eval_legendre(args, kwargs) (line 122)
    eval_legendre_call_result_534842 = invoke(stypy.reporting.localization.Localization(__file__, 122, 11), eval_legendre_534834, *[astype_call_result_534839, x_534840], **kwargs_534841)
    
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type', eval_legendre_call_result_534842)
    
    # ################# End of 'eval_legendre_ld(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eval_legendre_ld' in the type store
    # Getting the type of 'stypy_return_type' (line 121)
    stypy_return_type_534843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534843)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eval_legendre_ld'
    return stypy_return_type_534843

# Assigning a type to the variable 'eval_legendre_ld' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'eval_legendre_ld', eval_legendre_ld)

@norecursion
def eval_legendre_dd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eval_legendre_dd'
    module_type_store = module_type_store.open_function_context('eval_legendre_dd', 124, 0, False)
    
    # Passed parameters checking function
    eval_legendre_dd.stypy_localization = localization
    eval_legendre_dd.stypy_type_of_self = None
    eval_legendre_dd.stypy_type_store = module_type_store
    eval_legendre_dd.stypy_function_name = 'eval_legendre_dd'
    eval_legendre_dd.stypy_param_names_list = ['n', 'x']
    eval_legendre_dd.stypy_varargs_param_name = None
    eval_legendre_dd.stypy_kwargs_param_name = None
    eval_legendre_dd.stypy_call_defaults = defaults
    eval_legendre_dd.stypy_call_varargs = varargs
    eval_legendre_dd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eval_legendre_dd', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eval_legendre_dd', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eval_legendre_dd(...)' code ##################

    
    # Call to eval_legendre(...): (line 125)
    # Processing the call arguments (line 125)
    
    # Call to astype(...): (line 125)
    # Processing the call arguments (line 125)
    str_534847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 34), 'str', 'd')
    # Processing the call keyword arguments (line 125)
    kwargs_534848 = {}
    # Getting the type of 'n' (line 125)
    n_534845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'n', False)
    # Obtaining the member 'astype' of a type (line 125)
    astype_534846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 25), n_534845, 'astype')
    # Calling astype(args, kwargs) (line 125)
    astype_call_result_534849 = invoke(stypy.reporting.localization.Localization(__file__, 125, 25), astype_534846, *[str_534847], **kwargs_534848)
    
    # Getting the type of 'x' (line 125)
    x_534850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'x', False)
    # Processing the call keyword arguments (line 125)
    kwargs_534851 = {}
    # Getting the type of 'eval_legendre' (line 125)
    eval_legendre_534844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'eval_legendre', False)
    # Calling eval_legendre(args, kwargs) (line 125)
    eval_legendre_call_result_534852 = invoke(stypy.reporting.localization.Localization(__file__, 125, 11), eval_legendre_534844, *[astype_call_result_534849, x_534850], **kwargs_534851)
    
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type', eval_legendre_call_result_534852)
    
    # ################# End of 'eval_legendre_dd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eval_legendre_dd' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_534853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534853)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eval_legendre_dd'
    return stypy_return_type_534853

# Assigning a type to the variable 'eval_legendre_dd' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'eval_legendre_dd', eval_legendre_dd)

@norecursion
def eval_hermite_ld(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eval_hermite_ld'
    module_type_store = module_type_store.open_function_context('eval_hermite_ld', 127, 0, False)
    
    # Passed parameters checking function
    eval_hermite_ld.stypy_localization = localization
    eval_hermite_ld.stypy_type_of_self = None
    eval_hermite_ld.stypy_type_store = module_type_store
    eval_hermite_ld.stypy_function_name = 'eval_hermite_ld'
    eval_hermite_ld.stypy_param_names_list = ['n', 'x']
    eval_hermite_ld.stypy_varargs_param_name = None
    eval_hermite_ld.stypy_kwargs_param_name = None
    eval_hermite_ld.stypy_call_defaults = defaults
    eval_hermite_ld.stypy_call_varargs = varargs
    eval_hermite_ld.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eval_hermite_ld', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eval_hermite_ld', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eval_hermite_ld(...)' code ##################

    
    # Call to eval_hermite(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Call to astype(...): (line 128)
    # Processing the call arguments (line 128)
    str_534857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 33), 'str', 'l')
    # Processing the call keyword arguments (line 128)
    kwargs_534858 = {}
    # Getting the type of 'n' (line 128)
    n_534855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'n', False)
    # Obtaining the member 'astype' of a type (line 128)
    astype_534856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), n_534855, 'astype')
    # Calling astype(args, kwargs) (line 128)
    astype_call_result_534859 = invoke(stypy.reporting.localization.Localization(__file__, 128, 24), astype_534856, *[str_534857], **kwargs_534858)
    
    # Getting the type of 'x' (line 128)
    x_534860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 39), 'x', False)
    # Processing the call keyword arguments (line 128)
    kwargs_534861 = {}
    # Getting the type of 'eval_hermite' (line 128)
    eval_hermite_534854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'eval_hermite', False)
    # Calling eval_hermite(args, kwargs) (line 128)
    eval_hermite_call_result_534862 = invoke(stypy.reporting.localization.Localization(__file__, 128, 11), eval_hermite_534854, *[astype_call_result_534859, x_534860], **kwargs_534861)
    
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', eval_hermite_call_result_534862)
    
    # ################# End of 'eval_hermite_ld(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eval_hermite_ld' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_534863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eval_hermite_ld'
    return stypy_return_type_534863

# Assigning a type to the variable 'eval_hermite_ld' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'eval_hermite_ld', eval_hermite_ld)

@norecursion
def eval_laguerre_ld(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eval_laguerre_ld'
    module_type_store = module_type_store.open_function_context('eval_laguerre_ld', 130, 0, False)
    
    # Passed parameters checking function
    eval_laguerre_ld.stypy_localization = localization
    eval_laguerre_ld.stypy_type_of_self = None
    eval_laguerre_ld.stypy_type_store = module_type_store
    eval_laguerre_ld.stypy_function_name = 'eval_laguerre_ld'
    eval_laguerre_ld.stypy_param_names_list = ['n', 'x']
    eval_laguerre_ld.stypy_varargs_param_name = None
    eval_laguerre_ld.stypy_kwargs_param_name = None
    eval_laguerre_ld.stypy_call_defaults = defaults
    eval_laguerre_ld.stypy_call_varargs = varargs
    eval_laguerre_ld.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eval_laguerre_ld', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eval_laguerre_ld', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eval_laguerre_ld(...)' code ##################

    
    # Call to eval_laguerre(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Call to astype(...): (line 131)
    # Processing the call arguments (line 131)
    str_534867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 34), 'str', 'l')
    # Processing the call keyword arguments (line 131)
    kwargs_534868 = {}
    # Getting the type of 'n' (line 131)
    n_534865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'n', False)
    # Obtaining the member 'astype' of a type (line 131)
    astype_534866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 25), n_534865, 'astype')
    # Calling astype(args, kwargs) (line 131)
    astype_call_result_534869 = invoke(stypy.reporting.localization.Localization(__file__, 131, 25), astype_534866, *[str_534867], **kwargs_534868)
    
    # Getting the type of 'x' (line 131)
    x_534870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'x', False)
    # Processing the call keyword arguments (line 131)
    kwargs_534871 = {}
    # Getting the type of 'eval_laguerre' (line 131)
    eval_laguerre_534864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'eval_laguerre', False)
    # Calling eval_laguerre(args, kwargs) (line 131)
    eval_laguerre_call_result_534872 = invoke(stypy.reporting.localization.Localization(__file__, 131, 11), eval_laguerre_534864, *[astype_call_result_534869, x_534870], **kwargs_534871)
    
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type', eval_laguerre_call_result_534872)
    
    # ################# End of 'eval_laguerre_ld(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eval_laguerre_ld' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_534873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534873)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eval_laguerre_ld'
    return stypy_return_type_534873

# Assigning a type to the variable 'eval_laguerre_ld' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'eval_laguerre_ld', eval_laguerre_ld)

@norecursion
def eval_laguerre_dd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eval_laguerre_dd'
    module_type_store = module_type_store.open_function_context('eval_laguerre_dd', 133, 0, False)
    
    # Passed parameters checking function
    eval_laguerre_dd.stypy_localization = localization
    eval_laguerre_dd.stypy_type_of_self = None
    eval_laguerre_dd.stypy_type_store = module_type_store
    eval_laguerre_dd.stypy_function_name = 'eval_laguerre_dd'
    eval_laguerre_dd.stypy_param_names_list = ['n', 'x']
    eval_laguerre_dd.stypy_varargs_param_name = None
    eval_laguerre_dd.stypy_kwargs_param_name = None
    eval_laguerre_dd.stypy_call_defaults = defaults
    eval_laguerre_dd.stypy_call_varargs = varargs
    eval_laguerre_dd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eval_laguerre_dd', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eval_laguerre_dd', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eval_laguerre_dd(...)' code ##################

    
    # Call to eval_laguerre(...): (line 134)
    # Processing the call arguments (line 134)
    
    # Call to astype(...): (line 134)
    # Processing the call arguments (line 134)
    str_534877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'str', 'd')
    # Processing the call keyword arguments (line 134)
    kwargs_534878 = {}
    # Getting the type of 'n' (line 134)
    n_534875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'n', False)
    # Obtaining the member 'astype' of a type (line 134)
    astype_534876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 25), n_534875, 'astype')
    # Calling astype(args, kwargs) (line 134)
    astype_call_result_534879 = invoke(stypy.reporting.localization.Localization(__file__, 134, 25), astype_534876, *[str_534877], **kwargs_534878)
    
    # Getting the type of 'x' (line 134)
    x_534880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 40), 'x', False)
    # Processing the call keyword arguments (line 134)
    kwargs_534881 = {}
    # Getting the type of 'eval_laguerre' (line 134)
    eval_laguerre_534874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'eval_laguerre', False)
    # Calling eval_laguerre(args, kwargs) (line 134)
    eval_laguerre_call_result_534882 = invoke(stypy.reporting.localization.Localization(__file__, 134, 11), eval_laguerre_534874, *[astype_call_result_534879, x_534880], **kwargs_534881)
    
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type', eval_laguerre_call_result_534882)
    
    # ################# End of 'eval_laguerre_dd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eval_laguerre_dd' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_534883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534883)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eval_laguerre_dd'
    return stypy_return_type_534883

# Assigning a type to the variable 'eval_laguerre_dd' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'eval_laguerre_dd', eval_laguerre_dd)

@norecursion
def eval_genlaguerre_ldd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eval_genlaguerre_ldd'
    module_type_store = module_type_store.open_function_context('eval_genlaguerre_ldd', 136, 0, False)
    
    # Passed parameters checking function
    eval_genlaguerre_ldd.stypy_localization = localization
    eval_genlaguerre_ldd.stypy_type_of_self = None
    eval_genlaguerre_ldd.stypy_type_store = module_type_store
    eval_genlaguerre_ldd.stypy_function_name = 'eval_genlaguerre_ldd'
    eval_genlaguerre_ldd.stypy_param_names_list = ['n', 'a', 'x']
    eval_genlaguerre_ldd.stypy_varargs_param_name = None
    eval_genlaguerre_ldd.stypy_kwargs_param_name = None
    eval_genlaguerre_ldd.stypy_call_defaults = defaults
    eval_genlaguerre_ldd.stypy_call_varargs = varargs
    eval_genlaguerre_ldd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eval_genlaguerre_ldd', ['n', 'a', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eval_genlaguerre_ldd', localization, ['n', 'a', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eval_genlaguerre_ldd(...)' code ##################

    
    # Call to eval_genlaguerre(...): (line 137)
    # Processing the call arguments (line 137)
    
    # Call to astype(...): (line 137)
    # Processing the call arguments (line 137)
    str_534887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 37), 'str', 'l')
    # Processing the call keyword arguments (line 137)
    kwargs_534888 = {}
    # Getting the type of 'n' (line 137)
    n_534885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 28), 'n', False)
    # Obtaining the member 'astype' of a type (line 137)
    astype_534886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 28), n_534885, 'astype')
    # Calling astype(args, kwargs) (line 137)
    astype_call_result_534889 = invoke(stypy.reporting.localization.Localization(__file__, 137, 28), astype_534886, *[str_534887], **kwargs_534888)
    
    # Getting the type of 'a' (line 137)
    a_534890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 43), 'a', False)
    # Getting the type of 'x' (line 137)
    x_534891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 46), 'x', False)
    # Processing the call keyword arguments (line 137)
    kwargs_534892 = {}
    # Getting the type of 'eval_genlaguerre' (line 137)
    eval_genlaguerre_534884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'eval_genlaguerre', False)
    # Calling eval_genlaguerre(args, kwargs) (line 137)
    eval_genlaguerre_call_result_534893 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), eval_genlaguerre_534884, *[astype_call_result_534889, a_534890, x_534891], **kwargs_534892)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', eval_genlaguerre_call_result_534893)
    
    # ################# End of 'eval_genlaguerre_ldd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eval_genlaguerre_ldd' in the type store
    # Getting the type of 'stypy_return_type' (line 136)
    stypy_return_type_534894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534894)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eval_genlaguerre_ldd'
    return stypy_return_type_534894

# Assigning a type to the variable 'eval_genlaguerre_ldd' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'eval_genlaguerre_ldd', eval_genlaguerre_ldd)

@norecursion
def eval_genlaguerre_ddd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eval_genlaguerre_ddd'
    module_type_store = module_type_store.open_function_context('eval_genlaguerre_ddd', 139, 0, False)
    
    # Passed parameters checking function
    eval_genlaguerre_ddd.stypy_localization = localization
    eval_genlaguerre_ddd.stypy_type_of_self = None
    eval_genlaguerre_ddd.stypy_type_store = module_type_store
    eval_genlaguerre_ddd.stypy_function_name = 'eval_genlaguerre_ddd'
    eval_genlaguerre_ddd.stypy_param_names_list = ['n', 'a', 'x']
    eval_genlaguerre_ddd.stypy_varargs_param_name = None
    eval_genlaguerre_ddd.stypy_kwargs_param_name = None
    eval_genlaguerre_ddd.stypy_call_defaults = defaults
    eval_genlaguerre_ddd.stypy_call_varargs = varargs
    eval_genlaguerre_ddd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eval_genlaguerre_ddd', ['n', 'a', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eval_genlaguerre_ddd', localization, ['n', 'a', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eval_genlaguerre_ddd(...)' code ##################

    
    # Call to eval_genlaguerre(...): (line 140)
    # Processing the call arguments (line 140)
    
    # Call to astype(...): (line 140)
    # Processing the call arguments (line 140)
    str_534898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 37), 'str', 'd')
    # Processing the call keyword arguments (line 140)
    kwargs_534899 = {}
    # Getting the type of 'n' (line 140)
    n_534896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'n', False)
    # Obtaining the member 'astype' of a type (line 140)
    astype_534897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 28), n_534896, 'astype')
    # Calling astype(args, kwargs) (line 140)
    astype_call_result_534900 = invoke(stypy.reporting.localization.Localization(__file__, 140, 28), astype_534897, *[str_534898], **kwargs_534899)
    
    # Getting the type of 'a' (line 140)
    a_534901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'a', False)
    # Getting the type of 'x' (line 140)
    x_534902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 46), 'x', False)
    # Processing the call keyword arguments (line 140)
    kwargs_534903 = {}
    # Getting the type of 'eval_genlaguerre' (line 140)
    eval_genlaguerre_534895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'eval_genlaguerre', False)
    # Calling eval_genlaguerre(args, kwargs) (line 140)
    eval_genlaguerre_call_result_534904 = invoke(stypy.reporting.localization.Localization(__file__, 140, 11), eval_genlaguerre_534895, *[astype_call_result_534900, a_534901, x_534902], **kwargs_534903)
    
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type', eval_genlaguerre_call_result_534904)
    
    # ################# End of 'eval_genlaguerre_ddd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eval_genlaguerre_ddd' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_534905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534905)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eval_genlaguerre_ddd'
    return stypy_return_type_534905

# Assigning a type to the variable 'eval_genlaguerre_ddd' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'eval_genlaguerre_ddd', eval_genlaguerre_ddd)

@norecursion
def bdtrik_comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bdtrik_comp'
    module_type_store = module_type_store.open_function_context('bdtrik_comp', 142, 0, False)
    
    # Passed parameters checking function
    bdtrik_comp.stypy_localization = localization
    bdtrik_comp.stypy_type_of_self = None
    bdtrik_comp.stypy_type_store = module_type_store
    bdtrik_comp.stypy_function_name = 'bdtrik_comp'
    bdtrik_comp.stypy_param_names_list = ['y', 'n', 'p']
    bdtrik_comp.stypy_varargs_param_name = None
    bdtrik_comp.stypy_kwargs_param_name = None
    bdtrik_comp.stypy_call_defaults = defaults
    bdtrik_comp.stypy_call_varargs = varargs
    bdtrik_comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bdtrik_comp', ['y', 'n', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bdtrik_comp', localization, ['y', 'n', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bdtrik_comp(...)' code ##################

    
    # Call to bdtrik(...): (line 143)
    # Processing the call arguments (line 143)
    int_534907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 18), 'int')
    # Getting the type of 'y' (line 143)
    y_534908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'y', False)
    # Applying the binary operator '-' (line 143)
    result_sub_534909 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 18), '-', int_534907, y_534908)
    
    # Getting the type of 'n' (line 143)
    n_534910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'n', False)
    # Getting the type of 'p' (line 143)
    p_534911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'p', False)
    # Processing the call keyword arguments (line 143)
    kwargs_534912 = {}
    # Getting the type of 'bdtrik' (line 143)
    bdtrik_534906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'bdtrik', False)
    # Calling bdtrik(args, kwargs) (line 143)
    bdtrik_call_result_534913 = invoke(stypy.reporting.localization.Localization(__file__, 143, 11), bdtrik_534906, *[result_sub_534909, n_534910, p_534911], **kwargs_534912)
    
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type', bdtrik_call_result_534913)
    
    # ################# End of 'bdtrik_comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bdtrik_comp' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_534914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534914)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bdtrik_comp'
    return stypy_return_type_534914

# Assigning a type to the variable 'bdtrik_comp' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'bdtrik_comp', bdtrik_comp)

@norecursion
def btdtri_comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'btdtri_comp'
    module_type_store = module_type_store.open_function_context('btdtri_comp', 145, 0, False)
    
    # Passed parameters checking function
    btdtri_comp.stypy_localization = localization
    btdtri_comp.stypy_type_of_self = None
    btdtri_comp.stypy_type_store = module_type_store
    btdtri_comp.stypy_function_name = 'btdtri_comp'
    btdtri_comp.stypy_param_names_list = ['a', 'b', 'p']
    btdtri_comp.stypy_varargs_param_name = None
    btdtri_comp.stypy_kwargs_param_name = None
    btdtri_comp.stypy_call_defaults = defaults
    btdtri_comp.stypy_call_varargs = varargs
    btdtri_comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'btdtri_comp', ['a', 'b', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'btdtri_comp', localization, ['a', 'b', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'btdtri_comp(...)' code ##################

    
    # Call to btdtri(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'a' (line 146)
    a_534916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'a', False)
    # Getting the type of 'b' (line 146)
    b_534917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 21), 'b', False)
    int_534918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'int')
    # Getting the type of 'p' (line 146)
    p_534919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'p', False)
    # Applying the binary operator '-' (line 146)
    result_sub_534920 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 24), '-', int_534918, p_534919)
    
    # Processing the call keyword arguments (line 146)
    kwargs_534921 = {}
    # Getting the type of 'btdtri' (line 146)
    btdtri_534915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'btdtri', False)
    # Calling btdtri(args, kwargs) (line 146)
    btdtri_call_result_534922 = invoke(stypy.reporting.localization.Localization(__file__, 146, 11), btdtri_534915, *[a_534916, b_534917, result_sub_534920], **kwargs_534921)
    
    # Assigning a type to the variable 'stypy_return_type' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type', btdtri_call_result_534922)
    
    # ################# End of 'btdtri_comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'btdtri_comp' in the type store
    # Getting the type of 'stypy_return_type' (line 145)
    stypy_return_type_534923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534923)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'btdtri_comp'
    return stypy_return_type_534923

# Assigning a type to the variable 'btdtri_comp' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'btdtri_comp', btdtri_comp)

@norecursion
def btdtria_comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'btdtria_comp'
    module_type_store = module_type_store.open_function_context('btdtria_comp', 148, 0, False)
    
    # Passed parameters checking function
    btdtria_comp.stypy_localization = localization
    btdtria_comp.stypy_type_of_self = None
    btdtria_comp.stypy_type_store = module_type_store
    btdtria_comp.stypy_function_name = 'btdtria_comp'
    btdtria_comp.stypy_param_names_list = ['p', 'b', 'x']
    btdtria_comp.stypy_varargs_param_name = None
    btdtria_comp.stypy_kwargs_param_name = None
    btdtria_comp.stypy_call_defaults = defaults
    btdtria_comp.stypy_call_varargs = varargs
    btdtria_comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'btdtria_comp', ['p', 'b', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'btdtria_comp', localization, ['p', 'b', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'btdtria_comp(...)' code ##################

    
    # Call to btdtria(...): (line 149)
    # Processing the call arguments (line 149)
    int_534925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 19), 'int')
    # Getting the type of 'p' (line 149)
    p_534926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'p', False)
    # Applying the binary operator '-' (line 149)
    result_sub_534927 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 19), '-', int_534925, p_534926)
    
    # Getting the type of 'b' (line 149)
    b_534928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'b', False)
    # Getting the type of 'x' (line 149)
    x_534929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'x', False)
    # Processing the call keyword arguments (line 149)
    kwargs_534930 = {}
    # Getting the type of 'btdtria' (line 149)
    btdtria_534924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'btdtria', False)
    # Calling btdtria(args, kwargs) (line 149)
    btdtria_call_result_534931 = invoke(stypy.reporting.localization.Localization(__file__, 149, 11), btdtria_534924, *[result_sub_534927, b_534928, x_534929], **kwargs_534930)
    
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type', btdtria_call_result_534931)
    
    # ################# End of 'btdtria_comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'btdtria_comp' in the type store
    # Getting the type of 'stypy_return_type' (line 148)
    stypy_return_type_534932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534932)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'btdtria_comp'
    return stypy_return_type_534932

# Assigning a type to the variable 'btdtria_comp' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'btdtria_comp', btdtria_comp)

@norecursion
def btdtrib_comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'btdtrib_comp'
    module_type_store = module_type_store.open_function_context('btdtrib_comp', 151, 0, False)
    
    # Passed parameters checking function
    btdtrib_comp.stypy_localization = localization
    btdtrib_comp.stypy_type_of_self = None
    btdtrib_comp.stypy_type_store = module_type_store
    btdtrib_comp.stypy_function_name = 'btdtrib_comp'
    btdtrib_comp.stypy_param_names_list = ['a', 'p', 'x']
    btdtrib_comp.stypy_varargs_param_name = None
    btdtrib_comp.stypy_kwargs_param_name = None
    btdtrib_comp.stypy_call_defaults = defaults
    btdtrib_comp.stypy_call_varargs = varargs
    btdtrib_comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'btdtrib_comp', ['a', 'p', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'btdtrib_comp', localization, ['a', 'p', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'btdtrib_comp(...)' code ##################

    
    # Call to btdtrib(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'a' (line 152)
    a_534934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'a', False)
    int_534935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'int')
    # Getting the type of 'p' (line 152)
    p_534936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'p', False)
    # Applying the binary operator '-' (line 152)
    result_sub_534937 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 22), '-', int_534935, p_534936)
    
    # Getting the type of 'x' (line 152)
    x_534938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'x', False)
    # Processing the call keyword arguments (line 152)
    kwargs_534939 = {}
    # Getting the type of 'btdtrib' (line 152)
    btdtrib_534933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'btdtrib', False)
    # Calling btdtrib(args, kwargs) (line 152)
    btdtrib_call_result_534940 = invoke(stypy.reporting.localization.Localization(__file__, 152, 11), btdtrib_534933, *[a_534934, result_sub_534937, x_534938], **kwargs_534939)
    
    # Assigning a type to the variable 'stypy_return_type' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type', btdtrib_call_result_534940)
    
    # ################# End of 'btdtrib_comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'btdtrib_comp' in the type store
    # Getting the type of 'stypy_return_type' (line 151)
    stypy_return_type_534941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534941)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'btdtrib_comp'
    return stypy_return_type_534941

# Assigning a type to the variable 'btdtrib_comp' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'btdtrib_comp', btdtrib_comp)

@norecursion
def gdtr_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gdtr_'
    module_type_store = module_type_store.open_function_context('gdtr_', 154, 0, False)
    
    # Passed parameters checking function
    gdtr_.stypy_localization = localization
    gdtr_.stypy_type_of_self = None
    gdtr_.stypy_type_store = module_type_store
    gdtr_.stypy_function_name = 'gdtr_'
    gdtr_.stypy_param_names_list = ['p', 'x']
    gdtr_.stypy_varargs_param_name = None
    gdtr_.stypy_kwargs_param_name = None
    gdtr_.stypy_call_defaults = defaults
    gdtr_.stypy_call_varargs = varargs
    gdtr_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gdtr_', ['p', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gdtr_', localization, ['p', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gdtr_(...)' code ##################

    
    # Call to gdtr(...): (line 155)
    # Processing the call arguments (line 155)
    float_534943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'float')
    # Getting the type of 'p' (line 155)
    p_534944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'p', False)
    # Getting the type of 'x' (line 155)
    x_534945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'x', False)
    # Processing the call keyword arguments (line 155)
    kwargs_534946 = {}
    # Getting the type of 'gdtr' (line 155)
    gdtr_534942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'gdtr', False)
    # Calling gdtr(args, kwargs) (line 155)
    gdtr_call_result_534947 = invoke(stypy.reporting.localization.Localization(__file__, 155, 11), gdtr_534942, *[float_534943, p_534944, x_534945], **kwargs_534946)
    
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', gdtr_call_result_534947)
    
    # ################# End of 'gdtr_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gdtr_' in the type store
    # Getting the type of 'stypy_return_type' (line 154)
    stypy_return_type_534948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534948)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gdtr_'
    return stypy_return_type_534948

# Assigning a type to the variable 'gdtr_' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'gdtr_', gdtr_)

@norecursion
def gdtrc_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gdtrc_'
    module_type_store = module_type_store.open_function_context('gdtrc_', 157, 0, False)
    
    # Passed parameters checking function
    gdtrc_.stypy_localization = localization
    gdtrc_.stypy_type_of_self = None
    gdtrc_.stypy_type_store = module_type_store
    gdtrc_.stypy_function_name = 'gdtrc_'
    gdtrc_.stypy_param_names_list = ['p', 'x']
    gdtrc_.stypy_varargs_param_name = None
    gdtrc_.stypy_kwargs_param_name = None
    gdtrc_.stypy_call_defaults = defaults
    gdtrc_.stypy_call_varargs = varargs
    gdtrc_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gdtrc_', ['p', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gdtrc_', localization, ['p', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gdtrc_(...)' code ##################

    
    # Call to gdtrc(...): (line 158)
    # Processing the call arguments (line 158)
    float_534950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 17), 'float')
    # Getting the type of 'p' (line 158)
    p_534951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'p', False)
    # Getting the type of 'x' (line 158)
    x_534952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'x', False)
    # Processing the call keyword arguments (line 158)
    kwargs_534953 = {}
    # Getting the type of 'gdtrc' (line 158)
    gdtrc_534949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'gdtrc', False)
    # Calling gdtrc(args, kwargs) (line 158)
    gdtrc_call_result_534954 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), gdtrc_534949, *[float_534950, p_534951, x_534952], **kwargs_534953)
    
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type', gdtrc_call_result_534954)
    
    # ################# End of 'gdtrc_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gdtrc_' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_534955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534955)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gdtrc_'
    return stypy_return_type_534955

# Assigning a type to the variable 'gdtrc_' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'gdtrc_', gdtrc_)

@norecursion
def gdtrix_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gdtrix_'
    module_type_store = module_type_store.open_function_context('gdtrix_', 160, 0, False)
    
    # Passed parameters checking function
    gdtrix_.stypy_localization = localization
    gdtrix_.stypy_type_of_self = None
    gdtrix_.stypy_type_store = module_type_store
    gdtrix_.stypy_function_name = 'gdtrix_'
    gdtrix_.stypy_param_names_list = ['b', 'p']
    gdtrix_.stypy_varargs_param_name = None
    gdtrix_.stypy_kwargs_param_name = None
    gdtrix_.stypy_call_defaults = defaults
    gdtrix_.stypy_call_varargs = varargs
    gdtrix_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gdtrix_', ['b', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gdtrix_', localization, ['b', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gdtrix_(...)' code ##################

    
    # Call to gdtrix(...): (line 161)
    # Processing the call arguments (line 161)
    float_534957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'float')
    # Getting the type of 'b' (line 161)
    b_534958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'b', False)
    # Getting the type of 'p' (line 161)
    p_534959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'p', False)
    # Processing the call keyword arguments (line 161)
    kwargs_534960 = {}
    # Getting the type of 'gdtrix' (line 161)
    gdtrix_534956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'gdtrix', False)
    # Calling gdtrix(args, kwargs) (line 161)
    gdtrix_call_result_534961 = invoke(stypy.reporting.localization.Localization(__file__, 161, 11), gdtrix_534956, *[float_534957, b_534958, p_534959], **kwargs_534960)
    
    # Assigning a type to the variable 'stypy_return_type' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type', gdtrix_call_result_534961)
    
    # ################# End of 'gdtrix_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gdtrix_' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_534962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534962)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gdtrix_'
    return stypy_return_type_534962

# Assigning a type to the variable 'gdtrix_' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'gdtrix_', gdtrix_)

@norecursion
def gdtrix_comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gdtrix_comp'
    module_type_store = module_type_store.open_function_context('gdtrix_comp', 163, 0, False)
    
    # Passed parameters checking function
    gdtrix_comp.stypy_localization = localization
    gdtrix_comp.stypy_type_of_self = None
    gdtrix_comp.stypy_type_store = module_type_store
    gdtrix_comp.stypy_function_name = 'gdtrix_comp'
    gdtrix_comp.stypy_param_names_list = ['b', 'p']
    gdtrix_comp.stypy_varargs_param_name = None
    gdtrix_comp.stypy_kwargs_param_name = None
    gdtrix_comp.stypy_call_defaults = defaults
    gdtrix_comp.stypy_call_varargs = varargs
    gdtrix_comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gdtrix_comp', ['b', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gdtrix_comp', localization, ['b', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gdtrix_comp(...)' code ##################

    
    # Call to gdtrix(...): (line 164)
    # Processing the call arguments (line 164)
    float_534964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 18), 'float')
    # Getting the type of 'b' (line 164)
    b_534965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'b', False)
    int_534966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 26), 'int')
    # Getting the type of 'p' (line 164)
    p_534967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'p', False)
    # Applying the binary operator '-' (line 164)
    result_sub_534968 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 26), '-', int_534966, p_534967)
    
    # Processing the call keyword arguments (line 164)
    kwargs_534969 = {}
    # Getting the type of 'gdtrix' (line 164)
    gdtrix_534963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'gdtrix', False)
    # Calling gdtrix(args, kwargs) (line 164)
    gdtrix_call_result_534970 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), gdtrix_534963, *[float_534964, b_534965, result_sub_534968], **kwargs_534969)
    
    # Assigning a type to the variable 'stypy_return_type' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type', gdtrix_call_result_534970)
    
    # ################# End of 'gdtrix_comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gdtrix_comp' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_534971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534971)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gdtrix_comp'
    return stypy_return_type_534971

# Assigning a type to the variable 'gdtrix_comp' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'gdtrix_comp', gdtrix_comp)

@norecursion
def gdtrib_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gdtrib_'
    module_type_store = module_type_store.open_function_context('gdtrib_', 166, 0, False)
    
    # Passed parameters checking function
    gdtrib_.stypy_localization = localization
    gdtrib_.stypy_type_of_self = None
    gdtrib_.stypy_type_store = module_type_store
    gdtrib_.stypy_function_name = 'gdtrib_'
    gdtrib_.stypy_param_names_list = ['p', 'x']
    gdtrib_.stypy_varargs_param_name = None
    gdtrib_.stypy_kwargs_param_name = None
    gdtrib_.stypy_call_defaults = defaults
    gdtrib_.stypy_call_varargs = varargs
    gdtrib_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gdtrib_', ['p', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gdtrib_', localization, ['p', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gdtrib_(...)' code ##################

    
    # Call to gdtrib(...): (line 167)
    # Processing the call arguments (line 167)
    float_534973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 18), 'float')
    # Getting the type of 'p' (line 167)
    p_534974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'p', False)
    # Getting the type of 'x' (line 167)
    x_534975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 26), 'x', False)
    # Processing the call keyword arguments (line 167)
    kwargs_534976 = {}
    # Getting the type of 'gdtrib' (line 167)
    gdtrib_534972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'gdtrib', False)
    # Calling gdtrib(args, kwargs) (line 167)
    gdtrib_call_result_534977 = invoke(stypy.reporting.localization.Localization(__file__, 167, 11), gdtrib_534972, *[float_534973, p_534974, x_534975], **kwargs_534976)
    
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', gdtrib_call_result_534977)
    
    # ################# End of 'gdtrib_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gdtrib_' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_534978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534978)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gdtrib_'
    return stypy_return_type_534978

# Assigning a type to the variable 'gdtrib_' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'gdtrib_', gdtrib_)

@norecursion
def gdtrib_comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gdtrib_comp'
    module_type_store = module_type_store.open_function_context('gdtrib_comp', 169, 0, False)
    
    # Passed parameters checking function
    gdtrib_comp.stypy_localization = localization
    gdtrib_comp.stypy_type_of_self = None
    gdtrib_comp.stypy_type_store = module_type_store
    gdtrib_comp.stypy_function_name = 'gdtrib_comp'
    gdtrib_comp.stypy_param_names_list = ['p', 'x']
    gdtrib_comp.stypy_varargs_param_name = None
    gdtrib_comp.stypy_kwargs_param_name = None
    gdtrib_comp.stypy_call_defaults = defaults
    gdtrib_comp.stypy_call_varargs = varargs
    gdtrib_comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gdtrib_comp', ['p', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gdtrib_comp', localization, ['p', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gdtrib_comp(...)' code ##################

    
    # Call to gdtrib(...): (line 170)
    # Processing the call arguments (line 170)
    float_534980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 18), 'float')
    int_534981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'int')
    # Getting the type of 'p' (line 170)
    p_534982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 25), 'p', False)
    # Applying the binary operator '-' (line 170)
    result_sub_534983 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 23), '-', int_534981, p_534982)
    
    # Getting the type of 'x' (line 170)
    x_534984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'x', False)
    # Processing the call keyword arguments (line 170)
    kwargs_534985 = {}
    # Getting the type of 'gdtrib' (line 170)
    gdtrib_534979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'gdtrib', False)
    # Calling gdtrib(args, kwargs) (line 170)
    gdtrib_call_result_534986 = invoke(stypy.reporting.localization.Localization(__file__, 170, 11), gdtrib_534979, *[float_534980, result_sub_534983, x_534984], **kwargs_534985)
    
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type', gdtrib_call_result_534986)
    
    # ################# End of 'gdtrib_comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gdtrib_comp' in the type store
    # Getting the type of 'stypy_return_type' (line 169)
    stypy_return_type_534987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534987)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gdtrib_comp'
    return stypy_return_type_534987

# Assigning a type to the variable 'gdtrib_comp' (line 169)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'gdtrib_comp', gdtrib_comp)

@norecursion
def nbdtrik_comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nbdtrik_comp'
    module_type_store = module_type_store.open_function_context('nbdtrik_comp', 172, 0, False)
    
    # Passed parameters checking function
    nbdtrik_comp.stypy_localization = localization
    nbdtrik_comp.stypy_type_of_self = None
    nbdtrik_comp.stypy_type_store = module_type_store
    nbdtrik_comp.stypy_function_name = 'nbdtrik_comp'
    nbdtrik_comp.stypy_param_names_list = ['y', 'n', 'p']
    nbdtrik_comp.stypy_varargs_param_name = None
    nbdtrik_comp.stypy_kwargs_param_name = None
    nbdtrik_comp.stypy_call_defaults = defaults
    nbdtrik_comp.stypy_call_varargs = varargs
    nbdtrik_comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nbdtrik_comp', ['y', 'n', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nbdtrik_comp', localization, ['y', 'n', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nbdtrik_comp(...)' code ##################

    
    # Call to nbdtrik(...): (line 173)
    # Processing the call arguments (line 173)
    int_534989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 19), 'int')
    # Getting the type of 'y' (line 173)
    y_534990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'y', False)
    # Applying the binary operator '-' (line 173)
    result_sub_534991 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 19), '-', int_534989, y_534990)
    
    # Getting the type of 'n' (line 173)
    n_534992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'n', False)
    # Getting the type of 'p' (line 173)
    p_534993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'p', False)
    # Processing the call keyword arguments (line 173)
    kwargs_534994 = {}
    # Getting the type of 'nbdtrik' (line 173)
    nbdtrik_534988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'nbdtrik', False)
    # Calling nbdtrik(args, kwargs) (line 173)
    nbdtrik_call_result_534995 = invoke(stypy.reporting.localization.Localization(__file__, 173, 11), nbdtrik_534988, *[result_sub_534991, n_534992, p_534993], **kwargs_534994)
    
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type', nbdtrik_call_result_534995)
    
    # ################# End of 'nbdtrik_comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nbdtrik_comp' in the type store
    # Getting the type of 'stypy_return_type' (line 172)
    stypy_return_type_534996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_534996)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nbdtrik_comp'
    return stypy_return_type_534996

# Assigning a type to the variable 'nbdtrik_comp' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'nbdtrik_comp', nbdtrik_comp)

@norecursion
def pdtrik_comp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pdtrik_comp'
    module_type_store = module_type_store.open_function_context('pdtrik_comp', 175, 0, False)
    
    # Passed parameters checking function
    pdtrik_comp.stypy_localization = localization
    pdtrik_comp.stypy_type_of_self = None
    pdtrik_comp.stypy_type_store = module_type_store
    pdtrik_comp.stypy_function_name = 'pdtrik_comp'
    pdtrik_comp.stypy_param_names_list = ['p', 'm']
    pdtrik_comp.stypy_varargs_param_name = None
    pdtrik_comp.stypy_kwargs_param_name = None
    pdtrik_comp.stypy_call_defaults = defaults
    pdtrik_comp.stypy_call_varargs = varargs
    pdtrik_comp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pdtrik_comp', ['p', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pdtrik_comp', localization, ['p', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pdtrik_comp(...)' code ##################

    
    # Call to pdtrik(...): (line 176)
    # Processing the call arguments (line 176)
    int_534998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'int')
    # Getting the type of 'p' (line 176)
    p_534999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'p', False)
    # Applying the binary operator '-' (line 176)
    result_sub_535000 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 18), '-', int_534998, p_534999)
    
    # Getting the type of 'm' (line 176)
    m_535001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'm', False)
    # Processing the call keyword arguments (line 176)
    kwargs_535002 = {}
    # Getting the type of 'pdtrik' (line 176)
    pdtrik_534997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'pdtrik', False)
    # Calling pdtrik(args, kwargs) (line 176)
    pdtrik_call_result_535003 = invoke(stypy.reporting.localization.Localization(__file__, 176, 11), pdtrik_534997, *[result_sub_535000, m_535001], **kwargs_535002)
    
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type', pdtrik_call_result_535003)
    
    # ################# End of 'pdtrik_comp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pdtrik_comp' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_535004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_535004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pdtrik_comp'
    return stypy_return_type_535004

# Assigning a type to the variable 'pdtrik_comp' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'pdtrik_comp', pdtrik_comp)

@norecursion
def poch_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'poch_'
    module_type_store = module_type_store.open_function_context('poch_', 178, 0, False)
    
    # Passed parameters checking function
    poch_.stypy_localization = localization
    poch_.stypy_type_of_self = None
    poch_.stypy_type_store = module_type_store
    poch_.stypy_function_name = 'poch_'
    poch_.stypy_param_names_list = ['z', 'm']
    poch_.stypy_varargs_param_name = None
    poch_.stypy_kwargs_param_name = None
    poch_.stypy_call_defaults = defaults
    poch_.stypy_call_varargs = varargs
    poch_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'poch_', ['z', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'poch_', localization, ['z', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'poch_(...)' code ##################

    float_535005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 11), 'float')
    
    # Call to poch(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'z' (line 179)
    z_535007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'z', False)
    # Getting the type of 'm' (line 179)
    m_535008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'm', False)
    # Processing the call keyword arguments (line 179)
    kwargs_535009 = {}
    # Getting the type of 'poch' (line 179)
    poch_535006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'poch', False)
    # Calling poch(args, kwargs) (line 179)
    poch_call_result_535010 = invoke(stypy.reporting.localization.Localization(__file__, 179, 17), poch_535006, *[z_535007, m_535008], **kwargs_535009)
    
    # Applying the binary operator 'div' (line 179)
    result_div_535011 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 11), 'div', float_535005, poch_call_result_535010)
    
    # Assigning a type to the variable 'stypy_return_type' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type', result_div_535011)
    
    # ################# End of 'poch_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'poch_' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_535012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_535012)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'poch_'
    return stypy_return_type_535012

# Assigning a type to the variable 'poch_' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'poch_', poch_)

@norecursion
def poch_minus(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'poch_minus'
    module_type_store = module_type_store.open_function_context('poch_minus', 181, 0, False)
    
    # Passed parameters checking function
    poch_minus.stypy_localization = localization
    poch_minus.stypy_type_of_self = None
    poch_minus.stypy_type_store = module_type_store
    poch_minus.stypy_function_name = 'poch_minus'
    poch_minus.stypy_param_names_list = ['z', 'm']
    poch_minus.stypy_varargs_param_name = None
    poch_minus.stypy_kwargs_param_name = None
    poch_minus.stypy_call_defaults = defaults
    poch_minus.stypy_call_varargs = varargs
    poch_minus.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'poch_minus', ['z', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'poch_minus', localization, ['z', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'poch_minus(...)' code ##################

    float_535013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 11), 'float')
    
    # Call to poch(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'z' (line 182)
    z_535015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'z', False)
    
    # Getting the type of 'm' (line 182)
    m_535016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'm', False)
    # Applying the 'usub' unary operator (line 182)
    result___neg___535017 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 25), 'usub', m_535016)
    
    # Processing the call keyword arguments (line 182)
    kwargs_535018 = {}
    # Getting the type of 'poch' (line 182)
    poch_535014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 17), 'poch', False)
    # Calling poch(args, kwargs) (line 182)
    poch_call_result_535019 = invoke(stypy.reporting.localization.Localization(__file__, 182, 17), poch_535014, *[z_535015, result___neg___535017], **kwargs_535018)
    
    # Applying the binary operator 'div' (line 182)
    result_div_535020 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 11), 'div', float_535013, poch_call_result_535019)
    
    # Assigning a type to the variable 'stypy_return_type' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type', result_div_535020)
    
    # ################# End of 'poch_minus(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'poch_minus' in the type store
    # Getting the type of 'stypy_return_type' (line 181)
    stypy_return_type_535021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_535021)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'poch_minus'
    return stypy_return_type_535021

# Assigning a type to the variable 'poch_minus' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'poch_minus', poch_minus)

@norecursion
def spherical_jn_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'spherical_jn_'
    module_type_store = module_type_store.open_function_context('spherical_jn_', 184, 0, False)
    
    # Passed parameters checking function
    spherical_jn_.stypy_localization = localization
    spherical_jn_.stypy_type_of_self = None
    spherical_jn_.stypy_type_store = module_type_store
    spherical_jn_.stypy_function_name = 'spherical_jn_'
    spherical_jn_.stypy_param_names_list = ['n', 'x']
    spherical_jn_.stypy_varargs_param_name = None
    spherical_jn_.stypy_kwargs_param_name = None
    spherical_jn_.stypy_call_defaults = defaults
    spherical_jn_.stypy_call_varargs = varargs
    spherical_jn_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spherical_jn_', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spherical_jn_', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spherical_jn_(...)' code ##################

    
    # Call to spherical_jn(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Call to astype(...): (line 185)
    # Processing the call arguments (line 185)
    str_535025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 33), 'str', 'l')
    # Processing the call keyword arguments (line 185)
    kwargs_535026 = {}
    # Getting the type of 'n' (line 185)
    n_535023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'n', False)
    # Obtaining the member 'astype' of a type (line 185)
    astype_535024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 24), n_535023, 'astype')
    # Calling astype(args, kwargs) (line 185)
    astype_call_result_535027 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), astype_535024, *[str_535025], **kwargs_535026)
    
    # Getting the type of 'x' (line 185)
    x_535028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 39), 'x', False)
    # Processing the call keyword arguments (line 185)
    kwargs_535029 = {}
    # Getting the type of 'spherical_jn' (line 185)
    spherical_jn_535022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'spherical_jn', False)
    # Calling spherical_jn(args, kwargs) (line 185)
    spherical_jn_call_result_535030 = invoke(stypy.reporting.localization.Localization(__file__, 185, 11), spherical_jn_535022, *[astype_call_result_535027, x_535028], **kwargs_535029)
    
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type', spherical_jn_call_result_535030)
    
    # ################# End of 'spherical_jn_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spherical_jn_' in the type store
    # Getting the type of 'stypy_return_type' (line 184)
    stypy_return_type_535031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_535031)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spherical_jn_'
    return stypy_return_type_535031

# Assigning a type to the variable 'spherical_jn_' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'spherical_jn_', spherical_jn_)

@norecursion
def spherical_yn_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'spherical_yn_'
    module_type_store = module_type_store.open_function_context('spherical_yn_', 187, 0, False)
    
    # Passed parameters checking function
    spherical_yn_.stypy_localization = localization
    spherical_yn_.stypy_type_of_self = None
    spherical_yn_.stypy_type_store = module_type_store
    spherical_yn_.stypy_function_name = 'spherical_yn_'
    spherical_yn_.stypy_param_names_list = ['n', 'x']
    spherical_yn_.stypy_varargs_param_name = None
    spherical_yn_.stypy_kwargs_param_name = None
    spherical_yn_.stypy_call_defaults = defaults
    spherical_yn_.stypy_call_varargs = varargs
    spherical_yn_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spherical_yn_', ['n', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spherical_yn_', localization, ['n', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spherical_yn_(...)' code ##################

    
    # Call to spherical_yn(...): (line 188)
    # Processing the call arguments (line 188)
    
    # Call to astype(...): (line 188)
    # Processing the call arguments (line 188)
    str_535035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 33), 'str', 'l')
    # Processing the call keyword arguments (line 188)
    kwargs_535036 = {}
    # Getting the type of 'n' (line 188)
    n_535033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'n', False)
    # Obtaining the member 'astype' of a type (line 188)
    astype_535034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), n_535033, 'astype')
    # Calling astype(args, kwargs) (line 188)
    astype_call_result_535037 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), astype_535034, *[str_535035], **kwargs_535036)
    
    # Getting the type of 'x' (line 188)
    x_535038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 39), 'x', False)
    # Processing the call keyword arguments (line 188)
    kwargs_535039 = {}
    # Getting the type of 'spherical_yn' (line 188)
    spherical_yn_535032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'spherical_yn', False)
    # Calling spherical_yn(args, kwargs) (line 188)
    spherical_yn_call_result_535040 = invoke(stypy.reporting.localization.Localization(__file__, 188, 11), spherical_yn_535032, *[astype_call_result_535037, x_535038], **kwargs_535039)
    
    # Assigning a type to the variable 'stypy_return_type' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type', spherical_yn_call_result_535040)
    
    # ################# End of 'spherical_yn_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spherical_yn_' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_535041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_535041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spherical_yn_'
    return stypy_return_type_535041

# Assigning a type to the variable 'spherical_yn_' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'spherical_yn_', spherical_yn_)

@norecursion
def sph_harm_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sph_harm_'
    module_type_store = module_type_store.open_function_context('sph_harm_', 190, 0, False)
    
    # Passed parameters checking function
    sph_harm_.stypy_localization = localization
    sph_harm_.stypy_type_of_self = None
    sph_harm_.stypy_type_store = module_type_store
    sph_harm_.stypy_function_name = 'sph_harm_'
    sph_harm_.stypy_param_names_list = ['m', 'n', 'theta', 'phi']
    sph_harm_.stypy_varargs_param_name = None
    sph_harm_.stypy_kwargs_param_name = None
    sph_harm_.stypy_call_defaults = defaults
    sph_harm_.stypy_call_varargs = varargs
    sph_harm_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sph_harm_', ['m', 'n', 'theta', 'phi'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sph_harm_', localization, ['m', 'n', 'theta', 'phi'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sph_harm_(...)' code ##################

    
    # Assigning a Call to a Name (line 191):
    
    # Call to sph_harm(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'm' (line 191)
    m_535043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 17), 'm', False)
    # Getting the type of 'n' (line 191)
    n_535044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'n', False)
    # Getting the type of 'theta' (line 191)
    theta_535045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'theta', False)
    # Getting the type of 'phi' (line 191)
    phi_535046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'phi', False)
    # Processing the call keyword arguments (line 191)
    kwargs_535047 = {}
    # Getting the type of 'sph_harm' (line 191)
    sph_harm_535042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'sph_harm', False)
    # Calling sph_harm(args, kwargs) (line 191)
    sph_harm_call_result_535048 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), sph_harm_535042, *[m_535043, n_535044, theta_535045, phi_535046], **kwargs_535047)
    
    # Assigning a type to the variable 'y' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'y', sph_harm_call_result_535048)
    
    # Obtaining an instance of the builtin type 'tuple' (line 192)
    tuple_535049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 192)
    # Adding element type (line 192)
    # Getting the type of 'y' (line 192)
    y_535050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'y')
    # Obtaining the member 'real' of a type (line 192)
    real_535051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), y_535050, 'real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 12), tuple_535049, real_535051)
    # Adding element type (line 192)
    # Getting the type of 'y' (line 192)
    y_535052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'y')
    # Obtaining the member 'imag' of a type (line 192)
    imag_535053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 20), y_535052, 'imag')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 12), tuple_535049, imag_535053)
    
    # Assigning a type to the variable 'stypy_return_type' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type', tuple_535049)
    
    # ################# End of 'sph_harm_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sph_harm_' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_535054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_535054)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sph_harm_'
    return stypy_return_type_535054

# Assigning a type to the variable 'sph_harm_' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'sph_harm_', sph_harm_)

@norecursion
def cexpm1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cexpm1'
    module_type_store = module_type_store.open_function_context('cexpm1', 194, 0, False)
    
    # Passed parameters checking function
    cexpm1.stypy_localization = localization
    cexpm1.stypy_type_of_self = None
    cexpm1.stypy_type_store = module_type_store
    cexpm1.stypy_function_name = 'cexpm1'
    cexpm1.stypy_param_names_list = ['x', 'y']
    cexpm1.stypy_varargs_param_name = None
    cexpm1.stypy_kwargs_param_name = None
    cexpm1.stypy_call_defaults = defaults
    cexpm1.stypy_call_varargs = varargs
    cexpm1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cexpm1', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cexpm1', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cexpm1(...)' code ##################

    
    # Assigning a Call to a Name (line 195):
    
    # Call to expm1(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'x' (line 195)
    x_535056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'x', False)
    complex_535057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'complex')
    # Getting the type of 'y' (line 195)
    y_535058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 21), 'y', False)
    # Applying the binary operator '*' (line 195)
    result_mul_535059 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 18), '*', complex_535057, y_535058)
    
    # Applying the binary operator '+' (line 195)
    result_add_535060 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 14), '+', x_535056, result_mul_535059)
    
    # Processing the call keyword arguments (line 195)
    kwargs_535061 = {}
    # Getting the type of 'expm1' (line 195)
    expm1_535055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'expm1', False)
    # Calling expm1(args, kwargs) (line 195)
    expm1_call_result_535062 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), expm1_535055, *[result_add_535060], **kwargs_535061)
    
    # Assigning a type to the variable 'z' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'z', expm1_call_result_535062)
    
    # Obtaining an instance of the builtin type 'tuple' (line 196)
    tuple_535063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 196)
    # Adding element type (line 196)
    # Getting the type of 'z' (line 196)
    z_535064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'z')
    # Obtaining the member 'real' of a type (line 196)
    real_535065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 11), z_535064, 'real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 11), tuple_535063, real_535065)
    # Adding element type (line 196)
    # Getting the type of 'z' (line 196)
    z_535066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'z')
    # Obtaining the member 'imag' of a type (line 196)
    imag_535067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 19), z_535066, 'imag')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 11), tuple_535063, imag_535067)
    
    # Assigning a type to the variable 'stypy_return_type' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type', tuple_535063)
    
    # ################# End of 'cexpm1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cexpm1' in the type store
    # Getting the type of 'stypy_return_type' (line 194)
    stypy_return_type_535068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_535068)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cexpm1'
    return stypy_return_type_535068

# Assigning a type to the variable 'cexpm1' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'cexpm1', cexpm1)

@norecursion
def clog1p(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'clog1p'
    module_type_store = module_type_store.open_function_context('clog1p', 198, 0, False)
    
    # Passed parameters checking function
    clog1p.stypy_localization = localization
    clog1p.stypy_type_of_self = None
    clog1p.stypy_type_store = module_type_store
    clog1p.stypy_function_name = 'clog1p'
    clog1p.stypy_param_names_list = ['x', 'y']
    clog1p.stypy_varargs_param_name = None
    clog1p.stypy_kwargs_param_name = None
    clog1p.stypy_call_defaults = defaults
    clog1p.stypy_call_varargs = varargs
    clog1p.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'clog1p', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'clog1p', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'clog1p(...)' code ##################

    
    # Assigning a Call to a Name (line 199):
    
    # Call to log1p(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'x' (line 199)
    x_535070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 14), 'x', False)
    complex_535071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 18), 'complex')
    # Getting the type of 'y' (line 199)
    y_535072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'y', False)
    # Applying the binary operator '*' (line 199)
    result_mul_535073 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 18), '*', complex_535071, y_535072)
    
    # Applying the binary operator '+' (line 199)
    result_add_535074 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 14), '+', x_535070, result_mul_535073)
    
    # Processing the call keyword arguments (line 199)
    kwargs_535075 = {}
    # Getting the type of 'log1p' (line 199)
    log1p_535069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'log1p', False)
    # Calling log1p(args, kwargs) (line 199)
    log1p_call_result_535076 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), log1p_535069, *[result_add_535074], **kwargs_535075)
    
    # Assigning a type to the variable 'z' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'z', log1p_call_result_535076)
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_535077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    # Getting the type of 'z' (line 200)
    z_535078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'z')
    # Obtaining the member 'real' of a type (line 200)
    real_535079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), z_535078, 'real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 11), tuple_535077, real_535079)
    # Adding element type (line 200)
    # Getting the type of 'z' (line 200)
    z_535080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'z')
    # Obtaining the member 'imag' of a type (line 200)
    imag_535081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 19), z_535080, 'imag')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 11), tuple_535077, imag_535081)
    
    # Assigning a type to the variable 'stypy_return_type' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type', tuple_535077)
    
    # ################# End of 'clog1p(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'clog1p' in the type store
    # Getting the type of 'stypy_return_type' (line 198)
    stypy_return_type_535082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_535082)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'clog1p'
    return stypy_return_type_535082

# Assigning a type to the variable 'clog1p' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'clog1p', clog1p)

# Assigning a List to a Name (line 202):

# Obtaining an instance of the builtin type 'list' (line 202)
list_535083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 202)
# Adding element type (line 202)

# Call to data(...): (line 203)
# Processing the call arguments (line 203)
# Getting the type of 'arccosh' (line 203)
arccosh_535085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 13), 'arccosh', False)
str_535086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 22), 'str', 'acosh_data_ipp-acosh_data')
int_535087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 51), 'int')
int_535088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 54), 'int')
# Processing the call keyword arguments (line 203)
float_535089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 62), 'float')
keyword_535090 = float_535089
kwargs_535091 = {'rtol': keyword_535090}
# Getting the type of 'data' (line 203)
data_535084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'data', False)
# Calling data(args, kwargs) (line 203)
data_call_result_535092 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), data_535084, *[arccosh_535085, str_535086, int_535087, int_535088], **kwargs_535091)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535092)
# Adding element type (line 202)

# Call to data(...): (line 204)
# Processing the call arguments (line 204)
# Getting the type of 'arccosh' (line 204)
arccosh_535094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 13), 'arccosh', False)
str_535095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 22), 'str', 'acosh_data_ipp-acosh_data')
complex_535096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 51), 'complex')
int_535097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 55), 'int')
# Processing the call keyword arguments (line 204)
float_535098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 63), 'float')
keyword_535099 = float_535098
kwargs_535100 = {'rtol': keyword_535099}
# Getting the type of 'data' (line 204)
data_535093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'data', False)
# Calling data(args, kwargs) (line 204)
data_call_result_535101 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), data_535093, *[arccosh_535094, str_535095, complex_535096, int_535097], **kwargs_535100)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535101)
# Adding element type (line 202)

# Call to data(...): (line 206)
# Processing the call arguments (line 206)
# Getting the type of 'arcsinh' (line 206)
arcsinh_535103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'arcsinh', False)
str_535104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 22), 'str', 'asinh_data_ipp-asinh_data')
int_535105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 51), 'int')
int_535106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 54), 'int')
# Processing the call keyword arguments (line 206)
float_535107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 62), 'float')
keyword_535108 = float_535107
kwargs_535109 = {'rtol': keyword_535108}
# Getting the type of 'data' (line 206)
data_535102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'data', False)
# Calling data(args, kwargs) (line 206)
data_call_result_535110 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), data_535102, *[arcsinh_535103, str_535104, int_535105, int_535106], **kwargs_535109)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535110)
# Adding element type (line 202)

# Call to data(...): (line 207)
# Processing the call arguments (line 207)
# Getting the type of 'arcsinh' (line 207)
arcsinh_535112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'arcsinh', False)
str_535113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 22), 'str', 'asinh_data_ipp-asinh_data')
complex_535114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 51), 'complex')
int_535115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 55), 'int')
# Processing the call keyword arguments (line 207)
float_535116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 63), 'float')
keyword_535117 = float_535116
kwargs_535118 = {'rtol': keyword_535117}
# Getting the type of 'data' (line 207)
data_535111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'data', False)
# Calling data(args, kwargs) (line 207)
data_call_result_535119 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), data_535111, *[arcsinh_535112, str_535113, complex_535114, int_535115], **kwargs_535118)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535119)
# Adding element type (line 202)

# Call to data(...): (line 209)
# Processing the call arguments (line 209)
# Getting the type of 'arctanh' (line 209)
arctanh_535121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'arctanh', False)
str_535122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'str', 'atanh_data_ipp-atanh_data')
int_535123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 51), 'int')
int_535124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 54), 'int')
# Processing the call keyword arguments (line 209)
float_535125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 62), 'float')
keyword_535126 = float_535125
kwargs_535127 = {'rtol': keyword_535126}
# Getting the type of 'data' (line 209)
data_535120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'data', False)
# Calling data(args, kwargs) (line 209)
data_call_result_535128 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), data_535120, *[arctanh_535121, str_535122, int_535123, int_535124], **kwargs_535127)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535128)
# Adding element type (line 202)

# Call to data(...): (line 210)
# Processing the call arguments (line 210)
# Getting the type of 'arctanh' (line 210)
arctanh_535130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'arctanh', False)
str_535131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'str', 'atanh_data_ipp-atanh_data')
complex_535132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 51), 'complex')
int_535133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 55), 'int')
# Processing the call keyword arguments (line 210)
float_535134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 63), 'float')
keyword_535135 = float_535134
kwargs_535136 = {'rtol': keyword_535135}
# Getting the type of 'data' (line 210)
data_535129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'data', False)
# Calling data(args, kwargs) (line 210)
data_call_result_535137 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), data_535129, *[arctanh_535130, str_535131, complex_535132, int_535133], **kwargs_535136)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535137)
# Adding element type (line 202)

# Call to data(...): (line 212)
# Processing the call arguments (line 212)
# Getting the type of 'assoc_legendre_p_boost_' (line 212)
assoc_legendre_p_boost__535139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'assoc_legendre_p_boost_', False)
str_535140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 38), 'str', 'assoc_legendre_p_ipp-assoc_legendre_p')

# Obtaining an instance of the builtin type 'tuple' (line 212)
tuple_535141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 80), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 212)
# Adding element type (line 212)
int_535142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 80), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 80), tuple_535141, int_535142)
# Adding element type (line 212)
int_535143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 82), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 80), tuple_535141, int_535143)
# Adding element type (line 212)
int_535144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 84), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 80), tuple_535141, int_535144)

int_535145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 88), 'int')
# Processing the call keyword arguments (line 212)
float_535146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 96), 'float')
keyword_535147 = float_535146
kwargs_535148 = {'rtol': keyword_535147}
# Getting the type of 'data' (line 212)
data_535138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'data', False)
# Calling data(args, kwargs) (line 212)
data_call_result_535149 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), data_535138, *[assoc_legendre_p_boost__535139, str_535140, tuple_535141, int_535145], **kwargs_535148)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535149)
# Adding element type (line 202)

# Call to data(...): (line 214)
# Processing the call arguments (line 214)
# Getting the type of 'legendre_p_via_assoc_' (line 214)
legendre_p_via_assoc__535151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 13), 'legendre_p_via_assoc_', False)
str_535152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 36), 'str', 'legendre_p_ipp-legendre_p')

# Obtaining an instance of the builtin type 'tuple' (line 214)
tuple_535153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 66), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 214)
# Adding element type (line 214)
int_535154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 66), tuple_535153, int_535154)
# Adding element type (line 214)
int_535155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 66), tuple_535153, int_535155)

int_535156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 72), 'int')
# Processing the call keyword arguments (line 214)
float_535157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 80), 'float')
keyword_535158 = float_535157
kwargs_535159 = {'rtol': keyword_535158}
# Getting the type of 'data' (line 214)
data_535150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'data', False)
# Calling data(args, kwargs) (line 214)
data_call_result_535160 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), data_535150, *[legendre_p_via_assoc__535151, str_535152, tuple_535153, int_535156], **kwargs_535159)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535160)
# Adding element type (line 202)

# Call to data(...): (line 215)
# Processing the call arguments (line 215)
# Getting the type of 'legendre_p_via_assoc_' (line 215)
legendre_p_via_assoc__535162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'legendre_p_via_assoc_', False)
str_535163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 36), 'str', 'legendre_p_large_ipp-legendre_p_large')

# Obtaining an instance of the builtin type 'tuple' (line 215)
tuple_535164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 78), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 215)
# Adding element type (line 215)
int_535165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 78), tuple_535164, int_535165)
# Adding element type (line 215)
int_535166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 80), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 78), tuple_535164, int_535166)

int_535167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 84), 'int')
# Processing the call keyword arguments (line 215)
float_535168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 92), 'float')
keyword_535169 = float_535168
kwargs_535170 = {'rtol': keyword_535169}
# Getting the type of 'data' (line 215)
data_535161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'data', False)
# Calling data(args, kwargs) (line 215)
data_call_result_535171 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), data_535161, *[legendre_p_via_assoc__535162, str_535163, tuple_535164, int_535167], **kwargs_535170)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535171)
# Adding element type (line 202)

# Call to data(...): (line 216)
# Processing the call arguments (line 216)
# Getting the type of 'legendre_p_via_lpmn' (line 216)
legendre_p_via_lpmn_535173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'legendre_p_via_lpmn', False)
str_535174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 34), 'str', 'legendre_p_ipp-legendre_p')

# Obtaining an instance of the builtin type 'tuple' (line 216)
tuple_535175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 216)
# Adding element type (line 216)
int_535176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 64), tuple_535175, int_535176)
# Adding element type (line 216)
int_535177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 64), tuple_535175, int_535177)

int_535178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 70), 'int')
# Processing the call keyword arguments (line 216)
float_535179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 78), 'float')
keyword_535180 = float_535179
# Getting the type of 'False' (line 216)
False_535181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 96), 'False', False)
keyword_535182 = False_535181
kwargs_535183 = {'vectorized': keyword_535182, 'rtol': keyword_535180}
# Getting the type of 'data' (line 216)
data_535172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'data', False)
# Calling data(args, kwargs) (line 216)
data_call_result_535184 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), data_535172, *[legendre_p_via_lpmn_535173, str_535174, tuple_535175, int_535178], **kwargs_535183)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535184)
# Adding element type (line 202)

# Call to data(...): (line 217)
# Processing the call arguments (line 217)
# Getting the type of 'legendre_p_via_lpmn' (line 217)
legendre_p_via_lpmn_535186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'legendre_p_via_lpmn', False)
str_535187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'str', 'legendre_p_large_ipp-legendre_p_large')

# Obtaining an instance of the builtin type 'tuple' (line 217)
tuple_535188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 76), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 217)
# Adding element type (line 217)
int_535189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 76), tuple_535188, int_535189)
# Adding element type (line 217)
int_535190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 76), tuple_535188, int_535190)

int_535191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 82), 'int')
# Processing the call keyword arguments (line 217)
float_535192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 90), 'float')
keyword_535193 = float_535192
# Getting the type of 'False' (line 217)
False_535194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 108), 'False', False)
keyword_535195 = False_535194
kwargs_535196 = {'vectorized': keyword_535195, 'rtol': keyword_535193}
# Getting the type of 'data' (line 217)
data_535185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'data', False)
# Calling data(args, kwargs) (line 217)
data_call_result_535197 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), data_535185, *[legendre_p_via_lpmn_535186, str_535187, tuple_535188, int_535191], **kwargs_535196)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535197)
# Adding element type (line 202)

# Call to data(...): (line 218)
# Processing the call arguments (line 218)
# Getting the type of 'lpn_' (line 218)
lpn__535199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 13), 'lpn_', False)
str_535200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 19), 'str', 'legendre_p_ipp-legendre_p')

# Obtaining an instance of the builtin type 'tuple' (line 218)
tuple_535201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 49), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 218)
# Adding element type (line 218)
int_535202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 49), tuple_535201, int_535202)
# Adding element type (line 218)
int_535203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 49), tuple_535201, int_535203)

int_535204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 55), 'int')
# Processing the call keyword arguments (line 218)
float_535205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 63), 'float')
keyword_535206 = float_535205
# Getting the type of 'False' (line 218)
False_535207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 81), 'False', False)
keyword_535208 = False_535207
kwargs_535209 = {'vectorized': keyword_535208, 'rtol': keyword_535206}
# Getting the type of 'data' (line 218)
data_535198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'data', False)
# Calling data(args, kwargs) (line 218)
data_call_result_535210 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), data_535198, *[lpn__535199, str_535200, tuple_535201, int_535204], **kwargs_535209)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535210)
# Adding element type (line 202)

# Call to data(...): (line 219)
# Processing the call arguments (line 219)
# Getting the type of 'lpn_' (line 219)
lpn__535212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 13), 'lpn_', False)
str_535213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 19), 'str', 'legendre_p_large_ipp-legendre_p_large')

# Obtaining an instance of the builtin type 'tuple' (line 219)
tuple_535214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 219)
# Adding element type (line 219)
int_535215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 61), tuple_535214, int_535215)
# Adding element type (line 219)
int_535216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 61), tuple_535214, int_535216)

int_535217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 67), 'int')
# Processing the call keyword arguments (line 219)
float_535218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 75), 'float')
keyword_535219 = float_535218
# Getting the type of 'False' (line 219)
False_535220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 93), 'False', False)
keyword_535221 = False_535220
kwargs_535222 = {'vectorized': keyword_535221, 'rtol': keyword_535219}
# Getting the type of 'data' (line 219)
data_535211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'data', False)
# Calling data(args, kwargs) (line 219)
data_call_result_535223 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), data_535211, *[lpn__535212, str_535213, tuple_535214, int_535217], **kwargs_535222)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535223)
# Adding element type (line 202)

# Call to data(...): (line 220)
# Processing the call arguments (line 220)
# Getting the type of 'eval_legendre_ld' (line 220)
eval_legendre_ld_535225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 13), 'eval_legendre_ld', False)
str_535226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 31), 'str', 'legendre_p_ipp-legendre_p')

# Obtaining an instance of the builtin type 'tuple' (line 220)
tuple_535227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 220)
# Adding element type (line 220)
int_535228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 61), tuple_535227, int_535228)
# Adding element type (line 220)
int_535229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 61), tuple_535227, int_535229)

int_535230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 67), 'int')
# Processing the call keyword arguments (line 220)
float_535231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 75), 'float')
keyword_535232 = float_535231
kwargs_535233 = {'rtol': keyword_535232}
# Getting the type of 'data' (line 220)
data_535224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'data', False)
# Calling data(args, kwargs) (line 220)
data_call_result_535234 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), data_535224, *[eval_legendre_ld_535225, str_535226, tuple_535227, int_535230], **kwargs_535233)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535234)
# Adding element type (line 202)

# Call to data(...): (line 221)
# Processing the call arguments (line 221)
# Getting the type of 'eval_legendre_ld' (line 221)
eval_legendre_ld_535236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 13), 'eval_legendre_ld', False)
str_535237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 31), 'str', 'legendre_p_large_ipp-legendre_p_large')

# Obtaining an instance of the builtin type 'tuple' (line 221)
tuple_535238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 73), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 221)
# Adding element type (line 221)
int_535239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 73), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 73), tuple_535238, int_535239)
# Adding element type (line 221)
int_535240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 73), tuple_535238, int_535240)

int_535241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 79), 'int')
# Processing the call keyword arguments (line 221)
float_535242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 87), 'float')
keyword_535243 = float_535242
kwargs_535244 = {'rtol': keyword_535243}
# Getting the type of 'data' (line 221)
data_535235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'data', False)
# Calling data(args, kwargs) (line 221)
data_call_result_535245 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), data_535235, *[eval_legendre_ld_535236, str_535237, tuple_535238, int_535241], **kwargs_535244)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535245)
# Adding element type (line 202)

# Call to data(...): (line 222)
# Processing the call arguments (line 222)
# Getting the type of 'eval_legendre_dd' (line 222)
eval_legendre_dd_535247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 13), 'eval_legendre_dd', False)
str_535248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 31), 'str', 'legendre_p_ipp-legendre_p')

# Obtaining an instance of the builtin type 'tuple' (line 222)
tuple_535249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 222)
# Adding element type (line 222)
int_535250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 61), tuple_535249, int_535250)
# Adding element type (line 222)
int_535251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 61), tuple_535249, int_535251)

int_535252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 67), 'int')
# Processing the call keyword arguments (line 222)
float_535253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 75), 'float')
keyword_535254 = float_535253
kwargs_535255 = {'rtol': keyword_535254}
# Getting the type of 'data' (line 222)
data_535246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'data', False)
# Calling data(args, kwargs) (line 222)
data_call_result_535256 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), data_535246, *[eval_legendre_dd_535247, str_535248, tuple_535249, int_535252], **kwargs_535255)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535256)
# Adding element type (line 202)

# Call to data(...): (line 223)
# Processing the call arguments (line 223)
# Getting the type of 'eval_legendre_dd' (line 223)
eval_legendre_dd_535258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 13), 'eval_legendre_dd', False)
str_535259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 31), 'str', 'legendre_p_large_ipp-legendre_p_large')

# Obtaining an instance of the builtin type 'tuple' (line 223)
tuple_535260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 73), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 223)
# Adding element type (line 223)
int_535261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 73), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 73), tuple_535260, int_535261)
# Adding element type (line 223)
int_535262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 73), tuple_535260, int_535262)

int_535263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 79), 'int')
# Processing the call keyword arguments (line 223)
float_535264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 87), 'float')
keyword_535265 = float_535264
kwargs_535266 = {'rtol': keyword_535265}
# Getting the type of 'data' (line 223)
data_535257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'data', False)
# Calling data(args, kwargs) (line 223)
data_call_result_535267 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), data_535257, *[eval_legendre_dd_535258, str_535259, tuple_535260, int_535263], **kwargs_535266)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535267)
# Adding element type (line 202)

# Call to data(...): (line 225)
# Processing the call arguments (line 225)
# Getting the type of 'lqn_' (line 225)
lqn__535269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 13), 'lqn_', False)
str_535270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 19), 'str', 'legendre_p_ipp-legendre_p')

# Obtaining an instance of the builtin type 'tuple' (line 225)
tuple_535271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 49), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 225)
# Adding element type (line 225)
int_535272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 49), tuple_535271, int_535272)
# Adding element type (line 225)
int_535273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 49), tuple_535271, int_535273)

int_535274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 55), 'int')
# Processing the call keyword arguments (line 225)
float_535275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 63), 'float')
keyword_535276 = float_535275
# Getting the type of 'False' (line 225)
False_535277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 81), 'False', False)
keyword_535278 = False_535277
kwargs_535279 = {'vectorized': keyword_535278, 'rtol': keyword_535276}
# Getting the type of 'data' (line 225)
data_535268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'data', False)
# Calling data(args, kwargs) (line 225)
data_call_result_535280 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), data_535268, *[lqn__535269, str_535270, tuple_535271, int_535274], **kwargs_535279)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535280)
# Adding element type (line 202)

# Call to data(...): (line 226)
# Processing the call arguments (line 226)
# Getting the type of 'lqn_' (line 226)
lqn__535282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 13), 'lqn_', False)
str_535283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 19), 'str', 'legendre_p_large_ipp-legendre_p_large')

# Obtaining an instance of the builtin type 'tuple' (line 226)
tuple_535284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 226)
# Adding element type (line 226)
int_535285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 61), tuple_535284, int_535285)
# Adding element type (line 226)
int_535286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 61), tuple_535284, int_535286)

int_535287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 67), 'int')
# Processing the call keyword arguments (line 226)
float_535288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 75), 'float')
keyword_535289 = float_535288
# Getting the type of 'False' (line 226)
False_535290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 93), 'False', False)
keyword_535291 = False_535290
kwargs_535292 = {'vectorized': keyword_535291, 'rtol': keyword_535289}
# Getting the type of 'data' (line 226)
data_535281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'data', False)
# Calling data(args, kwargs) (line 226)
data_call_result_535293 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), data_535281, *[lqn__535282, str_535283, tuple_535284, int_535287], **kwargs_535292)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535293)
# Adding element type (line 202)

# Call to data(...): (line 227)
# Processing the call arguments (line 227)
# Getting the type of 'legendre_q_via_lqmn' (line 227)
legendre_q_via_lqmn_535295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 13), 'legendre_q_via_lqmn', False)
str_535296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 34), 'str', 'legendre_p_ipp-legendre_p')

# Obtaining an instance of the builtin type 'tuple' (line 227)
tuple_535297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 227)
# Adding element type (line 227)
int_535298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 64), tuple_535297, int_535298)
# Adding element type (line 227)
int_535299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 64), tuple_535297, int_535299)

int_535300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 70), 'int')
# Processing the call keyword arguments (line 227)
float_535301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 78), 'float')
keyword_535302 = float_535301
# Getting the type of 'False' (line 227)
False_535303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 96), 'False', False)
keyword_535304 = False_535303
kwargs_535305 = {'vectorized': keyword_535304, 'rtol': keyword_535302}
# Getting the type of 'data' (line 227)
data_535294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'data', False)
# Calling data(args, kwargs) (line 227)
data_call_result_535306 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), data_535294, *[legendre_q_via_lqmn_535295, str_535296, tuple_535297, int_535300], **kwargs_535305)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535306)
# Adding element type (line 202)

# Call to data(...): (line 228)
# Processing the call arguments (line 228)
# Getting the type of 'legendre_q_via_lqmn' (line 228)
legendre_q_via_lqmn_535308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 13), 'legendre_q_via_lqmn', False)
str_535309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 34), 'str', 'legendre_p_large_ipp-legendre_p_large')

# Obtaining an instance of the builtin type 'tuple' (line 228)
tuple_535310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 76), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 228)
# Adding element type (line 228)
int_535311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 76), tuple_535310, int_535311)
# Adding element type (line 228)
int_535312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 76), tuple_535310, int_535312)

int_535313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 82), 'int')
# Processing the call keyword arguments (line 228)
float_535314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 90), 'float')
keyword_535315 = float_535314
# Getting the type of 'False' (line 228)
False_535316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 108), 'False', False)
keyword_535317 = False_535316
kwargs_535318 = {'vectorized': keyword_535317, 'rtol': keyword_535315}
# Getting the type of 'data' (line 228)
data_535307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'data', False)
# Calling data(args, kwargs) (line 228)
data_call_result_535319 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), data_535307, *[legendre_q_via_lqmn_535308, str_535309, tuple_535310, int_535313], **kwargs_535318)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535319)
# Adding element type (line 202)

# Call to data(...): (line 230)
# Processing the call arguments (line 230)
# Getting the type of 'beta' (line 230)
beta_535321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 13), 'beta', False)
str_535322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 19), 'str', 'beta_exp_data_ipp-beta_exp_data')

# Obtaining an instance of the builtin type 'tuple' (line 230)
tuple_535323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 230)
# Adding element type (line 230)
int_535324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 55), tuple_535323, int_535324)
# Adding element type (line 230)
int_535325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 55), tuple_535323, int_535325)

int_535326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 61), 'int')
# Processing the call keyword arguments (line 230)
float_535327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 69), 'float')
keyword_535328 = float_535327
kwargs_535329 = {'rtol': keyword_535328}
# Getting the type of 'data' (line 230)
data_535320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'data', False)
# Calling data(args, kwargs) (line 230)
data_call_result_535330 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), data_535320, *[beta_535321, str_535322, tuple_535323, int_535326], **kwargs_535329)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535330)
# Adding element type (line 202)

# Call to data(...): (line 231)
# Processing the call arguments (line 231)
# Getting the type of 'beta' (line 231)
beta_535332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 13), 'beta', False)
str_535333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 19), 'str', 'beta_exp_data_ipp-beta_exp_data')

# Obtaining an instance of the builtin type 'tuple' (line 231)
tuple_535334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 231)
# Adding element type (line 231)
int_535335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 55), tuple_535334, int_535335)
# Adding element type (line 231)
int_535336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 55), tuple_535334, int_535336)

int_535337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 61), 'int')
# Processing the call keyword arguments (line 231)
float_535338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 69), 'float')
keyword_535339 = float_535338
kwargs_535340 = {'rtol': keyword_535339}
# Getting the type of 'data' (line 231)
data_535331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'data', False)
# Calling data(args, kwargs) (line 231)
data_call_result_535341 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), data_535331, *[beta_535332, str_535333, tuple_535334, int_535337], **kwargs_535340)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535341)
# Adding element type (line 202)

# Call to data(...): (line 232)
# Processing the call arguments (line 232)
# Getting the type of 'beta' (line 232)
beta_535343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 13), 'beta', False)
str_535344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 19), 'str', 'beta_small_data_ipp-beta_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 232)
tuple_535345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 59), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 232)
# Adding element type (line 232)
int_535346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 59), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 59), tuple_535345, int_535346)
# Adding element type (line 232)
int_535347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 59), tuple_535345, int_535347)

int_535348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 65), 'int')
# Processing the call keyword arguments (line 232)
kwargs_535349 = {}
# Getting the type of 'data' (line 232)
data_535342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'data', False)
# Calling data(args, kwargs) (line 232)
data_call_result_535350 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), data_535342, *[beta_535343, str_535344, tuple_535345, int_535348], **kwargs_535349)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535350)
# Adding element type (line 202)

# Call to data(...): (line 233)
# Processing the call arguments (line 233)
# Getting the type of 'beta' (line 233)
beta_535352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'beta', False)
str_535353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 19), 'str', 'beta_med_data_ipp-beta_med_data')

# Obtaining an instance of the builtin type 'tuple' (line 233)
tuple_535354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 233)
# Adding element type (line 233)
int_535355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 55), tuple_535354, int_535355)
# Adding element type (line 233)
int_535356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 55), tuple_535354, int_535356)

int_535357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 61), 'int')
# Processing the call keyword arguments (line 233)
float_535358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 69), 'float')
keyword_535359 = float_535358
kwargs_535360 = {'rtol': keyword_535359}
# Getting the type of 'data' (line 233)
data_535351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'data', False)
# Calling data(args, kwargs) (line 233)
data_call_result_535361 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), data_535351, *[beta_535352, str_535353, tuple_535354, int_535357], **kwargs_535360)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535361)
# Adding element type (line 202)

# Call to data(...): (line 235)
# Processing the call arguments (line 235)
# Getting the type of 'betainc' (line 235)
betainc_535363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'betainc', False)
str_535364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'str', 'ibeta_small_data_ipp-ibeta_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 235)
tuple_535365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 235)
# Adding element type (line 235)
int_535366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 64), tuple_535365, int_535366)
# Adding element type (line 235)
int_535367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 64), tuple_535365, int_535367)
# Adding element type (line 235)
int_535368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 64), tuple_535365, int_535368)

int_535369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 72), 'int')
# Processing the call keyword arguments (line 235)
float_535370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 80), 'float')
keyword_535371 = float_535370
kwargs_535372 = {'rtol': keyword_535371}
# Getting the type of 'data' (line 235)
data_535362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'data', False)
# Calling data(args, kwargs) (line 235)
data_call_result_535373 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), data_535362, *[betainc_535363, str_535364, tuple_535365, int_535369], **kwargs_535372)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535373)
# Adding element type (line 202)

# Call to data(...): (line 236)
# Processing the call arguments (line 236)
# Getting the type of 'betainc' (line 236)
betainc_535375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'betainc', False)
str_535376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 22), 'str', 'ibeta_data_ipp-ibeta_data')

# Obtaining an instance of the builtin type 'tuple' (line 236)
tuple_535377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 52), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 236)
# Adding element type (line 236)
int_535378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 52), tuple_535377, int_535378)
# Adding element type (line 236)
int_535379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 52), tuple_535377, int_535379)
# Adding element type (line 236)
int_535380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 52), tuple_535377, int_535380)

int_535381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 60), 'int')
# Processing the call keyword arguments (line 236)
float_535382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 68), 'float')
keyword_535383 = float_535382
kwargs_535384 = {'rtol': keyword_535383}
# Getting the type of 'data' (line 236)
data_535374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'data', False)
# Calling data(args, kwargs) (line 236)
data_call_result_535385 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), data_535374, *[betainc_535375, str_535376, tuple_535377, int_535381], **kwargs_535384)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535385)
# Adding element type (line 202)

# Call to data(...): (line 237)
# Processing the call arguments (line 237)
# Getting the type of 'betainc' (line 237)
betainc_535387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 13), 'betainc', False)
str_535388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 22), 'str', 'ibeta_int_data_ipp-ibeta_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 237)
tuple_535389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 60), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 237)
# Adding element type (line 237)
int_535390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 60), tuple_535389, int_535390)
# Adding element type (line 237)
int_535391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 60), tuple_535389, int_535391)
# Adding element type (line 237)
int_535392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 60), tuple_535389, int_535392)

int_535393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 68), 'int')
# Processing the call keyword arguments (line 237)
float_535394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 76), 'float')
keyword_535395 = float_535394
kwargs_535396 = {'rtol': keyword_535395}
# Getting the type of 'data' (line 237)
data_535386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'data', False)
# Calling data(args, kwargs) (line 237)
data_call_result_535397 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), data_535386, *[betainc_535387, str_535388, tuple_535389, int_535393], **kwargs_535396)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535397)
# Adding element type (line 202)

# Call to data(...): (line 238)
# Processing the call arguments (line 238)
# Getting the type of 'betainc' (line 238)
betainc_535399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 13), 'betainc', False)
str_535400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 22), 'str', 'ibeta_large_data_ipp-ibeta_large_data')

# Obtaining an instance of the builtin type 'tuple' (line 238)
tuple_535401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 238)
# Adding element type (line 238)
int_535402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 64), tuple_535401, int_535402)
# Adding element type (line 238)
int_535403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 64), tuple_535401, int_535403)
# Adding element type (line 238)
int_535404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 64), tuple_535401, int_535404)

int_535405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 72), 'int')
# Processing the call keyword arguments (line 238)
float_535406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 80), 'float')
keyword_535407 = float_535406
kwargs_535408 = {'rtol': keyword_535407}
# Getting the type of 'data' (line 238)
data_535398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'data', False)
# Calling data(args, kwargs) (line 238)
data_call_result_535409 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), data_535398, *[betainc_535399, str_535400, tuple_535401, int_535405], **kwargs_535408)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535409)
# Adding element type (line 202)

# Call to data(...): (line 240)
# Processing the call arguments (line 240)
# Getting the type of 'betaincinv' (line 240)
betaincinv_535411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 13), 'betaincinv', False)
str_535412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 25), 'str', 'ibeta_inv_data_ipp-ibeta_inv_data')

# Obtaining an instance of the builtin type 'tuple' (line 240)
tuple_535413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 240)
# Adding element type (line 240)
int_535414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 63), tuple_535413, int_535414)
# Adding element type (line 240)
int_535415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 63), tuple_535413, int_535415)
# Adding element type (line 240)
int_535416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 63), tuple_535413, int_535416)

int_535417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 71), 'int')
# Processing the call keyword arguments (line 240)
float_535418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 79), 'float')
keyword_535419 = float_535418
kwargs_535420 = {'rtol': keyword_535419}
# Getting the type of 'data' (line 240)
data_535410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'data', False)
# Calling data(args, kwargs) (line 240)
data_call_result_535421 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), data_535410, *[betaincinv_535411, str_535412, tuple_535413, int_535417], **kwargs_535420)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535421)
# Adding element type (line 202)

# Call to data(...): (line 242)
# Processing the call arguments (line 242)
# Getting the type of 'btdtr' (line 242)
btdtr_535423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 13), 'btdtr', False)
str_535424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 20), 'str', 'ibeta_small_data_ipp-ibeta_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 242)
tuple_535425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 242)
# Adding element type (line 242)
int_535426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 62), tuple_535425, int_535426)
# Adding element type (line 242)
int_535427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 62), tuple_535425, int_535427)
# Adding element type (line 242)
int_535428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 62), tuple_535425, int_535428)

int_535429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 70), 'int')
# Processing the call keyword arguments (line 242)
float_535430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 78), 'float')
keyword_535431 = float_535430
kwargs_535432 = {'rtol': keyword_535431}
# Getting the type of 'data' (line 242)
data_535422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'data', False)
# Calling data(args, kwargs) (line 242)
data_call_result_535433 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), data_535422, *[btdtr_535423, str_535424, tuple_535425, int_535429], **kwargs_535432)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535433)
# Adding element type (line 202)

# Call to data(...): (line 243)
# Processing the call arguments (line 243)
# Getting the type of 'btdtr' (line 243)
btdtr_535435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 13), 'btdtr', False)
str_535436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 20), 'str', 'ibeta_data_ipp-ibeta_data')

# Obtaining an instance of the builtin type 'tuple' (line 243)
tuple_535437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 50), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 243)
# Adding element type (line 243)
int_535438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 50), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 50), tuple_535437, int_535438)
# Adding element type (line 243)
int_535439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 50), tuple_535437, int_535439)
# Adding element type (line 243)
int_535440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 50), tuple_535437, int_535440)

int_535441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 58), 'int')
# Processing the call keyword arguments (line 243)
float_535442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 66), 'float')
keyword_535443 = float_535442
kwargs_535444 = {'rtol': keyword_535443}
# Getting the type of 'data' (line 243)
data_535434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'data', False)
# Calling data(args, kwargs) (line 243)
data_call_result_535445 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), data_535434, *[btdtr_535435, str_535436, tuple_535437, int_535441], **kwargs_535444)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535445)
# Adding element type (line 202)

# Call to data(...): (line 244)
# Processing the call arguments (line 244)
# Getting the type of 'btdtr' (line 244)
btdtr_535447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 13), 'btdtr', False)
str_535448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'str', 'ibeta_int_data_ipp-ibeta_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 244)
tuple_535449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 58), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 244)
# Adding element type (line 244)
int_535450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 58), tuple_535449, int_535450)
# Adding element type (line 244)
int_535451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 58), tuple_535449, int_535451)
# Adding element type (line 244)
int_535452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 58), tuple_535449, int_535452)

int_535453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 66), 'int')
# Processing the call keyword arguments (line 244)
float_535454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 74), 'float')
keyword_535455 = float_535454
kwargs_535456 = {'rtol': keyword_535455}
# Getting the type of 'data' (line 244)
data_535446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'data', False)
# Calling data(args, kwargs) (line 244)
data_call_result_535457 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), data_535446, *[btdtr_535447, str_535448, tuple_535449, int_535453], **kwargs_535456)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535457)
# Adding element type (line 202)

# Call to data(...): (line 245)
# Processing the call arguments (line 245)
# Getting the type of 'btdtr' (line 245)
btdtr_535459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'btdtr', False)
str_535460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 20), 'str', 'ibeta_large_data_ipp-ibeta_large_data')

# Obtaining an instance of the builtin type 'tuple' (line 245)
tuple_535461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 245)
# Adding element type (line 245)
int_535462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 62), tuple_535461, int_535462)
# Adding element type (line 245)
int_535463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 62), tuple_535461, int_535463)
# Adding element type (line 245)
int_535464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 62), tuple_535461, int_535464)

int_535465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 70), 'int')
# Processing the call keyword arguments (line 245)
float_535466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 78), 'float')
keyword_535467 = float_535466
kwargs_535468 = {'rtol': keyword_535467}
# Getting the type of 'data' (line 245)
data_535458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'data', False)
# Calling data(args, kwargs) (line 245)
data_call_result_535469 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), data_535458, *[btdtr_535459, str_535460, tuple_535461, int_535465], **kwargs_535468)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535469)
# Adding element type (line 202)

# Call to data(...): (line 247)
# Processing the call arguments (line 247)
# Getting the type of 'btdtri' (line 247)
btdtri_535471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'btdtri', False)
str_535472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 21), 'str', 'ibeta_inv_data_ipp-ibeta_inv_data')

# Obtaining an instance of the builtin type 'tuple' (line 247)
tuple_535473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 59), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 247)
# Adding element type (line 247)
int_535474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 59), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 59), tuple_535473, int_535474)
# Adding element type (line 247)
int_535475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 59), tuple_535473, int_535475)
# Adding element type (line 247)
int_535476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 59), tuple_535473, int_535476)

int_535477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 67), 'int')
# Processing the call keyword arguments (line 247)
float_535478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 75), 'float')
keyword_535479 = float_535478
kwargs_535480 = {'rtol': keyword_535479}
# Getting the type of 'data' (line 247)
data_535470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'data', False)
# Calling data(args, kwargs) (line 247)
data_call_result_535481 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), data_535470, *[btdtri_535471, str_535472, tuple_535473, int_535477], **kwargs_535480)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535481)
# Adding element type (line 202)

# Call to data(...): (line 248)
# Processing the call arguments (line 248)
# Getting the type of 'btdtri_comp' (line 248)
btdtri_comp_535483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 13), 'btdtri_comp', False)
str_535484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 26), 'str', 'ibeta_inv_data_ipp-ibeta_inv_data')

# Obtaining an instance of the builtin type 'tuple' (line 248)
tuple_535485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 248)
# Adding element type (line 248)
int_535486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 64), tuple_535485, int_535486)
# Adding element type (line 248)
int_535487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 64), tuple_535485, int_535487)
# Adding element type (line 248)
int_535488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 64), tuple_535485, int_535488)

int_535489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 72), 'int')
# Processing the call keyword arguments (line 248)
float_535490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 80), 'float')
keyword_535491 = float_535490
kwargs_535492 = {'rtol': keyword_535491}
# Getting the type of 'data' (line 248)
data_535482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'data', False)
# Calling data(args, kwargs) (line 248)
data_call_result_535493 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), data_535482, *[btdtri_comp_535483, str_535484, tuple_535485, int_535489], **kwargs_535492)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535493)
# Adding element type (line 202)

# Call to data(...): (line 250)
# Processing the call arguments (line 250)
# Getting the type of 'btdtria' (line 250)
btdtria_535495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 13), 'btdtria', False)
str_535496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 22), 'str', 'ibeta_inva_data_ipp-ibeta_inva_data')

# Obtaining an instance of the builtin type 'tuple' (line 250)
tuple_535497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 250)
# Adding element type (line 250)
int_535498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 62), tuple_535497, int_535498)
# Adding element type (line 250)
int_535499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 62), tuple_535497, int_535499)
# Adding element type (line 250)
int_535500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 62), tuple_535497, int_535500)

int_535501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 70), 'int')
# Processing the call keyword arguments (line 250)
float_535502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 78), 'float')
keyword_535503 = float_535502
kwargs_535504 = {'rtol': keyword_535503}
# Getting the type of 'data' (line 250)
data_535494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'data', False)
# Calling data(args, kwargs) (line 250)
data_call_result_535505 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), data_535494, *[btdtria_535495, str_535496, tuple_535497, int_535501], **kwargs_535504)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535505)
# Adding element type (line 202)

# Call to data(...): (line 251)
# Processing the call arguments (line 251)
# Getting the type of 'btdtria_comp' (line 251)
btdtria_comp_535507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'btdtria_comp', False)
str_535508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 27), 'str', 'ibeta_inva_data_ipp-ibeta_inva_data')

# Obtaining an instance of the builtin type 'tuple' (line 251)
tuple_535509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 67), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 251)
# Adding element type (line 251)
int_535510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 67), tuple_535509, int_535510)
# Adding element type (line 251)
int_535511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 69), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 67), tuple_535509, int_535511)
# Adding element type (line 251)
int_535512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 71), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 67), tuple_535509, int_535512)

int_535513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 75), 'int')
# Processing the call keyword arguments (line 251)
float_535514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 83), 'float')
keyword_535515 = float_535514
kwargs_535516 = {'rtol': keyword_535515}
# Getting the type of 'data' (line 251)
data_535506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'data', False)
# Calling data(args, kwargs) (line 251)
data_call_result_535517 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), data_535506, *[btdtria_comp_535507, str_535508, tuple_535509, int_535513], **kwargs_535516)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535517)
# Adding element type (line 202)

# Call to data(...): (line 253)
# Processing the call arguments (line 253)
# Getting the type of 'btdtrib' (line 253)
btdtrib_535519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 13), 'btdtrib', False)
str_535520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 22), 'str', 'ibeta_inva_data_ipp-ibeta_inva_data')

# Obtaining an instance of the builtin type 'tuple' (line 253)
tuple_535521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 62), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 253)
# Adding element type (line 253)
int_535522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 62), tuple_535521, int_535522)
# Adding element type (line 253)
int_535523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 62), tuple_535521, int_535523)
# Adding element type (line 253)
int_535524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 62), tuple_535521, int_535524)

int_535525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 70), 'int')
# Processing the call keyword arguments (line 253)
float_535526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 78), 'float')
keyword_535527 = float_535526
kwargs_535528 = {'rtol': keyword_535527}
# Getting the type of 'data' (line 253)
data_535518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'data', False)
# Calling data(args, kwargs) (line 253)
data_call_result_535529 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), data_535518, *[btdtrib_535519, str_535520, tuple_535521, int_535525], **kwargs_535528)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535529)
# Adding element type (line 202)

# Call to data(...): (line 254)
# Processing the call arguments (line 254)
# Getting the type of 'btdtrib_comp' (line 254)
btdtrib_comp_535531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 13), 'btdtrib_comp', False)
str_535532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 27), 'str', 'ibeta_inva_data_ipp-ibeta_inva_data')

# Obtaining an instance of the builtin type 'tuple' (line 254)
tuple_535533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 67), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 254)
# Adding element type (line 254)
int_535534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 67), tuple_535533, int_535534)
# Adding element type (line 254)
int_535535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 69), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 67), tuple_535533, int_535535)
# Adding element type (line 254)
int_535536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 71), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 67), tuple_535533, int_535536)

int_535537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 75), 'int')
# Processing the call keyword arguments (line 254)
float_535538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 83), 'float')
keyword_535539 = float_535538
kwargs_535540 = {'rtol': keyword_535539}
# Getting the type of 'data' (line 254)
data_535530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'data', False)
# Calling data(args, kwargs) (line 254)
data_call_result_535541 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), data_535530, *[btdtrib_comp_535531, str_535532, tuple_535533, int_535537], **kwargs_535540)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535541)
# Adding element type (line 202)

# Call to data(...): (line 256)
# Processing the call arguments (line 256)
# Getting the type of 'binom' (line 256)
binom_535543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 13), 'binom', False)
str_535544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'str', 'binomial_data_ipp-binomial_data')

# Obtaining an instance of the builtin type 'tuple' (line 256)
tuple_535545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 56), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 256)
# Adding element type (line 256)
int_535546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 56), tuple_535545, int_535546)
# Adding element type (line 256)
int_535547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 56), tuple_535545, int_535547)

int_535548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 62), 'int')
# Processing the call keyword arguments (line 256)
float_535549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 70), 'float')
keyword_535550 = float_535549
kwargs_535551 = {'rtol': keyword_535550}
# Getting the type of 'data' (line 256)
data_535542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'data', False)
# Calling data(args, kwargs) (line 256)
data_call_result_535552 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), data_535542, *[binom_535543, str_535544, tuple_535545, int_535548], **kwargs_535551)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535552)
# Adding element type (line 202)

# Call to data(...): (line 257)
# Processing the call arguments (line 257)
# Getting the type of 'binom' (line 257)
binom_535554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 13), 'binom', False)
str_535555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 20), 'str', 'binomial_large_data_ipp-binomial_large_data')

# Obtaining an instance of the builtin type 'tuple' (line 257)
tuple_535556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 68), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 257)
# Adding element type (line 257)
int_535557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 68), tuple_535556, int_535557)
# Adding element type (line 257)
int_535558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 68), tuple_535556, int_535558)

int_535559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 74), 'int')
# Processing the call keyword arguments (line 257)
float_535560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 82), 'float')
keyword_535561 = float_535560
kwargs_535562 = {'rtol': keyword_535561}
# Getting the type of 'data' (line 257)
data_535553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'data', False)
# Calling data(args, kwargs) (line 257)
data_call_result_535563 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), data_535553, *[binom_535554, str_535555, tuple_535556, int_535559], **kwargs_535562)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535563)
# Adding element type (line 202)

# Call to data(...): (line 259)
# Processing the call arguments (line 259)
# Getting the type of 'bdtrik' (line 259)
bdtrik_535565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 13), 'bdtrik', False)
str_535566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 21), 'str', 'binomial_quantile_ipp-binomial_quantile_data')

# Obtaining an instance of the builtin type 'tuple' (line 259)
tuple_535567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 70), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 259)
# Adding element type (line 259)
int_535568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 70), tuple_535567, int_535568)
# Adding element type (line 259)
int_535569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 70), tuple_535567, int_535569)
# Adding element type (line 259)
int_535570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 74), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 70), tuple_535567, int_535570)

int_535571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 78), 'int')
# Processing the call keyword arguments (line 259)
float_535572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 86), 'float')
keyword_535573 = float_535572
kwargs_535574 = {'rtol': keyword_535573}
# Getting the type of 'data' (line 259)
data_535564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'data', False)
# Calling data(args, kwargs) (line 259)
data_call_result_535575 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), data_535564, *[bdtrik_535565, str_535566, tuple_535567, int_535571], **kwargs_535574)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535575)
# Adding element type (line 202)

# Call to data(...): (line 260)
# Processing the call arguments (line 260)
# Getting the type of 'bdtrik_comp' (line 260)
bdtrik_comp_535577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 13), 'bdtrik_comp', False)
str_535578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 26), 'str', 'binomial_quantile_ipp-binomial_quantile_data')

# Obtaining an instance of the builtin type 'tuple' (line 260)
tuple_535579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 75), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 260)
# Adding element type (line 260)
int_535580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 75), tuple_535579, int_535580)
# Adding element type (line 260)
int_535581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 77), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 75), tuple_535579, int_535581)
# Adding element type (line 260)
int_535582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 79), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 75), tuple_535579, int_535582)

int_535583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 83), 'int')
# Processing the call keyword arguments (line 260)
float_535584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 91), 'float')
keyword_535585 = float_535584
kwargs_535586 = {'rtol': keyword_535585}
# Getting the type of 'data' (line 260)
data_535576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'data', False)
# Calling data(args, kwargs) (line 260)
data_call_result_535587 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), data_535576, *[bdtrik_comp_535577, str_535578, tuple_535579, int_535583], **kwargs_535586)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535587)
# Adding element type (line 202)

# Call to data(...): (line 262)
# Processing the call arguments (line 262)
# Getting the type of 'nbdtrik' (line 262)
nbdtrik_535589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 13), 'nbdtrik', False)
str_535590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 22), 'str', 'negative_binomial_quantile_ipp-negative_binomial_quantile_data')

# Obtaining an instance of the builtin type 'tuple' (line 262)
tuple_535591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 89), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 262)
# Adding element type (line 262)
int_535592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 89), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 89), tuple_535591, int_535592)
# Adding element type (line 262)
int_535593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 91), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 89), tuple_535591, int_535593)
# Adding element type (line 262)
int_535594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 93), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 89), tuple_535591, int_535594)

int_535595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 97), 'int')
# Processing the call keyword arguments (line 262)
float_535596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 105), 'float')
keyword_535597 = float_535596
kwargs_535598 = {'rtol': keyword_535597}
# Getting the type of 'data' (line 262)
data_535588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'data', False)
# Calling data(args, kwargs) (line 262)
data_call_result_535599 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), data_535588, *[nbdtrik_535589, str_535590, tuple_535591, int_535595], **kwargs_535598)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535599)
# Adding element type (line 202)

# Call to data(...): (line 263)
# Processing the call arguments (line 263)
# Getting the type of 'nbdtrik_comp' (line 263)
nbdtrik_comp_535601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'nbdtrik_comp', False)
str_535602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'str', 'negative_binomial_quantile_ipp-negative_binomial_quantile_data')

# Obtaining an instance of the builtin type 'tuple' (line 263)
tuple_535603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 94), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 263)
# Adding element type (line 263)
int_535604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 94), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 94), tuple_535603, int_535604)
# Adding element type (line 263)
int_535605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 96), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 94), tuple_535603, int_535605)
# Adding element type (line 263)
int_535606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 98), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 94), tuple_535603, int_535606)

int_535607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 102), 'int')
# Processing the call keyword arguments (line 263)
float_535608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 110), 'float')
keyword_535609 = float_535608
kwargs_535610 = {'rtol': keyword_535609}
# Getting the type of 'data' (line 263)
data_535600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'data', False)
# Calling data(args, kwargs) (line 263)
data_call_result_535611 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), data_535600, *[nbdtrik_comp_535601, str_535602, tuple_535603, int_535607], **kwargs_535610)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535611)
# Adding element type (line 202)

# Call to data(...): (line 265)
# Processing the call arguments (line 265)
# Getting the type of 'pdtrik' (line 265)
pdtrik_535613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'pdtrik', False)
str_535614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 21), 'str', 'poisson_quantile_ipp-poisson_quantile_data')

# Obtaining an instance of the builtin type 'tuple' (line 265)
tuple_535615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 68), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 265)
# Adding element type (line 265)
int_535616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 68), tuple_535615, int_535616)
# Adding element type (line 265)
int_535617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 68), tuple_535615, int_535617)

int_535618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 74), 'int')
# Processing the call keyword arguments (line 265)
float_535619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 82), 'float')
keyword_535620 = float_535619
kwargs_535621 = {'rtol': keyword_535620}
# Getting the type of 'data' (line 265)
data_535612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'data', False)
# Calling data(args, kwargs) (line 265)
data_call_result_535622 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), data_535612, *[pdtrik_535613, str_535614, tuple_535615, int_535618], **kwargs_535621)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535622)
# Adding element type (line 202)

# Call to data(...): (line 266)
# Processing the call arguments (line 266)
# Getting the type of 'pdtrik_comp' (line 266)
pdtrik_comp_535624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 13), 'pdtrik_comp', False)
str_535625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'str', 'poisson_quantile_ipp-poisson_quantile_data')

# Obtaining an instance of the builtin type 'tuple' (line 266)
tuple_535626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 73), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 266)
# Adding element type (line 266)
int_535627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 73), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 73), tuple_535626, int_535627)
# Adding element type (line 266)
int_535628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 73), tuple_535626, int_535628)

int_535629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 79), 'int')
# Processing the call keyword arguments (line 266)
float_535630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 87), 'float')
keyword_535631 = float_535630
kwargs_535632 = {'rtol': keyword_535631}
# Getting the type of 'data' (line 266)
data_535623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'data', False)
# Calling data(args, kwargs) (line 266)
data_call_result_535633 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), data_535623, *[pdtrik_comp_535624, str_535625, tuple_535626, int_535629], **kwargs_535632)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535633)
# Adding element type (line 202)

# Call to data(...): (line 268)
# Processing the call arguments (line 268)
# Getting the type of 'cbrt' (line 268)
cbrt_535635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 13), 'cbrt', False)
str_535636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 19), 'str', 'cbrt_data_ipp-cbrt_data')
int_535637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 46), 'int')
int_535638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 49), 'int')
# Processing the call keyword arguments (line 268)
kwargs_535639 = {}
# Getting the type of 'data' (line 268)
data_535634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'data', False)
# Calling data(args, kwargs) (line 268)
data_call_result_535640 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), data_535634, *[cbrt_535635, str_535636, int_535637, int_535638], **kwargs_535639)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535640)
# Adding element type (line 202)

# Call to data(...): (line 270)
# Processing the call arguments (line 270)
# Getting the type of 'digamma' (line 270)
digamma_535642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 13), 'digamma', False)
str_535643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 22), 'str', 'digamma_data_ipp-digamma_data')
int_535644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 55), 'int')
int_535645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 58), 'int')
# Processing the call keyword arguments (line 270)
kwargs_535646 = {}
# Getting the type of 'data' (line 270)
data_535641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'data', False)
# Calling data(args, kwargs) (line 270)
data_call_result_535647 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), data_535641, *[digamma_535642, str_535643, int_535644, int_535645], **kwargs_535646)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535647)
# Adding element type (line 202)

# Call to data(...): (line 271)
# Processing the call arguments (line 271)
# Getting the type of 'digamma' (line 271)
digamma_535649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'digamma', False)
str_535650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 22), 'str', 'digamma_data_ipp-digamma_data')
complex_535651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 55), 'complex')
int_535652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 59), 'int')
# Processing the call keyword arguments (line 271)
kwargs_535653 = {}
# Getting the type of 'data' (line 271)
data_535648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'data', False)
# Calling data(args, kwargs) (line 271)
data_call_result_535654 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), data_535648, *[digamma_535649, str_535650, complex_535651, int_535652], **kwargs_535653)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535654)
# Adding element type (line 202)

# Call to data(...): (line 272)
# Processing the call arguments (line 272)
# Getting the type of 'digamma' (line 272)
digamma_535656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 13), 'digamma', False)
str_535657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 22), 'str', 'digamma_neg_data_ipp-digamma_neg_data')
int_535658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 63), 'int')
int_535659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 66), 'int')
# Processing the call keyword arguments (line 272)
float_535660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 74), 'float')
keyword_535661 = float_535660
kwargs_535662 = {'rtol': keyword_535661}
# Getting the type of 'data' (line 272)
data_535655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'data', False)
# Calling data(args, kwargs) (line 272)
data_call_result_535663 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), data_535655, *[digamma_535656, str_535657, int_535658, int_535659], **kwargs_535662)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535663)
# Adding element type (line 202)

# Call to data(...): (line 273)
# Processing the call arguments (line 273)
# Getting the type of 'digamma' (line 273)
digamma_535665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 13), 'digamma', False)
str_535666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 22), 'str', 'digamma_neg_data_ipp-digamma_neg_data')
complex_535667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 63), 'complex')
int_535668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 67), 'int')
# Processing the call keyword arguments (line 273)
float_535669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 75), 'float')
keyword_535670 = float_535669
kwargs_535671 = {'rtol': keyword_535670}
# Getting the type of 'data' (line 273)
data_535664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'data', False)
# Calling data(args, kwargs) (line 273)
data_call_result_535672 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), data_535664, *[digamma_535665, str_535666, complex_535667, int_535668], **kwargs_535671)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535672)
# Adding element type (line 202)

# Call to data(...): (line 274)
# Processing the call arguments (line 274)
# Getting the type of 'digamma' (line 274)
digamma_535674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 13), 'digamma', False)
str_535675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 22), 'str', 'digamma_root_data_ipp-digamma_root_data')
int_535676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 65), 'int')
int_535677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 68), 'int')
# Processing the call keyword arguments (line 274)
float_535678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 76), 'float')
keyword_535679 = float_535678
kwargs_535680 = {'rtol': keyword_535679}
# Getting the type of 'data' (line 274)
data_535673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'data', False)
# Calling data(args, kwargs) (line 274)
data_call_result_535681 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), data_535673, *[digamma_535674, str_535675, int_535676, int_535677], **kwargs_535680)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535681)
# Adding element type (line 202)

# Call to data(...): (line 275)
# Processing the call arguments (line 275)
# Getting the type of 'digamma' (line 275)
digamma_535683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'digamma', False)
str_535684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 22), 'str', 'digamma_root_data_ipp-digamma_root_data')
complex_535685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 65), 'complex')
int_535686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 69), 'int')
# Processing the call keyword arguments (line 275)
float_535687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 77), 'float')
keyword_535688 = float_535687
kwargs_535689 = {'rtol': keyword_535688}
# Getting the type of 'data' (line 275)
data_535682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'data', False)
# Calling data(args, kwargs) (line 275)
data_call_result_535690 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), data_535682, *[digamma_535683, str_535684, complex_535685, int_535686], **kwargs_535689)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535690)
# Adding element type (line 202)

# Call to data(...): (line 276)
# Processing the call arguments (line 276)
# Getting the type of 'digamma' (line 276)
digamma_535692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 13), 'digamma', False)
str_535693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 22), 'str', 'digamma_small_data_ipp-digamma_small_data')
int_535694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 67), 'int')
int_535695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 70), 'int')
# Processing the call keyword arguments (line 276)
kwargs_535696 = {}
# Getting the type of 'data' (line 276)
data_535691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'data', False)
# Calling data(args, kwargs) (line 276)
data_call_result_535697 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), data_535691, *[digamma_535692, str_535693, int_535694, int_535695], **kwargs_535696)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535697)
# Adding element type (line 202)

# Call to data(...): (line 277)
# Processing the call arguments (line 277)
# Getting the type of 'digamma' (line 277)
digamma_535699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'digamma', False)
str_535700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 22), 'str', 'digamma_small_data_ipp-digamma_small_data')
complex_535701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 67), 'complex')
int_535702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 71), 'int')
# Processing the call keyword arguments (line 277)
float_535703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 79), 'float')
keyword_535704 = float_535703
kwargs_535705 = {'rtol': keyword_535704}
# Getting the type of 'data' (line 277)
data_535698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'data', False)
# Calling data(args, kwargs) (line 277)
data_call_result_535706 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), data_535698, *[digamma_535699, str_535700, complex_535701, int_535702], **kwargs_535705)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535706)
# Adding element type (line 202)

# Call to data(...): (line 279)
# Processing the call arguments (line 279)
# Getting the type of 'ellipk_' (line 279)
ellipk__535708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 13), 'ellipk_', False)
str_535709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 22), 'str', 'ellint_k_data_ipp-ellint_k_data')
int_535710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 57), 'int')
int_535711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 60), 'int')
# Processing the call keyword arguments (line 279)
kwargs_535712 = {}
# Getting the type of 'data' (line 279)
data_535707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'data', False)
# Calling data(args, kwargs) (line 279)
data_call_result_535713 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), data_535707, *[ellipk__535708, str_535709, int_535710, int_535711], **kwargs_535712)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535713)
# Adding element type (line 202)

# Call to data(...): (line 280)
# Processing the call arguments (line 280)
# Getting the type of 'ellipkinc_' (line 280)
ellipkinc__535715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 13), 'ellipkinc_', False)
str_535716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 25), 'str', 'ellint_f_data_ipp-ellint_f_data')

# Obtaining an instance of the builtin type 'tuple' (line 280)
tuple_535717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 280)
# Adding element type (line 280)
int_535718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 61), tuple_535717, int_535718)
# Adding element type (line 280)
int_535719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 61), tuple_535717, int_535719)

int_535720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 67), 'int')
# Processing the call keyword arguments (line 280)
float_535721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 75), 'float')
keyword_535722 = float_535721
kwargs_535723 = {'rtol': keyword_535722}
# Getting the type of 'data' (line 280)
data_535714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'data', False)
# Calling data(args, kwargs) (line 280)
data_call_result_535724 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), data_535714, *[ellipkinc__535715, str_535716, tuple_535717, int_535720], **kwargs_535723)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535724)
# Adding element type (line 202)

# Call to data(...): (line 281)
# Processing the call arguments (line 281)
# Getting the type of 'ellipe_' (line 281)
ellipe__535726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'ellipe_', False)
str_535727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 22), 'str', 'ellint_e_data_ipp-ellint_e_data')
int_535728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 57), 'int')
int_535729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 60), 'int')
# Processing the call keyword arguments (line 281)
kwargs_535730 = {}
# Getting the type of 'data' (line 281)
data_535725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'data', False)
# Calling data(args, kwargs) (line 281)
data_call_result_535731 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), data_535725, *[ellipe__535726, str_535727, int_535728, int_535729], **kwargs_535730)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535731)
# Adding element type (line 202)

# Call to data(...): (line 282)
# Processing the call arguments (line 282)
# Getting the type of 'ellipeinc_' (line 282)
ellipeinc__535733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 13), 'ellipeinc_', False)
str_535734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 25), 'str', 'ellint_e2_data_ipp-ellint_e2_data')

# Obtaining an instance of the builtin type 'tuple' (line 282)
tuple_535735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 282)
# Adding element type (line 282)
int_535736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 63), tuple_535735, int_535736)
# Adding element type (line 282)
int_535737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 63), tuple_535735, int_535737)

int_535738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 69), 'int')
# Processing the call keyword arguments (line 282)
float_535739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 77), 'float')
keyword_535740 = float_535739
kwargs_535741 = {'rtol': keyword_535740}
# Getting the type of 'data' (line 282)
data_535732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'data', False)
# Calling data(args, kwargs) (line 282)
data_call_result_535742 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), data_535732, *[ellipeinc__535733, str_535734, tuple_535735, int_535738], **kwargs_535741)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535742)
# Adding element type (line 202)

# Call to data(...): (line 284)
# Processing the call arguments (line 284)
# Getting the type of 'erf' (line 284)
erf_535744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 13), 'erf', False)
str_535745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 18), 'str', 'erf_data_ipp-erf_data')
int_535746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 43), 'int')
int_535747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 46), 'int')
# Processing the call keyword arguments (line 284)
kwargs_535748 = {}
# Getting the type of 'data' (line 284)
data_535743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'data', False)
# Calling data(args, kwargs) (line 284)
data_call_result_535749 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), data_535743, *[erf_535744, str_535745, int_535746, int_535747], **kwargs_535748)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535749)
# Adding element type (line 202)

# Call to data(...): (line 285)
# Processing the call arguments (line 285)
# Getting the type of 'erf' (line 285)
erf_535751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 13), 'erf', False)
str_535752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 18), 'str', 'erf_data_ipp-erf_data')
complex_535753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 43), 'complex')
int_535754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 47), 'int')
# Processing the call keyword arguments (line 285)
float_535755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 55), 'float')
keyword_535756 = float_535755
kwargs_535757 = {'rtol': keyword_535756}
# Getting the type of 'data' (line 285)
data_535750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'data', False)
# Calling data(args, kwargs) (line 285)
data_call_result_535758 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), data_535750, *[erf_535751, str_535752, complex_535753, int_535754], **kwargs_535757)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535758)
# Adding element type (line 202)

# Call to data(...): (line 286)
# Processing the call arguments (line 286)
# Getting the type of 'erfc' (line 286)
erfc_535760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'erfc', False)
str_535761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 19), 'str', 'erf_data_ipp-erf_data')
int_535762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 44), 'int')
int_535763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 47), 'int')
# Processing the call keyword arguments (line 286)
float_535764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 55), 'float')
keyword_535765 = float_535764
kwargs_535766 = {'rtol': keyword_535765}
# Getting the type of 'data' (line 286)
data_535759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'data', False)
# Calling data(args, kwargs) (line 286)
data_call_result_535767 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), data_535759, *[erfc_535760, str_535761, int_535762, int_535763], **kwargs_535766)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535767)
# Adding element type (line 202)

# Call to data(...): (line 287)
# Processing the call arguments (line 287)
# Getting the type of 'erf' (line 287)
erf_535769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 13), 'erf', False)
str_535770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 18), 'str', 'erf_large_data_ipp-erf_large_data')
int_535771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 55), 'int')
int_535772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 58), 'int')
# Processing the call keyword arguments (line 287)
kwargs_535773 = {}
# Getting the type of 'data' (line 287)
data_535768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'data', False)
# Calling data(args, kwargs) (line 287)
data_call_result_535774 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), data_535768, *[erf_535769, str_535770, int_535771, int_535772], **kwargs_535773)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535774)
# Adding element type (line 202)

# Call to data(...): (line 288)
# Processing the call arguments (line 288)
# Getting the type of 'erf' (line 288)
erf_535776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 13), 'erf', False)
str_535777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 18), 'str', 'erf_large_data_ipp-erf_large_data')
complex_535778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 55), 'complex')
int_535779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 59), 'int')
# Processing the call keyword arguments (line 288)
kwargs_535780 = {}
# Getting the type of 'data' (line 288)
data_535775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'data', False)
# Calling data(args, kwargs) (line 288)
data_call_result_535781 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), data_535775, *[erf_535776, str_535777, complex_535778, int_535779], **kwargs_535780)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535781)
# Adding element type (line 202)

# Call to data(...): (line 289)
# Processing the call arguments (line 289)
# Getting the type of 'erfc' (line 289)
erfc_535783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 13), 'erfc', False)
str_535784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 19), 'str', 'erf_large_data_ipp-erf_large_data')
int_535785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 56), 'int')
int_535786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 59), 'int')
# Processing the call keyword arguments (line 289)
float_535787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 67), 'float')
keyword_535788 = float_535787
kwargs_535789 = {'rtol': keyword_535788}
# Getting the type of 'data' (line 289)
data_535782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'data', False)
# Calling data(args, kwargs) (line 289)
data_call_result_535790 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), data_535782, *[erfc_535783, str_535784, int_535785, int_535786], **kwargs_535789)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535790)
# Adding element type (line 202)

# Call to data(...): (line 290)
# Processing the call arguments (line 290)
# Getting the type of 'erf' (line 290)
erf_535792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 13), 'erf', False)
str_535793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 18), 'str', 'erf_small_data_ipp-erf_small_data')
int_535794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 55), 'int')
int_535795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 58), 'int')
# Processing the call keyword arguments (line 290)
kwargs_535796 = {}
# Getting the type of 'data' (line 290)
data_535791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'data', False)
# Calling data(args, kwargs) (line 290)
data_call_result_535797 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), data_535791, *[erf_535792, str_535793, int_535794, int_535795], **kwargs_535796)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535797)
# Adding element type (line 202)

# Call to data(...): (line 291)
# Processing the call arguments (line 291)
# Getting the type of 'erf' (line 291)
erf_535799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 13), 'erf', False)
str_535800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 18), 'str', 'erf_small_data_ipp-erf_small_data')
complex_535801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 55), 'complex')
int_535802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 59), 'int')
# Processing the call keyword arguments (line 291)
float_535803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 67), 'float')
keyword_535804 = float_535803
kwargs_535805 = {'rtol': keyword_535804}
# Getting the type of 'data' (line 291)
data_535798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'data', False)
# Calling data(args, kwargs) (line 291)
data_call_result_535806 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), data_535798, *[erf_535799, str_535800, complex_535801, int_535802], **kwargs_535805)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535806)
# Adding element type (line 202)

# Call to data(...): (line 292)
# Processing the call arguments (line 292)
# Getting the type of 'erfc' (line 292)
erfc_535808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 13), 'erfc', False)
str_535809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 19), 'str', 'erf_small_data_ipp-erf_small_data')
int_535810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 56), 'int')
int_535811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 59), 'int')
# Processing the call keyword arguments (line 292)
kwargs_535812 = {}
# Getting the type of 'data' (line 292)
data_535807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'data', False)
# Calling data(args, kwargs) (line 292)
data_call_result_535813 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), data_535807, *[erfc_535808, str_535809, int_535810, int_535811], **kwargs_535812)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535813)
# Adding element type (line 202)

# Call to data(...): (line 294)
# Processing the call arguments (line 294)
# Getting the type of 'erfinv' (line 294)
erfinv_535815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 13), 'erfinv', False)
str_535816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 21), 'str', 'erf_inv_data_ipp-erf_inv_data')
int_535817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 54), 'int')
int_535818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 57), 'int')
# Processing the call keyword arguments (line 294)
kwargs_535819 = {}
# Getting the type of 'data' (line 294)
data_535814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'data', False)
# Calling data(args, kwargs) (line 294)
data_call_result_535820 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), data_535814, *[erfinv_535815, str_535816, int_535817, int_535818], **kwargs_535819)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535820)
# Adding element type (line 202)

# Call to data(...): (line 295)
# Processing the call arguments (line 295)
# Getting the type of 'erfcinv' (line 295)
erfcinv_535822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 13), 'erfcinv', False)
str_535823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 22), 'str', 'erfc_inv_data_ipp-erfc_inv_data')
int_535824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 57), 'int')
int_535825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 60), 'int')
# Processing the call keyword arguments (line 295)
kwargs_535826 = {}
# Getting the type of 'data' (line 295)
data_535821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'data', False)
# Calling data(args, kwargs) (line 295)
data_call_result_535827 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), data_535821, *[erfcinv_535822, str_535823, int_535824, int_535825], **kwargs_535826)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535827)
# Adding element type (line 202)

# Call to data(...): (line 296)
# Processing the call arguments (line 296)
# Getting the type of 'erfcinv' (line 296)
erfcinv_535829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 13), 'erfcinv', False)
str_535830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 22), 'str', 'erfc_inv_big_data_ipp-erfc_inv_big_data2')
int_535831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 66), 'int')
int_535832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 69), 'int')
# Processing the call keyword arguments (line 296)
kwargs_535833 = {}
# Getting the type of 'data' (line 296)
data_535828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'data', False)
# Calling data(args, kwargs) (line 296)
data_call_result_535834 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), data_535828, *[erfcinv_535829, str_535830, int_535831, int_535832], **kwargs_535833)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535834)
# Adding element type (line 202)

# Call to data(...): (line 298)
# Processing the call arguments (line 298)
# Getting the type of 'exp1' (line 298)
exp1_535836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 13), 'exp1', False)
str_535837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 19), 'str', 'expint_1_data_ipp-expint_1_data')
int_535838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 54), 'int')
int_535839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 57), 'int')
# Processing the call keyword arguments (line 298)
float_535840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 65), 'float')
keyword_535841 = float_535840
kwargs_535842 = {'rtol': keyword_535841}
# Getting the type of 'data' (line 298)
data_535835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'data', False)
# Calling data(args, kwargs) (line 298)
data_call_result_535843 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), data_535835, *[exp1_535836, str_535837, int_535838, int_535839], **kwargs_535842)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535843)
# Adding element type (line 202)

# Call to data(...): (line 299)
# Processing the call arguments (line 299)
# Getting the type of 'exp1' (line 299)
exp1_535845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 13), 'exp1', False)
str_535846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 19), 'str', 'expint_1_data_ipp-expint_1_data')
complex_535847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 54), 'complex')
int_535848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 58), 'int')
# Processing the call keyword arguments (line 299)
float_535849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 66), 'float')
keyword_535850 = float_535849
kwargs_535851 = {'rtol': keyword_535850}
# Getting the type of 'data' (line 299)
data_535844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'data', False)
# Calling data(args, kwargs) (line 299)
data_call_result_535852 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), data_535844, *[exp1_535845, str_535846, complex_535847, int_535848], **kwargs_535851)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535852)
# Adding element type (line 202)

# Call to data(...): (line 300)
# Processing the call arguments (line 300)
# Getting the type of 'expi' (line 300)
expi_535854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 13), 'expi', False)
str_535855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 19), 'str', 'expinti_data_ipp-expinti_data')
int_535856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 52), 'int')
int_535857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 55), 'int')
# Processing the call keyword arguments (line 300)
float_535858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 63), 'float')
keyword_535859 = float_535858
kwargs_535860 = {'rtol': keyword_535859}
# Getting the type of 'data' (line 300)
data_535853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'data', False)
# Calling data(args, kwargs) (line 300)
data_call_result_535861 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), data_535853, *[expi_535854, str_535855, int_535856, int_535857], **kwargs_535860)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535861)
# Adding element type (line 202)

# Call to data(...): (line 301)
# Processing the call arguments (line 301)
# Getting the type of 'expi' (line 301)
expi_535863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 13), 'expi', False)
str_535864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 19), 'str', 'expinti_data_double_ipp-expinti_data_double')
int_535865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 66), 'int')
int_535866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 69), 'int')
# Processing the call keyword arguments (line 301)
float_535867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 77), 'float')
keyword_535868 = float_535867
kwargs_535869 = {'rtol': keyword_535868}
# Getting the type of 'data' (line 301)
data_535862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'data', False)
# Calling data(args, kwargs) (line 301)
data_call_result_535870 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), data_535862, *[expi_535863, str_535864, int_535865, int_535866], **kwargs_535869)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535870)
# Adding element type (line 202)

# Call to data(...): (line 303)
# Processing the call arguments (line 303)
# Getting the type of 'expn' (line 303)
expn_535872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 13), 'expn', False)
str_535873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 19), 'str', 'expint_small_data_ipp-expint_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 303)
tuple_535874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 303)
# Adding element type (line 303)
int_535875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 63), tuple_535874, int_535875)
# Adding element type (line 303)
int_535876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 63), tuple_535874, int_535876)

int_535877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 69), 'int')
# Processing the call keyword arguments (line 303)
kwargs_535878 = {}
# Getting the type of 'data' (line 303)
data_535871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'data', False)
# Calling data(args, kwargs) (line 303)
data_call_result_535879 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), data_535871, *[expn_535872, str_535873, tuple_535874, int_535877], **kwargs_535878)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535879)
# Adding element type (line 202)

# Call to data(...): (line 304)
# Processing the call arguments (line 304)
# Getting the type of 'expn' (line 304)
expn_535881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 13), 'expn', False)
str_535882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 19), 'str', 'expint_data_ipp-expint_data')

# Obtaining an instance of the builtin type 'tuple' (line 304)
tuple_535883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 51), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 304)
# Adding element type (line 304)
int_535884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 51), tuple_535883, int_535884)
# Adding element type (line 304)
int_535885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 51), tuple_535883, int_535885)

int_535886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 57), 'int')
# Processing the call keyword arguments (line 304)
float_535887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 65), 'float')
keyword_535888 = float_535887
kwargs_535889 = {'rtol': keyword_535888}
# Getting the type of 'data' (line 304)
data_535880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'data', False)
# Calling data(args, kwargs) (line 304)
data_call_result_535890 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), data_535880, *[expn_535881, str_535882, tuple_535883, int_535886], **kwargs_535889)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535890)
# Adding element type (line 202)

# Call to data(...): (line 306)
# Processing the call arguments (line 306)
# Getting the type of 'gamma' (line 306)
gamma_535892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 13), 'gamma', False)
str_535893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 20), 'str', 'test_gamma_data_ipp-near_0')
int_535894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 50), 'int')
int_535895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 53), 'int')
# Processing the call keyword arguments (line 306)
kwargs_535896 = {}
# Getting the type of 'data' (line 306)
data_535891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'data', False)
# Calling data(args, kwargs) (line 306)
data_call_result_535897 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), data_535891, *[gamma_535892, str_535893, int_535894, int_535895], **kwargs_535896)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535897)
# Adding element type (line 202)

# Call to data(...): (line 307)
# Processing the call arguments (line 307)
# Getting the type of 'gamma' (line 307)
gamma_535899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 13), 'gamma', False)
str_535900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 20), 'str', 'test_gamma_data_ipp-near_1')
int_535901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 50), 'int')
int_535902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 53), 'int')
# Processing the call keyword arguments (line 307)
kwargs_535903 = {}
# Getting the type of 'data' (line 307)
data_535898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'data', False)
# Calling data(args, kwargs) (line 307)
data_call_result_535904 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), data_535898, *[gamma_535899, str_535900, int_535901, int_535902], **kwargs_535903)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535904)
# Adding element type (line 202)

# Call to data(...): (line 308)
# Processing the call arguments (line 308)
# Getting the type of 'gamma' (line 308)
gamma_535906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 13), 'gamma', False)
str_535907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 20), 'str', 'test_gamma_data_ipp-near_2')
int_535908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 50), 'int')
int_535909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 53), 'int')
# Processing the call keyword arguments (line 308)
kwargs_535910 = {}
# Getting the type of 'data' (line 308)
data_535905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'data', False)
# Calling data(args, kwargs) (line 308)
data_call_result_535911 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), data_535905, *[gamma_535906, str_535907, int_535908, int_535909], **kwargs_535910)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535911)
# Adding element type (line 202)

# Call to data(...): (line 309)
# Processing the call arguments (line 309)
# Getting the type of 'gamma' (line 309)
gamma_535913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 13), 'gamma', False)
str_535914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 20), 'str', 'test_gamma_data_ipp-near_m10')
int_535915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 52), 'int')
int_535916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 55), 'int')
# Processing the call keyword arguments (line 309)
kwargs_535917 = {}
# Getting the type of 'data' (line 309)
data_535912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'data', False)
# Calling data(args, kwargs) (line 309)
data_call_result_535918 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), data_535912, *[gamma_535913, str_535914, int_535915, int_535916], **kwargs_535917)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535918)
# Adding element type (line 202)

# Call to data(...): (line 310)
# Processing the call arguments (line 310)
# Getting the type of 'gamma' (line 310)
gamma_535920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 13), 'gamma', False)
str_535921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 20), 'str', 'test_gamma_data_ipp-near_m55')
int_535922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 52), 'int')
int_535923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 55), 'int')
# Processing the call keyword arguments (line 310)
float_535924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 63), 'float')
keyword_535925 = float_535924
kwargs_535926 = {'rtol': keyword_535925}
# Getting the type of 'data' (line 310)
data_535919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'data', False)
# Calling data(args, kwargs) (line 310)
data_call_result_535927 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), data_535919, *[gamma_535920, str_535921, int_535922, int_535923], **kwargs_535926)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535927)
# Adding element type (line 202)

# Call to data(...): (line 311)
# Processing the call arguments (line 311)
# Getting the type of 'gamma' (line 311)
gamma_535929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 13), 'gamma', False)
str_535930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 20), 'str', 'test_gamma_data_ipp-factorials')
int_535931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 54), 'int')
int_535932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 57), 'int')
# Processing the call keyword arguments (line 311)
float_535933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 65), 'float')
keyword_535934 = float_535933
kwargs_535935 = {'rtol': keyword_535934}
# Getting the type of 'data' (line 311)
data_535928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'data', False)
# Calling data(args, kwargs) (line 311)
data_call_result_535936 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), data_535928, *[gamma_535929, str_535930, int_535931, int_535932], **kwargs_535935)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535936)
# Adding element type (line 202)

# Call to data(...): (line 312)
# Processing the call arguments (line 312)
# Getting the type of 'gamma' (line 312)
gamma_535938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 13), 'gamma', False)
str_535939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 20), 'str', 'test_gamma_data_ipp-near_0')
complex_535940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 50), 'complex')
int_535941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 54), 'int')
# Processing the call keyword arguments (line 312)
float_535942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 62), 'float')
keyword_535943 = float_535942
kwargs_535944 = {'rtol': keyword_535943}
# Getting the type of 'data' (line 312)
data_535937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'data', False)
# Calling data(args, kwargs) (line 312)
data_call_result_535945 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), data_535937, *[gamma_535938, str_535939, complex_535940, int_535941], **kwargs_535944)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535945)
# Adding element type (line 202)

# Call to data(...): (line 313)
# Processing the call arguments (line 313)
# Getting the type of 'gamma' (line 313)
gamma_535947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'gamma', False)
str_535948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 20), 'str', 'test_gamma_data_ipp-near_1')
complex_535949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 50), 'complex')
int_535950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 54), 'int')
# Processing the call keyword arguments (line 313)
float_535951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 62), 'float')
keyword_535952 = float_535951
kwargs_535953 = {'rtol': keyword_535952}
# Getting the type of 'data' (line 313)
data_535946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'data', False)
# Calling data(args, kwargs) (line 313)
data_call_result_535954 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), data_535946, *[gamma_535947, str_535948, complex_535949, int_535950], **kwargs_535953)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535954)
# Adding element type (line 202)

# Call to data(...): (line 314)
# Processing the call arguments (line 314)
# Getting the type of 'gamma' (line 314)
gamma_535956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 13), 'gamma', False)
str_535957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 20), 'str', 'test_gamma_data_ipp-near_2')
complex_535958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 50), 'complex')
int_535959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 54), 'int')
# Processing the call keyword arguments (line 314)
float_535960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 62), 'float')
keyword_535961 = float_535960
kwargs_535962 = {'rtol': keyword_535961}
# Getting the type of 'data' (line 314)
data_535955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'data', False)
# Calling data(args, kwargs) (line 314)
data_call_result_535963 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), data_535955, *[gamma_535956, str_535957, complex_535958, int_535959], **kwargs_535962)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535963)
# Adding element type (line 202)

# Call to data(...): (line 315)
# Processing the call arguments (line 315)
# Getting the type of 'gamma' (line 315)
gamma_535965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 13), 'gamma', False)
str_535966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 20), 'str', 'test_gamma_data_ipp-near_m10')
complex_535967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 52), 'complex')
int_535968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 56), 'int')
# Processing the call keyword arguments (line 315)
float_535969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 64), 'float')
keyword_535970 = float_535969
kwargs_535971 = {'rtol': keyword_535970}
# Getting the type of 'data' (line 315)
data_535964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'data', False)
# Calling data(args, kwargs) (line 315)
data_call_result_535972 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), data_535964, *[gamma_535965, str_535966, complex_535967, int_535968], **kwargs_535971)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535972)
# Adding element type (line 202)

# Call to data(...): (line 316)
# Processing the call arguments (line 316)
# Getting the type of 'gamma' (line 316)
gamma_535974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'gamma', False)
str_535975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 20), 'str', 'test_gamma_data_ipp-near_m55')
complex_535976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 52), 'complex')
int_535977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 56), 'int')
# Processing the call keyword arguments (line 316)
float_535978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 64), 'float')
keyword_535979 = float_535978
kwargs_535980 = {'rtol': keyword_535979}
# Getting the type of 'data' (line 316)
data_535973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'data', False)
# Calling data(args, kwargs) (line 316)
data_call_result_535981 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), data_535973, *[gamma_535974, str_535975, complex_535976, int_535977], **kwargs_535980)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535981)
# Adding element type (line 202)

# Call to data(...): (line 317)
# Processing the call arguments (line 317)
# Getting the type of 'gamma' (line 317)
gamma_535983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 13), 'gamma', False)
str_535984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 20), 'str', 'test_gamma_data_ipp-factorials')
complex_535985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 54), 'complex')
int_535986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 58), 'int')
# Processing the call keyword arguments (line 317)
float_535987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 66), 'float')
keyword_535988 = float_535987
kwargs_535989 = {'rtol': keyword_535988}
# Getting the type of 'data' (line 317)
data_535982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'data', False)
# Calling data(args, kwargs) (line 317)
data_call_result_535990 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), data_535982, *[gamma_535983, str_535984, complex_535985, int_535986], **kwargs_535989)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535990)
# Adding element type (line 202)

# Call to data(...): (line 318)
# Processing the call arguments (line 318)
# Getting the type of 'gammaln' (line 318)
gammaln_535992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'gammaln', False)
str_535993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 22), 'str', 'test_gamma_data_ipp-near_0')
int_535994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 52), 'int')
int_535995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 55), 'int')
# Processing the call keyword arguments (line 318)
float_535996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 63), 'float')
keyword_535997 = float_535996
kwargs_535998 = {'rtol': keyword_535997}
# Getting the type of 'data' (line 318)
data_535991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'data', False)
# Calling data(args, kwargs) (line 318)
data_call_result_535999 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), data_535991, *[gammaln_535992, str_535993, int_535994, int_535995], **kwargs_535998)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_535999)
# Adding element type (line 202)

# Call to data(...): (line 319)
# Processing the call arguments (line 319)
# Getting the type of 'gammaln' (line 319)
gammaln_536001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 13), 'gammaln', False)
str_536002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 22), 'str', 'test_gamma_data_ipp-near_1')
int_536003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 52), 'int')
int_536004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 55), 'int')
# Processing the call keyword arguments (line 319)
float_536005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 63), 'float')
keyword_536006 = float_536005
kwargs_536007 = {'rtol': keyword_536006}
# Getting the type of 'data' (line 319)
data_536000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'data', False)
# Calling data(args, kwargs) (line 319)
data_call_result_536008 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), data_536000, *[gammaln_536001, str_536002, int_536003, int_536004], **kwargs_536007)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536008)
# Adding element type (line 202)

# Call to data(...): (line 320)
# Processing the call arguments (line 320)
# Getting the type of 'gammaln' (line 320)
gammaln_536010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 13), 'gammaln', False)
str_536011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 22), 'str', 'test_gamma_data_ipp-near_2')
int_536012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 52), 'int')
int_536013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 55), 'int')
# Processing the call keyword arguments (line 320)
float_536014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 63), 'float')
keyword_536015 = float_536014
kwargs_536016 = {'rtol': keyword_536015}
# Getting the type of 'data' (line 320)
data_536009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'data', False)
# Calling data(args, kwargs) (line 320)
data_call_result_536017 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), data_536009, *[gammaln_536010, str_536011, int_536012, int_536013], **kwargs_536016)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536017)
# Adding element type (line 202)

# Call to data(...): (line 321)
# Processing the call arguments (line 321)
# Getting the type of 'gammaln' (line 321)
gammaln_536019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 13), 'gammaln', False)
str_536020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 22), 'str', 'test_gamma_data_ipp-near_m10')
int_536021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 54), 'int')
int_536022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 57), 'int')
# Processing the call keyword arguments (line 321)
float_536023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 65), 'float')
keyword_536024 = float_536023
kwargs_536025 = {'rtol': keyword_536024}
# Getting the type of 'data' (line 321)
data_536018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'data', False)
# Calling data(args, kwargs) (line 321)
data_call_result_536026 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), data_536018, *[gammaln_536019, str_536020, int_536021, int_536022], **kwargs_536025)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536026)
# Adding element type (line 202)

# Call to data(...): (line 322)
# Processing the call arguments (line 322)
# Getting the type of 'gammaln' (line 322)
gammaln_536028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 13), 'gammaln', False)
str_536029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 22), 'str', 'test_gamma_data_ipp-near_m55')
int_536030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 54), 'int')
int_536031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 57), 'int')
# Processing the call keyword arguments (line 322)
float_536032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 65), 'float')
keyword_536033 = float_536032
kwargs_536034 = {'rtol': keyword_536033}
# Getting the type of 'data' (line 322)
data_536027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'data', False)
# Calling data(args, kwargs) (line 322)
data_call_result_536035 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), data_536027, *[gammaln_536028, str_536029, int_536030, int_536031], **kwargs_536034)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536035)
# Adding element type (line 202)

# Call to data(...): (line 323)
# Processing the call arguments (line 323)
# Getting the type of 'gammaln' (line 323)
gammaln_536037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'gammaln', False)
str_536038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 22), 'str', 'test_gamma_data_ipp-factorials')
int_536039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 56), 'int')
int_536040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 59), 'int')
# Processing the call keyword arguments (line 323)
kwargs_536041 = {}
# Getting the type of 'data' (line 323)
data_536036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'data', False)
# Calling data(args, kwargs) (line 323)
data_call_result_536042 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), data_536036, *[gammaln_536037, str_536038, int_536039, int_536040], **kwargs_536041)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536042)
# Adding element type (line 202)

# Call to data(...): (line 325)
# Processing the call arguments (line 325)
# Getting the type of 'gammainc' (line 325)
gammainc_536044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 13), 'gammainc', False)
str_536045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 23), 'str', 'igamma_small_data_ipp-igamma_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 325)
tuple_536046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 67), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 325)
# Adding element type (line 325)
int_536047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 67), tuple_536046, int_536047)
# Adding element type (line 325)
int_536048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 69), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 67), tuple_536046, int_536048)

int_536049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 73), 'int')
# Processing the call keyword arguments (line 325)
float_536050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 81), 'float')
keyword_536051 = float_536050
kwargs_536052 = {'rtol': keyword_536051}
# Getting the type of 'data' (line 325)
data_536043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'data', False)
# Calling data(args, kwargs) (line 325)
data_call_result_536053 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), data_536043, *[gammainc_536044, str_536045, tuple_536046, int_536049], **kwargs_536052)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536053)
# Adding element type (line 202)

# Call to data(...): (line 326)
# Processing the call arguments (line 326)
# Getting the type of 'gammainc' (line 326)
gammainc_536055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'gammainc', False)
str_536056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 23), 'str', 'igamma_med_data_ipp-igamma_med_data')

# Obtaining an instance of the builtin type 'tuple' (line 326)
tuple_536057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 326)
# Adding element type (line 326)
int_536058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 63), tuple_536057, int_536058)
# Adding element type (line 326)
int_536059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 63), tuple_536057, int_536059)

int_536060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 69), 'int')
# Processing the call keyword arguments (line 326)
float_536061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 77), 'float')
keyword_536062 = float_536061
kwargs_536063 = {'rtol': keyword_536062}
# Getting the type of 'data' (line 326)
data_536054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'data', False)
# Calling data(args, kwargs) (line 326)
data_call_result_536064 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), data_536054, *[gammainc_536055, str_536056, tuple_536057, int_536060], **kwargs_536063)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536064)
# Adding element type (line 202)

# Call to data(...): (line 327)
# Processing the call arguments (line 327)
# Getting the type of 'gammainc' (line 327)
gammainc_536066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 13), 'gammainc', False)
str_536067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 23), 'str', 'igamma_int_data_ipp-igamma_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 327)
tuple_536068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 327)
# Adding element type (line 327)
int_536069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 63), tuple_536068, int_536069)
# Adding element type (line 327)
int_536070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 63), tuple_536068, int_536070)

int_536071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 69), 'int')
# Processing the call keyword arguments (line 327)
float_536072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 77), 'float')
keyword_536073 = float_536072
kwargs_536074 = {'rtol': keyword_536073}
# Getting the type of 'data' (line 327)
data_536065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'data', False)
# Calling data(args, kwargs) (line 327)
data_call_result_536075 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), data_536065, *[gammainc_536066, str_536067, tuple_536068, int_536071], **kwargs_536074)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536075)
# Adding element type (line 202)

# Call to data(...): (line 328)
# Processing the call arguments (line 328)
# Getting the type of 'gammainc' (line 328)
gammainc_536077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 13), 'gammainc', False)
str_536078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 23), 'str', 'igamma_big_data_ipp-igamma_big_data')

# Obtaining an instance of the builtin type 'tuple' (line 328)
tuple_536079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 328)
# Adding element type (line 328)
int_536080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 63), tuple_536079, int_536080)
# Adding element type (line 328)
int_536081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 63), tuple_536079, int_536081)

int_536082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 69), 'int')
# Processing the call keyword arguments (line 328)
float_536083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 77), 'float')
keyword_536084 = float_536083
kwargs_536085 = {'rtol': keyword_536084}
# Getting the type of 'data' (line 328)
data_536076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'data', False)
# Calling data(args, kwargs) (line 328)
data_call_result_536086 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), data_536076, *[gammainc_536077, str_536078, tuple_536079, int_536082], **kwargs_536085)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536086)
# Adding element type (line 202)

# Call to data(...): (line 330)
# Processing the call arguments (line 330)
# Getting the type of 'gdtr_' (line 330)
gdtr__536088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 13), 'gdtr_', False)
str_536089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 20), 'str', 'igamma_small_data_ipp-igamma_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 330)
tuple_536090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 330)
# Adding element type (line 330)
int_536091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 64), tuple_536090, int_536091)
# Adding element type (line 330)
int_536092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 64), tuple_536090, int_536092)

int_536093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 70), 'int')
# Processing the call keyword arguments (line 330)
float_536094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 78), 'float')
keyword_536095 = float_536094
kwargs_536096 = {'rtol': keyword_536095}
# Getting the type of 'data' (line 330)
data_536087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'data', False)
# Calling data(args, kwargs) (line 330)
data_call_result_536097 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), data_536087, *[gdtr__536088, str_536089, tuple_536090, int_536093], **kwargs_536096)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536097)
# Adding element type (line 202)

# Call to data(...): (line 331)
# Processing the call arguments (line 331)
# Getting the type of 'gdtr_' (line 331)
gdtr__536099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 13), 'gdtr_', False)
str_536100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 20), 'str', 'igamma_med_data_ipp-igamma_med_data')

# Obtaining an instance of the builtin type 'tuple' (line 331)
tuple_536101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 60), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 331)
# Adding element type (line 331)
int_536102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 60), tuple_536101, int_536102)
# Adding element type (line 331)
int_536103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 60), tuple_536101, int_536103)

int_536104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 66), 'int')
# Processing the call keyword arguments (line 331)
float_536105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 74), 'float')
keyword_536106 = float_536105
kwargs_536107 = {'rtol': keyword_536106}
# Getting the type of 'data' (line 331)
data_536098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'data', False)
# Calling data(args, kwargs) (line 331)
data_call_result_536108 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), data_536098, *[gdtr__536099, str_536100, tuple_536101, int_536104], **kwargs_536107)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536108)
# Adding element type (line 202)

# Call to data(...): (line 332)
# Processing the call arguments (line 332)
# Getting the type of 'gdtr_' (line 332)
gdtr__536110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 13), 'gdtr_', False)
str_536111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 20), 'str', 'igamma_int_data_ipp-igamma_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 332)
tuple_536112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 60), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 332)
# Adding element type (line 332)
int_536113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 60), tuple_536112, int_536113)
# Adding element type (line 332)
int_536114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 60), tuple_536112, int_536114)

int_536115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 66), 'int')
# Processing the call keyword arguments (line 332)
float_536116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 74), 'float')
keyword_536117 = float_536116
kwargs_536118 = {'rtol': keyword_536117}
# Getting the type of 'data' (line 332)
data_536109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'data', False)
# Calling data(args, kwargs) (line 332)
data_call_result_536119 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), data_536109, *[gdtr__536110, str_536111, tuple_536112, int_536115], **kwargs_536118)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536119)
# Adding element type (line 202)

# Call to data(...): (line 333)
# Processing the call arguments (line 333)
# Getting the type of 'gdtr_' (line 333)
gdtr__536121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 13), 'gdtr_', False)
str_536122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 20), 'str', 'igamma_big_data_ipp-igamma_big_data')

# Obtaining an instance of the builtin type 'tuple' (line 333)
tuple_536123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 60), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 333)
# Adding element type (line 333)
int_536124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 60), tuple_536123, int_536124)
# Adding element type (line 333)
int_536125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 60), tuple_536123, int_536125)

int_536126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 66), 'int')
# Processing the call keyword arguments (line 333)
float_536127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 74), 'float')
keyword_536128 = float_536127
kwargs_536129 = {'rtol': keyword_536128}
# Getting the type of 'data' (line 333)
data_536120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'data', False)
# Calling data(args, kwargs) (line 333)
data_call_result_536130 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), data_536120, *[gdtr__536121, str_536122, tuple_536123, int_536126], **kwargs_536129)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536130)
# Adding element type (line 202)

# Call to data(...): (line 335)
# Processing the call arguments (line 335)
# Getting the type of 'gammaincc' (line 335)
gammaincc_536132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 13), 'gammaincc', False)
str_536133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 24), 'str', 'igamma_small_data_ipp-igamma_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 335)
tuple_536134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 68), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 335)
# Adding element type (line 335)
int_536135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 68), tuple_536134, int_536135)
# Adding element type (line 335)
int_536136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 68), tuple_536134, int_536136)

int_536137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 74), 'int')
# Processing the call keyword arguments (line 335)
float_536138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 82), 'float')
keyword_536139 = float_536138
kwargs_536140 = {'rtol': keyword_536139}
# Getting the type of 'data' (line 335)
data_536131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'data', False)
# Calling data(args, kwargs) (line 335)
data_call_result_536141 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), data_536131, *[gammaincc_536132, str_536133, tuple_536134, int_536137], **kwargs_536140)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536141)
# Adding element type (line 202)

# Call to data(...): (line 336)
# Processing the call arguments (line 336)
# Getting the type of 'gammaincc' (line 336)
gammaincc_536143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 13), 'gammaincc', False)
str_536144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 24), 'str', 'igamma_med_data_ipp-igamma_med_data')

# Obtaining an instance of the builtin type 'tuple' (line 336)
tuple_536145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 336)
# Adding element type (line 336)
int_536146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 64), tuple_536145, int_536146)
# Adding element type (line 336)
int_536147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 64), tuple_536145, int_536147)

int_536148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 70), 'int')
# Processing the call keyword arguments (line 336)
float_536149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 78), 'float')
keyword_536150 = float_536149
kwargs_536151 = {'rtol': keyword_536150}
# Getting the type of 'data' (line 336)
data_536142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'data', False)
# Calling data(args, kwargs) (line 336)
data_call_result_536152 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), data_536142, *[gammaincc_536143, str_536144, tuple_536145, int_536148], **kwargs_536151)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536152)
# Adding element type (line 202)

# Call to data(...): (line 337)
# Processing the call arguments (line 337)
# Getting the type of 'gammaincc' (line 337)
gammaincc_536154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 13), 'gammaincc', False)
str_536155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 24), 'str', 'igamma_int_data_ipp-igamma_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 337)
tuple_536156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 337)
# Adding element type (line 337)
int_536157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 64), tuple_536156, int_536157)
# Adding element type (line 337)
int_536158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 64), tuple_536156, int_536158)

int_536159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 70), 'int')
# Processing the call keyword arguments (line 337)
float_536160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 78), 'float')
keyword_536161 = float_536160
kwargs_536162 = {'rtol': keyword_536161}
# Getting the type of 'data' (line 337)
data_536153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'data', False)
# Calling data(args, kwargs) (line 337)
data_call_result_536163 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), data_536153, *[gammaincc_536154, str_536155, tuple_536156, int_536159], **kwargs_536162)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536163)
# Adding element type (line 202)

# Call to data(...): (line 338)
# Processing the call arguments (line 338)
# Getting the type of 'gammaincc' (line 338)
gammaincc_536165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'gammaincc', False)
str_536166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 24), 'str', 'igamma_big_data_ipp-igamma_big_data')

# Obtaining an instance of the builtin type 'tuple' (line 338)
tuple_536167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 338)
# Adding element type (line 338)
int_536168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 64), tuple_536167, int_536168)
# Adding element type (line 338)
int_536169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 64), tuple_536167, int_536169)

int_536170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 70), 'int')
# Processing the call keyword arguments (line 338)
float_536171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 78), 'float')
keyword_536172 = float_536171
kwargs_536173 = {'rtol': keyword_536172}
# Getting the type of 'data' (line 338)
data_536164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'data', False)
# Calling data(args, kwargs) (line 338)
data_call_result_536174 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), data_536164, *[gammaincc_536165, str_536166, tuple_536167, int_536170], **kwargs_536173)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536174)
# Adding element type (line 202)

# Call to data(...): (line 340)
# Processing the call arguments (line 340)
# Getting the type of 'gdtrc_' (line 340)
gdtrc__536176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'gdtrc_', False)
str_536177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 21), 'str', 'igamma_small_data_ipp-igamma_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 340)
tuple_536178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 65), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 340)
# Adding element type (line 340)
int_536179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 65), tuple_536178, int_536179)
# Adding element type (line 340)
int_536180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 65), tuple_536178, int_536180)

int_536181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 71), 'int')
# Processing the call keyword arguments (line 340)
float_536182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 79), 'float')
keyword_536183 = float_536182
kwargs_536184 = {'rtol': keyword_536183}
# Getting the type of 'data' (line 340)
data_536175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'data', False)
# Calling data(args, kwargs) (line 340)
data_call_result_536185 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), data_536175, *[gdtrc__536176, str_536177, tuple_536178, int_536181], **kwargs_536184)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536185)
# Adding element type (line 202)

# Call to data(...): (line 341)
# Processing the call arguments (line 341)
# Getting the type of 'gdtrc_' (line 341)
gdtrc__536187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 13), 'gdtrc_', False)
str_536188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 21), 'str', 'igamma_med_data_ipp-igamma_med_data')

# Obtaining an instance of the builtin type 'tuple' (line 341)
tuple_536189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 341)
# Adding element type (line 341)
int_536190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 61), tuple_536189, int_536190)
# Adding element type (line 341)
int_536191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 61), tuple_536189, int_536191)

int_536192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 67), 'int')
# Processing the call keyword arguments (line 341)
float_536193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 75), 'float')
keyword_536194 = float_536193
kwargs_536195 = {'rtol': keyword_536194}
# Getting the type of 'data' (line 341)
data_536186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'data', False)
# Calling data(args, kwargs) (line 341)
data_call_result_536196 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), data_536186, *[gdtrc__536187, str_536188, tuple_536189, int_536192], **kwargs_536195)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536196)
# Adding element type (line 202)

# Call to data(...): (line 342)
# Processing the call arguments (line 342)
# Getting the type of 'gdtrc_' (line 342)
gdtrc__536198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'gdtrc_', False)
str_536199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 21), 'str', 'igamma_int_data_ipp-igamma_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 342)
tuple_536200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 342)
# Adding element type (line 342)
int_536201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 61), tuple_536200, int_536201)
# Adding element type (line 342)
int_536202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 61), tuple_536200, int_536202)

int_536203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 67), 'int')
# Processing the call keyword arguments (line 342)
float_536204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 75), 'float')
keyword_536205 = float_536204
kwargs_536206 = {'rtol': keyword_536205}
# Getting the type of 'data' (line 342)
data_536197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'data', False)
# Calling data(args, kwargs) (line 342)
data_call_result_536207 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), data_536197, *[gdtrc__536198, str_536199, tuple_536200, int_536203], **kwargs_536206)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536207)
# Adding element type (line 202)

# Call to data(...): (line 343)
# Processing the call arguments (line 343)
# Getting the type of 'gdtrc_' (line 343)
gdtrc__536209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 13), 'gdtrc_', False)
str_536210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 21), 'str', 'igamma_big_data_ipp-igamma_big_data')

# Obtaining an instance of the builtin type 'tuple' (line 343)
tuple_536211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 343)
# Adding element type (line 343)
int_536212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 61), tuple_536211, int_536212)
# Adding element type (line 343)
int_536213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 61), tuple_536211, int_536213)

int_536214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 67), 'int')
# Processing the call keyword arguments (line 343)
float_536215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 75), 'float')
keyword_536216 = float_536215
kwargs_536217 = {'rtol': keyword_536216}
# Getting the type of 'data' (line 343)
data_536208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'data', False)
# Calling data(args, kwargs) (line 343)
data_call_result_536218 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), data_536208, *[gdtrc__536209, str_536210, tuple_536211, int_536214], **kwargs_536217)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536218)
# Adding element type (line 202)

# Call to data(...): (line 345)
# Processing the call arguments (line 345)
# Getting the type of 'gdtrib_' (line 345)
gdtrib__536220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 13), 'gdtrib_', False)
str_536221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 22), 'str', 'igamma_inva_data_ipp-igamma_inva_data')

# Obtaining an instance of the builtin type 'tuple' (line 345)
tuple_536222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 345)
# Adding element type (line 345)
int_536223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 64), tuple_536222, int_536223)
# Adding element type (line 345)
int_536224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 64), tuple_536222, int_536224)

int_536225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 70), 'int')
# Processing the call keyword arguments (line 345)
float_536226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 78), 'float')
keyword_536227 = float_536226
kwargs_536228 = {'rtol': keyword_536227}
# Getting the type of 'data' (line 345)
data_536219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'data', False)
# Calling data(args, kwargs) (line 345)
data_call_result_536229 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), data_536219, *[gdtrib__536220, str_536221, tuple_536222, int_536225], **kwargs_536228)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536229)
# Adding element type (line 202)

# Call to data(...): (line 346)
# Processing the call arguments (line 346)
# Getting the type of 'gdtrib_comp' (line 346)
gdtrib_comp_536231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 13), 'gdtrib_comp', False)
str_536232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 26), 'str', 'igamma_inva_data_ipp-igamma_inva_data')

# Obtaining an instance of the builtin type 'tuple' (line 346)
tuple_536233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 68), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 346)
# Adding element type (line 346)
int_536234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 68), tuple_536233, int_536234)
# Adding element type (line 346)
int_536235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 68), tuple_536233, int_536235)

int_536236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 74), 'int')
# Processing the call keyword arguments (line 346)
float_536237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 82), 'float')
keyword_536238 = float_536237
kwargs_536239 = {'rtol': keyword_536238}
# Getting the type of 'data' (line 346)
data_536230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'data', False)
# Calling data(args, kwargs) (line 346)
data_call_result_536240 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), data_536230, *[gdtrib_comp_536231, str_536232, tuple_536233, int_536236], **kwargs_536239)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536240)
# Adding element type (line 202)

# Call to data(...): (line 348)
# Processing the call arguments (line 348)
# Getting the type of 'poch_' (line 348)
poch__536242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 13), 'poch_', False)
str_536243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 20), 'str', 'tgamma_delta_ratio_data_ipp-tgamma_delta_ratio_data')

# Obtaining an instance of the builtin type 'tuple' (line 348)
tuple_536244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 76), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 348)
# Adding element type (line 348)
int_536245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 76), tuple_536244, int_536245)
# Adding element type (line 348)
int_536246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 76), tuple_536244, int_536246)

int_536247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 82), 'int')
# Processing the call keyword arguments (line 348)
float_536248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 90), 'float')
keyword_536249 = float_536248
kwargs_536250 = {'rtol': keyword_536249}
# Getting the type of 'data' (line 348)
data_536241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'data', False)
# Calling data(args, kwargs) (line 348)
data_call_result_536251 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), data_536241, *[poch__536242, str_536243, tuple_536244, int_536247], **kwargs_536250)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536251)
# Adding element type (line 202)

# Call to data(...): (line 349)
# Processing the call arguments (line 349)
# Getting the type of 'poch_' (line 349)
poch__536253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 13), 'poch_', False)
str_536254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 20), 'str', 'tgamma_delta_ratio_int_ipp-tgamma_delta_ratio_int')

# Obtaining an instance of the builtin type 'tuple' (line 349)
tuple_536255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 74), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 349)
# Adding element type (line 349)
int_536256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 74), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 74), tuple_536255, int_536256)
# Adding element type (line 349)
int_536257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 74), tuple_536255, int_536257)

int_536258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 80), 'int')
# Processing the call keyword arguments (line 349)
kwargs_536259 = {}
# Getting the type of 'data' (line 349)
data_536252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'data', False)
# Calling data(args, kwargs) (line 349)
data_call_result_536260 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), data_536252, *[poch__536253, str_536254, tuple_536255, int_536258], **kwargs_536259)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536260)
# Adding element type (line 202)

# Call to data(...): (line 350)
# Processing the call arguments (line 350)
# Getting the type of 'poch_' (line 350)
poch__536262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'poch_', False)
str_536263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 20), 'str', 'tgamma_delta_ratio_int2_ipp-tgamma_delta_ratio_int2')

# Obtaining an instance of the builtin type 'tuple' (line 350)
tuple_536264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 76), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 350)
# Adding element type (line 350)
int_536265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 76), tuple_536264, int_536265)
# Adding element type (line 350)
int_536266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 76), tuple_536264, int_536266)

int_536267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 82), 'int')
# Processing the call keyword arguments (line 350)
kwargs_536268 = {}
# Getting the type of 'data' (line 350)
data_536261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'data', False)
# Calling data(args, kwargs) (line 350)
data_call_result_536269 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), data_536261, *[poch__536262, str_536263, tuple_536264, int_536267], **kwargs_536268)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536269)
# Adding element type (line 202)

# Call to data(...): (line 351)
# Processing the call arguments (line 351)
# Getting the type of 'poch_minus' (line 351)
poch_minus_536271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'poch_minus', False)
str_536272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 25), 'str', 'tgamma_delta_ratio_data_ipp-tgamma_delta_ratio_data')

# Obtaining an instance of the builtin type 'tuple' (line 351)
tuple_536273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 81), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 351)
# Adding element type (line 351)
int_536274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 81), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 81), tuple_536273, int_536274)
# Adding element type (line 351)
int_536275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 83), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 81), tuple_536273, int_536275)

int_536276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 87), 'int')
# Processing the call keyword arguments (line 351)
float_536277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 95), 'float')
keyword_536278 = float_536277
kwargs_536279 = {'rtol': keyword_536278}
# Getting the type of 'data' (line 351)
data_536270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'data', False)
# Calling data(args, kwargs) (line 351)
data_call_result_536280 = invoke(stypy.reporting.localization.Localization(__file__, 351, 8), data_536270, *[poch_minus_536271, str_536272, tuple_536273, int_536276], **kwargs_536279)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536280)
# Adding element type (line 202)

# Call to data(...): (line 352)
# Processing the call arguments (line 352)
# Getting the type of 'poch_minus' (line 352)
poch_minus_536282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 13), 'poch_minus', False)
str_536283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 25), 'str', 'tgamma_delta_ratio_int_ipp-tgamma_delta_ratio_int')

# Obtaining an instance of the builtin type 'tuple' (line 352)
tuple_536284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 79), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 352)
# Adding element type (line 352)
int_536285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 79), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 79), tuple_536284, int_536285)
# Adding element type (line 352)
int_536286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 81), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 79), tuple_536284, int_536286)

int_536287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 85), 'int')
# Processing the call keyword arguments (line 352)
kwargs_536288 = {}
# Getting the type of 'data' (line 352)
data_536281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'data', False)
# Calling data(args, kwargs) (line 352)
data_call_result_536289 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), data_536281, *[poch_minus_536282, str_536283, tuple_536284, int_536287], **kwargs_536288)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536289)
# Adding element type (line 202)

# Call to data(...): (line 353)
# Processing the call arguments (line 353)
# Getting the type of 'poch_minus' (line 353)
poch_minus_536291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 13), 'poch_minus', False)
str_536292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 25), 'str', 'tgamma_delta_ratio_int2_ipp-tgamma_delta_ratio_int2')

# Obtaining an instance of the builtin type 'tuple' (line 353)
tuple_536293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 81), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 353)
# Adding element type (line 353)
int_536294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 81), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 81), tuple_536293, int_536294)
# Adding element type (line 353)
int_536295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 83), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 81), tuple_536293, int_536295)

int_536296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 87), 'int')
# Processing the call keyword arguments (line 353)
kwargs_536297 = {}
# Getting the type of 'data' (line 353)
data_536290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'data', False)
# Calling data(args, kwargs) (line 353)
data_call_result_536298 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), data_536290, *[poch_minus_536291, str_536292, tuple_536293, int_536296], **kwargs_536297)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536298)
# Adding element type (line 202)

# Call to data(...): (line 356)
# Processing the call arguments (line 356)
# Getting the type of 'eval_hermite_ld' (line 356)
eval_hermite_ld_536300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 13), 'eval_hermite_ld', False)
str_536301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 30), 'str', 'hermite_ipp-hermite')

# Obtaining an instance of the builtin type 'tuple' (line 356)
tuple_536302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 54), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 356)
# Adding element type (line 356)
int_536303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 54), tuple_536302, int_536303)
# Adding element type (line 356)
int_536304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 54), tuple_536302, int_536304)

int_536305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 60), 'int')
# Processing the call keyword arguments (line 356)
float_536306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 68), 'float')
keyword_536307 = float_536306
kwargs_536308 = {'rtol': keyword_536307}
# Getting the type of 'data' (line 356)
data_536299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'data', False)
# Calling data(args, kwargs) (line 356)
data_call_result_536309 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), data_536299, *[eval_hermite_ld_536300, str_536301, tuple_536302, int_536305], **kwargs_536308)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536309)
# Adding element type (line 202)

# Call to data(...): (line 357)
# Processing the call arguments (line 357)
# Getting the type of 'eval_laguerre_ld' (line 357)
eval_laguerre_ld_536311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'eval_laguerre_ld', False)
str_536312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 31), 'str', 'laguerre2_ipp-laguerre2')

# Obtaining an instance of the builtin type 'tuple' (line 357)
tuple_536313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 59), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 357)
# Adding element type (line 357)
int_536314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 59), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 59), tuple_536313, int_536314)
# Adding element type (line 357)
int_536315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 59), tuple_536313, int_536315)

int_536316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 65), 'int')
# Processing the call keyword arguments (line 357)
float_536317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 73), 'float')
keyword_536318 = float_536317
kwargs_536319 = {'rtol': keyword_536318}
# Getting the type of 'data' (line 357)
data_536310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'data', False)
# Calling data(args, kwargs) (line 357)
data_call_result_536320 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), data_536310, *[eval_laguerre_ld_536311, str_536312, tuple_536313, int_536316], **kwargs_536319)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536320)
# Adding element type (line 202)

# Call to data(...): (line 358)
# Processing the call arguments (line 358)
# Getting the type of 'eval_laguerre_dd' (line 358)
eval_laguerre_dd_536322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 13), 'eval_laguerre_dd', False)
str_536323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 31), 'str', 'laguerre2_ipp-laguerre2')

# Obtaining an instance of the builtin type 'tuple' (line 358)
tuple_536324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 59), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 358)
# Adding element type (line 358)
int_536325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 59), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 59), tuple_536324, int_536325)
# Adding element type (line 358)
int_536326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 59), tuple_536324, int_536326)

int_536327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 65), 'int')
# Processing the call keyword arguments (line 358)
str_536328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 81), 'str', 'hyp2f1 insufficiently accurate.')
keyword_536329 = str_536328
kwargs_536330 = {'knownfailure': keyword_536329}
# Getting the type of 'data' (line 358)
data_536321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'data', False)
# Calling data(args, kwargs) (line 358)
data_call_result_536331 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), data_536321, *[eval_laguerre_dd_536322, str_536323, tuple_536324, int_536327], **kwargs_536330)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536331)
# Adding element type (line 202)

# Call to data(...): (line 359)
# Processing the call arguments (line 359)
# Getting the type of 'eval_genlaguerre_ldd' (line 359)
eval_genlaguerre_ldd_536333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'eval_genlaguerre_ldd', False)
str_536334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 35), 'str', 'laguerre3_ipp-laguerre3')

# Obtaining an instance of the builtin type 'tuple' (line 359)
tuple_536335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 359)
# Adding element type (line 359)
int_536336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 63), tuple_536335, int_536336)
# Adding element type (line 359)
int_536337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 63), tuple_536335, int_536337)
# Adding element type (line 359)
int_536338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 63), tuple_536335, int_536338)

int_536339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 71), 'int')
# Processing the call keyword arguments (line 359)
float_536340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 79), 'float')
keyword_536341 = float_536340
kwargs_536342 = {'rtol': keyword_536341}
# Getting the type of 'data' (line 359)
data_536332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'data', False)
# Calling data(args, kwargs) (line 359)
data_call_result_536343 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), data_536332, *[eval_genlaguerre_ldd_536333, str_536334, tuple_536335, int_536339], **kwargs_536342)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536343)
# Adding element type (line 202)

# Call to data(...): (line 360)
# Processing the call arguments (line 360)
# Getting the type of 'eval_genlaguerre_ddd' (line 360)
eval_genlaguerre_ddd_536345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 13), 'eval_genlaguerre_ddd', False)
str_536346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 35), 'str', 'laguerre3_ipp-laguerre3')

# Obtaining an instance of the builtin type 'tuple' (line 360)
tuple_536347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 63), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 360)
# Adding element type (line 360)
int_536348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 63), tuple_536347, int_536348)
# Adding element type (line 360)
int_536349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 63), tuple_536347, int_536349)
# Adding element type (line 360)
int_536350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 63), tuple_536347, int_536350)

int_536351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 71), 'int')
# Processing the call keyword arguments (line 360)
str_536352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 87), 'str', 'hyp2f1 insufficiently accurate.')
keyword_536353 = str_536352
kwargs_536354 = {'knownfailure': keyword_536353}
# Getting the type of 'data' (line 360)
data_536344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'data', False)
# Calling data(args, kwargs) (line 360)
data_call_result_536355 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), data_536344, *[eval_genlaguerre_ddd_536345, str_536346, tuple_536347, int_536351], **kwargs_536354)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536355)
# Adding element type (line 202)

# Call to data(...): (line 362)
# Processing the call arguments (line 362)
# Getting the type of 'log1p' (line 362)
log1p_536357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 13), 'log1p', False)
str_536358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 20), 'str', 'log1p_expm1_data_ipp-log1p_expm1_data')
int_536359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 61), 'int')
int_536360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 64), 'int')
# Processing the call keyword arguments (line 362)
kwargs_536361 = {}
# Getting the type of 'data' (line 362)
data_536356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'data', False)
# Calling data(args, kwargs) (line 362)
data_call_result_536362 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), data_536356, *[log1p_536357, str_536358, int_536359, int_536360], **kwargs_536361)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536362)
# Adding element type (line 202)

# Call to data(...): (line 363)
# Processing the call arguments (line 363)
# Getting the type of 'expm1' (line 363)
expm1_536364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 13), 'expm1', False)
str_536365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 20), 'str', 'log1p_expm1_data_ipp-log1p_expm1_data')
int_536366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 61), 'int')
int_536367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 64), 'int')
# Processing the call keyword arguments (line 363)
kwargs_536368 = {}
# Getting the type of 'data' (line 363)
data_536363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'data', False)
# Calling data(args, kwargs) (line 363)
data_call_result_536369 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), data_536363, *[expm1_536364, str_536365, int_536366, int_536367], **kwargs_536368)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536369)
# Adding element type (line 202)

# Call to data(...): (line 365)
# Processing the call arguments (line 365)
# Getting the type of 'iv' (line 365)
iv_536371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 13), 'iv', False)
str_536372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 17), 'str', 'bessel_i_data_ipp-bessel_i_data')

# Obtaining an instance of the builtin type 'tuple' (line 365)
tuple_536373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 53), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 365)
# Adding element type (line 365)
int_536374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 53), tuple_536373, int_536374)
# Adding element type (line 365)
int_536375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 53), tuple_536373, int_536375)

int_536376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 59), 'int')
# Processing the call keyword arguments (line 365)
float_536377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 67), 'float')
keyword_536378 = float_536377
kwargs_536379 = {'rtol': keyword_536378}
# Getting the type of 'data' (line 365)
data_536370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'data', False)
# Calling data(args, kwargs) (line 365)
data_call_result_536380 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), data_536370, *[iv_536371, str_536372, tuple_536373, int_536376], **kwargs_536379)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536380)
# Adding element type (line 202)

# Call to data(...): (line 366)
# Processing the call arguments (line 366)
# Getting the type of 'iv' (line 366)
iv_536382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 13), 'iv', False)
str_536383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 17), 'str', 'bessel_i_data_ipp-bessel_i_data')

# Obtaining an instance of the builtin type 'tuple' (line 366)
tuple_536384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 53), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 366)
# Adding element type (line 366)
int_536385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 53), tuple_536384, int_536385)
# Adding element type (line 366)
complex_536386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 55), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 53), tuple_536384, complex_536386)

int_536387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 60), 'int')
# Processing the call keyword arguments (line 366)
float_536388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 68), 'float')
keyword_536389 = float_536388
float_536390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 80), 'float')
keyword_536391 = float_536390
kwargs_536392 = {'rtol': keyword_536389, 'atol': keyword_536391}
# Getting the type of 'data' (line 366)
data_536381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'data', False)
# Calling data(args, kwargs) (line 366)
data_call_result_536393 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), data_536381, *[iv_536382, str_536383, tuple_536384, int_536387], **kwargs_536392)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536393)
# Adding element type (line 202)

# Call to data(...): (line 367)
# Processing the call arguments (line 367)
# Getting the type of 'iv' (line 367)
iv_536395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 'iv', False)
str_536396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 17), 'str', 'bessel_i_int_data_ipp-bessel_i_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 367)
tuple_536397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 367)
# Adding element type (line 367)
int_536398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 61), tuple_536397, int_536398)
# Adding element type (line 367)
int_536399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 61), tuple_536397, int_536399)

int_536400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 67), 'int')
# Processing the call keyword arguments (line 367)
float_536401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 75), 'float')
keyword_536402 = float_536401
kwargs_536403 = {'rtol': keyword_536402}
# Getting the type of 'data' (line 367)
data_536394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'data', False)
# Calling data(args, kwargs) (line 367)
data_call_result_536404 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), data_536394, *[iv_536395, str_536396, tuple_536397, int_536400], **kwargs_536403)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536404)
# Adding element type (line 202)

# Call to data(...): (line 368)
# Processing the call arguments (line 368)
# Getting the type of 'iv' (line 368)
iv_536406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 13), 'iv', False)
str_536407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 17), 'str', 'bessel_i_int_data_ipp-bessel_i_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 368)
tuple_536408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 368)
# Adding element type (line 368)
int_536409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 61), tuple_536408, int_536409)
# Adding element type (line 368)
complex_536410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 63), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 61), tuple_536408, complex_536410)

int_536411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 68), 'int')
# Processing the call keyword arguments (line 368)
float_536412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 76), 'float')
keyword_536413 = float_536412
kwargs_536414 = {'rtol': keyword_536413}
# Getting the type of 'data' (line 368)
data_536405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'data', False)
# Calling data(args, kwargs) (line 368)
data_call_result_536415 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), data_536405, *[iv_536406, str_536407, tuple_536408, int_536411], **kwargs_536414)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536415)
# Adding element type (line 202)

# Call to data(...): (line 370)
# Processing the call arguments (line 370)
# Getting the type of 'jn' (line 370)
jn_536417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 13), 'jn', False)
str_536418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 17), 'str', 'bessel_j_int_data_ipp-bessel_j_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 370)
tuple_536419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 370)
# Adding element type (line 370)
int_536420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 61), tuple_536419, int_536420)
# Adding element type (line 370)
int_536421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 61), tuple_536419, int_536421)

int_536422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 67), 'int')
# Processing the call keyword arguments (line 370)
float_536423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 75), 'float')
keyword_536424 = float_536423
kwargs_536425 = {'rtol': keyword_536424}
# Getting the type of 'data' (line 370)
data_536416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'data', False)
# Calling data(args, kwargs) (line 370)
data_call_result_536426 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), data_536416, *[jn_536417, str_536418, tuple_536419, int_536422], **kwargs_536425)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536426)
# Adding element type (line 202)

# Call to data(...): (line 371)
# Processing the call arguments (line 371)
# Getting the type of 'jn' (line 371)
jn_536428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 13), 'jn', False)
str_536429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 17), 'str', 'bessel_j_int_data_ipp-bessel_j_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 371)
tuple_536430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 371)
# Adding element type (line 371)
int_536431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 61), tuple_536430, int_536431)
# Adding element type (line 371)
complex_536432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 63), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 61), tuple_536430, complex_536432)

int_536433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 68), 'int')
# Processing the call keyword arguments (line 371)
float_536434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 76), 'float')
keyword_536435 = float_536434
kwargs_536436 = {'rtol': keyword_536435}
# Getting the type of 'data' (line 371)
data_536427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'data', False)
# Calling data(args, kwargs) (line 371)
data_call_result_536437 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), data_536427, *[jn_536428, str_536429, tuple_536430, int_536433], **kwargs_536436)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536437)
# Adding element type (line 202)

# Call to data(...): (line 372)
# Processing the call arguments (line 372)
# Getting the type of 'jn' (line 372)
jn_536439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 13), 'jn', False)
str_536440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 17), 'str', 'bessel_j_large_data_ipp-bessel_j_large_data')

# Obtaining an instance of the builtin type 'tuple' (line 372)
tuple_536441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 65), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 372)
# Adding element type (line 372)
int_536442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 65), tuple_536441, int_536442)
# Adding element type (line 372)
int_536443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 65), tuple_536441, int_536443)

int_536444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 71), 'int')
# Processing the call keyword arguments (line 372)
float_536445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 79), 'float')
keyword_536446 = float_536445
kwargs_536447 = {'rtol': keyword_536446}
# Getting the type of 'data' (line 372)
data_536438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'data', False)
# Calling data(args, kwargs) (line 372)
data_call_result_536448 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), data_536438, *[jn_536439, str_536440, tuple_536441, int_536444], **kwargs_536447)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536448)
# Adding element type (line 202)

# Call to data(...): (line 373)
# Processing the call arguments (line 373)
# Getting the type of 'jn' (line 373)
jn_536450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 13), 'jn', False)
str_536451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 17), 'str', 'bessel_j_large_data_ipp-bessel_j_large_data')

# Obtaining an instance of the builtin type 'tuple' (line 373)
tuple_536452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 65), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 373)
# Adding element type (line 373)
int_536453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 65), tuple_536452, int_536453)
# Adding element type (line 373)
complex_536454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 67), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 65), tuple_536452, complex_536454)

int_536455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 72), 'int')
# Processing the call keyword arguments (line 373)
float_536456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 80), 'float')
keyword_536457 = float_536456
kwargs_536458 = {'rtol': keyword_536457}
# Getting the type of 'data' (line 373)
data_536449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'data', False)
# Calling data(args, kwargs) (line 373)
data_call_result_536459 = invoke(stypy.reporting.localization.Localization(__file__, 373, 8), data_536449, *[jn_536450, str_536451, tuple_536452, int_536455], **kwargs_536458)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536459)
# Adding element type (line 202)

# Call to data(...): (line 375)
# Processing the call arguments (line 375)
# Getting the type of 'jv' (line 375)
jv_536461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 13), 'jv', False)
str_536462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 17), 'str', 'bessel_j_int_data_ipp-bessel_j_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 375)
tuple_536463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 375)
# Adding element type (line 375)
int_536464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 61), tuple_536463, int_536464)
# Adding element type (line 375)
int_536465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 61), tuple_536463, int_536465)

int_536466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 67), 'int')
# Processing the call keyword arguments (line 375)
float_536467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 75), 'float')
keyword_536468 = float_536467
kwargs_536469 = {'rtol': keyword_536468}
# Getting the type of 'data' (line 375)
data_536460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'data', False)
# Calling data(args, kwargs) (line 375)
data_call_result_536470 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), data_536460, *[jv_536461, str_536462, tuple_536463, int_536466], **kwargs_536469)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536470)
# Adding element type (line 202)

# Call to data(...): (line 376)
# Processing the call arguments (line 376)
# Getting the type of 'jv' (line 376)
jv_536472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 13), 'jv', False)
str_536473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 17), 'str', 'bessel_j_int_data_ipp-bessel_j_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 376)
tuple_536474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 376)
# Adding element type (line 376)
int_536475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 61), tuple_536474, int_536475)
# Adding element type (line 376)
complex_536476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 63), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 61), tuple_536474, complex_536476)

int_536477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 68), 'int')
# Processing the call keyword arguments (line 376)
float_536478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 76), 'float')
keyword_536479 = float_536478
kwargs_536480 = {'rtol': keyword_536479}
# Getting the type of 'data' (line 376)
data_536471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'data', False)
# Calling data(args, kwargs) (line 376)
data_call_result_536481 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), data_536471, *[jv_536472, str_536473, tuple_536474, int_536477], **kwargs_536480)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536481)
# Adding element type (line 202)

# Call to data(...): (line 377)
# Processing the call arguments (line 377)
# Getting the type of 'jv' (line 377)
jv_536483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 13), 'jv', False)
str_536484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 17), 'str', 'bessel_j_data_ipp-bessel_j_data')

# Obtaining an instance of the builtin type 'tuple' (line 377)
tuple_536485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 53), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 377)
# Adding element type (line 377)
int_536486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 53), tuple_536485, int_536486)
# Adding element type (line 377)
int_536487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 53), tuple_536485, int_536487)

int_536488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 59), 'int')
# Processing the call keyword arguments (line 377)
float_536489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 67), 'float')
keyword_536490 = float_536489
kwargs_536491 = {'rtol': keyword_536490}
# Getting the type of 'data' (line 377)
data_536482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'data', False)
# Calling data(args, kwargs) (line 377)
data_call_result_536492 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), data_536482, *[jv_536483, str_536484, tuple_536485, int_536488], **kwargs_536491)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536492)
# Adding element type (line 202)

# Call to data(...): (line 378)
# Processing the call arguments (line 378)
# Getting the type of 'jv' (line 378)
jv_536494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 13), 'jv', False)
str_536495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 17), 'str', 'bessel_j_data_ipp-bessel_j_data')

# Obtaining an instance of the builtin type 'tuple' (line 378)
tuple_536496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 53), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 378)
# Adding element type (line 378)
int_536497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 53), tuple_536496, int_536497)
# Adding element type (line 378)
complex_536498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 55), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 53), tuple_536496, complex_536498)

int_536499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 60), 'int')
# Processing the call keyword arguments (line 378)
float_536500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 68), 'float')
keyword_536501 = float_536500
kwargs_536502 = {'rtol': keyword_536501}
# Getting the type of 'data' (line 378)
data_536493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'data', False)
# Calling data(args, kwargs) (line 378)
data_call_result_536503 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), data_536493, *[jv_536494, str_536495, tuple_536496, int_536499], **kwargs_536502)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536503)
# Adding element type (line 202)

# Call to data(...): (line 380)
# Processing the call arguments (line 380)
# Getting the type of 'kn' (line 380)
kn_536505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 13), 'kn', False)
str_536506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 17), 'str', 'bessel_k_int_data_ipp-bessel_k_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 380)
tuple_536507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 380)
# Adding element type (line 380)
int_536508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 61), tuple_536507, int_536508)
# Adding element type (line 380)
int_536509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 61), tuple_536507, int_536509)

int_536510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 67), 'int')
# Processing the call keyword arguments (line 380)
float_536511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 75), 'float')
keyword_536512 = float_536511
kwargs_536513 = {'rtol': keyword_536512}
# Getting the type of 'data' (line 380)
data_536504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'data', False)
# Calling data(args, kwargs) (line 380)
data_call_result_536514 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), data_536504, *[kn_536505, str_536506, tuple_536507, int_536510], **kwargs_536513)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536514)
# Adding element type (line 202)

# Call to data(...): (line 382)
# Processing the call arguments (line 382)
# Getting the type of 'kv' (line 382)
kv_536516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 13), 'kv', False)
str_536517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 17), 'str', 'bessel_k_int_data_ipp-bessel_k_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 382)
tuple_536518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 382)
# Adding element type (line 382)
int_536519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 61), tuple_536518, int_536519)
# Adding element type (line 382)
int_536520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 61), tuple_536518, int_536520)

int_536521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 67), 'int')
# Processing the call keyword arguments (line 382)
float_536522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 75), 'float')
keyword_536523 = float_536522
kwargs_536524 = {'rtol': keyword_536523}
# Getting the type of 'data' (line 382)
data_536515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'data', False)
# Calling data(args, kwargs) (line 382)
data_call_result_536525 = invoke(stypy.reporting.localization.Localization(__file__, 382, 8), data_536515, *[kv_536516, str_536517, tuple_536518, int_536521], **kwargs_536524)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536525)
# Adding element type (line 202)

# Call to data(...): (line 383)
# Processing the call arguments (line 383)
# Getting the type of 'kv' (line 383)
kv_536527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 13), 'kv', False)
str_536528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 17), 'str', 'bessel_k_int_data_ipp-bessel_k_int_data')

# Obtaining an instance of the builtin type 'tuple' (line 383)
tuple_536529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 61), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 383)
# Adding element type (line 383)
int_536530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 61), tuple_536529, int_536530)
# Adding element type (line 383)
complex_536531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 63), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 61), tuple_536529, complex_536531)

int_536532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 68), 'int')
# Processing the call keyword arguments (line 383)
float_536533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 76), 'float')
keyword_536534 = float_536533
kwargs_536535 = {'rtol': keyword_536534}
# Getting the type of 'data' (line 383)
data_536526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'data', False)
# Calling data(args, kwargs) (line 383)
data_call_result_536536 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), data_536526, *[kv_536527, str_536528, tuple_536529, int_536532], **kwargs_536535)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536536)
# Adding element type (line 202)

# Call to data(...): (line 384)
# Processing the call arguments (line 384)
# Getting the type of 'kv' (line 384)
kv_536538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 13), 'kv', False)
str_536539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 17), 'str', 'bessel_k_data_ipp-bessel_k_data')

# Obtaining an instance of the builtin type 'tuple' (line 384)
tuple_536540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 53), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 384)
# Adding element type (line 384)
int_536541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 53), tuple_536540, int_536541)
# Adding element type (line 384)
int_536542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 53), tuple_536540, int_536542)

int_536543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 59), 'int')
# Processing the call keyword arguments (line 384)
float_536544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 67), 'float')
keyword_536545 = float_536544
kwargs_536546 = {'rtol': keyword_536545}
# Getting the type of 'data' (line 384)
data_536537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'data', False)
# Calling data(args, kwargs) (line 384)
data_call_result_536547 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), data_536537, *[kv_536538, str_536539, tuple_536540, int_536543], **kwargs_536546)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536547)
# Adding element type (line 202)

# Call to data(...): (line 385)
# Processing the call arguments (line 385)
# Getting the type of 'kv' (line 385)
kv_536549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 13), 'kv', False)
str_536550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 17), 'str', 'bessel_k_data_ipp-bessel_k_data')

# Obtaining an instance of the builtin type 'tuple' (line 385)
tuple_536551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 53), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 385)
# Adding element type (line 385)
int_536552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 53), tuple_536551, int_536552)
# Adding element type (line 385)
complex_536553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 55), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 53), tuple_536551, complex_536553)

int_536554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 60), 'int')
# Processing the call keyword arguments (line 385)
float_536555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 68), 'float')
keyword_536556 = float_536555
kwargs_536557 = {'rtol': keyword_536556}
# Getting the type of 'data' (line 385)
data_536548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'data', False)
# Calling data(args, kwargs) (line 385)
data_call_result_536558 = invoke(stypy.reporting.localization.Localization(__file__, 385, 8), data_536548, *[kv_536549, str_536550, tuple_536551, int_536554], **kwargs_536557)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536558)
# Adding element type (line 202)

# Call to data(...): (line 387)
# Processing the call arguments (line 387)
# Getting the type of 'yn' (line 387)
yn_536560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 13), 'yn', False)
str_536561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 17), 'str', 'bessel_y01_data_ipp-bessel_y01_data')

# Obtaining an instance of the builtin type 'tuple' (line 387)
tuple_536562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 57), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 387)
# Adding element type (line 387)
int_536563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 57), tuple_536562, int_536563)
# Adding element type (line 387)
int_536564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 59), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 57), tuple_536562, int_536564)

int_536565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 63), 'int')
# Processing the call keyword arguments (line 387)
float_536566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 71), 'float')
keyword_536567 = float_536566
kwargs_536568 = {'rtol': keyword_536567}
# Getting the type of 'data' (line 387)
data_536559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'data', False)
# Calling data(args, kwargs) (line 387)
data_call_result_536569 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), data_536559, *[yn_536560, str_536561, tuple_536562, int_536565], **kwargs_536568)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536569)
# Adding element type (line 202)

# Call to data(...): (line 388)
# Processing the call arguments (line 388)
# Getting the type of 'yn' (line 388)
yn_536571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 13), 'yn', False)
str_536572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 17), 'str', 'bessel_yn_data_ipp-bessel_yn_data')

# Obtaining an instance of the builtin type 'tuple' (line 388)
tuple_536573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 388)
# Adding element type (line 388)
int_536574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 55), tuple_536573, int_536574)
# Adding element type (line 388)
int_536575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 55), tuple_536573, int_536575)

int_536576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 61), 'int')
# Processing the call keyword arguments (line 388)
float_536577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 69), 'float')
keyword_536578 = float_536577
kwargs_536579 = {'rtol': keyword_536578}
# Getting the type of 'data' (line 388)
data_536570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'data', False)
# Calling data(args, kwargs) (line 388)
data_call_result_536580 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), data_536570, *[yn_536571, str_536572, tuple_536573, int_536576], **kwargs_536579)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536580)
# Adding element type (line 202)

# Call to data(...): (line 390)
# Processing the call arguments (line 390)
# Getting the type of 'yv' (line 390)
yv_536582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 13), 'yv', False)
str_536583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 17), 'str', 'bessel_yn_data_ipp-bessel_yn_data')

# Obtaining an instance of the builtin type 'tuple' (line 390)
tuple_536584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 390)
# Adding element type (line 390)
int_536585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 55), tuple_536584, int_536585)
# Adding element type (line 390)
int_536586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 55), tuple_536584, int_536586)

int_536587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 61), 'int')
# Processing the call keyword arguments (line 390)
float_536588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 69), 'float')
keyword_536589 = float_536588
kwargs_536590 = {'rtol': keyword_536589}
# Getting the type of 'data' (line 390)
data_536581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'data', False)
# Calling data(args, kwargs) (line 390)
data_call_result_536591 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), data_536581, *[yv_536582, str_536583, tuple_536584, int_536587], **kwargs_536590)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536591)
# Adding element type (line 202)

# Call to data(...): (line 391)
# Processing the call arguments (line 391)
# Getting the type of 'yv' (line 391)
yv_536593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 13), 'yv', False)
str_536594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 17), 'str', 'bessel_yn_data_ipp-bessel_yn_data')

# Obtaining an instance of the builtin type 'tuple' (line 391)
tuple_536595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 391)
# Adding element type (line 391)
int_536596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 55), tuple_536595, int_536596)
# Adding element type (line 391)
complex_536597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 57), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 55), tuple_536595, complex_536597)

int_536598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 62), 'int')
# Processing the call keyword arguments (line 391)
float_536599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 70), 'float')
keyword_536600 = float_536599
kwargs_536601 = {'rtol': keyword_536600}
# Getting the type of 'data' (line 391)
data_536592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'data', False)
# Calling data(args, kwargs) (line 391)
data_call_result_536602 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), data_536592, *[yv_536593, str_536594, tuple_536595, int_536598], **kwargs_536601)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536602)
# Adding element type (line 202)

# Call to data(...): (line 392)
# Processing the call arguments (line 392)
# Getting the type of 'yv' (line 392)
yv_536604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 13), 'yv', False)
str_536605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 17), 'str', 'bessel_yv_data_ipp-bessel_yv_data')

# Obtaining an instance of the builtin type 'tuple' (line 392)
tuple_536606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 392)
# Adding element type (line 392)
int_536607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 55), tuple_536606, int_536607)
# Adding element type (line 392)
int_536608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 55), tuple_536606, int_536608)

int_536609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 61), 'int')
# Processing the call keyword arguments (line 392)
float_536610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 69), 'float')
keyword_536611 = float_536610
kwargs_536612 = {'rtol': keyword_536611}
# Getting the type of 'data' (line 392)
data_536603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'data', False)
# Calling data(args, kwargs) (line 392)
data_call_result_536613 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), data_536603, *[yv_536604, str_536605, tuple_536606, int_536609], **kwargs_536612)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536613)
# Adding element type (line 202)

# Call to data(...): (line 393)
# Processing the call arguments (line 393)
# Getting the type of 'yv' (line 393)
yv_536615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 13), 'yv', False)
str_536616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 17), 'str', 'bessel_yv_data_ipp-bessel_yv_data')

# Obtaining an instance of the builtin type 'tuple' (line 393)
tuple_536617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 393)
# Adding element type (line 393)
int_536618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 55), tuple_536617, int_536618)
# Adding element type (line 393)
complex_536619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 57), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 55), tuple_536617, complex_536619)

int_536620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 62), 'int')
# Processing the call keyword arguments (line 393)
float_536621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 70), 'float')
keyword_536622 = float_536621
kwargs_536623 = {'rtol': keyword_536622}
# Getting the type of 'data' (line 393)
data_536614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'data', False)
# Calling data(args, kwargs) (line 393)
data_call_result_536624 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), data_536614, *[yv_536615, str_536616, tuple_536617, int_536620], **kwargs_536623)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536624)
# Adding element type (line 202)

# Call to data(...): (line 395)
# Processing the call arguments (line 395)
# Getting the type of 'zeta_' (line 395)
zeta__536626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 13), 'zeta_', False)
str_536627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 20), 'str', 'zeta_data_ipp-zeta_data')
int_536628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 47), 'int')
int_536629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 50), 'int')
# Processing the call keyword arguments (line 395)

@norecursion
def _stypy_temp_lambda_326(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_326'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_326', 395, 67, True)
    # Passed parameters checking function
    _stypy_temp_lambda_326.stypy_localization = localization
    _stypy_temp_lambda_326.stypy_type_of_self = None
    _stypy_temp_lambda_326.stypy_type_store = module_type_store
    _stypy_temp_lambda_326.stypy_function_name = '_stypy_temp_lambda_326'
    _stypy_temp_lambda_326.stypy_param_names_list = ['s']
    _stypy_temp_lambda_326.stypy_varargs_param_name = None
    _stypy_temp_lambda_326.stypy_kwargs_param_name = None
    _stypy_temp_lambda_326.stypy_call_defaults = defaults
    _stypy_temp_lambda_326.stypy_call_varargs = varargs
    _stypy_temp_lambda_326.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_326', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_326', ['s'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Getting the type of 's' (line 395)
    s_536630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 77), 's', False)
    int_536631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 81), 'int')
    # Applying the binary operator '>' (line 395)
    result_gt_536632 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 77), '>', s_536630, int_536631)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 67), 'stypy_return_type', result_gt_536632)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_326' in the type store
    # Getting the type of 'stypy_return_type' (line 395)
    stypy_return_type_536633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 67), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536633)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_326'
    return stypy_return_type_536633

# Assigning a type to the variable '_stypy_temp_lambda_326' (line 395)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 67), '_stypy_temp_lambda_326', _stypy_temp_lambda_326)
# Getting the type of '_stypy_temp_lambda_326' (line 395)
_stypy_temp_lambda_326_536634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 67), '_stypy_temp_lambda_326')
keyword_536635 = _stypy_temp_lambda_326_536634
kwargs_536636 = {'param_filter': keyword_536635}
# Getting the type of 'data' (line 395)
data_536625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'data', False)
# Calling data(args, kwargs) (line 395)
data_call_result_536637 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), data_536625, *[zeta__536626, str_536627, int_536628, int_536629], **kwargs_536636)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536637)
# Adding element type (line 202)

# Call to data(...): (line 396)
# Processing the call arguments (line 396)
# Getting the type of 'zeta_' (line 396)
zeta__536639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 13), 'zeta_', False)
str_536640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 20), 'str', 'zeta_neg_data_ipp-zeta_neg_data')
int_536641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 55), 'int')
int_536642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 58), 'int')
# Processing the call keyword arguments (line 396)

@norecursion
def _stypy_temp_lambda_327(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_327'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_327', 396, 75, True)
    # Passed parameters checking function
    _stypy_temp_lambda_327.stypy_localization = localization
    _stypy_temp_lambda_327.stypy_type_of_self = None
    _stypy_temp_lambda_327.stypy_type_store = module_type_store
    _stypy_temp_lambda_327.stypy_function_name = '_stypy_temp_lambda_327'
    _stypy_temp_lambda_327.stypy_param_names_list = ['s']
    _stypy_temp_lambda_327.stypy_varargs_param_name = None
    _stypy_temp_lambda_327.stypy_kwargs_param_name = None
    _stypy_temp_lambda_327.stypy_call_defaults = defaults
    _stypy_temp_lambda_327.stypy_call_varargs = varargs
    _stypy_temp_lambda_327.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_327', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_327', ['s'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Getting the type of 's' (line 396)
    s_536643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 85), 's', False)
    int_536644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 89), 'int')
    # Applying the binary operator '>' (line 396)
    result_gt_536645 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 85), '>', s_536643, int_536644)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 75), 'stypy_return_type', result_gt_536645)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_327' in the type store
    # Getting the type of 'stypy_return_type' (line 396)
    stypy_return_type_536646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 75), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536646)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_327'
    return stypy_return_type_536646

# Assigning a type to the variable '_stypy_temp_lambda_327' (line 396)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 75), '_stypy_temp_lambda_327', _stypy_temp_lambda_327)
# Getting the type of '_stypy_temp_lambda_327' (line 396)
_stypy_temp_lambda_327_536647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 75), '_stypy_temp_lambda_327')
keyword_536648 = _stypy_temp_lambda_327_536647
kwargs_536649 = {'param_filter': keyword_536648}
# Getting the type of 'data' (line 396)
data_536638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'data', False)
# Calling data(args, kwargs) (line 396)
data_call_result_536650 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), data_536638, *[zeta__536639, str_536640, int_536641, int_536642], **kwargs_536649)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536650)
# Adding element type (line 202)

# Call to data(...): (line 397)
# Processing the call arguments (line 397)
# Getting the type of 'zeta_' (line 397)
zeta__536652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 13), 'zeta_', False)
str_536653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 20), 'str', 'zeta_1_up_data_ipp-zeta_1_up_data')
int_536654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 57), 'int')
int_536655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 60), 'int')
# Processing the call keyword arguments (line 397)

@norecursion
def _stypy_temp_lambda_328(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_328'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_328', 397, 77, True)
    # Passed parameters checking function
    _stypy_temp_lambda_328.stypy_localization = localization
    _stypy_temp_lambda_328.stypy_type_of_self = None
    _stypy_temp_lambda_328.stypy_type_store = module_type_store
    _stypy_temp_lambda_328.stypy_function_name = '_stypy_temp_lambda_328'
    _stypy_temp_lambda_328.stypy_param_names_list = ['s']
    _stypy_temp_lambda_328.stypy_varargs_param_name = None
    _stypy_temp_lambda_328.stypy_kwargs_param_name = None
    _stypy_temp_lambda_328.stypy_call_defaults = defaults
    _stypy_temp_lambda_328.stypy_call_varargs = varargs
    _stypy_temp_lambda_328.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_328', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_328', ['s'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Getting the type of 's' (line 397)
    s_536656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 87), 's', False)
    int_536657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 91), 'int')
    # Applying the binary operator '>' (line 397)
    result_gt_536658 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 87), '>', s_536656, int_536657)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 77), 'stypy_return_type', result_gt_536658)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_328' in the type store
    # Getting the type of 'stypy_return_type' (line 397)
    stypy_return_type_536659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 77), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_328'
    return stypy_return_type_536659

# Assigning a type to the variable '_stypy_temp_lambda_328' (line 397)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 77), '_stypy_temp_lambda_328', _stypy_temp_lambda_328)
# Getting the type of '_stypy_temp_lambda_328' (line 397)
_stypy_temp_lambda_328_536660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 77), '_stypy_temp_lambda_328')
keyword_536661 = _stypy_temp_lambda_328_536660
kwargs_536662 = {'param_filter': keyword_536661}
# Getting the type of 'data' (line 397)
data_536651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'data', False)
# Calling data(args, kwargs) (line 397)
data_call_result_536663 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), data_536651, *[zeta__536652, str_536653, int_536654, int_536655], **kwargs_536662)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536663)
# Adding element type (line 202)

# Call to data(...): (line 398)
# Processing the call arguments (line 398)
# Getting the type of 'zeta_' (line 398)
zeta__536665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 13), 'zeta_', False)
str_536666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 20), 'str', 'zeta_1_below_data_ipp-zeta_1_below_data')
int_536667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 63), 'int')
int_536668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 66), 'int')
# Processing the call keyword arguments (line 398)

@norecursion
def _stypy_temp_lambda_329(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_329'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_329', 398, 83, True)
    # Passed parameters checking function
    _stypy_temp_lambda_329.stypy_localization = localization
    _stypy_temp_lambda_329.stypy_type_of_self = None
    _stypy_temp_lambda_329.stypy_type_store = module_type_store
    _stypy_temp_lambda_329.stypy_function_name = '_stypy_temp_lambda_329'
    _stypy_temp_lambda_329.stypy_param_names_list = ['s']
    _stypy_temp_lambda_329.stypy_varargs_param_name = None
    _stypy_temp_lambda_329.stypy_kwargs_param_name = None
    _stypy_temp_lambda_329.stypy_call_defaults = defaults
    _stypy_temp_lambda_329.stypy_call_varargs = varargs
    _stypy_temp_lambda_329.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_329', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_329', ['s'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Getting the type of 's' (line 398)
    s_536669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 93), 's', False)
    int_536670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 97), 'int')
    # Applying the binary operator '>' (line 398)
    result_gt_536671 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 93), '>', s_536669, int_536670)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 83), 'stypy_return_type', result_gt_536671)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_329' in the type store
    # Getting the type of 'stypy_return_type' (line 398)
    stypy_return_type_536672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 83), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536672)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_329'
    return stypy_return_type_536672

# Assigning a type to the variable '_stypy_temp_lambda_329' (line 398)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 83), '_stypy_temp_lambda_329', _stypy_temp_lambda_329)
# Getting the type of '_stypy_temp_lambda_329' (line 398)
_stypy_temp_lambda_329_536673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 83), '_stypy_temp_lambda_329')
keyword_536674 = _stypy_temp_lambda_329_536673
kwargs_536675 = {'param_filter': keyword_536674}
# Getting the type of 'data' (line 398)
data_536664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'data', False)
# Calling data(args, kwargs) (line 398)
data_call_result_536676 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), data_536664, *[zeta__536665, str_536666, int_536667, int_536668], **kwargs_536675)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536676)
# Adding element type (line 202)

# Call to data(...): (line 400)
# Processing the call arguments (line 400)
# Getting the type of 'gammaincinv' (line 400)
gammaincinv_536678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'gammaincinv', False)
str_536679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 26), 'str', 'gamma_inv_small_data_ipp-gamma_inv_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 400)
tuple_536680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 76), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 400)
# Adding element type (line 400)
int_536681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 76), tuple_536680, int_536681)
# Adding element type (line 400)
int_536682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 76), tuple_536680, int_536682)

int_536683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 82), 'int')
# Processing the call keyword arguments (line 400)
float_536684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 90), 'float')
keyword_536685 = float_536684
str_536686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 110), 'str', 'gammaincinv bad few small points')
keyword_536687 = str_536686
kwargs_536688 = {'knownfailure': keyword_536687, 'rtol': keyword_536685}
# Getting the type of 'data' (line 400)
data_536677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'data', False)
# Calling data(args, kwargs) (line 400)
data_call_result_536689 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), data_536677, *[gammaincinv_536678, str_536679, tuple_536680, int_536683], **kwargs_536688)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536689)
# Adding element type (line 202)

# Call to data(...): (line 401)
# Processing the call arguments (line 401)
# Getting the type of 'gammaincinv' (line 401)
gammaincinv_536691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 13), 'gammaincinv', False)
str_536692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 26), 'str', 'gamma_inv_data_ipp-gamma_inv_data')

# Obtaining an instance of the builtin type 'tuple' (line 401)
tuple_536693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 401)
# Adding element type (line 401)
int_536694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 64), tuple_536693, int_536694)
# Adding element type (line 401)
int_536695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 64), tuple_536693, int_536695)

int_536696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 70), 'int')
# Processing the call keyword arguments (line 401)
float_536697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 78), 'float')
keyword_536698 = float_536697
kwargs_536699 = {'rtol': keyword_536698}
# Getting the type of 'data' (line 401)
data_536690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'data', False)
# Calling data(args, kwargs) (line 401)
data_call_result_536700 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), data_536690, *[gammaincinv_536691, str_536692, tuple_536693, int_536696], **kwargs_536699)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536700)
# Adding element type (line 202)

# Call to data(...): (line 402)
# Processing the call arguments (line 402)
# Getting the type of 'gammaincinv' (line 402)
gammaincinv_536702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 13), 'gammaincinv', False)
str_536703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 26), 'str', 'gamma_inv_big_data_ipp-gamma_inv_big_data')

# Obtaining an instance of the builtin type 'tuple' (line 402)
tuple_536704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 72), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 402)
# Adding element type (line 402)
int_536705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 72), tuple_536704, int_536705)
# Adding element type (line 402)
int_536706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 74), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 72), tuple_536704, int_536706)

int_536707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 78), 'int')
# Processing the call keyword arguments (line 402)
float_536708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 86), 'float')
keyword_536709 = float_536708
kwargs_536710 = {'rtol': keyword_536709}
# Getting the type of 'data' (line 402)
data_536701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'data', False)
# Calling data(args, kwargs) (line 402)
data_call_result_536711 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), data_536701, *[gammaincinv_536702, str_536703, tuple_536704, int_536707], **kwargs_536710)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536711)
# Adding element type (line 202)

# Call to data(...): (line 404)
# Processing the call arguments (line 404)
# Getting the type of 'gammainccinv' (line 404)
gammainccinv_536713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 13), 'gammainccinv', False)
str_536714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 27), 'str', 'gamma_inv_small_data_ipp-gamma_inv_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 404)
tuple_536715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 77), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 404)
# Adding element type (line 404)
int_536716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 77), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 77), tuple_536715, int_536716)
# Adding element type (line 404)
int_536717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 79), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 77), tuple_536715, int_536717)

int_536718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 83), 'int')
# Processing the call keyword arguments (line 404)
float_536719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 91), 'float')
keyword_536720 = float_536719
kwargs_536721 = {'rtol': keyword_536720}
# Getting the type of 'data' (line 404)
data_536712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'data', False)
# Calling data(args, kwargs) (line 404)
data_call_result_536722 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), data_536712, *[gammainccinv_536713, str_536714, tuple_536715, int_536718], **kwargs_536721)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536722)
# Adding element type (line 202)

# Call to data(...): (line 405)
# Processing the call arguments (line 405)
# Getting the type of 'gammainccinv' (line 405)
gammainccinv_536724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'gammainccinv', False)
str_536725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 27), 'str', 'gamma_inv_data_ipp-gamma_inv_data')

# Obtaining an instance of the builtin type 'tuple' (line 405)
tuple_536726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 65), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 405)
# Adding element type (line 405)
int_536727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 65), tuple_536726, int_536727)
# Adding element type (line 405)
int_536728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 67), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 65), tuple_536726, int_536728)

int_536729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 71), 'int')
# Processing the call keyword arguments (line 405)
float_536730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 79), 'float')
keyword_536731 = float_536730
kwargs_536732 = {'rtol': keyword_536731}
# Getting the type of 'data' (line 405)
data_536723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'data', False)
# Calling data(args, kwargs) (line 405)
data_call_result_536733 = invoke(stypy.reporting.localization.Localization(__file__, 405, 8), data_536723, *[gammainccinv_536724, str_536725, tuple_536726, int_536729], **kwargs_536732)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536733)
# Adding element type (line 202)

# Call to data(...): (line 406)
# Processing the call arguments (line 406)
# Getting the type of 'gammainccinv' (line 406)
gammainccinv_536735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 13), 'gammainccinv', False)
str_536736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 27), 'str', 'gamma_inv_big_data_ipp-gamma_inv_big_data')

# Obtaining an instance of the builtin type 'tuple' (line 406)
tuple_536737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 73), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 406)
# Adding element type (line 406)
int_536738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 73), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 73), tuple_536737, int_536738)
# Adding element type (line 406)
int_536739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 73), tuple_536737, int_536739)

int_536740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 79), 'int')
# Processing the call keyword arguments (line 406)
float_536741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 87), 'float')
keyword_536742 = float_536741
kwargs_536743 = {'rtol': keyword_536742}
# Getting the type of 'data' (line 406)
data_536734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'data', False)
# Calling data(args, kwargs) (line 406)
data_call_result_536744 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), data_536734, *[gammainccinv_536735, str_536736, tuple_536737, int_536740], **kwargs_536743)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536744)
# Adding element type (line 202)

# Call to data(...): (line 408)
# Processing the call arguments (line 408)
# Getting the type of 'gdtrix_' (line 408)
gdtrix__536746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 13), 'gdtrix_', False)
str_536747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 22), 'str', 'gamma_inv_small_data_ipp-gamma_inv_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 408)
tuple_536748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 72), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 408)
# Adding element type (line 408)
int_536749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 72), tuple_536748, int_536749)
# Adding element type (line 408)
int_536750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 74), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 72), tuple_536748, int_536750)

int_536751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 78), 'int')
# Processing the call keyword arguments (line 408)
float_536752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 86), 'float')
keyword_536753 = float_536752
str_536754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 106), 'str', 'gdtrix unflow some points')
keyword_536755 = str_536754
kwargs_536756 = {'knownfailure': keyword_536755, 'rtol': keyword_536753}
# Getting the type of 'data' (line 408)
data_536745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'data', False)
# Calling data(args, kwargs) (line 408)
data_call_result_536757 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), data_536745, *[gdtrix__536746, str_536747, tuple_536748, int_536751], **kwargs_536756)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536757)
# Adding element type (line 202)

# Call to data(...): (line 409)
# Processing the call arguments (line 409)
# Getting the type of 'gdtrix_' (line 409)
gdtrix__536759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 13), 'gdtrix_', False)
str_536760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 22), 'str', 'gamma_inv_data_ipp-gamma_inv_data')

# Obtaining an instance of the builtin type 'tuple' (line 409)
tuple_536761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 60), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 409)
# Adding element type (line 409)
int_536762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 60), tuple_536761, int_536762)
# Adding element type (line 409)
int_536763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 60), tuple_536761, int_536763)

int_536764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 66), 'int')
# Processing the call keyword arguments (line 409)
float_536765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 74), 'float')
keyword_536766 = float_536765
kwargs_536767 = {'rtol': keyword_536766}
# Getting the type of 'data' (line 409)
data_536758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'data', False)
# Calling data(args, kwargs) (line 409)
data_call_result_536768 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), data_536758, *[gdtrix__536759, str_536760, tuple_536761, int_536764], **kwargs_536767)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536768)
# Adding element type (line 202)

# Call to data(...): (line 410)
# Processing the call arguments (line 410)
# Getting the type of 'gdtrix_' (line 410)
gdtrix__536770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 13), 'gdtrix_', False)
str_536771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 22), 'str', 'gamma_inv_big_data_ipp-gamma_inv_big_data')

# Obtaining an instance of the builtin type 'tuple' (line 410)
tuple_536772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 68), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 410)
# Adding element type (line 410)
int_536773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 68), tuple_536772, int_536773)
# Adding element type (line 410)
int_536774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 68), tuple_536772, int_536774)

int_536775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 74), 'int')
# Processing the call keyword arguments (line 410)
kwargs_536776 = {}
# Getting the type of 'data' (line 410)
data_536769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'data', False)
# Calling data(args, kwargs) (line 410)
data_call_result_536777 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), data_536769, *[gdtrix__536770, str_536771, tuple_536772, int_536775], **kwargs_536776)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536777)
# Adding element type (line 202)

# Call to data(...): (line 411)
# Processing the call arguments (line 411)
# Getting the type of 'gdtrix_comp' (line 411)
gdtrix_comp_536779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 13), 'gdtrix_comp', False)
str_536780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 26), 'str', 'gamma_inv_small_data_ipp-gamma_inv_small_data')

# Obtaining an instance of the builtin type 'tuple' (line 411)
tuple_536781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 76), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 411)
# Adding element type (line 411)
int_536782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 76), tuple_536781, int_536782)
# Adding element type (line 411)
int_536783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 78), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 76), tuple_536781, int_536783)

int_536784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 82), 'int')
# Processing the call keyword arguments (line 411)
str_536785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 98), 'str', 'gdtrix bad some points')
keyword_536786 = str_536785
kwargs_536787 = {'knownfailure': keyword_536786}
# Getting the type of 'data' (line 411)
data_536778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'data', False)
# Calling data(args, kwargs) (line 411)
data_call_result_536788 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), data_536778, *[gdtrix_comp_536779, str_536780, tuple_536781, int_536784], **kwargs_536787)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536788)
# Adding element type (line 202)

# Call to data(...): (line 412)
# Processing the call arguments (line 412)
# Getting the type of 'gdtrix_comp' (line 412)
gdtrix_comp_536790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 13), 'gdtrix_comp', False)
str_536791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 26), 'str', 'gamma_inv_data_ipp-gamma_inv_data')

# Obtaining an instance of the builtin type 'tuple' (line 412)
tuple_536792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 64), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 412)
# Adding element type (line 412)
int_536793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 64), tuple_536792, int_536793)
# Adding element type (line 412)
int_536794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 64), tuple_536792, int_536794)

int_536795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 70), 'int')
# Processing the call keyword arguments (line 412)
float_536796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 78), 'float')
keyword_536797 = float_536796
kwargs_536798 = {'rtol': keyword_536797}
# Getting the type of 'data' (line 412)
data_536789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'data', False)
# Calling data(args, kwargs) (line 412)
data_call_result_536799 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), data_536789, *[gdtrix_comp_536790, str_536791, tuple_536792, int_536795], **kwargs_536798)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536799)
# Adding element type (line 202)

# Call to data(...): (line 413)
# Processing the call arguments (line 413)
# Getting the type of 'gdtrix_comp' (line 413)
gdtrix_comp_536801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 13), 'gdtrix_comp', False)
str_536802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 26), 'str', 'gamma_inv_big_data_ipp-gamma_inv_big_data')

# Obtaining an instance of the builtin type 'tuple' (line 413)
tuple_536803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 72), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 413)
# Adding element type (line 413)
int_536804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 72), tuple_536803, int_536804)
# Adding element type (line 413)
int_536805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 74), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 72), tuple_536803, int_536805)

int_536806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 78), 'int')
# Processing the call keyword arguments (line 413)
kwargs_536807 = {}
# Getting the type of 'data' (line 413)
data_536800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'data', False)
# Calling data(args, kwargs) (line 413)
data_call_result_536808 = invoke(stypy.reporting.localization.Localization(__file__, 413, 8), data_536800, *[gdtrix_comp_536801, str_536802, tuple_536803, int_536806], **kwargs_536807)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536808)
# Adding element type (line 202)

# Call to data(...): (line 415)
# Processing the call arguments (line 415)
# Getting the type of 'chndtr' (line 415)
chndtr_536810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 13), 'chndtr', False)
str_536811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 21), 'str', 'nccs_ipp-nccs')

# Obtaining an instance of the builtin type 'tuple' (line 415)
tuple_536812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 415)
# Adding element type (line 415)
int_536813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 39), tuple_536812, int_536813)
# Adding element type (line 415)
int_536814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 39), tuple_536812, int_536814)
# Adding element type (line 415)
int_536815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 415, 39), tuple_536812, int_536815)

int_536816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 47), 'int')
# Processing the call keyword arguments (line 415)
float_536817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 55), 'float')
keyword_536818 = float_536817
kwargs_536819 = {'rtol': keyword_536818}
# Getting the type of 'data' (line 415)
data_536809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'data', False)
# Calling data(args, kwargs) (line 415)
data_call_result_536820 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), data_536809, *[chndtr_536810, str_536811, tuple_536812, int_536816], **kwargs_536819)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536820)
# Adding element type (line 202)

# Call to data(...): (line 416)
# Processing the call arguments (line 416)
# Getting the type of 'chndtr' (line 416)
chndtr_536822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 13), 'chndtr', False)
str_536823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 21), 'str', 'nccs_big_ipp-nccs_big')

# Obtaining an instance of the builtin type 'tuple' (line 416)
tuple_536824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 47), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 416)
# Adding element type (line 416)
int_536825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 47), tuple_536824, int_536825)
# Adding element type (line 416)
int_536826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 47), tuple_536824, int_536826)
# Adding element type (line 416)
int_536827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 47), tuple_536824, int_536827)

int_536828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 55), 'int')
# Processing the call keyword arguments (line 416)
float_536829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 63), 'float')
keyword_536830 = float_536829
str_536831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 82), 'str', 'chndtr inaccurate some points')
keyword_536832 = str_536831
kwargs_536833 = {'knownfailure': keyword_536832, 'rtol': keyword_536830}
# Getting the type of 'data' (line 416)
data_536821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'data', False)
# Calling data(args, kwargs) (line 416)
data_call_result_536834 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), data_536821, *[chndtr_536822, str_536823, tuple_536824, int_536828], **kwargs_536833)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536834)
# Adding element type (line 202)

# Call to data(...): (line 418)
# Processing the call arguments (line 418)
# Getting the type of 'sph_harm_' (line 418)
sph_harm__536836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 13), 'sph_harm_', False)
str_536837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 24), 'str', 'spherical_harmonic_ipp-spherical_harmonic')

# Obtaining an instance of the builtin type 'tuple' (line 418)
tuple_536838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 70), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 418)
# Adding element type (line 418)
int_536839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 70), tuple_536838, int_536839)
# Adding element type (line 418)
int_536840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 70), tuple_536838, int_536840)
# Adding element type (line 418)
int_536841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 74), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 70), tuple_536838, int_536841)
# Adding element type (line 418)
int_536842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 70), tuple_536838, int_536842)


# Obtaining an instance of the builtin type 'tuple' (line 418)
tuple_536843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 81), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 418)
# Adding element type (line 418)
int_536844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 81), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 81), tuple_536843, int_536844)
# Adding element type (line 418)
int_536845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 83), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 81), tuple_536843, int_536845)

# Processing the call keyword arguments (line 418)
float_536846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 92), 'float')
keyword_536847 = float_536846

# Obtaining an instance of the builtin type 'tuple' (line 419)
tuple_536848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 27), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 419)
# Adding element type (line 419)

@norecursion
def _stypy_temp_lambda_330(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_330'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_330', 419, 27, True)
    # Passed parameters checking function
    _stypy_temp_lambda_330.stypy_localization = localization
    _stypy_temp_lambda_330.stypy_type_of_self = None
    _stypy_temp_lambda_330.stypy_type_store = module_type_store
    _stypy_temp_lambda_330.stypy_function_name = '_stypy_temp_lambda_330'
    _stypy_temp_lambda_330.stypy_param_names_list = ['p']
    _stypy_temp_lambda_330.stypy_varargs_param_name = None
    _stypy_temp_lambda_330.stypy_kwargs_param_name = None
    _stypy_temp_lambda_330.stypy_call_defaults = defaults
    _stypy_temp_lambda_330.stypy_call_varargs = varargs
    _stypy_temp_lambda_330.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_330', ['p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_330', ['p'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to ones(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'p' (line 419)
    p_536851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 45), 'p', False)
    # Obtaining the member 'shape' of a type (line 419)
    shape_536852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 45), p_536851, 'shape')
    str_536853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 54), 'str', '?')
    # Processing the call keyword arguments (line 419)
    kwargs_536854 = {}
    # Getting the type of 'np' (line 419)
    np_536849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 37), 'np', False)
    # Obtaining the member 'ones' of a type (line 419)
    ones_536850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 37), np_536849, 'ones')
    # Calling ones(args, kwargs) (line 419)
    ones_call_result_536855 = invoke(stypy.reporting.localization.Localization(__file__, 419, 37), ones_536850, *[shape_536852, str_536853], **kwargs_536854)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 27), 'stypy_return_type', ones_call_result_536855)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_330' in the type store
    # Getting the type of 'stypy_return_type' (line 419)
    stypy_return_type_536856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 27), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536856)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_330'
    return stypy_return_type_536856

# Assigning a type to the variable '_stypy_temp_lambda_330' (line 419)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 27), '_stypy_temp_lambda_330', _stypy_temp_lambda_330)
# Getting the type of '_stypy_temp_lambda_330' (line 419)
_stypy_temp_lambda_330_536857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 27), '_stypy_temp_lambda_330')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_536848, _stypy_temp_lambda_330_536857)
# Adding element type (line 419)

@norecursion
def _stypy_temp_lambda_331(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_331'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_331', 420, 27, True)
    # Passed parameters checking function
    _stypy_temp_lambda_331.stypy_localization = localization
    _stypy_temp_lambda_331.stypy_type_of_self = None
    _stypy_temp_lambda_331.stypy_type_store = module_type_store
    _stypy_temp_lambda_331.stypy_function_name = '_stypy_temp_lambda_331'
    _stypy_temp_lambda_331.stypy_param_names_list = ['p']
    _stypy_temp_lambda_331.stypy_varargs_param_name = None
    _stypy_temp_lambda_331.stypy_kwargs_param_name = None
    _stypy_temp_lambda_331.stypy_call_defaults = defaults
    _stypy_temp_lambda_331.stypy_call_varargs = varargs
    _stypy_temp_lambda_331.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_331', ['p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_331', ['p'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to ones(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'p' (line 420)
    p_536860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 45), 'p', False)
    # Obtaining the member 'shape' of a type (line 420)
    shape_536861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 45), p_536860, 'shape')
    str_536862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 54), 'str', '?')
    # Processing the call keyword arguments (line 420)
    kwargs_536863 = {}
    # Getting the type of 'np' (line 420)
    np_536858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 37), 'np', False)
    # Obtaining the member 'ones' of a type (line 420)
    ones_536859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 37), np_536858, 'ones')
    # Calling ones(args, kwargs) (line 420)
    ones_call_result_536864 = invoke(stypy.reporting.localization.Localization(__file__, 420, 37), ones_536859, *[shape_536861, str_536862], **kwargs_536863)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 27), 'stypy_return_type', ones_call_result_536864)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_331' in the type store
    # Getting the type of 'stypy_return_type' (line 420)
    stypy_return_type_536865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 27), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536865)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_331'
    return stypy_return_type_536865

# Assigning a type to the variable '_stypy_temp_lambda_331' (line 420)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 27), '_stypy_temp_lambda_331', _stypy_temp_lambda_331)
# Getting the type of '_stypy_temp_lambda_331' (line 420)
_stypy_temp_lambda_331_536866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 27), '_stypy_temp_lambda_331')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_536848, _stypy_temp_lambda_331_536866)
# Adding element type (line 419)

@norecursion
def _stypy_temp_lambda_332(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_332'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_332', 421, 27, True)
    # Passed parameters checking function
    _stypy_temp_lambda_332.stypy_localization = localization
    _stypy_temp_lambda_332.stypy_type_of_self = None
    _stypy_temp_lambda_332.stypy_type_store = module_type_store
    _stypy_temp_lambda_332.stypy_function_name = '_stypy_temp_lambda_332'
    _stypy_temp_lambda_332.stypy_param_names_list = ['p']
    _stypy_temp_lambda_332.stypy_varargs_param_name = None
    _stypy_temp_lambda_332.stypy_kwargs_param_name = None
    _stypy_temp_lambda_332.stypy_call_defaults = defaults
    _stypy_temp_lambda_332.stypy_call_varargs = varargs
    _stypy_temp_lambda_332.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_332', ['p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_332', ['p'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to logical_and(...): (line 421)
    # Processing the call arguments (line 421)
    
    # Getting the type of 'p' (line 421)
    p_536869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 52), 'p', False)
    int_536870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 56), 'int')
    # Getting the type of 'np' (line 421)
    np_536871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 58), 'np', False)
    # Obtaining the member 'pi' of a type (line 421)
    pi_536872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 58), np_536871, 'pi')
    # Applying the binary operator '*' (line 421)
    result_mul_536873 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 56), '*', int_536870, pi_536872)
    
    # Applying the binary operator '<' (line 421)
    result_lt_536874 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 52), '<', p_536869, result_mul_536873)
    
    
    # Getting the type of 'p' (line 421)
    p_536875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 65), 'p', False)
    int_536876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 70), 'int')
    # Applying the binary operator '>=' (line 421)
    result_ge_536877 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 65), '>=', p_536875, int_536876)
    
    # Processing the call keyword arguments (line 421)
    kwargs_536878 = {}
    # Getting the type of 'np' (line 421)
    np_536867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 37), 'np', False)
    # Obtaining the member 'logical_and' of a type (line 421)
    logical_and_536868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 37), np_536867, 'logical_and')
    # Calling logical_and(args, kwargs) (line 421)
    logical_and_call_result_536879 = invoke(stypy.reporting.localization.Localization(__file__, 421, 37), logical_and_536868, *[result_lt_536874, result_ge_536877], **kwargs_536878)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), 'stypy_return_type', logical_and_call_result_536879)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_332' in the type store
    # Getting the type of 'stypy_return_type' (line 421)
    stypy_return_type_536880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536880)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_332'
    return stypy_return_type_536880

# Assigning a type to the variable '_stypy_temp_lambda_332' (line 421)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), '_stypy_temp_lambda_332', _stypy_temp_lambda_332)
# Getting the type of '_stypy_temp_lambda_332' (line 421)
_stypy_temp_lambda_332_536881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), '_stypy_temp_lambda_332')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_536848, _stypy_temp_lambda_332_536881)
# Adding element type (line 419)

@norecursion
def _stypy_temp_lambda_333(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_333'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_333', 422, 27, True)
    # Passed parameters checking function
    _stypy_temp_lambda_333.stypy_localization = localization
    _stypy_temp_lambda_333.stypy_type_of_self = None
    _stypy_temp_lambda_333.stypy_type_store = module_type_store
    _stypy_temp_lambda_333.stypy_function_name = '_stypy_temp_lambda_333'
    _stypy_temp_lambda_333.stypy_param_names_list = ['p']
    _stypy_temp_lambda_333.stypy_varargs_param_name = None
    _stypy_temp_lambda_333.stypy_kwargs_param_name = None
    _stypy_temp_lambda_333.stypy_call_defaults = defaults
    _stypy_temp_lambda_333.stypy_call_varargs = varargs
    _stypy_temp_lambda_333.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_333', ['p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_333', ['p'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to logical_and(...): (line 422)
    # Processing the call arguments (line 422)
    
    # Getting the type of 'p' (line 422)
    p_536884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 52), 'p', False)
    # Getting the type of 'np' (line 422)
    np_536885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 56), 'np', False)
    # Obtaining the member 'pi' of a type (line 422)
    pi_536886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 56), np_536885, 'pi')
    # Applying the binary operator '<' (line 422)
    result_lt_536887 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 52), '<', p_536884, pi_536886)
    
    
    # Getting the type of 'p' (line 422)
    p_536888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 63), 'p', False)
    int_536889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 68), 'int')
    # Applying the binary operator '>=' (line 422)
    result_ge_536890 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 63), '>=', p_536888, int_536889)
    
    # Processing the call keyword arguments (line 422)
    kwargs_536891 = {}
    # Getting the type of 'np' (line 422)
    np_536882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 37), 'np', False)
    # Obtaining the member 'logical_and' of a type (line 422)
    logical_and_536883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 37), np_536882, 'logical_and')
    # Calling logical_and(args, kwargs) (line 422)
    logical_and_call_result_536892 = invoke(stypy.reporting.localization.Localization(__file__, 422, 37), logical_and_536883, *[result_lt_536887, result_ge_536890], **kwargs_536891)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 27), 'stypy_return_type', logical_and_call_result_536892)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_333' in the type store
    # Getting the type of 'stypy_return_type' (line 422)
    stypy_return_type_536893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 27), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536893)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_333'
    return stypy_return_type_536893

# Assigning a type to the variable '_stypy_temp_lambda_333' (line 422)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 27), '_stypy_temp_lambda_333', _stypy_temp_lambda_333)
# Getting the type of '_stypy_temp_lambda_333' (line 422)
_stypy_temp_lambda_333_536894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 27), '_stypy_temp_lambda_333')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 27), tuple_536848, _stypy_temp_lambda_333_536894)

keyword_536895 = tuple_536848
kwargs_536896 = {'param_filter': keyword_536895, 'rtol': keyword_536847}
# Getting the type of 'data' (line 418)
data_536835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'data', False)
# Calling data(args, kwargs) (line 418)
data_call_result_536897 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), data_536835, *[sph_harm__536836, str_536837, tuple_536838, tuple_536843], **kwargs_536896)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536897)
# Adding element type (line 202)

# Call to data(...): (line 424)
# Processing the call arguments (line 424)
# Getting the type of 'spherical_jn_' (line 424)
spherical_jn__536899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 13), 'spherical_jn_', False)
str_536900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 28), 'str', 'sph_bessel_data_ipp-sph_bessel_data')

# Obtaining an instance of the builtin type 'tuple' (line 424)
tuple_536901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 68), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 424)
# Adding element type (line 424)
int_536902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 68), tuple_536901, int_536902)
# Adding element type (line 424)
int_536903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 68), tuple_536901, int_536903)

int_536904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 74), 'int')
# Processing the call keyword arguments (line 424)
float_536905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 82), 'float')
keyword_536906 = float_536905
kwargs_536907 = {'rtol': keyword_536906}
# Getting the type of 'data' (line 424)
data_536898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'data', False)
# Calling data(args, kwargs) (line 424)
data_call_result_536908 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), data_536898, *[spherical_jn__536899, str_536900, tuple_536901, int_536904], **kwargs_536907)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536908)
# Adding element type (line 202)

# Call to data(...): (line 425)
# Processing the call arguments (line 425)
# Getting the type of 'spherical_yn_' (line 425)
spherical_yn__536910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 13), 'spherical_yn_', False)
str_536911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 28), 'str', 'sph_neumann_data_ipp-sph_neumann_data')

# Obtaining an instance of the builtin type 'tuple' (line 425)
tuple_536912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 70), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 425)
# Adding element type (line 425)
int_536913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 70), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 70), tuple_536912, int_536913)
# Adding element type (line 425)
int_536914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 70), tuple_536912, int_536914)

int_536915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 76), 'int')
# Processing the call keyword arguments (line 425)
float_536916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 84), 'float')
keyword_536917 = float_536916
kwargs_536918 = {'rtol': keyword_536917}
# Getting the type of 'data' (line 425)
data_536909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'data', False)
# Calling data(args, kwargs) (line 425)
data_call_result_536919 = invoke(stypy.reporting.localization.Localization(__file__, 425, 8), data_536909, *[spherical_yn__536910, str_536911, tuple_536912, int_536915], **kwargs_536918)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 14), list_535083, data_call_result_536919)

# Assigning a type to the variable 'BOOST_TESTS' (line 202)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'BOOST_TESTS', list_535083)

@norecursion
def test_boost(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_boost'
    module_type_store = module_type_store.open_function_context('test_boost', 444, 0, False)
    
    # Passed parameters checking function
    test_boost.stypy_localization = localization
    test_boost.stypy_type_of_self = None
    test_boost.stypy_type_store = module_type_store
    test_boost.stypy_function_name = 'test_boost'
    test_boost.stypy_param_names_list = ['test']
    test_boost.stypy_varargs_param_name = None
    test_boost.stypy_kwargs_param_name = None
    test_boost.stypy_call_defaults = defaults
    test_boost.stypy_call_varargs = varargs
    test_boost.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_boost', ['test'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_boost', localization, ['test'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_boost(...)' code ##################

    
    # Call to _test_factory(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'test' (line 446)
    test_536921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 18), 'test', False)
    # Processing the call keyword arguments (line 446)
    kwargs_536922 = {}
    # Getting the type of '_test_factory' (line 446)
    _test_factory_536920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), '_test_factory', False)
    # Calling _test_factory(args, kwargs) (line 446)
    _test_factory_call_result_536923 = invoke(stypy.reporting.localization.Localization(__file__, 446, 4), _test_factory_536920, *[test_536921], **kwargs_536922)
    
    
    # ################# End of 'test_boost(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_boost' in the type store
    # Getting the type of 'stypy_return_type' (line 444)
    stypy_return_type_536924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_536924)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_boost'
    return stypy_return_type_536924

# Assigning a type to the variable 'test_boost' (line 444)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 0), 'test_boost', test_boost)

# Assigning a List to a Name (line 449):

# Obtaining an instance of the builtin type 'list' (line 449)
list_536925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 449)
# Adding element type (line 449)

# Call to data_gsl(...): (line 450)
# Processing the call arguments (line 450)
# Getting the type of 'mathieu_a' (line 450)
mathieu_a_536927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 17), 'mathieu_a', False)
str_536928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 28), 'str', 'mathieu_ab')

# Obtaining an instance of the builtin type 'tuple' (line 450)
tuple_536929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 43), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 450)
# Adding element type (line 450)
int_536930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 43), tuple_536929, int_536930)
# Adding element type (line 450)
int_536931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 43), tuple_536929, int_536931)

int_536932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 50), 'int')
# Processing the call keyword arguments (line 450)
float_536933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 58), 'float')
keyword_536934 = float_536933
float_536935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 70), 'float')
keyword_536936 = float_536935
kwargs_536937 = {'rtol': keyword_536934, 'atol': keyword_536936}
# Getting the type of 'data_gsl' (line 450)
data_gsl_536926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'data_gsl', False)
# Calling data_gsl(args, kwargs) (line 450)
data_gsl_call_result_536938 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), data_gsl_536926, *[mathieu_a_536927, str_536928, tuple_536929, int_536932], **kwargs_536937)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), list_536925, data_gsl_call_result_536938)
# Adding element type (line 449)

# Call to data_gsl(...): (line 451)
# Processing the call arguments (line 451)
# Getting the type of 'mathieu_b' (line 451)
mathieu_b_536940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 17), 'mathieu_b', False)
str_536941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 28), 'str', 'mathieu_ab')

# Obtaining an instance of the builtin type 'tuple' (line 451)
tuple_536942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 43), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 451)
# Adding element type (line 451)
int_536943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 43), tuple_536942, int_536943)
# Adding element type (line 451)
int_536944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 43), tuple_536942, int_536944)

int_536945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 50), 'int')
# Processing the call keyword arguments (line 451)
float_536946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 58), 'float')
keyword_536947 = float_536946
float_536948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 70), 'float')
keyword_536949 = float_536948
kwargs_536950 = {'rtol': keyword_536947, 'atol': keyword_536949}
# Getting the type of 'data_gsl' (line 451)
data_gsl_536939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'data_gsl', False)
# Calling data_gsl(args, kwargs) (line 451)
data_gsl_call_result_536951 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), data_gsl_536939, *[mathieu_b_536940, str_536941, tuple_536942, int_536945], **kwargs_536950)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), list_536925, data_gsl_call_result_536951)
# Adding element type (line 449)

# Call to data_gsl(...): (line 454)
# Processing the call arguments (line 454)
# Getting the type of 'mathieu_ce_rad' (line 454)
mathieu_ce_rad_536953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 17), 'mathieu_ce_rad', False)
str_536954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 33), 'str', 'mathieu_ce_se')

# Obtaining an instance of the builtin type 'tuple' (line 454)
tuple_536955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 51), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 454)
# Adding element type (line 454)
int_536956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 51), tuple_536955, int_536956)
# Adding element type (line 454)
int_536957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 51), tuple_536955, int_536957)
# Adding element type (line 454)
int_536958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 51), tuple_536955, int_536958)

int_536959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 61), 'int')
# Processing the call keyword arguments (line 454)
float_536960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 69), 'float')
keyword_536961 = float_536960
float_536962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 80), 'float')
keyword_536963 = float_536962
kwargs_536964 = {'rtol': keyword_536961, 'atol': keyword_536963}
# Getting the type of 'data_gsl' (line 454)
data_gsl_536952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'data_gsl', False)
# Calling data_gsl(args, kwargs) (line 454)
data_gsl_call_result_536965 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), data_gsl_536952, *[mathieu_ce_rad_536953, str_536954, tuple_536955, int_536959], **kwargs_536964)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), list_536925, data_gsl_call_result_536965)
# Adding element type (line 449)

# Call to data_gsl(...): (line 455)
# Processing the call arguments (line 455)
# Getting the type of 'mathieu_se_rad' (line 455)
mathieu_se_rad_536967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 17), 'mathieu_se_rad', False)
str_536968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 33), 'str', 'mathieu_ce_se')

# Obtaining an instance of the builtin type 'tuple' (line 455)
tuple_536969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 51), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 455)
# Adding element type (line 455)
int_536970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 51), tuple_536969, int_536970)
# Adding element type (line 455)
int_536971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 51), tuple_536969, int_536971)
# Adding element type (line 455)
int_536972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 51), tuple_536969, int_536972)

int_536973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 61), 'int')
# Processing the call keyword arguments (line 455)
float_536974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 69), 'float')
keyword_536975 = float_536974
float_536976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 80), 'float')
keyword_536977 = float_536976
kwargs_536978 = {'rtol': keyword_536975, 'atol': keyword_536977}
# Getting the type of 'data_gsl' (line 455)
data_gsl_536966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'data_gsl', False)
# Calling data_gsl(args, kwargs) (line 455)
data_gsl_call_result_536979 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), data_gsl_536966, *[mathieu_se_rad_536967, str_536968, tuple_536969, int_536973], **kwargs_536978)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), list_536925, data_gsl_call_result_536979)
# Adding element type (line 449)

# Call to data_gsl(...): (line 457)
# Processing the call arguments (line 457)
# Getting the type of 'mathieu_mc1_scaled' (line 457)
mathieu_mc1_scaled_536981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 17), 'mathieu_mc1_scaled', False)
str_536982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 37), 'str', 'mathieu_mc_ms')

# Obtaining an instance of the builtin type 'tuple' (line 457)
tuple_536983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 457)
# Adding element type (line 457)
int_536984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 55), tuple_536983, int_536984)
# Adding element type (line 457)
int_536985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 55), tuple_536983, int_536985)
# Adding element type (line 457)
int_536986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 55), tuple_536983, int_536986)

int_536987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 65), 'int')
# Processing the call keyword arguments (line 457)
float_536988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 73), 'float')
keyword_536989 = float_536988
float_536990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 84), 'float')
keyword_536991 = float_536990
kwargs_536992 = {'rtol': keyword_536989, 'atol': keyword_536991}
# Getting the type of 'data_gsl' (line 457)
data_gsl_536980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'data_gsl', False)
# Calling data_gsl(args, kwargs) (line 457)
data_gsl_call_result_536993 = invoke(stypy.reporting.localization.Localization(__file__, 457, 8), data_gsl_536980, *[mathieu_mc1_scaled_536981, str_536982, tuple_536983, int_536987], **kwargs_536992)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), list_536925, data_gsl_call_result_536993)
# Adding element type (line 449)

# Call to data_gsl(...): (line 458)
# Processing the call arguments (line 458)
# Getting the type of 'mathieu_ms1_scaled' (line 458)
mathieu_ms1_scaled_536995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 17), 'mathieu_ms1_scaled', False)
str_536996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 37), 'str', 'mathieu_mc_ms')

# Obtaining an instance of the builtin type 'tuple' (line 458)
tuple_536997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 458)
# Adding element type (line 458)
int_536998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 55), tuple_536997, int_536998)
# Adding element type (line 458)
int_536999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 55), tuple_536997, int_536999)
# Adding element type (line 458)
int_537000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 55), tuple_536997, int_537000)

int_537001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 65), 'int')
# Processing the call keyword arguments (line 458)
float_537002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 73), 'float')
keyword_537003 = float_537002
float_537004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 84), 'float')
keyword_537005 = float_537004
kwargs_537006 = {'rtol': keyword_537003, 'atol': keyword_537005}
# Getting the type of 'data_gsl' (line 458)
data_gsl_536994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'data_gsl', False)
# Calling data_gsl(args, kwargs) (line 458)
data_gsl_call_result_537007 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), data_gsl_536994, *[mathieu_ms1_scaled_536995, str_536996, tuple_536997, int_537001], **kwargs_537006)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), list_536925, data_gsl_call_result_537007)
# Adding element type (line 449)

# Call to data_gsl(...): (line 460)
# Processing the call arguments (line 460)
# Getting the type of 'mathieu_mc2_scaled' (line 460)
mathieu_mc2_scaled_537009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 17), 'mathieu_mc2_scaled', False)
str_537010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 37), 'str', 'mathieu_mc_ms')

# Obtaining an instance of the builtin type 'tuple' (line 460)
tuple_537011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 460)
# Adding element type (line 460)
int_537012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 55), tuple_537011, int_537012)
# Adding element type (line 460)
int_537013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 55), tuple_537011, int_537013)
# Adding element type (line 460)
int_537014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 55), tuple_537011, int_537014)

int_537015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 65), 'int')
# Processing the call keyword arguments (line 460)
float_537016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 73), 'float')
keyword_537017 = float_537016
float_537018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 84), 'float')
keyword_537019 = float_537018
kwargs_537020 = {'rtol': keyword_537017, 'atol': keyword_537019}
# Getting the type of 'data_gsl' (line 460)
data_gsl_537008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'data_gsl', False)
# Calling data_gsl(args, kwargs) (line 460)
data_gsl_call_result_537021 = invoke(stypy.reporting.localization.Localization(__file__, 460, 8), data_gsl_537008, *[mathieu_mc2_scaled_537009, str_537010, tuple_537011, int_537015], **kwargs_537020)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), list_536925, data_gsl_call_result_537021)
# Adding element type (line 449)

# Call to data_gsl(...): (line 461)
# Processing the call arguments (line 461)
# Getting the type of 'mathieu_ms2_scaled' (line 461)
mathieu_ms2_scaled_537023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 17), 'mathieu_ms2_scaled', False)
str_537024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 37), 'str', 'mathieu_mc_ms')

# Obtaining an instance of the builtin type 'tuple' (line 461)
tuple_537025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 55), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 461)
# Adding element type (line 461)
int_537026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 55), tuple_537025, int_537026)
# Adding element type (line 461)
int_537027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 55), tuple_537025, int_537027)
# Adding element type (line 461)
int_537028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 55), tuple_537025, int_537028)

int_537029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 65), 'int')
# Processing the call keyword arguments (line 461)
float_537030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 73), 'float')
keyword_537031 = float_537030
float_537032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 84), 'float')
keyword_537033 = float_537032
kwargs_537034 = {'rtol': keyword_537031, 'atol': keyword_537033}
# Getting the type of 'data_gsl' (line 461)
data_gsl_537022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'data_gsl', False)
# Calling data_gsl(args, kwargs) (line 461)
data_gsl_call_result_537035 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), data_gsl_537022, *[mathieu_ms2_scaled_537023, str_537024, tuple_537025, int_537029], **kwargs_537034)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 12), list_536925, data_gsl_call_result_537035)

# Assigning a type to the variable 'GSL_TESTS' (line 449)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 0), 'GSL_TESTS', list_536925)

@norecursion
def test_gsl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gsl'
    module_type_store = module_type_store.open_function_context('test_gsl', 465, 0, False)
    
    # Passed parameters checking function
    test_gsl.stypy_localization = localization
    test_gsl.stypy_type_of_self = None
    test_gsl.stypy_type_store = module_type_store
    test_gsl.stypy_function_name = 'test_gsl'
    test_gsl.stypy_param_names_list = ['test']
    test_gsl.stypy_varargs_param_name = None
    test_gsl.stypy_kwargs_param_name = None
    test_gsl.stypy_call_defaults = defaults
    test_gsl.stypy_call_varargs = varargs
    test_gsl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gsl', ['test'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gsl', localization, ['test'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gsl(...)' code ##################

    
    # Call to _test_factory(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'test' (line 467)
    test_537037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 18), 'test', False)
    # Processing the call keyword arguments (line 467)
    kwargs_537038 = {}
    # Getting the type of '_test_factory' (line 467)
    _test_factory_537036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), '_test_factory', False)
    # Calling _test_factory(args, kwargs) (line 467)
    _test_factory_call_result_537039 = invoke(stypy.reporting.localization.Localization(__file__, 467, 4), _test_factory_537036, *[test_537037], **kwargs_537038)
    
    
    # ################# End of 'test_gsl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gsl' in the type store
    # Getting the type of 'stypy_return_type' (line 465)
    stypy_return_type_537040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_537040)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gsl'
    return stypy_return_type_537040

# Assigning a type to the variable 'test_gsl' (line 465)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'test_gsl', test_gsl)

# Assigning a List to a Name (line 470):

# Obtaining an instance of the builtin type 'list' (line 470)
list_537041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 470)
# Adding element type (line 470)

# Call to data_local(...): (line 471)
# Processing the call arguments (line 471)
# Getting the type of 'ellipkinc' (line 471)
ellipkinc_537043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 15), 'ellipkinc', False)
str_537044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 26), 'str', 'ellipkinc_neg_m')

# Obtaining an instance of the builtin type 'tuple' (line 471)
tuple_537045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 471)
# Adding element type (line 471)
int_537046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 46), tuple_537045, int_537046)
# Adding element type (line 471)
int_537047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 471, 46), tuple_537045, int_537047)

int_537048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 53), 'int')
# Processing the call keyword arguments (line 471)
kwargs_537049 = {}
# Getting the type of 'data_local' (line 471)
data_local_537042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 471)
data_local_call_result_537050 = invoke(stypy.reporting.localization.Localization(__file__, 471, 4), data_local_537042, *[ellipkinc_537043, str_537044, tuple_537045, int_537048], **kwargs_537049)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537050)
# Adding element type (line 470)

# Call to data_local(...): (line 472)
# Processing the call arguments (line 472)
# Getting the type of 'ellipkm1' (line 472)
ellipkm1_537052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 15), 'ellipkm1', False)
str_537053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 25), 'str', 'ellipkm1')
int_537054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 37), 'int')
int_537055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 40), 'int')
# Processing the call keyword arguments (line 472)
kwargs_537056 = {}
# Getting the type of 'data_local' (line 472)
data_local_537051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 472)
data_local_call_result_537057 = invoke(stypy.reporting.localization.Localization(__file__, 472, 4), data_local_537051, *[ellipkm1_537052, str_537053, int_537054, int_537055], **kwargs_537056)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537057)
# Adding element type (line 470)

# Call to data_local(...): (line 473)
# Processing the call arguments (line 473)
# Getting the type of 'ellipeinc' (line 473)
ellipeinc_537059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 15), 'ellipeinc', False)
str_537060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 26), 'str', 'ellipeinc_neg_m')

# Obtaining an instance of the builtin type 'tuple' (line 473)
tuple_537061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 473)
# Adding element type (line 473)
int_537062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 46), tuple_537061, int_537062)
# Adding element type (line 473)
int_537063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 46), tuple_537061, int_537063)

int_537064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 53), 'int')
# Processing the call keyword arguments (line 473)
kwargs_537065 = {}
# Getting the type of 'data_local' (line 473)
data_local_537058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 473)
data_local_call_result_537066 = invoke(stypy.reporting.localization.Localization(__file__, 473, 4), data_local_537058, *[ellipeinc_537059, str_537060, tuple_537061, int_537064], **kwargs_537065)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537066)
# Adding element type (line 470)

# Call to data_local(...): (line 474)
# Processing the call arguments (line 474)
# Getting the type of 'clog1p' (line 474)
clog1p_537068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'clog1p', False)
str_537069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 23), 'str', 'log1p_expm1_complex')

# Obtaining an instance of the builtin type 'tuple' (line 474)
tuple_537070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 47), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 474)
# Adding element type (line 474)
int_537071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 47), tuple_537070, int_537071)
# Adding element type (line 474)
int_537072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 47), tuple_537070, int_537072)


# Obtaining an instance of the builtin type 'tuple' (line 474)
tuple_537073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 54), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 474)
# Adding element type (line 474)
int_537074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 54), tuple_537073, int_537074)
# Adding element type (line 474)
int_537075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 54), tuple_537073, int_537075)

# Processing the call keyword arguments (line 474)
float_537076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 65), 'float')
keyword_537077 = float_537076
kwargs_537078 = {'rtol': keyword_537077}
# Getting the type of 'data_local' (line 474)
data_local_537067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 474)
data_local_call_result_537079 = invoke(stypy.reporting.localization.Localization(__file__, 474, 4), data_local_537067, *[clog1p_537068, str_537069, tuple_537070, tuple_537073], **kwargs_537078)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537079)
# Adding element type (line 470)

# Call to data_local(...): (line 475)
# Processing the call arguments (line 475)
# Getting the type of 'cexpm1' (line 475)
cexpm1_537081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), 'cexpm1', False)
str_537082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 23), 'str', 'log1p_expm1_complex')

# Obtaining an instance of the builtin type 'tuple' (line 475)
tuple_537083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 47), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 475)
# Adding element type (line 475)
int_537084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 47), tuple_537083, int_537084)
# Adding element type (line 475)
int_537085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 47), tuple_537083, int_537085)


# Obtaining an instance of the builtin type 'tuple' (line 475)
tuple_537086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 54), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 475)
# Adding element type (line 475)
int_537087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 54), tuple_537086, int_537087)
# Adding element type (line 475)
int_537088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 54), tuple_537086, int_537088)

# Processing the call keyword arguments (line 475)
float_537089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 65), 'float')
keyword_537090 = float_537089
kwargs_537091 = {'rtol': keyword_537090}
# Getting the type of 'data_local' (line 475)
data_local_537080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 475)
data_local_call_result_537092 = invoke(stypy.reporting.localization.Localization(__file__, 475, 4), data_local_537080, *[cexpm1_537081, str_537082, tuple_537083, tuple_537086], **kwargs_537091)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537092)
# Adding element type (line 470)

# Call to data_local(...): (line 476)
# Processing the call arguments (line 476)
# Getting the type of 'gammainc' (line 476)
gammainc_537094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'gammainc', False)
str_537095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 25), 'str', 'gammainc')

# Obtaining an instance of the builtin type 'tuple' (line 476)
tuple_537096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 476)
# Adding element type (line 476)
int_537097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 38), tuple_537096, int_537097)
# Adding element type (line 476)
int_537098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 476, 38), tuple_537096, int_537098)

int_537099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 45), 'int')
# Processing the call keyword arguments (line 476)
float_537100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 53), 'float')
keyword_537101 = float_537100
kwargs_537102 = {'rtol': keyword_537101}
# Getting the type of 'data_local' (line 476)
data_local_537093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 476)
data_local_call_result_537103 = invoke(stypy.reporting.localization.Localization(__file__, 476, 4), data_local_537093, *[gammainc_537094, str_537095, tuple_537096, int_537099], **kwargs_537102)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537103)
# Adding element type (line 470)

# Call to data_local(...): (line 477)
# Processing the call arguments (line 477)
# Getting the type of 'gammaincc' (line 477)
gammaincc_537105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'gammaincc', False)
str_537106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 26), 'str', 'gammaincc')

# Obtaining an instance of the builtin type 'tuple' (line 477)
tuple_537107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 40), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 477)
# Adding element type (line 477)
int_537108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 40), tuple_537107, int_537108)
# Adding element type (line 477)
int_537109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 40), tuple_537107, int_537109)

int_537110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 47), 'int')
# Processing the call keyword arguments (line 477)
float_537111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 55), 'float')
keyword_537112 = float_537111
kwargs_537113 = {'rtol': keyword_537112}
# Getting the type of 'data_local' (line 477)
data_local_537104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 477)
data_local_call_result_537114 = invoke(stypy.reporting.localization.Localization(__file__, 477, 4), data_local_537104, *[gammaincc_537105, str_537106, tuple_537107, int_537110], **kwargs_537113)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537114)
# Adding element type (line 470)

# Call to data_local(...): (line 478)
# Processing the call arguments (line 478)
# Getting the type of 'ellip_harm_2' (line 478)
ellip_harm_2_537116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 15), 'ellip_harm_2', False)
str_537117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 29), 'str', 'ellip')

# Obtaining an instance of the builtin type 'tuple' (line 478)
tuple_537118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 478)
# Adding element type (line 478)
int_537119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 38), tuple_537118, int_537119)
# Adding element type (line 478)
int_537120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 38), tuple_537118, int_537120)
# Adding element type (line 478)
int_537121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 44), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 38), tuple_537118, int_537121)
# Adding element type (line 478)
int_537122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 38), tuple_537118, int_537122)
# Adding element type (line 478)
int_537123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 50), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 38), tuple_537118, int_537123)

int_537124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 54), 'int')
# Processing the call keyword arguments (line 478)
float_537125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 62), 'float')
keyword_537126 = float_537125
float_537127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 74), 'float')
keyword_537128 = float_537127
kwargs_537129 = {'rtol': keyword_537126, 'atol': keyword_537128}
# Getting the type of 'data_local' (line 478)
data_local_537115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 478)
data_local_call_result_537130 = invoke(stypy.reporting.localization.Localization(__file__, 478, 4), data_local_537115, *[ellip_harm_2_537116, str_537117, tuple_537118, int_537124], **kwargs_537129)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537130)
# Adding element type (line 470)

# Call to data_local(...): (line 479)
# Processing the call arguments (line 479)
# Getting the type of 'ellip_harm' (line 479)
ellip_harm_537132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'ellip_harm', False)
str_537133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 27), 'str', 'ellip')

# Obtaining an instance of the builtin type 'tuple' (line 479)
tuple_537134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 479)
# Adding element type (line 479)
int_537135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 36), tuple_537134, int_537135)
# Adding element type (line 479)
int_537136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 36), tuple_537134, int_537136)
# Adding element type (line 479)
int_537137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 36), tuple_537134, int_537137)
# Adding element type (line 479)
int_537138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 36), tuple_537134, int_537138)
# Adding element type (line 479)
int_537139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 36), tuple_537134, int_537139)

int_537140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 52), 'int')
# Processing the call keyword arguments (line 479)
float_537141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 60), 'float')
keyword_537142 = float_537141
float_537143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 72), 'float')
keyword_537144 = float_537143
kwargs_537145 = {'rtol': keyword_537142, 'atol': keyword_537144}
# Getting the type of 'data_local' (line 479)
data_local_537131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'data_local', False)
# Calling data_local(args, kwargs) (line 479)
data_local_call_result_537146 = invoke(stypy.reporting.localization.Localization(__file__, 479, 4), data_local_537131, *[ellip_harm_537132, str_537133, tuple_537134, int_537140], **kwargs_537145)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 14), list_537041, data_local_call_result_537146)

# Assigning a type to the variable 'LOCAL_TESTS' (line 470)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 0), 'LOCAL_TESTS', list_537041)

@norecursion
def test_local(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_local'
    module_type_store = module_type_store.open_function_context('test_local', 483, 0, False)
    
    # Passed parameters checking function
    test_local.stypy_localization = localization
    test_local.stypy_type_of_self = None
    test_local.stypy_type_store = module_type_store
    test_local.stypy_function_name = 'test_local'
    test_local.stypy_param_names_list = ['test']
    test_local.stypy_varargs_param_name = None
    test_local.stypy_kwargs_param_name = None
    test_local.stypy_call_defaults = defaults
    test_local.stypy_call_varargs = varargs
    test_local.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_local', ['test'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_local', localization, ['test'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_local(...)' code ##################

    
    # Call to _test_factory(...): (line 485)
    # Processing the call arguments (line 485)
    # Getting the type of 'test' (line 485)
    test_537148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 18), 'test', False)
    # Processing the call keyword arguments (line 485)
    kwargs_537149 = {}
    # Getting the type of '_test_factory' (line 485)
    _test_factory_537147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), '_test_factory', False)
    # Calling _test_factory(args, kwargs) (line 485)
    _test_factory_call_result_537150 = invoke(stypy.reporting.localization.Localization(__file__, 485, 4), _test_factory_537147, *[test_537148], **kwargs_537149)
    
    
    # ################# End of 'test_local(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_local' in the type store
    # Getting the type of 'stypy_return_type' (line 483)
    stypy_return_type_537151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_537151)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_local'
    return stypy_return_type_537151

# Assigning a type to the variable 'test_local' (line 483)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'test_local', test_local)

@norecursion
def _test_factory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'np' (line 488)
    np_537152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 30), 'np')
    # Obtaining the member 'double' of a type (line 488)
    double_537153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 30), np_537152, 'double')
    defaults = [double_537153]
    # Create a new context for function '_test_factory'
    module_type_store = module_type_store.open_function_context('_test_factory', 488, 0, False)
    
    # Passed parameters checking function
    _test_factory.stypy_localization = localization
    _test_factory.stypy_type_of_self = None
    _test_factory.stypy_type_store = module_type_store
    _test_factory.stypy_function_name = '_test_factory'
    _test_factory.stypy_param_names_list = ['test', 'dtype']
    _test_factory.stypy_varargs_param_name = None
    _test_factory.stypy_kwargs_param_name = None
    _test_factory.stypy_call_defaults = defaults
    _test_factory.stypy_call_varargs = varargs
    _test_factory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_test_factory', ['test', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_test_factory', localization, ['test', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_test_factory(...)' code ##################

    str_537154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 4), 'str', 'Boost test')
    
    # Call to suppress_warnings(...): (line 490)
    # Processing the call keyword arguments (line 490)
    kwargs_537156 = {}
    # Getting the type of 'suppress_warnings' (line 490)
    suppress_warnings_537155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 490)
    suppress_warnings_call_result_537157 = invoke(stypy.reporting.localization.Localization(__file__, 490, 9), suppress_warnings_537155, *[], **kwargs_537156)
    
    with_537158 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 490, 9), suppress_warnings_call_result_537157, 'with parameter', '__enter__', '__exit__')

    if with_537158:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 490)
        enter___537159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 9), suppress_warnings_call_result_537157, '__enter__')
        with_enter_537160 = invoke(stypy.reporting.localization.Localization(__file__, 490, 9), enter___537159)
        # Assigning a type to the variable 'sup' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 9), 'sup', with_enter_537160)
        
        # Call to filter(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'IntegrationWarning' (line 491)
        IntegrationWarning_537163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 19), 'IntegrationWarning', False)
        str_537164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 39), 'str', 'The occurrence of roundoff error is detected')
        # Processing the call keyword arguments (line 491)
        kwargs_537165 = {}
        # Getting the type of 'sup' (line 491)
        sup_537161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 491)
        filter_537162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), sup_537161, 'filter')
        # Calling filter(args, kwargs) (line 491)
        filter_call_result_537166 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), filter_537162, *[IntegrationWarning_537163, str_537164], **kwargs_537165)
        
        
        # Assigning a Call to a Name (line 492):
        
        # Call to seterr(...): (line 492)
        # Processing the call keyword arguments (line 492)
        str_537169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 31), 'str', 'ignore')
        keyword_537170 = str_537169
        kwargs_537171 = {'all': keyword_537170}
        # Getting the type of 'np' (line 492)
        np_537167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 17), 'np', False)
        # Obtaining the member 'seterr' of a type (line 492)
        seterr_537168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 17), np_537167, 'seterr')
        # Calling seterr(args, kwargs) (line 492)
        seterr_call_result_537172 = invoke(stypy.reporting.localization.Localization(__file__, 492, 17), seterr_537168, *[], **kwargs_537171)
        
        # Assigning a type to the variable 'olderr' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'olderr', seterr_call_result_537172)
        
        # Try-finally block (line 493)
        
        # Call to check(...): (line 494)
        # Processing the call keyword arguments (line 494)
        # Getting the type of 'dtype' (line 494)
        dtype_537175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 29), 'dtype', False)
        keyword_537176 = dtype_537175
        kwargs_537177 = {'dtype': keyword_537176}
        # Getting the type of 'test' (line 494)
        test_537173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'test', False)
        # Obtaining the member 'check' of a type (line 494)
        check_537174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 12), test_537173, 'check')
        # Calling check(args, kwargs) (line 494)
        check_call_result_537178 = invoke(stypy.reporting.localization.Localization(__file__, 494, 12), check_537174, *[], **kwargs_537177)
        
        
        # finally branch of the try-finally block (line 493)
        
        # Call to seterr(...): (line 496)
        # Processing the call keyword arguments (line 496)
        # Getting the type of 'olderr' (line 496)
        olderr_537181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 24), 'olderr', False)
        kwargs_537182 = {'olderr_537181': olderr_537181}
        # Getting the type of 'np' (line 496)
        np_537179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'np', False)
        # Obtaining the member 'seterr' of a type (line 496)
        seterr_537180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 12), np_537179, 'seterr')
        # Calling seterr(args, kwargs) (line 496)
        seterr_call_result_537183 = invoke(stypy.reporting.localization.Localization(__file__, 496, 12), seterr_537180, *[], **kwargs_537182)
        
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 490)
        exit___537184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 9), suppress_warnings_call_result_537157, '__exit__')
        with_exit_537185 = invoke(stypy.reporting.localization.Localization(__file__, 490, 9), exit___537184, None, None, None)

    
    # ################# End of '_test_factory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_test_factory' in the type store
    # Getting the type of 'stypy_return_type' (line 488)
    stypy_return_type_537186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_537186)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_test_factory'
    return stypy_return_type_537186

# Assigning a type to the variable '_test_factory' (line 488)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), '_test_factory', _test_factory)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

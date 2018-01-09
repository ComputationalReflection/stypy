
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: # Scipy imports.
4: import numpy as np
5: from numpy import pi
6: from numpy.testing import (assert_array_almost_equal,
7:                            assert_equal, assert_warns)
8: from pytest import raises as assert_raises
9: from scipy.odr import Data, Model, ODR, RealData, OdrStop, OdrWarning
10: 
11: 
12: class TestODR(object):
13: 
14:     # Bad Data for 'x'
15: 
16:     def test_bad_data(self):
17:         assert_raises(ValueError, Data, 2, 1)
18:         assert_raises(ValueError, RealData, 2, 1)
19: 
20:     # Empty Data for 'x'
21:     def empty_data_func(self, B, x):
22:         return B[0]*x + B[1]
23: 
24:     def test_empty_data(self):
25:         beta0 = [0.02, 0.0]
26:         linear = Model(self.empty_data_func)
27: 
28:         empty_dat = Data([], [])
29:         assert_warns(OdrWarning, ODR,
30:                      empty_dat, linear, beta0=beta0)
31: 
32:         empty_dat = RealData([], [])
33:         assert_warns(OdrWarning, ODR,
34:                      empty_dat, linear, beta0=beta0)
35: 
36:     # Explicit Example
37: 
38:     def explicit_fcn(self, B, x):
39:         ret = B[0] + B[1] * np.power(np.exp(B[2]*x) - 1.0, 2)
40:         return ret
41: 
42:     def explicit_fjd(self, B, x):
43:         eBx = np.exp(B[2]*x)
44:         ret = B[1] * 2.0 * (eBx-1.0) * B[2] * eBx
45:         return ret
46: 
47:     def explicit_fjb(self, B, x):
48:         eBx = np.exp(B[2]*x)
49:         res = np.vstack([np.ones(x.shape[-1]),
50:                          np.power(eBx-1.0, 2),
51:                          B[1]*2.0*(eBx-1.0)*eBx*x])
52:         return res
53: 
54:     def test_explicit(self):
55:         explicit_mod = Model(
56:             self.explicit_fcn,
57:             fjacb=self.explicit_fjb,
58:             fjacd=self.explicit_fjd,
59:             meta=dict(name='Sample Explicit Model',
60:                       ref='ODRPACK UG, pg. 39'),
61:         )
62:         explicit_dat = Data([0.,0.,5.,7.,7.5,10.,16.,26.,30.,34.,34.5,100.],
63:                         [1265.,1263.6,1258.,1254.,1253.,1249.8,1237.,1218.,1220.6,
64:                          1213.8,1215.5,1212.])
65:         explicit_odr = ODR(explicit_dat, explicit_mod, beta0=[1500.0, -50.0, -0.1],
66:                        ifixx=[0,0,1,1,1,1,1,1,1,1,1,0])
67:         explicit_odr.set_job(deriv=2)
68:         explicit_odr.set_iprint(init=0, iter=0, final=0)
69: 
70:         out = explicit_odr.run()
71:         assert_array_almost_equal(
72:             out.beta,
73:             np.array([1.2646548050648876e+03, -5.4018409956678255e+01,
74:                 -8.7849712165253724e-02]),
75:         )
76:         assert_array_almost_equal(
77:             out.sd_beta,
78:             np.array([1.0349270280543437, 1.583997785262061, 0.0063321988657267]),
79:         )
80:         assert_array_almost_equal(
81:             out.cov_beta,
82:             np.array([[4.4949592379003039e-01, -3.7421976890364739e-01,
83:                  -8.0978217468468912e-04],
84:                [-3.7421976890364739e-01, 1.0529686462751804e+00,
85:                  -1.9453521827942002e-03],
86:                [-8.0978217468468912e-04, -1.9453521827942002e-03,
87:                   1.6827336938454476e-05]]),
88:         )
89: 
90:     # Implicit Example
91: 
92:     def implicit_fcn(self, B, x):
93:         return (B[2]*np.power(x[0]-B[0], 2) +
94:                 2.0*B[3]*(x[0]-B[0])*(x[1]-B[1]) +
95:                 B[4]*np.power(x[1]-B[1], 2) - 1.0)
96: 
97:     def test_implicit(self):
98:         implicit_mod = Model(
99:             self.implicit_fcn,
100:             implicit=1,
101:             meta=dict(name='Sample Implicit Model',
102:                       ref='ODRPACK UG, pg. 49'),
103:         )
104:         implicit_dat = Data([
105:             [0.5,1.2,1.6,1.86,2.12,2.36,2.44,2.36,2.06,1.74,1.34,0.9,-0.28,
106:              -0.78,-1.36,-1.9,-2.5,-2.88,-3.18,-3.44],
107:             [-0.12,-0.6,-1.,-1.4,-2.54,-3.36,-4.,-4.75,-5.25,-5.64,-5.97,-6.32,
108:              -6.44,-6.44,-6.41,-6.25,-5.88,-5.5,-5.24,-4.86]],
109:             1,
110:         )
111:         implicit_odr = ODR(implicit_dat, implicit_mod,
112:             beta0=[-1.0, -3.0, 0.09, 0.02, 0.08])
113: 
114:         out = implicit_odr.run()
115:         assert_array_almost_equal(
116:             out.beta,
117:             np.array([-0.9993809167281279, -2.9310484652026476, 0.0875730502693354,
118:                 0.0162299708984738, 0.0797537982976416]),
119:         )
120:         assert_array_almost_equal(
121:             out.sd_beta,
122:             np.array([0.1113840353364371, 0.1097673310686467, 0.0041060738314314,
123:                 0.0027500347539902, 0.0034962501532468]),
124:         )
125:         assert_array_almost_equal(
126:             out.cov_beta,
127:             np.array([[2.1089274602333052e+00, -1.9437686411979040e+00,
128:                   7.0263550868344446e-02, -4.7175267373474862e-02,
129:                   5.2515575927380355e-02],
130:                [-1.9437686411979040e+00, 2.0481509222414456e+00,
131:                  -6.1600515853057307e-02, 4.6268827806232933e-02,
132:                  -5.8822307501391467e-02],
133:                [7.0263550868344446e-02, -6.1600515853057307e-02,
134:                   2.8659542561579308e-03, -1.4628662260014491e-03,
135:                   1.4528860663055824e-03],
136:                [-4.7175267373474862e-02, 4.6268827806232933e-02,
137:                  -1.4628662260014491e-03, 1.2855592885514335e-03,
138:                  -1.2692942951415293e-03],
139:                [5.2515575927380355e-02, -5.8822307501391467e-02,
140:                   1.4528860663055824e-03, -1.2692942951415293e-03,
141:                   2.0778813389755596e-03]]),
142:         )
143: 
144:     # Multi-variable Example
145: 
146:     def multi_fcn(self, B, x):
147:         if (x < 0.0).any():
148:             raise OdrStop
149:         theta = pi*B[3]/2.
150:         ctheta = np.cos(theta)
151:         stheta = np.sin(theta)
152:         omega = np.power(2.*pi*x*np.exp(-B[2]), B[3])
153:         phi = np.arctan2((omega*stheta), (1.0 + omega*ctheta))
154:         r = (B[0] - B[1]) * np.power(np.sqrt(np.power(1.0 + omega*ctheta, 2) +
155:              np.power(omega*stheta, 2)), -B[4])
156:         ret = np.vstack([B[1] + r*np.cos(B[4]*phi),
157:                          r*np.sin(B[4]*phi)])
158:         return ret
159: 
160:     def test_multi(self):
161:         multi_mod = Model(
162:             self.multi_fcn,
163:             meta=dict(name='Sample Multi-Response Model',
164:                       ref='ODRPACK UG, pg. 56'),
165:         )
166: 
167:         multi_x = np.array([30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0,
168:             700.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0,
169:             15000.0, 20000.0, 30000.0, 50000.0, 70000.0, 100000.0, 150000.0])
170:         multi_y = np.array([
171:             [4.22, 4.167, 4.132, 4.038, 4.019, 3.956, 3.884, 3.784, 3.713,
172:              3.633, 3.54, 3.433, 3.358, 3.258, 3.193, 3.128, 3.059, 2.984,
173:              2.934, 2.876, 2.838, 2.798, 2.759],
174:             [0.136, 0.167, 0.188, 0.212, 0.236, 0.257, 0.276, 0.297, 0.309,
175:              0.311, 0.314, 0.311, 0.305, 0.289, 0.277, 0.255, 0.24, 0.218,
176:              0.202, 0.182, 0.168, 0.153, 0.139],
177:         ])
178:         n = len(multi_x)
179:         multi_we = np.zeros((2, 2, n), dtype=float)
180:         multi_ifixx = np.ones(n, dtype=int)
181:         multi_delta = np.zeros(n, dtype=float)
182: 
183:         multi_we[0,0,:] = 559.6
184:         multi_we[1,0,:] = multi_we[0,1,:] = -1634.0
185:         multi_we[1,1,:] = 8397.0
186: 
187:         for i in range(n):
188:             if multi_x[i] < 100.0:
189:                 multi_ifixx[i] = 0
190:             elif multi_x[i] <= 150.0:
191:                 pass  # defaults are fine
192:             elif multi_x[i] <= 1000.0:
193:                 multi_delta[i] = 25.0
194:             elif multi_x[i] <= 10000.0:
195:                 multi_delta[i] = 560.0
196:             elif multi_x[i] <= 100000.0:
197:                 multi_delta[i] = 9500.0
198:             else:
199:                 multi_delta[i] = 144000.0
200:             if multi_x[i] == 100.0 or multi_x[i] == 150.0:
201:                 multi_we[:,:,i] = 0.0
202: 
203:         multi_dat = Data(multi_x, multi_y, wd=1e-4/np.power(multi_x, 2),
204:             we=multi_we)
205:         multi_odr = ODR(multi_dat, multi_mod, beta0=[4.,2.,7.,.4,.5],
206:             delta0=multi_delta, ifixx=multi_ifixx)
207:         multi_odr.set_job(deriv=1, del_init=1)
208: 
209:         out = multi_odr.run()
210:         assert_array_almost_equal(
211:             out.beta,
212:             np.array([4.3799880305938963, 2.4333057577497703, 8.0028845899503978,
213:                 0.5101147161764654, 0.5173902330489161]),
214:         )
215:         assert_array_almost_equal(
216:             out.sd_beta,
217:             np.array([0.0130625231081944, 0.0130499785273277, 0.1167085962217757,
218:                 0.0132642749596149, 0.0288529201353984]),
219:         )
220:         assert_array_almost_equal(
221:             out.cov_beta,
222:             np.array([[0.0064918418231375, 0.0036159705923791, 0.0438637051470406,
223:                 -0.0058700836512467, 0.011281212888768],
224:                [0.0036159705923791, 0.0064793789429006, 0.0517610978353126,
225:                 -0.0051181304940204, 0.0130726943624117],
226:                [0.0438637051470406, 0.0517610978353126, 0.5182263323095322,
227:                 -0.0563083340093696, 0.1269490939468611],
228:                [-0.0058700836512467, -0.0051181304940204, -0.0563083340093696,
229:                  0.0066939246261263, -0.0140184391377962],
230:                [0.011281212888768, 0.0130726943624117, 0.1269490939468611,
231:                 -0.0140184391377962, 0.0316733013820852]]),
232:         )
233: 
234:     # Pearson's Data
235:     # K. Pearson, Philosophical Magazine, 2, 559 (1901)
236: 
237:     def pearson_fcn(self, B, x):
238:         return B[0] + B[1]*x
239: 
240:     def test_pearson(self):
241:         p_x = np.array([0.,.9,1.8,2.6,3.3,4.4,5.2,6.1,6.5,7.4])
242:         p_y = np.array([5.9,5.4,4.4,4.6,3.5,3.7,2.8,2.8,2.4,1.5])
243:         p_sx = np.array([.03,.03,.04,.035,.07,.11,.13,.22,.74,1.])
244:         p_sy = np.array([1.,.74,.5,.35,.22,.22,.12,.12,.1,.04])
245: 
246:         p_dat = RealData(p_x, p_y, sx=p_sx, sy=p_sy)
247: 
248:         # Reverse the data to test invariance of results
249:         pr_dat = RealData(p_y, p_x, sx=p_sy, sy=p_sx)
250: 
251:         p_mod = Model(self.pearson_fcn, meta=dict(name='Uni-linear Fit'))
252: 
253:         p_odr = ODR(p_dat, p_mod, beta0=[1.,1.])
254:         pr_odr = ODR(pr_dat, p_mod, beta0=[1.,1.])
255: 
256:         out = p_odr.run()
257:         assert_array_almost_equal(
258:             out.beta,
259:             np.array([5.4767400299231674, -0.4796082367610305]),
260:         )
261:         assert_array_almost_equal(
262:             out.sd_beta,
263:             np.array([0.3590121690702467, 0.0706291186037444]),
264:         )
265:         assert_array_almost_equal(
266:             out.cov_beta,
267:             np.array([[0.0854275622946333, -0.0161807025443155],
268:                [-0.0161807025443155, 0.003306337993922]]),
269:         )
270: 
271:         rout = pr_odr.run()
272:         assert_array_almost_equal(
273:             rout.beta,
274:             np.array([11.4192022410781231, -2.0850374506165474]),
275:         )
276:         assert_array_almost_equal(
277:             rout.sd_beta,
278:             np.array([0.9820231665657161, 0.3070515616198911]),
279:         )
280:         assert_array_almost_equal(
281:             rout.cov_beta,
282:             np.array([[0.6391799462548782, -0.1955657291119177],
283:                [-0.1955657291119177, 0.0624888159223392]]),
284:         )
285: 
286:     # Lorentz Peak
287:     # The data is taken from one of the undergraduate physics labs I performed.
288: 
289:     def lorentz(self, beta, x):
290:         return (beta[0]*beta[1]*beta[2] / np.sqrt(np.power(x*x -
291:             beta[2]*beta[2], 2.0) + np.power(beta[1]*x, 2.0)))
292: 
293:     def test_lorentz(self):
294:         l_sy = np.array([.29]*18)
295:         l_sx = np.array([.000972971,.000948268,.000707632,.000706679,
296:             .000706074, .000703918,.000698955,.000456856,
297:             .000455207,.000662717,.000654619,.000652694,
298:             .000000859202,.00106589,.00106378,.00125483, .00140818,.00241839])
299: 
300:         l_dat = RealData(
301:             [3.9094, 3.85945, 3.84976, 3.84716, 3.84551, 3.83964, 3.82608,
302:              3.78847, 3.78163, 3.72558, 3.70274, 3.6973, 3.67373, 3.65982,
303:              3.6562, 3.62498, 3.55525, 3.41886],
304:             [652, 910.5, 984, 1000, 1007.5, 1053, 1160.5, 1409.5, 1430, 1122,
305:              957.5, 920, 777.5, 709.5, 698, 578.5, 418.5, 275.5],
306:             sx=l_sx,
307:             sy=l_sy,
308:         )
309:         l_mod = Model(self.lorentz, meta=dict(name='Lorentz Peak'))
310:         l_odr = ODR(l_dat, l_mod, beta0=(1000., .1, 3.8))
311: 
312:         out = l_odr.run()
313:         assert_array_almost_equal(
314:             out.beta,
315:             np.array([1.4306780846149925e+03, 1.3390509034538309e-01,
316:                  3.7798193600109009e+00]),
317:         )
318:         assert_array_almost_equal(
319:             out.sd_beta,
320:             np.array([7.3621186811330963e-01, 3.5068899941471650e-04,
321:                  2.4451209281408992e-04]),
322:         )
323:         assert_array_almost_equal(
324:             out.cov_beta,
325:             np.array([[2.4714409064597873e-01, -6.9067261911110836e-05,
326:                  -3.1236953270424990e-05],
327:                [-6.9067261911110836e-05, 5.6077531517333009e-08,
328:                   3.6133261832722601e-08],
329:                [-3.1236953270424990e-05, 3.6133261832722601e-08,
330:                   2.7261220025171730e-08]]),
331:         )
332: 
333:     def test_ticket_1253(self):
334:         def linear(c, x):
335:             return c[0]*x+c[1]
336: 
337:         c = [2.0, 3.0]
338:         x = np.linspace(0, 10)
339:         y = linear(c, x)
340: 
341:         model = Model(linear)
342:         data = Data(x, y, wd=1.0, we=1.0)
343:         job = ODR(data, model, beta0=[1.0, 1.0])
344:         result = job.run()
345:         assert_equal(result.info, 2)
346: 
347: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/tests/')
import_165777 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_165777) is not StypyTypeError):

    if (import_165777 != 'pyd_module'):
        __import__(import_165777)
        sys_modules_165778 = sys.modules[import_165777]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_165778.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_165777)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy import pi' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/tests/')
import_165779 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_165779) is not StypyTypeError):

    if (import_165779 != 'pyd_module'):
        __import__(import_165779)
        sys_modules_165780 = sys.modules[import_165779]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', sys_modules_165780.module_type_store, module_type_store, ['pi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_165780, sys_modules_165780.module_type_store, module_type_store)
    else:
        from numpy import pi

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', None, module_type_store, ['pi'], [pi])

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_165779)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_equal, assert_warns' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/tests/')
import_165781 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_165781) is not StypyTypeError):

    if (import_165781 != 'pyd_module'):
        __import__(import_165781)
        sys_modules_165782 = sys.modules[import_165781]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_165782.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_equal', 'assert_warns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_165782, sys_modules_165782.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_equal, assert_warns

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_equal', 'assert_warns'], [assert_array_almost_equal, assert_equal, assert_warns])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_165781)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/tests/')
import_165783 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_165783) is not StypyTypeError):

    if (import_165783 != 'pyd_module'):
        __import__(import_165783)
        sys_modules_165784 = sys.modules[import_165783]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_165784.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_165784, sys_modules_165784.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_165783)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.odr import Data, Model, ODR, RealData, OdrStop, OdrWarning' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/tests/')
import_165785 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.odr')

if (type(import_165785) is not StypyTypeError):

    if (import_165785 != 'pyd_module'):
        __import__(import_165785)
        sys_modules_165786 = sys.modules[import_165785]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.odr', sys_modules_165786.module_type_store, module_type_store, ['Data', 'Model', 'ODR', 'RealData', 'OdrStop', 'OdrWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_165786, sys_modules_165786.module_type_store, module_type_store)
    else:
        from scipy.odr import Data, Model, ODR, RealData, OdrStop, OdrWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.odr', None, module_type_store, ['Data', 'Model', 'ODR', 'RealData', 'OdrStop', 'OdrWarning'], [Data, Model, ODR, RealData, OdrStop, OdrWarning])

else:
    # Assigning a type to the variable 'scipy.odr' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.odr', import_165785)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/tests/')

# Declaration of the 'TestODR' class

class TestODR(object, ):

    @norecursion
    def test_bad_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_data'
        module_type_store = module_type_store.open_function_context('test_bad_data', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.test_bad_data.__dict__.__setitem__('stypy_localization', localization)
        TestODR.test_bad_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.test_bad_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.test_bad_data.__dict__.__setitem__('stypy_function_name', 'TestODR.test_bad_data')
        TestODR.test_bad_data.__dict__.__setitem__('stypy_param_names_list', [])
        TestODR.test_bad_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.test_bad_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.test_bad_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.test_bad_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.test_bad_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.test_bad_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.test_bad_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_data(...)' code ##################

        
        # Call to assert_raises(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'ValueError' (line 17)
        ValueError_165788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'ValueError', False)
        # Getting the type of 'Data' (line 17)
        Data_165789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 34), 'Data', False)
        int_165790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 40), 'int')
        int_165791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 43), 'int')
        # Processing the call keyword arguments (line 17)
        kwargs_165792 = {}
        # Getting the type of 'assert_raises' (line 17)
        assert_raises_165787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 17)
        assert_raises_call_result_165793 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), assert_raises_165787, *[ValueError_165788, Data_165789, int_165790, int_165791], **kwargs_165792)
        
        
        # Call to assert_raises(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'ValueError' (line 18)
        ValueError_165795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'ValueError', False)
        # Getting the type of 'RealData' (line 18)
        RealData_165796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 34), 'RealData', False)
        int_165797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 44), 'int')
        int_165798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'int')
        # Processing the call keyword arguments (line 18)
        kwargs_165799 = {}
        # Getting the type of 'assert_raises' (line 18)
        assert_raises_165794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 18)
        assert_raises_call_result_165800 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), assert_raises_165794, *[ValueError_165795, RealData_165796, int_165797, int_165798], **kwargs_165799)
        
        
        # ################# End of 'test_bad_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_data' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_165801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_data'
        return stypy_return_type_165801


    @norecursion
    def empty_data_func(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'empty_data_func'
        module_type_store = module_type_store.open_function_context('empty_data_func', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.empty_data_func.__dict__.__setitem__('stypy_localization', localization)
        TestODR.empty_data_func.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.empty_data_func.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.empty_data_func.__dict__.__setitem__('stypy_function_name', 'TestODR.empty_data_func')
        TestODR.empty_data_func.__dict__.__setitem__('stypy_param_names_list', ['B', 'x'])
        TestODR.empty_data_func.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.empty_data_func.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.empty_data_func.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.empty_data_func.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.empty_data_func.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.empty_data_func.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.empty_data_func', ['B', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'empty_data_func', localization, ['B', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'empty_data_func(...)' code ##################

        
        # Obtaining the type of the subscript
        int_165802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
        # Getting the type of 'B' (line 22)
        B_165803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'B')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___165804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 15), B_165803, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_165805 = invoke(stypy.reporting.localization.Localization(__file__, 22, 15), getitem___165804, int_165802)
        
        # Getting the type of 'x' (line 22)
        x_165806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'x')
        # Applying the binary operator '*' (line 22)
        result_mul_165807 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '*', subscript_call_result_165805, x_165806)
        
        
        # Obtaining the type of the subscript
        int_165808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'int')
        # Getting the type of 'B' (line 22)
        B_165809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'B')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___165810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), B_165809, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_165811 = invoke(stypy.reporting.localization.Localization(__file__, 22, 24), getitem___165810, int_165808)
        
        # Applying the binary operator '+' (line 22)
        result_add_165812 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '+', result_mul_165807, subscript_call_result_165811)
        
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', result_add_165812)
        
        # ################# End of 'empty_data_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'empty_data_func' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_165813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165813)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'empty_data_func'
        return stypy_return_type_165813


    @norecursion
    def test_empty_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty_data'
        module_type_store = module_type_store.open_function_context('test_empty_data', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.test_empty_data.__dict__.__setitem__('stypy_localization', localization)
        TestODR.test_empty_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.test_empty_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.test_empty_data.__dict__.__setitem__('stypy_function_name', 'TestODR.test_empty_data')
        TestODR.test_empty_data.__dict__.__setitem__('stypy_param_names_list', [])
        TestODR.test_empty_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.test_empty_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.test_empty_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.test_empty_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.test_empty_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.test_empty_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.test_empty_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_empty_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_empty_data(...)' code ##################

        
        # Assigning a List to a Name (line 25):
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_165814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        float_165815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_165814, float_165815)
        # Adding element type (line 25)
        float_165816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_165814, float_165816)
        
        # Assigning a type to the variable 'beta0' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'beta0', list_165814)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to Model(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'self' (line 26)
        self_165818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'self', False)
        # Obtaining the member 'empty_data_func' of a type (line 26)
        empty_data_func_165819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 23), self_165818, 'empty_data_func')
        # Processing the call keyword arguments (line 26)
        kwargs_165820 = {}
        # Getting the type of 'Model' (line 26)
        Model_165817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'Model', False)
        # Calling Model(args, kwargs) (line 26)
        Model_call_result_165821 = invoke(stypy.reporting.localization.Localization(__file__, 26, 17), Model_165817, *[empty_data_func_165819], **kwargs_165820)
        
        # Assigning a type to the variable 'linear' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'linear', Model_call_result_165821)
        
        # Assigning a Call to a Name (line 28):
        
        # Call to Data(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_165823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_165824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        
        # Processing the call keyword arguments (line 28)
        kwargs_165825 = {}
        # Getting the type of 'Data' (line 28)
        Data_165822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'Data', False)
        # Calling Data(args, kwargs) (line 28)
        Data_call_result_165826 = invoke(stypy.reporting.localization.Localization(__file__, 28, 20), Data_165822, *[list_165823, list_165824], **kwargs_165825)
        
        # Assigning a type to the variable 'empty_dat' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'empty_dat', Data_call_result_165826)
        
        # Call to assert_warns(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'OdrWarning' (line 29)
        OdrWarning_165828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'OdrWarning', False)
        # Getting the type of 'ODR' (line 29)
        ODR_165829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'ODR', False)
        # Getting the type of 'empty_dat' (line 30)
        empty_dat_165830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'empty_dat', False)
        # Getting the type of 'linear' (line 30)
        linear_165831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'linear', False)
        # Processing the call keyword arguments (line 29)
        # Getting the type of 'beta0' (line 30)
        beta0_165832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 46), 'beta0', False)
        keyword_165833 = beta0_165832
        kwargs_165834 = {'beta0': keyword_165833}
        # Getting the type of 'assert_warns' (line 29)
        assert_warns_165827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 29)
        assert_warns_call_result_165835 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert_warns_165827, *[OdrWarning_165828, ODR_165829, empty_dat_165830, linear_165831], **kwargs_165834)
        
        
        # Assigning a Call to a Name (line 32):
        
        # Call to RealData(...): (line 32)
        # Processing the call arguments (line 32)
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_165837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_165838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        
        # Processing the call keyword arguments (line 32)
        kwargs_165839 = {}
        # Getting the type of 'RealData' (line 32)
        RealData_165836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'RealData', False)
        # Calling RealData(args, kwargs) (line 32)
        RealData_call_result_165840 = invoke(stypy.reporting.localization.Localization(__file__, 32, 20), RealData_165836, *[list_165837, list_165838], **kwargs_165839)
        
        # Assigning a type to the variable 'empty_dat' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'empty_dat', RealData_call_result_165840)
        
        # Call to assert_warns(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'OdrWarning' (line 33)
        OdrWarning_165842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'OdrWarning', False)
        # Getting the type of 'ODR' (line 33)
        ODR_165843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'ODR', False)
        # Getting the type of 'empty_dat' (line 34)
        empty_dat_165844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'empty_dat', False)
        # Getting the type of 'linear' (line 34)
        linear_165845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'linear', False)
        # Processing the call keyword arguments (line 33)
        # Getting the type of 'beta0' (line 34)
        beta0_165846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 46), 'beta0', False)
        keyword_165847 = beta0_165846
        kwargs_165848 = {'beta0': keyword_165847}
        # Getting the type of 'assert_warns' (line 33)
        assert_warns_165841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 33)
        assert_warns_call_result_165849 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assert_warns_165841, *[OdrWarning_165842, ODR_165843, empty_dat_165844, linear_165845], **kwargs_165848)
        
        
        # ################# End of 'test_empty_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty_data' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_165850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165850)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty_data'
        return stypy_return_type_165850


    @norecursion
    def explicit_fcn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'explicit_fcn'
        module_type_store = module_type_store.open_function_context('explicit_fcn', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_localization', localization)
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_function_name', 'TestODR.explicit_fcn')
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_param_names_list', ['B', 'x'])
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.explicit_fcn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.explicit_fcn', ['B', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'explicit_fcn', localization, ['B', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'explicit_fcn(...)' code ##################

        
        # Assigning a BinOp to a Name (line 39):
        
        # Obtaining the type of the subscript
        int_165851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 16), 'int')
        # Getting the type of 'B' (line 39)
        B_165852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'B')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___165853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 14), B_165852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_165854 = invoke(stypy.reporting.localization.Localization(__file__, 39, 14), getitem___165853, int_165851)
        
        
        # Obtaining the type of the subscript
        int_165855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'int')
        # Getting the type of 'B' (line 39)
        B_165856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'B')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___165857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 21), B_165856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_165858 = invoke(stypy.reporting.localization.Localization(__file__, 39, 21), getitem___165857, int_165855)
        
        
        # Call to power(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to exp(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining the type of the subscript
        int_165863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
        # Getting the type of 'B' (line 39)
        B_165864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 44), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___165865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 44), B_165864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_165866 = invoke(stypy.reporting.localization.Localization(__file__, 39, 44), getitem___165865, int_165863)
        
        # Getting the type of 'x' (line 39)
        x_165867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 49), 'x', False)
        # Applying the binary operator '*' (line 39)
        result_mul_165868 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 44), '*', subscript_call_result_165866, x_165867)
        
        # Processing the call keyword arguments (line 39)
        kwargs_165869 = {}
        # Getting the type of 'np' (line 39)
        np_165861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'np', False)
        # Obtaining the member 'exp' of a type (line 39)
        exp_165862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 37), np_165861, 'exp')
        # Calling exp(args, kwargs) (line 39)
        exp_call_result_165870 = invoke(stypy.reporting.localization.Localization(__file__, 39, 37), exp_165862, *[result_mul_165868], **kwargs_165869)
        
        float_165871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 54), 'float')
        # Applying the binary operator '-' (line 39)
        result_sub_165872 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 37), '-', exp_call_result_165870, float_165871)
        
        int_165873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 59), 'int')
        # Processing the call keyword arguments (line 39)
        kwargs_165874 = {}
        # Getting the type of 'np' (line 39)
        np_165859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'np', False)
        # Obtaining the member 'power' of a type (line 39)
        power_165860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 28), np_165859, 'power')
        # Calling power(args, kwargs) (line 39)
        power_call_result_165875 = invoke(stypy.reporting.localization.Localization(__file__, 39, 28), power_165860, *[result_sub_165872, int_165873], **kwargs_165874)
        
        # Applying the binary operator '*' (line 39)
        result_mul_165876 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 21), '*', subscript_call_result_165858, power_call_result_165875)
        
        # Applying the binary operator '+' (line 39)
        result_add_165877 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 14), '+', subscript_call_result_165854, result_mul_165876)
        
        # Assigning a type to the variable 'ret' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'ret', result_add_165877)
        # Getting the type of 'ret' (line 40)
        ret_165878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', ret_165878)
        
        # ################# End of 'explicit_fcn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'explicit_fcn' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_165879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165879)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'explicit_fcn'
        return stypy_return_type_165879


    @norecursion
    def explicit_fjd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'explicit_fjd'
        module_type_store = module_type_store.open_function_context('explicit_fjd', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_localization', localization)
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_function_name', 'TestODR.explicit_fjd')
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_param_names_list', ['B', 'x'])
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.explicit_fjd.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.explicit_fjd', ['B', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'explicit_fjd', localization, ['B', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'explicit_fjd(...)' code ##################

        
        # Assigning a Call to a Name (line 43):
        
        # Call to exp(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Obtaining the type of the subscript
        int_165882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'int')
        # Getting the type of 'B' (line 43)
        B_165883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___165884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 21), B_165883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_165885 = invoke(stypy.reporting.localization.Localization(__file__, 43, 21), getitem___165884, int_165882)
        
        # Getting the type of 'x' (line 43)
        x_165886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'x', False)
        # Applying the binary operator '*' (line 43)
        result_mul_165887 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 21), '*', subscript_call_result_165885, x_165886)
        
        # Processing the call keyword arguments (line 43)
        kwargs_165888 = {}
        # Getting the type of 'np' (line 43)
        np_165880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'np', False)
        # Obtaining the member 'exp' of a type (line 43)
        exp_165881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 14), np_165880, 'exp')
        # Calling exp(args, kwargs) (line 43)
        exp_call_result_165889 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), exp_165881, *[result_mul_165887], **kwargs_165888)
        
        # Assigning a type to the variable 'eBx' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'eBx', exp_call_result_165889)
        
        # Assigning a BinOp to a Name (line 44):
        
        # Obtaining the type of the subscript
        int_165890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'int')
        # Getting the type of 'B' (line 44)
        B_165891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'B')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___165892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 14), B_165891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_165893 = invoke(stypy.reporting.localization.Localization(__file__, 44, 14), getitem___165892, int_165890)
        
        float_165894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'float')
        # Applying the binary operator '*' (line 44)
        result_mul_165895 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 14), '*', subscript_call_result_165893, float_165894)
        
        # Getting the type of 'eBx' (line 44)
        eBx_165896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 28), 'eBx')
        float_165897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 32), 'float')
        # Applying the binary operator '-' (line 44)
        result_sub_165898 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 28), '-', eBx_165896, float_165897)
        
        # Applying the binary operator '*' (line 44)
        result_mul_165899 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 25), '*', result_mul_165895, result_sub_165898)
        
        
        # Obtaining the type of the subscript
        int_165900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'int')
        # Getting the type of 'B' (line 44)
        B_165901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'B')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___165902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 39), B_165901, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_165903 = invoke(stypy.reporting.localization.Localization(__file__, 44, 39), getitem___165902, int_165900)
        
        # Applying the binary operator '*' (line 44)
        result_mul_165904 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 37), '*', result_mul_165899, subscript_call_result_165903)
        
        # Getting the type of 'eBx' (line 44)
        eBx_165905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 46), 'eBx')
        # Applying the binary operator '*' (line 44)
        result_mul_165906 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 44), '*', result_mul_165904, eBx_165905)
        
        # Assigning a type to the variable 'ret' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'ret', result_mul_165906)
        # Getting the type of 'ret' (line 45)
        ret_165907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', ret_165907)
        
        # ################# End of 'explicit_fjd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'explicit_fjd' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_165908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165908)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'explicit_fjd'
        return stypy_return_type_165908


    @norecursion
    def explicit_fjb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'explicit_fjb'
        module_type_store = module_type_store.open_function_context('explicit_fjb', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_localization', localization)
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_function_name', 'TestODR.explicit_fjb')
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_param_names_list', ['B', 'x'])
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.explicit_fjb.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.explicit_fjb', ['B', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'explicit_fjb', localization, ['B', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'explicit_fjb(...)' code ##################

        
        # Assigning a Call to a Name (line 48):
        
        # Call to exp(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Obtaining the type of the subscript
        int_165911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'int')
        # Getting the type of 'B' (line 48)
        B_165912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___165913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 21), B_165912, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_165914 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), getitem___165913, int_165911)
        
        # Getting the type of 'x' (line 48)
        x_165915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'x', False)
        # Applying the binary operator '*' (line 48)
        result_mul_165916 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 21), '*', subscript_call_result_165914, x_165915)
        
        # Processing the call keyword arguments (line 48)
        kwargs_165917 = {}
        # Getting the type of 'np' (line 48)
        np_165909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 14), 'np', False)
        # Obtaining the member 'exp' of a type (line 48)
        exp_165910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 14), np_165909, 'exp')
        # Calling exp(args, kwargs) (line 48)
        exp_call_result_165918 = invoke(stypy.reporting.localization.Localization(__file__, 48, 14), exp_165910, *[result_mul_165916], **kwargs_165917)
        
        # Assigning a type to the variable 'eBx' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'eBx', exp_call_result_165918)
        
        # Assigning a Call to a Name (line 49):
        
        # Call to vstack(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_165921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        
        # Call to ones(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Obtaining the type of the subscript
        int_165924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 41), 'int')
        # Getting the type of 'x' (line 49)
        x_165925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'x', False)
        # Obtaining the member 'shape' of a type (line 49)
        shape_165926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 33), x_165925, 'shape')
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___165927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 33), shape_165926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_165928 = invoke(stypy.reporting.localization.Localization(__file__, 49, 33), getitem___165927, int_165924)
        
        # Processing the call keyword arguments (line 49)
        kwargs_165929 = {}
        # Getting the type of 'np' (line 49)
        np_165922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'np', False)
        # Obtaining the member 'ones' of a type (line 49)
        ones_165923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 25), np_165922, 'ones')
        # Calling ones(args, kwargs) (line 49)
        ones_call_result_165930 = invoke(stypy.reporting.localization.Localization(__file__, 49, 25), ones_165923, *[subscript_call_result_165928], **kwargs_165929)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), list_165921, ones_call_result_165930)
        # Adding element type (line 49)
        
        # Call to power(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'eBx' (line 50)
        eBx_165933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 34), 'eBx', False)
        float_165934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 38), 'float')
        # Applying the binary operator '-' (line 50)
        result_sub_165935 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 34), '-', eBx_165933, float_165934)
        
        int_165936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 43), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_165937 = {}
        # Getting the type of 'np' (line 50)
        np_165931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'np', False)
        # Obtaining the member 'power' of a type (line 50)
        power_165932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), np_165931, 'power')
        # Calling power(args, kwargs) (line 50)
        power_call_result_165938 = invoke(stypy.reporting.localization.Localization(__file__, 50, 25), power_165932, *[result_sub_165935, int_165936], **kwargs_165937)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), list_165921, power_call_result_165938)
        # Adding element type (line 49)
        
        # Obtaining the type of the subscript
        int_165939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'int')
        # Getting the type of 'B' (line 51)
        B_165940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___165941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), B_165940, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_165942 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), getitem___165941, int_165939)
        
        float_165943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'float')
        # Applying the binary operator '*' (line 51)
        result_mul_165944 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 25), '*', subscript_call_result_165942, float_165943)
        
        # Getting the type of 'eBx' (line 51)
        eBx_165945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 35), 'eBx', False)
        float_165946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 39), 'float')
        # Applying the binary operator '-' (line 51)
        result_sub_165947 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 35), '-', eBx_165945, float_165946)
        
        # Applying the binary operator '*' (line 51)
        result_mul_165948 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 33), '*', result_mul_165944, result_sub_165947)
        
        # Getting the type of 'eBx' (line 51)
        eBx_165949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 44), 'eBx', False)
        # Applying the binary operator '*' (line 51)
        result_mul_165950 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 43), '*', result_mul_165948, eBx_165949)
        
        # Getting the type of 'x' (line 51)
        x_165951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 48), 'x', False)
        # Applying the binary operator '*' (line 51)
        result_mul_165952 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 47), '*', result_mul_165950, x_165951)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 24), list_165921, result_mul_165952)
        
        # Processing the call keyword arguments (line 49)
        kwargs_165953 = {}
        # Getting the type of 'np' (line 49)
        np_165919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'np', False)
        # Obtaining the member 'vstack' of a type (line 49)
        vstack_165920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 14), np_165919, 'vstack')
        # Calling vstack(args, kwargs) (line 49)
        vstack_call_result_165954 = invoke(stypy.reporting.localization.Localization(__file__, 49, 14), vstack_165920, *[list_165921], **kwargs_165953)
        
        # Assigning a type to the variable 'res' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'res', vstack_call_result_165954)
        # Getting the type of 'res' (line 52)
        res_165955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', res_165955)
        
        # ################# End of 'explicit_fjb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'explicit_fjb' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_165956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_165956)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'explicit_fjb'
        return stypy_return_type_165956


    @norecursion
    def test_explicit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_explicit'
        module_type_store = module_type_store.open_function_context('test_explicit', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.test_explicit.__dict__.__setitem__('stypy_localization', localization)
        TestODR.test_explicit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.test_explicit.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.test_explicit.__dict__.__setitem__('stypy_function_name', 'TestODR.test_explicit')
        TestODR.test_explicit.__dict__.__setitem__('stypy_param_names_list', [])
        TestODR.test_explicit.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.test_explicit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.test_explicit.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.test_explicit.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.test_explicit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.test_explicit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.test_explicit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_explicit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_explicit(...)' code ##################

        
        # Assigning a Call to a Name (line 55):
        
        # Call to Model(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 56)
        self_165958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self', False)
        # Obtaining the member 'explicit_fcn' of a type (line 56)
        explicit_fcn_165959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_165958, 'explicit_fcn')
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'self' (line 57)
        self_165960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'self', False)
        # Obtaining the member 'explicit_fjb' of a type (line 57)
        explicit_fjb_165961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), self_165960, 'explicit_fjb')
        keyword_165962 = explicit_fjb_165961
        # Getting the type of 'self' (line 58)
        self_165963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'self', False)
        # Obtaining the member 'explicit_fjd' of a type (line 58)
        explicit_fjd_165964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 18), self_165963, 'explicit_fjd')
        keyword_165965 = explicit_fjd_165964
        
        # Call to dict(...): (line 59)
        # Processing the call keyword arguments (line 59)
        str_165967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'str', 'Sample Explicit Model')
        keyword_165968 = str_165967
        str_165969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 26), 'str', 'ODRPACK UG, pg. 39')
        keyword_165970 = str_165969
        kwargs_165971 = {'ref': keyword_165970, 'name': keyword_165968}
        # Getting the type of 'dict' (line 59)
        dict_165966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'dict', False)
        # Calling dict(args, kwargs) (line 59)
        dict_call_result_165972 = invoke(stypy.reporting.localization.Localization(__file__, 59, 17), dict_165966, *[], **kwargs_165971)
        
        keyword_165973 = dict_call_result_165972
        kwargs_165974 = {'fjacd': keyword_165965, 'meta': keyword_165973, 'fjacb': keyword_165962}
        # Getting the type of 'Model' (line 55)
        Model_165957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'Model', False)
        # Calling Model(args, kwargs) (line 55)
        Model_call_result_165975 = invoke(stypy.reporting.localization.Localization(__file__, 55, 23), Model_165957, *[explicit_fcn_165959], **kwargs_165974)
        
        # Assigning a type to the variable 'explicit_mod' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'explicit_mod', Model_call_result_165975)
        
        # Assigning a Call to a Name (line 62):
        
        # Call to Data(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_165977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        float_165978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165978)
        # Adding element type (line 62)
        float_165979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165979)
        # Adding element type (line 62)
        float_165980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165980)
        # Adding element type (line 62)
        float_165981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165981)
        # Adding element type (line 62)
        float_165982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165982)
        # Adding element type (line 62)
        float_165983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165983)
        # Adding element type (line 62)
        float_165984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165984)
        # Adding element type (line 62)
        float_165985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165985)
        # Adding element type (line 62)
        float_165986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165986)
        # Adding element type (line 62)
        float_165987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165987)
        # Adding element type (line 62)
        float_165988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165988)
        # Adding element type (line 62)
        float_165989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 28), list_165977, float_165989)
        
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_165990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        float_165991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165991)
        # Adding element type (line 63)
        float_165992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165992)
        # Adding element type (line 63)
        float_165993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165993)
        # Adding element type (line 63)
        float_165994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165994)
        # Adding element type (line 63)
        float_165995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165995)
        # Adding element type (line 63)
        float_165996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165996)
        # Adding element type (line 63)
        float_165997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165997)
        # Adding element type (line 63)
        float_165998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165998)
        # Adding element type (line 63)
        float_165999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 75), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_165999)
        # Adding element type (line 63)
        float_166000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_166000)
        # Adding element type (line 63)
        float_166001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_166001)
        # Adding element type (line 63)
        float_166002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 24), list_165990, float_166002)
        
        # Processing the call keyword arguments (line 62)
        kwargs_166003 = {}
        # Getting the type of 'Data' (line 62)
        Data_165976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'Data', False)
        # Calling Data(args, kwargs) (line 62)
        Data_call_result_166004 = invoke(stypy.reporting.localization.Localization(__file__, 62, 23), Data_165976, *[list_165977, list_165990], **kwargs_166003)
        
        # Assigning a type to the variable 'explicit_dat' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'explicit_dat', Data_call_result_166004)
        
        # Assigning a Call to a Name (line 65):
        
        # Call to ODR(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'explicit_dat' (line 65)
        explicit_dat_166006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'explicit_dat', False)
        # Getting the type of 'explicit_mod' (line 65)
        explicit_mod_166007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), 'explicit_mod', False)
        # Processing the call keyword arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_166008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        float_166009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 61), list_166008, float_166009)
        # Adding element type (line 65)
        float_166010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 61), list_166008, float_166010)
        # Adding element type (line 65)
        float_166011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 77), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 61), list_166008, float_166011)
        
        keyword_166012 = list_166008
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_166013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        int_166014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166014)
        # Adding element type (line 66)
        int_166015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166015)
        # Adding element type (line 66)
        int_166016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166016)
        # Adding element type (line 66)
        int_166017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166017)
        # Adding element type (line 66)
        int_166018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166018)
        # Adding element type (line 66)
        int_166019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166019)
        # Adding element type (line 66)
        int_166020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166020)
        # Adding element type (line 66)
        int_166021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166021)
        # Adding element type (line 66)
        int_166022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166022)
        # Adding element type (line 66)
        int_166023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166023)
        # Adding element type (line 66)
        int_166024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166024)
        # Adding element type (line 66)
        int_166025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 29), list_166013, int_166025)
        
        keyword_166026 = list_166013
        kwargs_166027 = {'ifixx': keyword_166026, 'beta0': keyword_166012}
        # Getting the type of 'ODR' (line 65)
        ODR_166005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'ODR', False)
        # Calling ODR(args, kwargs) (line 65)
        ODR_call_result_166028 = invoke(stypy.reporting.localization.Localization(__file__, 65, 23), ODR_166005, *[explicit_dat_166006, explicit_mod_166007], **kwargs_166027)
        
        # Assigning a type to the variable 'explicit_odr' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'explicit_odr', ODR_call_result_166028)
        
        # Call to set_job(...): (line 67)
        # Processing the call keyword arguments (line 67)
        int_166031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 35), 'int')
        keyword_166032 = int_166031
        kwargs_166033 = {'deriv': keyword_166032}
        # Getting the type of 'explicit_odr' (line 67)
        explicit_odr_166029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'explicit_odr', False)
        # Obtaining the member 'set_job' of a type (line 67)
        set_job_166030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), explicit_odr_166029, 'set_job')
        # Calling set_job(args, kwargs) (line 67)
        set_job_call_result_166034 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), set_job_166030, *[], **kwargs_166033)
        
        
        # Call to set_iprint(...): (line 68)
        # Processing the call keyword arguments (line 68)
        int_166037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 37), 'int')
        keyword_166038 = int_166037
        int_166039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 45), 'int')
        keyword_166040 = int_166039
        int_166041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 54), 'int')
        keyword_166042 = int_166041
        kwargs_166043 = {'init': keyword_166038, 'final': keyword_166042, 'iter': keyword_166040}
        # Getting the type of 'explicit_odr' (line 68)
        explicit_odr_166035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'explicit_odr', False)
        # Obtaining the member 'set_iprint' of a type (line 68)
        set_iprint_166036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), explicit_odr_166035, 'set_iprint')
        # Calling set_iprint(args, kwargs) (line 68)
        set_iprint_call_result_166044 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), set_iprint_166036, *[], **kwargs_166043)
        
        
        # Assigning a Call to a Name (line 70):
        
        # Call to run(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_166047 = {}
        # Getting the type of 'explicit_odr' (line 70)
        explicit_odr_166045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'explicit_odr', False)
        # Obtaining the member 'run' of a type (line 70)
        run_166046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 14), explicit_odr_166045, 'run')
        # Calling run(args, kwargs) (line 70)
        run_call_result_166048 = invoke(stypy.reporting.localization.Localization(__file__, 70, 14), run_166046, *[], **kwargs_166047)
        
        # Assigning a type to the variable 'out' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'out', run_call_result_166048)
        
        # Call to assert_array_almost_equal(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'out' (line 72)
        out_166050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'out', False)
        # Obtaining the member 'beta' of a type (line 72)
        beta_166051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), out_166050, 'beta')
        
        # Call to array(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_166054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        float_166055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 21), list_166054, float_166055)
        # Adding element type (line 73)
        float_166056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 21), list_166054, float_166056)
        # Adding element type (line 73)
        float_166057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 21), list_166054, float_166057)
        
        # Processing the call keyword arguments (line 73)
        kwargs_166058 = {}
        # Getting the type of 'np' (line 73)
        np_166052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 73)
        array_166053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), np_166052, 'array')
        # Calling array(args, kwargs) (line 73)
        array_call_result_166059 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), array_166053, *[list_166054], **kwargs_166058)
        
        # Processing the call keyword arguments (line 71)
        kwargs_166060 = {}
        # Getting the type of 'assert_array_almost_equal' (line 71)
        assert_array_almost_equal_166049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 71)
        assert_array_almost_equal_call_result_166061 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), assert_array_almost_equal_166049, *[beta_166051, array_call_result_166059], **kwargs_166060)
        
        
        # Call to assert_array_almost_equal(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'out' (line 77)
        out_166063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'out', False)
        # Obtaining the member 'sd_beta' of a type (line 77)
        sd_beta_166064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), out_166063, 'sd_beta')
        
        # Call to array(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_166067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        float_166068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_166067, float_166068)
        # Adding element type (line 78)
        float_166069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_166067, float_166069)
        # Adding element type (line 78)
        float_166070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 21), list_166067, float_166070)
        
        # Processing the call keyword arguments (line 78)
        kwargs_166071 = {}
        # Getting the type of 'np' (line 78)
        np_166065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 78)
        array_166066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), np_166065, 'array')
        # Calling array(args, kwargs) (line 78)
        array_call_result_166072 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), array_166066, *[list_166067], **kwargs_166071)
        
        # Processing the call keyword arguments (line 76)
        kwargs_166073 = {}
        # Getting the type of 'assert_array_almost_equal' (line 76)
        assert_array_almost_equal_166062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 76)
        assert_array_almost_equal_call_result_166074 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_array_almost_equal_166062, *[sd_beta_166064, array_call_result_166072], **kwargs_166073)
        
        
        # Call to assert_array_almost_equal(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'out' (line 81)
        out_166076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'out', False)
        # Obtaining the member 'cov_beta' of a type (line 81)
        cov_beta_166077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), out_166076, 'cov_beta')
        
        # Call to array(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_166080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_166081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        float_166082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_166081, float_166082)
        # Adding element type (line 82)
        float_166083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_166081, float_166083)
        # Adding element type (line 82)
        float_166084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 22), list_166081, float_166084)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 21), list_166080, list_166081)
        # Adding element type (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_166085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        float_166086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), list_166085, float_166086)
        # Adding element type (line 84)
        float_166087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), list_166085, float_166087)
        # Adding element type (line 84)
        float_166088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 15), list_166085, float_166088)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 21), list_166080, list_166085)
        # Adding element type (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_166089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        float_166090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), list_166089, float_166090)
        # Adding element type (line 86)
        float_166091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), list_166089, float_166091)
        # Adding element type (line 86)
        float_166092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), list_166089, float_166092)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 21), list_166080, list_166089)
        
        # Processing the call keyword arguments (line 82)
        kwargs_166093 = {}
        # Getting the type of 'np' (line 82)
        np_166078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 82)
        array_166079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), np_166078, 'array')
        # Calling array(args, kwargs) (line 82)
        array_call_result_166094 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), array_166079, *[list_166080], **kwargs_166093)
        
        # Processing the call keyword arguments (line 80)
        kwargs_166095 = {}
        # Getting the type of 'assert_array_almost_equal' (line 80)
        assert_array_almost_equal_166075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 80)
        assert_array_almost_equal_call_result_166096 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_array_almost_equal_166075, *[cov_beta_166077, array_call_result_166094], **kwargs_166095)
        
        
        # ################# End of 'test_explicit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_explicit' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_166097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_166097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_explicit'
        return stypy_return_type_166097


    @norecursion
    def implicit_fcn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'implicit_fcn'
        module_type_store = module_type_store.open_function_context('implicit_fcn', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_localization', localization)
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_function_name', 'TestODR.implicit_fcn')
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_param_names_list', ['B', 'x'])
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.implicit_fcn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.implicit_fcn', ['B', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'implicit_fcn', localization, ['B', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'implicit_fcn(...)' code ##################

        
        # Obtaining the type of the subscript
        int_166098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 18), 'int')
        # Getting the type of 'B' (line 93)
        B_166099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'B')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___166100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), B_166099, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_166101 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), getitem___166100, int_166098)
        
        
        # Call to power(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining the type of the subscript
        int_166104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 32), 'int')
        # Getting the type of 'x' (line 93)
        x_166105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 30), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___166106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 30), x_166105, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_166107 = invoke(stypy.reporting.localization.Localization(__file__, 93, 30), getitem___166106, int_166104)
        
        
        # Obtaining the type of the subscript
        int_166108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 37), 'int')
        # Getting the type of 'B' (line 93)
        B_166109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___166110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 35), B_166109, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_166111 = invoke(stypy.reporting.localization.Localization(__file__, 93, 35), getitem___166110, int_166108)
        
        # Applying the binary operator '-' (line 93)
        result_sub_166112 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 30), '-', subscript_call_result_166107, subscript_call_result_166111)
        
        int_166113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 41), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_166114 = {}
        # Getting the type of 'np' (line 93)
        np_166102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'np', False)
        # Obtaining the member 'power' of a type (line 93)
        power_166103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 21), np_166102, 'power')
        # Calling power(args, kwargs) (line 93)
        power_call_result_166115 = invoke(stypy.reporting.localization.Localization(__file__, 93, 21), power_166103, *[result_sub_166112, int_166113], **kwargs_166114)
        
        # Applying the binary operator '*' (line 93)
        result_mul_166116 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 16), '*', subscript_call_result_166101, power_call_result_166115)
        
        float_166117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 16), 'float')
        
        # Obtaining the type of the subscript
        int_166118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'int')
        # Getting the type of 'B' (line 94)
        B_166119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'B')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___166120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 20), B_166119, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_166121 = invoke(stypy.reporting.localization.Localization(__file__, 94, 20), getitem___166120, int_166118)
        
        # Applying the binary operator '*' (line 94)
        result_mul_166122 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 16), '*', float_166117, subscript_call_result_166121)
        
        
        # Obtaining the type of the subscript
        int_166123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 28), 'int')
        # Getting the type of 'x' (line 94)
        x_166124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'x')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___166125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 26), x_166124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_166126 = invoke(stypy.reporting.localization.Localization(__file__, 94, 26), getitem___166125, int_166123)
        
        
        # Obtaining the type of the subscript
        int_166127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 33), 'int')
        # Getting the type of 'B' (line 94)
        B_166128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'B')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___166129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), B_166128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_166130 = invoke(stypy.reporting.localization.Localization(__file__, 94, 31), getitem___166129, int_166127)
        
        # Applying the binary operator '-' (line 94)
        result_sub_166131 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 26), '-', subscript_call_result_166126, subscript_call_result_166130)
        
        # Applying the binary operator '*' (line 94)
        result_mul_166132 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 24), '*', result_mul_166122, result_sub_166131)
        
        
        # Obtaining the type of the subscript
        int_166133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 40), 'int')
        # Getting the type of 'x' (line 94)
        x_166134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'x')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___166135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 38), x_166134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_166136 = invoke(stypy.reporting.localization.Localization(__file__, 94, 38), getitem___166135, int_166133)
        
        
        # Obtaining the type of the subscript
        int_166137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 45), 'int')
        # Getting the type of 'B' (line 94)
        B_166138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 43), 'B')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___166139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 43), B_166138, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_166140 = invoke(stypy.reporting.localization.Localization(__file__, 94, 43), getitem___166139, int_166137)
        
        # Applying the binary operator '-' (line 94)
        result_sub_166141 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 38), '-', subscript_call_result_166136, subscript_call_result_166140)
        
        # Applying the binary operator '*' (line 94)
        result_mul_166142 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 36), '*', result_mul_166132, result_sub_166141)
        
        # Applying the binary operator '+' (line 93)
        result_add_166143 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 16), '+', result_mul_166116, result_mul_166142)
        
        
        # Obtaining the type of the subscript
        int_166144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 18), 'int')
        # Getting the type of 'B' (line 95)
        B_166145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'B')
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___166146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), B_166145, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_166147 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), getitem___166146, int_166144)
        
        
        # Call to power(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining the type of the subscript
        int_166150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 32), 'int')
        # Getting the type of 'x' (line 95)
        x_166151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___166152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 30), x_166151, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_166153 = invoke(stypy.reporting.localization.Localization(__file__, 95, 30), getitem___166152, int_166150)
        
        
        # Obtaining the type of the subscript
        int_166154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 37), 'int')
        # Getting the type of 'B' (line 95)
        B_166155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___166156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 35), B_166155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_166157 = invoke(stypy.reporting.localization.Localization(__file__, 95, 35), getitem___166156, int_166154)
        
        # Applying the binary operator '-' (line 95)
        result_sub_166158 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 30), '-', subscript_call_result_166153, subscript_call_result_166157)
        
        int_166159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 41), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_166160 = {}
        # Getting the type of 'np' (line 95)
        np_166148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'np', False)
        # Obtaining the member 'power' of a type (line 95)
        power_166149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 21), np_166148, 'power')
        # Calling power(args, kwargs) (line 95)
        power_call_result_166161 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), power_166149, *[result_sub_166158, int_166159], **kwargs_166160)
        
        # Applying the binary operator '*' (line 95)
        result_mul_166162 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 16), '*', subscript_call_result_166147, power_call_result_166161)
        
        # Applying the binary operator '+' (line 94)
        result_add_166163 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 49), '+', result_add_166143, result_mul_166162)
        
        float_166164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 46), 'float')
        # Applying the binary operator '-' (line 95)
        result_sub_166165 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 44), '-', result_add_166163, float_166164)
        
        # Assigning a type to the variable 'stypy_return_type' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', result_sub_166165)
        
        # ################# End of 'implicit_fcn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'implicit_fcn' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_166166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_166166)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'implicit_fcn'
        return stypy_return_type_166166


    @norecursion
    def test_implicit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_implicit'
        module_type_store = module_type_store.open_function_context('test_implicit', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.test_implicit.__dict__.__setitem__('stypy_localization', localization)
        TestODR.test_implicit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.test_implicit.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.test_implicit.__dict__.__setitem__('stypy_function_name', 'TestODR.test_implicit')
        TestODR.test_implicit.__dict__.__setitem__('stypy_param_names_list', [])
        TestODR.test_implicit.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.test_implicit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.test_implicit.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.test_implicit.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.test_implicit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.test_implicit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.test_implicit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_implicit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_implicit(...)' code ##################

        
        # Assigning a Call to a Name (line 98):
        
        # Call to Model(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'self' (line 99)
        self_166168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'self', False)
        # Obtaining the member 'implicit_fcn' of a type (line 99)
        implicit_fcn_166169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), self_166168, 'implicit_fcn')
        # Processing the call keyword arguments (line 98)
        int_166170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 21), 'int')
        keyword_166171 = int_166170
        
        # Call to dict(...): (line 101)
        # Processing the call keyword arguments (line 101)
        str_166173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'str', 'Sample Implicit Model')
        keyword_166174 = str_166173
        str_166175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 26), 'str', 'ODRPACK UG, pg. 49')
        keyword_166176 = str_166175
        kwargs_166177 = {'ref': keyword_166176, 'name': keyword_166174}
        # Getting the type of 'dict' (line 101)
        dict_166172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'dict', False)
        # Calling dict(args, kwargs) (line 101)
        dict_call_result_166178 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), dict_166172, *[], **kwargs_166177)
        
        keyword_166179 = dict_call_result_166178
        kwargs_166180 = {'meta': keyword_166179, 'implicit': keyword_166171}
        # Getting the type of 'Model' (line 98)
        Model_166167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'Model', False)
        # Calling Model(args, kwargs) (line 98)
        Model_call_result_166181 = invoke(stypy.reporting.localization.Localization(__file__, 98, 23), Model_166167, *[implicit_fcn_166169], **kwargs_166180)
        
        # Assigning a type to the variable 'implicit_mod' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'implicit_mod', Model_call_result_166181)
        
        # Assigning a Call to a Name (line 104):
        
        # Call to Data(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_166183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_166184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        float_166185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166185)
        # Adding element type (line 105)
        float_166186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166186)
        # Adding element type (line 105)
        float_166187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166187)
        # Adding element type (line 105)
        float_166188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166188)
        # Adding element type (line 105)
        float_166189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166189)
        # Adding element type (line 105)
        float_166190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166190)
        # Adding element type (line 105)
        float_166191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166191)
        # Adding element type (line 105)
        float_166192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166192)
        # Adding element type (line 105)
        float_166193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166193)
        # Adding element type (line 105)
        float_166194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166194)
        # Adding element type (line 105)
        float_166195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166195)
        # Adding element type (line 105)
        float_166196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166196)
        # Adding element type (line 105)
        float_166197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166197)
        # Adding element type (line 105)
        float_166198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166198)
        # Adding element type (line 105)
        float_166199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166199)
        # Adding element type (line 105)
        float_166200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166200)
        # Adding element type (line 105)
        float_166201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166201)
        # Adding element type (line 105)
        float_166202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166202)
        # Adding element type (line 105)
        float_166203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166203)
        # Adding element type (line 105)
        float_166204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), list_166184, float_166204)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 28), list_166183, list_166184)
        # Adding element type (line 104)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_166205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        float_166206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166206)
        # Adding element type (line 107)
        float_166207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166207)
        # Adding element type (line 107)
        float_166208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166208)
        # Adding element type (line 107)
        float_166209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166209)
        # Adding element type (line 107)
        float_166210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166210)
        # Adding element type (line 107)
        float_166211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166211)
        # Adding element type (line 107)
        float_166212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166212)
        # Adding element type (line 107)
        float_166213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166213)
        # Adding element type (line 107)
        float_166214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166214)
        # Adding element type (line 107)
        float_166215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166215)
        # Adding element type (line 107)
        float_166216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166216)
        # Adding element type (line 107)
        float_166217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166217)
        # Adding element type (line 107)
        float_166218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166218)
        # Adding element type (line 107)
        float_166219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166219)
        # Adding element type (line 107)
        float_166220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166220)
        # Adding element type (line 107)
        float_166221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166221)
        # Adding element type (line 107)
        float_166222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166222)
        # Adding element type (line 107)
        float_166223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166223)
        # Adding element type (line 107)
        float_166224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166224)
        # Adding element type (line 107)
        float_166225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 12), list_166205, float_166225)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 28), list_166183, list_166205)
        
        int_166226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'int')
        # Processing the call keyword arguments (line 104)
        kwargs_166227 = {}
        # Getting the type of 'Data' (line 104)
        Data_166182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'Data', False)
        # Calling Data(args, kwargs) (line 104)
        Data_call_result_166228 = invoke(stypy.reporting.localization.Localization(__file__, 104, 23), Data_166182, *[list_166183, int_166226], **kwargs_166227)
        
        # Assigning a type to the variable 'implicit_dat' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'implicit_dat', Data_call_result_166228)
        
        # Assigning a Call to a Name (line 111):
        
        # Call to ODR(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'implicit_dat' (line 111)
        implicit_dat_166230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'implicit_dat', False)
        # Getting the type of 'implicit_mod' (line 111)
        implicit_mod_166231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'implicit_mod', False)
        # Processing the call keyword arguments (line 111)
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_166232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        float_166233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 18), list_166232, float_166233)
        # Adding element type (line 112)
        float_166234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 18), list_166232, float_166234)
        # Adding element type (line 112)
        float_166235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 18), list_166232, float_166235)
        # Adding element type (line 112)
        float_166236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 18), list_166232, float_166236)
        # Adding element type (line 112)
        float_166237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 18), list_166232, float_166237)
        
        keyword_166238 = list_166232
        kwargs_166239 = {'beta0': keyword_166238}
        # Getting the type of 'ODR' (line 111)
        ODR_166229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'ODR', False)
        # Calling ODR(args, kwargs) (line 111)
        ODR_call_result_166240 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), ODR_166229, *[implicit_dat_166230, implicit_mod_166231], **kwargs_166239)
        
        # Assigning a type to the variable 'implicit_odr' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'implicit_odr', ODR_call_result_166240)
        
        # Assigning a Call to a Name (line 114):
        
        # Call to run(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_166243 = {}
        # Getting the type of 'implicit_odr' (line 114)
        implicit_odr_166241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'implicit_odr', False)
        # Obtaining the member 'run' of a type (line 114)
        run_166242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 14), implicit_odr_166241, 'run')
        # Calling run(args, kwargs) (line 114)
        run_call_result_166244 = invoke(stypy.reporting.localization.Localization(__file__, 114, 14), run_166242, *[], **kwargs_166243)
        
        # Assigning a type to the variable 'out' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'out', run_call_result_166244)
        
        # Call to assert_array_almost_equal(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'out' (line 116)
        out_166246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'out', False)
        # Obtaining the member 'beta' of a type (line 116)
        beta_166247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), out_166246, 'beta')
        
        # Call to array(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_166250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        float_166251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_166250, float_166251)
        # Adding element type (line 117)
        float_166252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_166250, float_166252)
        # Adding element type (line 117)
        float_166253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_166250, float_166253)
        # Adding element type (line 117)
        float_166254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_166250, float_166254)
        # Adding element type (line 117)
        float_166255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_166250, float_166255)
        
        # Processing the call keyword arguments (line 117)
        kwargs_166256 = {}
        # Getting the type of 'np' (line 117)
        np_166248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 117)
        array_166249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), np_166248, 'array')
        # Calling array(args, kwargs) (line 117)
        array_call_result_166257 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), array_166249, *[list_166250], **kwargs_166256)
        
        # Processing the call keyword arguments (line 115)
        kwargs_166258 = {}
        # Getting the type of 'assert_array_almost_equal' (line 115)
        assert_array_almost_equal_166245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 115)
        assert_array_almost_equal_call_result_166259 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), assert_array_almost_equal_166245, *[beta_166247, array_call_result_166257], **kwargs_166258)
        
        
        # Call to assert_array_almost_equal(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'out' (line 121)
        out_166261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'out', False)
        # Obtaining the member 'sd_beta' of a type (line 121)
        sd_beta_166262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), out_166261, 'sd_beta')
        
        # Call to array(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_166265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        float_166266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_166265, float_166266)
        # Adding element type (line 122)
        float_166267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_166265, float_166267)
        # Adding element type (line 122)
        float_166268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_166265, float_166268)
        # Adding element type (line 122)
        float_166269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_166265, float_166269)
        # Adding element type (line 122)
        float_166270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_166265, float_166270)
        
        # Processing the call keyword arguments (line 122)
        kwargs_166271 = {}
        # Getting the type of 'np' (line 122)
        np_166263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 122)
        array_166264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), np_166263, 'array')
        # Calling array(args, kwargs) (line 122)
        array_call_result_166272 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), array_166264, *[list_166265], **kwargs_166271)
        
        # Processing the call keyword arguments (line 120)
        kwargs_166273 = {}
        # Getting the type of 'assert_array_almost_equal' (line 120)
        assert_array_almost_equal_166260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 120)
        assert_array_almost_equal_call_result_166274 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), assert_array_almost_equal_166260, *[sd_beta_166262, array_call_result_166272], **kwargs_166273)
        
        
        # Call to assert_array_almost_equal(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'out' (line 126)
        out_166276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'out', False)
        # Obtaining the member 'cov_beta' of a type (line 126)
        cov_beta_166277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), out_166276, 'cov_beta')
        
        # Call to array(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_166280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_166281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        float_166282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), list_166281, float_166282)
        # Adding element type (line 127)
        float_166283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), list_166281, float_166283)
        # Adding element type (line 127)
        float_166284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), list_166281, float_166284)
        # Adding element type (line 127)
        float_166285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), list_166281, float_166285)
        # Adding element type (line 127)
        float_166286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 22), list_166281, float_166286)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 21), list_166280, list_166281)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 130)
        list_166287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 130)
        # Adding element type (line 130)
        float_166288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), list_166287, float_166288)
        # Adding element type (line 130)
        float_166289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), list_166287, float_166289)
        # Adding element type (line 130)
        float_166290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), list_166287, float_166290)
        # Adding element type (line 130)
        float_166291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), list_166287, float_166291)
        # Adding element type (line 130)
        float_166292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 15), list_166287, float_166292)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 21), list_166280, list_166287)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_166293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        float_166294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 15), list_166293, float_166294)
        # Adding element type (line 133)
        float_166295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 15), list_166293, float_166295)
        # Adding element type (line 133)
        float_166296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 15), list_166293, float_166296)
        # Adding element type (line 133)
        float_166297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 15), list_166293, float_166297)
        # Adding element type (line 133)
        float_166298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 15), list_166293, float_166298)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 21), list_166280, list_166293)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_166299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        float_166300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 15), list_166299, float_166300)
        # Adding element type (line 136)
        float_166301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 15), list_166299, float_166301)
        # Adding element type (line 136)
        float_166302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 15), list_166299, float_166302)
        # Adding element type (line 136)
        float_166303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 15), list_166299, float_166303)
        # Adding element type (line 136)
        float_166304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 15), list_166299, float_166304)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 21), list_166280, list_166299)
        # Adding element type (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_166305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        float_166306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 15), list_166305, float_166306)
        # Adding element type (line 139)
        float_166307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 15), list_166305, float_166307)
        # Adding element type (line 139)
        float_166308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 15), list_166305, float_166308)
        # Adding element type (line 139)
        float_166309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 15), list_166305, float_166309)
        # Adding element type (line 139)
        float_166310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 15), list_166305, float_166310)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 21), list_166280, list_166305)
        
        # Processing the call keyword arguments (line 127)
        kwargs_166311 = {}
        # Getting the type of 'np' (line 127)
        np_166278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 127)
        array_166279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), np_166278, 'array')
        # Calling array(args, kwargs) (line 127)
        array_call_result_166312 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), array_166279, *[list_166280], **kwargs_166311)
        
        # Processing the call keyword arguments (line 125)
        kwargs_166313 = {}
        # Getting the type of 'assert_array_almost_equal' (line 125)
        assert_array_almost_equal_166275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 125)
        assert_array_almost_equal_call_result_166314 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assert_array_almost_equal_166275, *[cov_beta_166277, array_call_result_166312], **kwargs_166313)
        
        
        # ################# End of 'test_implicit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_implicit' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_166315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_166315)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_implicit'
        return stypy_return_type_166315


    @norecursion
    def multi_fcn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'multi_fcn'
        module_type_store = module_type_store.open_function_context('multi_fcn', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.multi_fcn.__dict__.__setitem__('stypy_localization', localization)
        TestODR.multi_fcn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.multi_fcn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.multi_fcn.__dict__.__setitem__('stypy_function_name', 'TestODR.multi_fcn')
        TestODR.multi_fcn.__dict__.__setitem__('stypy_param_names_list', ['B', 'x'])
        TestODR.multi_fcn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.multi_fcn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.multi_fcn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.multi_fcn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.multi_fcn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.multi_fcn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.multi_fcn', ['B', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'multi_fcn', localization, ['B', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'multi_fcn(...)' code ##################

        
        
        # Call to any(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_166320 = {}
        
        # Getting the type of 'x' (line 147)
        x_166316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'x', False)
        float_166317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 16), 'float')
        # Applying the binary operator '<' (line 147)
        result_lt_166318 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 12), '<', x_166316, float_166317)
        
        # Obtaining the member 'any' of a type (line 147)
        any_166319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), result_lt_166318, 'any')
        # Calling any(args, kwargs) (line 147)
        any_call_result_166321 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), any_166319, *[], **kwargs_166320)
        
        # Testing the type of an if condition (line 147)
        if_condition_166322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), any_call_result_166321)
        # Assigning a type to the variable 'if_condition_166322' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'if_condition_166322', if_condition_166322)
        # SSA begins for if statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'OdrStop' (line 148)
        OdrStop_166323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 18), 'OdrStop')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 148, 12), OdrStop_166323, 'raise parameter', BaseException)
        # SSA join for if statement (line 147)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 149):
        # Getting the type of 'pi' (line 149)
        pi_166324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'pi')
        
        # Obtaining the type of the subscript
        int_166325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 21), 'int')
        # Getting the type of 'B' (line 149)
        B_166326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'B')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___166327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), B_166326, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_166328 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), getitem___166327, int_166325)
        
        # Applying the binary operator '*' (line 149)
        result_mul_166329 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 16), '*', pi_166324, subscript_call_result_166328)
        
        float_166330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 24), 'float')
        # Applying the binary operator 'div' (line 149)
        result_div_166331 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 23), 'div', result_mul_166329, float_166330)
        
        # Assigning a type to the variable 'theta' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'theta', result_div_166331)
        
        # Assigning a Call to a Name (line 150):
        
        # Call to cos(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'theta' (line 150)
        theta_166334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'theta', False)
        # Processing the call keyword arguments (line 150)
        kwargs_166335 = {}
        # Getting the type of 'np' (line 150)
        np_166332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'np', False)
        # Obtaining the member 'cos' of a type (line 150)
        cos_166333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 17), np_166332, 'cos')
        # Calling cos(args, kwargs) (line 150)
        cos_call_result_166336 = invoke(stypy.reporting.localization.Localization(__file__, 150, 17), cos_166333, *[theta_166334], **kwargs_166335)
        
        # Assigning a type to the variable 'ctheta' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'ctheta', cos_call_result_166336)
        
        # Assigning a Call to a Name (line 151):
        
        # Call to sin(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'theta' (line 151)
        theta_166339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'theta', False)
        # Processing the call keyword arguments (line 151)
        kwargs_166340 = {}
        # Getting the type of 'np' (line 151)
        np_166337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'np', False)
        # Obtaining the member 'sin' of a type (line 151)
        sin_166338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 17), np_166337, 'sin')
        # Calling sin(args, kwargs) (line 151)
        sin_call_result_166341 = invoke(stypy.reporting.localization.Localization(__file__, 151, 17), sin_166338, *[theta_166339], **kwargs_166340)
        
        # Assigning a type to the variable 'stheta' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stheta', sin_call_result_166341)
        
        # Assigning a Call to a Name (line 152):
        
        # Call to power(...): (line 152)
        # Processing the call arguments (line 152)
        float_166344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'float')
        # Getting the type of 'pi' (line 152)
        pi_166345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'pi', False)
        # Applying the binary operator '*' (line 152)
        result_mul_166346 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 25), '*', float_166344, pi_166345)
        
        # Getting the type of 'x' (line 152)
        x_166347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'x', False)
        # Applying the binary operator '*' (line 152)
        result_mul_166348 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 30), '*', result_mul_166346, x_166347)
        
        
        # Call to exp(...): (line 152)
        # Processing the call arguments (line 152)
        
        
        # Obtaining the type of the subscript
        int_166351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 43), 'int')
        # Getting the type of 'B' (line 152)
        B_166352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 41), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___166353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 41), B_166352, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_166354 = invoke(stypy.reporting.localization.Localization(__file__, 152, 41), getitem___166353, int_166351)
        
        # Applying the 'usub' unary operator (line 152)
        result___neg___166355 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 40), 'usub', subscript_call_result_166354)
        
        # Processing the call keyword arguments (line 152)
        kwargs_166356 = {}
        # Getting the type of 'np' (line 152)
        np_166349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 33), 'np', False)
        # Obtaining the member 'exp' of a type (line 152)
        exp_166350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 33), np_166349, 'exp')
        # Calling exp(args, kwargs) (line 152)
        exp_call_result_166357 = invoke(stypy.reporting.localization.Localization(__file__, 152, 33), exp_166350, *[result___neg___166355], **kwargs_166356)
        
        # Applying the binary operator '*' (line 152)
        result_mul_166358 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 32), '*', result_mul_166348, exp_call_result_166357)
        
        
        # Obtaining the type of the subscript
        int_166359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 50), 'int')
        # Getting the type of 'B' (line 152)
        B_166360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 48), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___166361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 48), B_166360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_166362 = invoke(stypy.reporting.localization.Localization(__file__, 152, 48), getitem___166361, int_166359)
        
        # Processing the call keyword arguments (line 152)
        kwargs_166363 = {}
        # Getting the type of 'np' (line 152)
        np_166342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'np', False)
        # Obtaining the member 'power' of a type (line 152)
        power_166343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), np_166342, 'power')
        # Calling power(args, kwargs) (line 152)
        power_call_result_166364 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), power_166343, *[result_mul_166358, subscript_call_result_166362], **kwargs_166363)
        
        # Assigning a type to the variable 'omega' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'omega', power_call_result_166364)
        
        # Assigning a Call to a Name (line 153):
        
        # Call to arctan2(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'omega' (line 153)
        omega_166367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 26), 'omega', False)
        # Getting the type of 'stheta' (line 153)
        stheta_166368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 32), 'stheta', False)
        # Applying the binary operator '*' (line 153)
        result_mul_166369 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 26), '*', omega_166367, stheta_166368)
        
        float_166370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 42), 'float')
        # Getting the type of 'omega' (line 153)
        omega_166371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 48), 'omega', False)
        # Getting the type of 'ctheta' (line 153)
        ctheta_166372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 54), 'ctheta', False)
        # Applying the binary operator '*' (line 153)
        result_mul_166373 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 48), '*', omega_166371, ctheta_166372)
        
        # Applying the binary operator '+' (line 153)
        result_add_166374 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 42), '+', float_166370, result_mul_166373)
        
        # Processing the call keyword arguments (line 153)
        kwargs_166375 = {}
        # Getting the type of 'np' (line 153)
        np_166365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 14), 'np', False)
        # Obtaining the member 'arctan2' of a type (line 153)
        arctan2_166366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 14), np_166365, 'arctan2')
        # Calling arctan2(args, kwargs) (line 153)
        arctan2_call_result_166376 = invoke(stypy.reporting.localization.Localization(__file__, 153, 14), arctan2_166366, *[result_mul_166369, result_add_166374], **kwargs_166375)
        
        # Assigning a type to the variable 'phi' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'phi', arctan2_call_result_166376)
        
        # Assigning a BinOp to a Name (line 154):
        
        # Obtaining the type of the subscript
        int_166377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 15), 'int')
        # Getting the type of 'B' (line 154)
        B_166378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'B')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___166379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 13), B_166378, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_166380 = invoke(stypy.reporting.localization.Localization(__file__, 154, 13), getitem___166379, int_166377)
        
        
        # Obtaining the type of the subscript
        int_166381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 22), 'int')
        # Getting the type of 'B' (line 154)
        B_166382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'B')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___166383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 20), B_166382, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_166384 = invoke(stypy.reporting.localization.Localization(__file__, 154, 20), getitem___166383, int_166381)
        
        # Applying the binary operator '-' (line 154)
        result_sub_166385 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 13), '-', subscript_call_result_166380, subscript_call_result_166384)
        
        
        # Call to power(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Call to sqrt(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Call to power(...): (line 154)
        # Processing the call arguments (line 154)
        float_166392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 54), 'float')
        # Getting the type of 'omega' (line 154)
        omega_166393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 60), 'omega', False)
        # Getting the type of 'ctheta' (line 154)
        ctheta_166394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 66), 'ctheta', False)
        # Applying the binary operator '*' (line 154)
        result_mul_166395 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 60), '*', omega_166393, ctheta_166394)
        
        # Applying the binary operator '+' (line 154)
        result_add_166396 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 54), '+', float_166392, result_mul_166395)
        
        int_166397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 74), 'int')
        # Processing the call keyword arguments (line 154)
        kwargs_166398 = {}
        # Getting the type of 'np' (line 154)
        np_166390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 45), 'np', False)
        # Obtaining the member 'power' of a type (line 154)
        power_166391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 45), np_166390, 'power')
        # Calling power(args, kwargs) (line 154)
        power_call_result_166399 = invoke(stypy.reporting.localization.Localization(__file__, 154, 45), power_166391, *[result_add_166396, int_166397], **kwargs_166398)
        
        
        # Call to power(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'omega' (line 155)
        omega_166402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'omega', False)
        # Getting the type of 'stheta' (line 155)
        stheta_166403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'stheta', False)
        # Applying the binary operator '*' (line 155)
        result_mul_166404 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 22), '*', omega_166402, stheta_166403)
        
        int_166405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 36), 'int')
        # Processing the call keyword arguments (line 155)
        kwargs_166406 = {}
        # Getting the type of 'np' (line 155)
        np_166400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 13), 'np', False)
        # Obtaining the member 'power' of a type (line 155)
        power_166401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 13), np_166400, 'power')
        # Calling power(args, kwargs) (line 155)
        power_call_result_166407 = invoke(stypy.reporting.localization.Localization(__file__, 155, 13), power_166401, *[result_mul_166404, int_166405], **kwargs_166406)
        
        # Applying the binary operator '+' (line 154)
        result_add_166408 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 45), '+', power_call_result_166399, power_call_result_166407)
        
        # Processing the call keyword arguments (line 154)
        kwargs_166409 = {}
        # Getting the type of 'np' (line 154)
        np_166388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 154)
        sqrt_166389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 37), np_166388, 'sqrt')
        # Calling sqrt(args, kwargs) (line 154)
        sqrt_call_result_166410 = invoke(stypy.reporting.localization.Localization(__file__, 154, 37), sqrt_166389, *[result_add_166408], **kwargs_166409)
        
        
        
        # Obtaining the type of the subscript
        int_166411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 44), 'int')
        # Getting the type of 'B' (line 155)
        B_166412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 42), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___166413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 42), B_166412, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_166414 = invoke(stypy.reporting.localization.Localization(__file__, 155, 42), getitem___166413, int_166411)
        
        # Applying the 'usub' unary operator (line 155)
        result___neg___166415 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 41), 'usub', subscript_call_result_166414)
        
        # Processing the call keyword arguments (line 154)
        kwargs_166416 = {}
        # Getting the type of 'np' (line 154)
        np_166386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'np', False)
        # Obtaining the member 'power' of a type (line 154)
        power_166387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 28), np_166386, 'power')
        # Calling power(args, kwargs) (line 154)
        power_call_result_166417 = invoke(stypy.reporting.localization.Localization(__file__, 154, 28), power_166387, *[sqrt_call_result_166410, result___neg___166415], **kwargs_166416)
        
        # Applying the binary operator '*' (line 154)
        result_mul_166418 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 12), '*', result_sub_166385, power_call_result_166417)
        
        # Assigning a type to the variable 'r' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'r', result_mul_166418)
        
        # Assigning a Call to a Name (line 156):
        
        # Call to vstack(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_166421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        
        # Obtaining the type of the subscript
        int_166422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 27), 'int')
        # Getting the type of 'B' (line 156)
        B_166423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___166424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 25), B_166423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_166425 = invoke(stypy.reporting.localization.Localization(__file__, 156, 25), getitem___166424, int_166422)
        
        # Getting the type of 'r' (line 156)
        r_166426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'r', False)
        
        # Call to cos(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining the type of the subscript
        int_166429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 43), 'int')
        # Getting the type of 'B' (line 156)
        B_166430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 41), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___166431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 41), B_166430, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_166432 = invoke(stypy.reporting.localization.Localization(__file__, 156, 41), getitem___166431, int_166429)
        
        # Getting the type of 'phi' (line 156)
        phi_166433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 46), 'phi', False)
        # Applying the binary operator '*' (line 156)
        result_mul_166434 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 41), '*', subscript_call_result_166432, phi_166433)
        
        # Processing the call keyword arguments (line 156)
        kwargs_166435 = {}
        # Getting the type of 'np' (line 156)
        np_166427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'np', False)
        # Obtaining the member 'cos' of a type (line 156)
        cos_166428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 34), np_166427, 'cos')
        # Calling cos(args, kwargs) (line 156)
        cos_call_result_166436 = invoke(stypy.reporting.localization.Localization(__file__, 156, 34), cos_166428, *[result_mul_166434], **kwargs_166435)
        
        # Applying the binary operator '*' (line 156)
        result_mul_166437 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 32), '*', r_166426, cos_call_result_166436)
        
        # Applying the binary operator '+' (line 156)
        result_add_166438 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 25), '+', subscript_call_result_166425, result_mul_166437)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 24), list_166421, result_add_166438)
        # Adding element type (line 156)
        # Getting the type of 'r' (line 157)
        r_166439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 'r', False)
        
        # Call to sin(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Obtaining the type of the subscript
        int_166442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 36), 'int')
        # Getting the type of 'B' (line 157)
        B_166443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 34), 'B', False)
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___166444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 34), B_166443, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_166445 = invoke(stypy.reporting.localization.Localization(__file__, 157, 34), getitem___166444, int_166442)
        
        # Getting the type of 'phi' (line 157)
        phi_166446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 39), 'phi', False)
        # Applying the binary operator '*' (line 157)
        result_mul_166447 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 34), '*', subscript_call_result_166445, phi_166446)
        
        # Processing the call keyword arguments (line 157)
        kwargs_166448 = {}
        # Getting the type of 'np' (line 157)
        np_166440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'np', False)
        # Obtaining the member 'sin' of a type (line 157)
        sin_166441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 27), np_166440, 'sin')
        # Calling sin(args, kwargs) (line 157)
        sin_call_result_166449 = invoke(stypy.reporting.localization.Localization(__file__, 157, 27), sin_166441, *[result_mul_166447], **kwargs_166448)
        
        # Applying the binary operator '*' (line 157)
        result_mul_166450 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 25), '*', r_166439, sin_call_result_166449)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 24), list_166421, result_mul_166450)
        
        # Processing the call keyword arguments (line 156)
        kwargs_166451 = {}
        # Getting the type of 'np' (line 156)
        np_166419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'np', False)
        # Obtaining the member 'vstack' of a type (line 156)
        vstack_166420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 14), np_166419, 'vstack')
        # Calling vstack(args, kwargs) (line 156)
        vstack_call_result_166452 = invoke(stypy.reporting.localization.Localization(__file__, 156, 14), vstack_166420, *[list_166421], **kwargs_166451)
        
        # Assigning a type to the variable 'ret' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'ret', vstack_call_result_166452)
        # Getting the type of 'ret' (line 158)
        ret_166453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', ret_166453)
        
        # ################# End of 'multi_fcn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'multi_fcn' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_166454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_166454)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'multi_fcn'
        return stypy_return_type_166454


    @norecursion
    def test_multi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_multi'
        module_type_store = module_type_store.open_function_context('test_multi', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.test_multi.__dict__.__setitem__('stypy_localization', localization)
        TestODR.test_multi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.test_multi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.test_multi.__dict__.__setitem__('stypy_function_name', 'TestODR.test_multi')
        TestODR.test_multi.__dict__.__setitem__('stypy_param_names_list', [])
        TestODR.test_multi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.test_multi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.test_multi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.test_multi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.test_multi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.test_multi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.test_multi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_multi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_multi(...)' code ##################

        
        # Assigning a Call to a Name (line 161):
        
        # Call to Model(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'self' (line 162)
        self_166456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'self', False)
        # Obtaining the member 'multi_fcn' of a type (line 162)
        multi_fcn_166457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), self_166456, 'multi_fcn')
        # Processing the call keyword arguments (line 161)
        
        # Call to dict(...): (line 163)
        # Processing the call keyword arguments (line 163)
        str_166459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 27), 'str', 'Sample Multi-Response Model')
        keyword_166460 = str_166459
        str_166461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 26), 'str', 'ODRPACK UG, pg. 56')
        keyword_166462 = str_166461
        kwargs_166463 = {'ref': keyword_166462, 'name': keyword_166460}
        # Getting the type of 'dict' (line 163)
        dict_166458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 17), 'dict', False)
        # Calling dict(args, kwargs) (line 163)
        dict_call_result_166464 = invoke(stypy.reporting.localization.Localization(__file__, 163, 17), dict_166458, *[], **kwargs_166463)
        
        keyword_166465 = dict_call_result_166464
        kwargs_166466 = {'meta': keyword_166465}
        # Getting the type of 'Model' (line 161)
        Model_166455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'Model', False)
        # Calling Model(args, kwargs) (line 161)
        Model_call_result_166467 = invoke(stypy.reporting.localization.Localization(__file__, 161, 20), Model_166455, *[multi_fcn_166457], **kwargs_166466)
        
        # Assigning a type to the variable 'multi_mod' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'multi_mod', Model_call_result_166467)
        
        # Assigning a Call to a Name (line 167):
        
        # Call to array(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Obtaining an instance of the builtin type 'list' (line 167)
        list_166470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 167)
        # Adding element type (line 167)
        float_166471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166471)
        # Adding element type (line 167)
        float_166472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166472)
        # Adding element type (line 167)
        float_166473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166473)
        # Adding element type (line 167)
        float_166474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166474)
        # Adding element type (line 167)
        float_166475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166475)
        # Adding element type (line 167)
        float_166476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166476)
        # Adding element type (line 167)
        float_166477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166477)
        # Adding element type (line 167)
        float_166478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 74), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166478)
        # Adding element type (line 167)
        float_166479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166479)
        # Adding element type (line 167)
        float_166480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166480)
        # Adding element type (line 167)
        float_166481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166481)
        # Adding element type (line 167)
        float_166482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166482)
        # Adding element type (line 167)
        float_166483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166483)
        # Adding element type (line 167)
        float_166484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166484)
        # Adding element type (line 167)
        float_166485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166485)
        # Adding element type (line 167)
        float_166486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166486)
        # Adding element type (line 167)
        float_166487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166487)
        # Adding element type (line 167)
        float_166488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166488)
        # Adding element type (line 167)
        float_166489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166489)
        # Adding element type (line 167)
        float_166490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166490)
        # Adding element type (line 167)
        float_166491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166491)
        # Adding element type (line 167)
        float_166492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166492)
        # Adding element type (line 167)
        float_166493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 27), list_166470, float_166493)
        
        # Processing the call keyword arguments (line 167)
        kwargs_166494 = {}
        # Getting the type of 'np' (line 167)
        np_166468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 167)
        array_166469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 18), np_166468, 'array')
        # Calling array(args, kwargs) (line 167)
        array_call_result_166495 = invoke(stypy.reporting.localization.Localization(__file__, 167, 18), array_166469, *[list_166470], **kwargs_166494)
        
        # Assigning a type to the variable 'multi_x' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'multi_x', array_call_result_166495)
        
        # Assigning a Call to a Name (line 170):
        
        # Call to array(...): (line 170)
        # Processing the call arguments (line 170)
        
        # Obtaining an instance of the builtin type 'list' (line 170)
        list_166498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 170)
        # Adding element type (line 170)
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_166499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        float_166500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166500)
        # Adding element type (line 171)
        float_166501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166501)
        # Adding element type (line 171)
        float_166502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166502)
        # Adding element type (line 171)
        float_166503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166503)
        # Adding element type (line 171)
        float_166504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166504)
        # Adding element type (line 171)
        float_166505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166505)
        # Adding element type (line 171)
        float_166506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166506)
        # Adding element type (line 171)
        float_166507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166507)
        # Adding element type (line 171)
        float_166508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166508)
        # Adding element type (line 171)
        float_166509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166509)
        # Adding element type (line 171)
        float_166510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166510)
        # Adding element type (line 171)
        float_166511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166511)
        # Adding element type (line 171)
        float_166512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166512)
        # Adding element type (line 171)
        float_166513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166513)
        # Adding element type (line 171)
        float_166514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166514)
        # Adding element type (line 171)
        float_166515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166515)
        # Adding element type (line 171)
        float_166516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166516)
        # Adding element type (line 171)
        float_166517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166517)
        # Adding element type (line 171)
        float_166518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166518)
        # Adding element type (line 171)
        float_166519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166519)
        # Adding element type (line 171)
        float_166520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166520)
        # Adding element type (line 171)
        float_166521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166521)
        # Adding element type (line 171)
        float_166522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 12), list_166499, float_166522)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 27), list_166498, list_166499)
        # Adding element type (line 170)
        
        # Obtaining an instance of the builtin type 'list' (line 174)
        list_166523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 174)
        # Adding element type (line 174)
        float_166524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166524)
        # Adding element type (line 174)
        float_166525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166525)
        # Adding element type (line 174)
        float_166526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166526)
        # Adding element type (line 174)
        float_166527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166527)
        # Adding element type (line 174)
        float_166528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166528)
        # Adding element type (line 174)
        float_166529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166529)
        # Adding element type (line 174)
        float_166530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166530)
        # Adding element type (line 174)
        float_166531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166531)
        # Adding element type (line 174)
        float_166532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166532)
        # Adding element type (line 174)
        float_166533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166533)
        # Adding element type (line 174)
        float_166534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166534)
        # Adding element type (line 174)
        float_166535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166535)
        # Adding element type (line 174)
        float_166536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166536)
        # Adding element type (line 174)
        float_166537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166537)
        # Adding element type (line 174)
        float_166538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166538)
        # Adding element type (line 174)
        float_166539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166539)
        # Adding element type (line 174)
        float_166540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166540)
        # Adding element type (line 174)
        float_166541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166541)
        # Adding element type (line 174)
        float_166542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166542)
        # Adding element type (line 174)
        float_166543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166543)
        # Adding element type (line 174)
        float_166544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166544)
        # Adding element type (line 174)
        float_166545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166545)
        # Adding element type (line 174)
        float_166546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 12), list_166523, float_166546)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 27), list_166498, list_166523)
        
        # Processing the call keyword arguments (line 170)
        kwargs_166547 = {}
        # Getting the type of 'np' (line 170)
        np_166496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 170)
        array_166497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 18), np_166496, 'array')
        # Calling array(args, kwargs) (line 170)
        array_call_result_166548 = invoke(stypy.reporting.localization.Localization(__file__, 170, 18), array_166497, *[list_166498], **kwargs_166547)
        
        # Assigning a type to the variable 'multi_y' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'multi_y', array_call_result_166548)
        
        # Assigning a Call to a Name (line 178):
        
        # Call to len(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'multi_x' (line 178)
        multi_x_166550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'multi_x', False)
        # Processing the call keyword arguments (line 178)
        kwargs_166551 = {}
        # Getting the type of 'len' (line 178)
        len_166549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'len', False)
        # Calling len(args, kwargs) (line 178)
        len_call_result_166552 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), len_166549, *[multi_x_166550], **kwargs_166551)
        
        # Assigning a type to the variable 'n' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'n', len_call_result_166552)
        
        # Assigning a Call to a Name (line 179):
        
        # Call to zeros(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Obtaining an instance of the builtin type 'tuple' (line 179)
        tuple_166555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 179)
        # Adding element type (line 179)
        int_166556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 29), tuple_166555, int_166556)
        # Adding element type (line 179)
        int_166557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 29), tuple_166555, int_166557)
        # Adding element type (line 179)
        # Getting the type of 'n' (line 179)
        n_166558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 35), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 29), tuple_166555, n_166558)
        
        # Processing the call keyword arguments (line 179)
        # Getting the type of 'float' (line 179)
        float_166559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 45), 'float', False)
        keyword_166560 = float_166559
        kwargs_166561 = {'dtype': keyword_166560}
        # Getting the type of 'np' (line 179)
        np_166553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'np', False)
        # Obtaining the member 'zeros' of a type (line 179)
        zeros_166554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 19), np_166553, 'zeros')
        # Calling zeros(args, kwargs) (line 179)
        zeros_call_result_166562 = invoke(stypy.reporting.localization.Localization(__file__, 179, 19), zeros_166554, *[tuple_166555], **kwargs_166561)
        
        # Assigning a type to the variable 'multi_we' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'multi_we', zeros_call_result_166562)
        
        # Assigning a Call to a Name (line 180):
        
        # Call to ones(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'n' (line 180)
        n_166565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'n', False)
        # Processing the call keyword arguments (line 180)
        # Getting the type of 'int' (line 180)
        int_166566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'int', False)
        keyword_166567 = int_166566
        kwargs_166568 = {'dtype': keyword_166567}
        # Getting the type of 'np' (line 180)
        np_166563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'np', False)
        # Obtaining the member 'ones' of a type (line 180)
        ones_166564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 22), np_166563, 'ones')
        # Calling ones(args, kwargs) (line 180)
        ones_call_result_166569 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), ones_166564, *[n_166565], **kwargs_166568)
        
        # Assigning a type to the variable 'multi_ifixx' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'multi_ifixx', ones_call_result_166569)
        
        # Assigning a Call to a Name (line 181):
        
        # Call to zeros(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'n' (line 181)
        n_166572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'n', False)
        # Processing the call keyword arguments (line 181)
        # Getting the type of 'float' (line 181)
        float_166573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 40), 'float', False)
        keyword_166574 = float_166573
        kwargs_166575 = {'dtype': keyword_166574}
        # Getting the type of 'np' (line 181)
        np_166570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'np', False)
        # Obtaining the member 'zeros' of a type (line 181)
        zeros_166571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 22), np_166570, 'zeros')
        # Calling zeros(args, kwargs) (line 181)
        zeros_call_result_166576 = invoke(stypy.reporting.localization.Localization(__file__, 181, 22), zeros_166571, *[n_166572], **kwargs_166575)
        
        # Assigning a type to the variable 'multi_delta' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'multi_delta', zeros_call_result_166576)
        
        # Assigning a Num to a Subscript (line 183):
        float_166577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 26), 'float')
        # Getting the type of 'multi_we' (line 183)
        multi_we_166578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'multi_we')
        int_166579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 17), 'int')
        int_166580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 19), 'int')
        slice_166581 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 183, 8), None, None, None)
        # Storing an element on a container (line 183)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 8), multi_we_166578, ((int_166579, int_166580, slice_166581), float_166577))
        
        # Multiple assignment of 2 elements.
        float_166582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 44), 'float')
        # Getting the type of 'multi_we' (line 184)
        multi_we_166583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'multi_we')
        int_166584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 35), 'int')
        int_166585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 37), 'int')
        slice_166586 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 184, 26), None, None, None)
        # Storing an element on a container (line 184)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 26), multi_we_166583, ((int_166584, int_166585, slice_166586), float_166582))
        
        # Obtaining the type of the subscript
        int_166587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 35), 'int')
        int_166588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 37), 'int')
        slice_166589 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 184, 26), None, None, None)
        # Getting the type of 'multi_we' (line 184)
        multi_we_166590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'multi_we')
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___166591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 26), multi_we_166590, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 184)
        subscript_call_result_166592 = invoke(stypy.reporting.localization.Localization(__file__, 184, 26), getitem___166591, (int_166587, int_166588, slice_166589))
        
        # Getting the type of 'multi_we' (line 184)
        multi_we_166593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'multi_we')
        int_166594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 17), 'int')
        int_166595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 19), 'int')
        slice_166596 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 184, 8), None, None, None)
        # Storing an element on a container (line 184)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 8), multi_we_166593, ((int_166594, int_166595, slice_166596), subscript_call_result_166592))
        
        # Assigning a Num to a Subscript (line 185):
        float_166597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 26), 'float')
        # Getting the type of 'multi_we' (line 185)
        multi_we_166598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'multi_we')
        int_166599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 17), 'int')
        int_166600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'int')
        slice_166601 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 185, 8), None, None, None)
        # Storing an element on a container (line 185)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 8), multi_we_166598, ((int_166599, int_166600, slice_166601), float_166597))
        
        
        # Call to range(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'n' (line 187)
        n_166603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'n', False)
        # Processing the call keyword arguments (line 187)
        kwargs_166604 = {}
        # Getting the type of 'range' (line 187)
        range_166602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 17), 'range', False)
        # Calling range(args, kwargs) (line 187)
        range_call_result_166605 = invoke(stypy.reporting.localization.Localization(__file__, 187, 17), range_166602, *[n_166603], **kwargs_166604)
        
        # Testing the type of a for loop iterable (line 187)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 187, 8), range_call_result_166605)
        # Getting the type of the for loop variable (line 187)
        for_loop_var_166606 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 187, 8), range_call_result_166605)
        # Assigning a type to the variable 'i' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'i', for_loop_var_166606)
        # SSA begins for a for statement (line 187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 188)
        i_166607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'i')
        # Getting the type of 'multi_x' (line 188)
        multi_x_166608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'multi_x')
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___166609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 15), multi_x_166608, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_166610 = invoke(stypy.reporting.localization.Localization(__file__, 188, 15), getitem___166609, i_166607)
        
        float_166611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 28), 'float')
        # Applying the binary operator '<' (line 188)
        result_lt_166612 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 15), '<', subscript_call_result_166610, float_166611)
        
        # Testing the type of an if condition (line 188)
        if_condition_166613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 12), result_lt_166612)
        # Assigning a type to the variable 'if_condition_166613' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'if_condition_166613', if_condition_166613)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 189):
        int_166614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 33), 'int')
        # Getting the type of 'multi_ifixx' (line 189)
        multi_ifixx_166615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'multi_ifixx')
        # Getting the type of 'i' (line 189)
        i_166616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 28), 'i')
        # Storing an element on a container (line 189)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 16), multi_ifixx_166615, (i_166616, int_166614))
        # SSA branch for the else part of an if statement (line 188)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 190)
        i_166617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'i')
        # Getting the type of 'multi_x' (line 190)
        multi_x_166618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'multi_x')
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___166619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 17), multi_x_166618, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_166620 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), getitem___166619, i_166617)
        
        float_166621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 31), 'float')
        # Applying the binary operator '<=' (line 190)
        result_le_166622 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 17), '<=', subscript_call_result_166620, float_166621)
        
        # Testing the type of an if condition (line 190)
        if_condition_166623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 17), result_le_166622)
        # Assigning a type to the variable 'if_condition_166623' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'if_condition_166623', if_condition_166623)
        # SSA begins for if statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA branch for the else part of an if statement (line 190)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 192)
        i_166624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 'i')
        # Getting the type of 'multi_x' (line 192)
        multi_x_166625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'multi_x')
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___166626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 17), multi_x_166625, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_166627 = invoke(stypy.reporting.localization.Localization(__file__, 192, 17), getitem___166626, i_166624)
        
        float_166628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 31), 'float')
        # Applying the binary operator '<=' (line 192)
        result_le_166629 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 17), '<=', subscript_call_result_166627, float_166628)
        
        # Testing the type of an if condition (line 192)
        if_condition_166630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 17), result_le_166629)
        # Assigning a type to the variable 'if_condition_166630' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'if_condition_166630', if_condition_166630)
        # SSA begins for if statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 193):
        float_166631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 33), 'float')
        # Getting the type of 'multi_delta' (line 193)
        multi_delta_166632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'multi_delta')
        # Getting the type of 'i' (line 193)
        i_166633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'i')
        # Storing an element on a container (line 193)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 16), multi_delta_166632, (i_166633, float_166631))
        # SSA branch for the else part of an if statement (line 192)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 194)
        i_166634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 25), 'i')
        # Getting the type of 'multi_x' (line 194)
        multi_x_166635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'multi_x')
        # Obtaining the member '__getitem__' of a type (line 194)
        getitem___166636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), multi_x_166635, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 194)
        subscript_call_result_166637 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), getitem___166636, i_166634)
        
        float_166638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'float')
        # Applying the binary operator '<=' (line 194)
        result_le_166639 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 17), '<=', subscript_call_result_166637, float_166638)
        
        # Testing the type of an if condition (line 194)
        if_condition_166640 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 17), result_le_166639)
        # Assigning a type to the variable 'if_condition_166640' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'if_condition_166640', if_condition_166640)
        # SSA begins for if statement (line 194)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 195):
        float_166641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 33), 'float')
        # Getting the type of 'multi_delta' (line 195)
        multi_delta_166642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'multi_delta')
        # Getting the type of 'i' (line 195)
        i_166643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'i')
        # Storing an element on a container (line 195)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), multi_delta_166642, (i_166643, float_166641))
        # SSA branch for the else part of an if statement (line 194)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 196)
        i_166644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 25), 'i')
        # Getting the type of 'multi_x' (line 196)
        multi_x_166645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'multi_x')
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___166646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 17), multi_x_166645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_166647 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), getitem___166646, i_166644)
        
        float_166648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'float')
        # Applying the binary operator '<=' (line 196)
        result_le_166649 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 17), '<=', subscript_call_result_166647, float_166648)
        
        # Testing the type of an if condition (line 196)
        if_condition_166650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 17), result_le_166649)
        # Assigning a type to the variable 'if_condition_166650' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'if_condition_166650', if_condition_166650)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 197):
        float_166651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 33), 'float')
        # Getting the type of 'multi_delta' (line 197)
        multi_delta_166652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'multi_delta')
        # Getting the type of 'i' (line 197)
        i_166653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 28), 'i')
        # Storing an element on a container (line 197)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 16), multi_delta_166652, (i_166653, float_166651))
        # SSA branch for the else part of an if statement (line 196)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Subscript (line 199):
        float_166654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 33), 'float')
        # Getting the type of 'multi_delta' (line 199)
        multi_delta_166655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'multi_delta')
        # Getting the type of 'i' (line 199)
        i_166656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'i')
        # Storing an element on a container (line 199)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 16), multi_delta_166655, (i_166656, float_166654))
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 194)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 192)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 200)
        i_166657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'i')
        # Getting the type of 'multi_x' (line 200)
        multi_x_166658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'multi_x')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___166659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), multi_x_166658, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_166660 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), getitem___166659, i_166657)
        
        float_166661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'float')
        # Applying the binary operator '==' (line 200)
        result_eq_166662 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), '==', subscript_call_result_166660, float_166661)
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 200)
        i_166663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 46), 'i')
        # Getting the type of 'multi_x' (line 200)
        multi_x_166664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 38), 'multi_x')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___166665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 38), multi_x_166664, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_166666 = invoke(stypy.reporting.localization.Localization(__file__, 200, 38), getitem___166665, i_166663)
        
        float_166667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 52), 'float')
        # Applying the binary operator '==' (line 200)
        result_eq_166668 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 38), '==', subscript_call_result_166666, float_166667)
        
        # Applying the binary operator 'or' (line 200)
        result_or_keyword_166669 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), 'or', result_eq_166662, result_eq_166668)
        
        # Testing the type of an if condition (line 200)
        if_condition_166670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), result_or_keyword_166669)
        # Assigning a type to the variable 'if_condition_166670' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_166670', if_condition_166670)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 201):
        float_166671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 34), 'float')
        # Getting the type of 'multi_we' (line 201)
        multi_we_166672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'multi_we')
        slice_166673 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 201, 16), None, None, None)
        slice_166674 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 201, 16), None, None, None)
        # Getting the type of 'i' (line 201)
        i_166675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 29), 'i')
        # Storing an element on a container (line 201)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 16), multi_we_166672, ((slice_166673, slice_166674, i_166675), float_166671))
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 203):
        
        # Call to Data(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'multi_x' (line 203)
        multi_x_166677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'multi_x', False)
        # Getting the type of 'multi_y' (line 203)
        multi_y_166678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 34), 'multi_y', False)
        # Processing the call keyword arguments (line 203)
        float_166679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 46), 'float')
        
        # Call to power(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'multi_x' (line 203)
        multi_x_166682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 60), 'multi_x', False)
        int_166683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 69), 'int')
        # Processing the call keyword arguments (line 203)
        kwargs_166684 = {}
        # Getting the type of 'np' (line 203)
        np_166680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 51), 'np', False)
        # Obtaining the member 'power' of a type (line 203)
        power_166681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 51), np_166680, 'power')
        # Calling power(args, kwargs) (line 203)
        power_call_result_166685 = invoke(stypy.reporting.localization.Localization(__file__, 203, 51), power_166681, *[multi_x_166682, int_166683], **kwargs_166684)
        
        # Applying the binary operator 'div' (line 203)
        result_div_166686 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 46), 'div', float_166679, power_call_result_166685)
        
        keyword_166687 = result_div_166686
        # Getting the type of 'multi_we' (line 204)
        multi_we_166688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 15), 'multi_we', False)
        keyword_166689 = multi_we_166688
        kwargs_166690 = {'we': keyword_166689, 'wd': keyword_166687}
        # Getting the type of 'Data' (line 203)
        Data_166676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'Data', False)
        # Calling Data(args, kwargs) (line 203)
        Data_call_result_166691 = invoke(stypy.reporting.localization.Localization(__file__, 203, 20), Data_166676, *[multi_x_166677, multi_y_166678], **kwargs_166690)
        
        # Assigning a type to the variable 'multi_dat' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'multi_dat', Data_call_result_166691)
        
        # Assigning a Call to a Name (line 205):
        
        # Call to ODR(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'multi_dat' (line 205)
        multi_dat_166693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'multi_dat', False)
        # Getting the type of 'multi_mod' (line 205)
        multi_mod_166694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 35), 'multi_mod', False)
        # Processing the call keyword arguments (line 205)
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_166695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        float_166696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 52), list_166695, float_166696)
        # Adding element type (line 205)
        float_166697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 52), list_166695, float_166697)
        # Adding element type (line 205)
        float_166698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 52), list_166695, float_166698)
        # Adding element type (line 205)
        float_166699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 52), list_166695, float_166699)
        # Adding element type (line 205)
        float_166700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 52), list_166695, float_166700)
        
        keyword_166701 = list_166695
        # Getting the type of 'multi_delta' (line 206)
        multi_delta_166702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'multi_delta', False)
        keyword_166703 = multi_delta_166702
        # Getting the type of 'multi_ifixx' (line 206)
        multi_ifixx_166704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 38), 'multi_ifixx', False)
        keyword_166705 = multi_ifixx_166704
        kwargs_166706 = {'delta0': keyword_166703, 'ifixx': keyword_166705, 'beta0': keyword_166701}
        # Getting the type of 'ODR' (line 205)
        ODR_166692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'ODR', False)
        # Calling ODR(args, kwargs) (line 205)
        ODR_call_result_166707 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), ODR_166692, *[multi_dat_166693, multi_mod_166694], **kwargs_166706)
        
        # Assigning a type to the variable 'multi_odr' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'multi_odr', ODR_call_result_166707)
        
        # Call to set_job(...): (line 207)
        # Processing the call keyword arguments (line 207)
        int_166710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 32), 'int')
        keyword_166711 = int_166710
        int_166712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 44), 'int')
        keyword_166713 = int_166712
        kwargs_166714 = {'del_init': keyword_166713, 'deriv': keyword_166711}
        # Getting the type of 'multi_odr' (line 207)
        multi_odr_166708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'multi_odr', False)
        # Obtaining the member 'set_job' of a type (line 207)
        set_job_166709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), multi_odr_166708, 'set_job')
        # Calling set_job(args, kwargs) (line 207)
        set_job_call_result_166715 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), set_job_166709, *[], **kwargs_166714)
        
        
        # Assigning a Call to a Name (line 209):
        
        # Call to run(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_166718 = {}
        # Getting the type of 'multi_odr' (line 209)
        multi_odr_166716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'multi_odr', False)
        # Obtaining the member 'run' of a type (line 209)
        run_166717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 14), multi_odr_166716, 'run')
        # Calling run(args, kwargs) (line 209)
        run_call_result_166719 = invoke(stypy.reporting.localization.Localization(__file__, 209, 14), run_166717, *[], **kwargs_166718)
        
        # Assigning a type to the variable 'out' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'out', run_call_result_166719)
        
        # Call to assert_array_almost_equal(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'out' (line 211)
        out_166721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'out', False)
        # Obtaining the member 'beta' of a type (line 211)
        beta_166722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), out_166721, 'beta')
        
        # Call to array(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'list' (line 212)
        list_166725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 212)
        # Adding element type (line 212)
        float_166726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 21), list_166725, float_166726)
        # Adding element type (line 212)
        float_166727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 21), list_166725, float_166727)
        # Adding element type (line 212)
        float_166728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 21), list_166725, float_166728)
        # Adding element type (line 212)
        float_166729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 21), list_166725, float_166729)
        # Adding element type (line 212)
        float_166730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 21), list_166725, float_166730)
        
        # Processing the call keyword arguments (line 212)
        kwargs_166731 = {}
        # Getting the type of 'np' (line 212)
        np_166723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 212)
        array_166724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), np_166723, 'array')
        # Calling array(args, kwargs) (line 212)
        array_call_result_166732 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), array_166724, *[list_166725], **kwargs_166731)
        
        # Processing the call keyword arguments (line 210)
        kwargs_166733 = {}
        # Getting the type of 'assert_array_almost_equal' (line 210)
        assert_array_almost_equal_166720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 210)
        assert_array_almost_equal_call_result_166734 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), assert_array_almost_equal_166720, *[beta_166722, array_call_result_166732], **kwargs_166733)
        
        
        # Call to assert_array_almost_equal(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'out' (line 216)
        out_166736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'out', False)
        # Obtaining the member 'sd_beta' of a type (line 216)
        sd_beta_166737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), out_166736, 'sd_beta')
        
        # Call to array(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_166740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        float_166741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 21), list_166740, float_166741)
        # Adding element type (line 217)
        float_166742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 21), list_166740, float_166742)
        # Adding element type (line 217)
        float_166743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 21), list_166740, float_166743)
        # Adding element type (line 217)
        float_166744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 21), list_166740, float_166744)
        # Adding element type (line 217)
        float_166745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 21), list_166740, float_166745)
        
        # Processing the call keyword arguments (line 217)
        kwargs_166746 = {}
        # Getting the type of 'np' (line 217)
        np_166738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 217)
        array_166739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), np_166738, 'array')
        # Calling array(args, kwargs) (line 217)
        array_call_result_166747 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), array_166739, *[list_166740], **kwargs_166746)
        
        # Processing the call keyword arguments (line 215)
        kwargs_166748 = {}
        # Getting the type of 'assert_array_almost_equal' (line 215)
        assert_array_almost_equal_166735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 215)
        assert_array_almost_equal_call_result_166749 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), assert_array_almost_equal_166735, *[sd_beta_166737, array_call_result_166747], **kwargs_166748)
        
        
        # Call to assert_array_almost_equal(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'out' (line 221)
        out_166751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'out', False)
        # Obtaining the member 'cov_beta' of a type (line 221)
        cov_beta_166752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), out_166751, 'cov_beta')
        
        # Call to array(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_166755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_166756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        float_166757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), list_166756, float_166757)
        # Adding element type (line 222)
        float_166758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), list_166756, float_166758)
        # Adding element type (line 222)
        float_166759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), list_166756, float_166759)
        # Adding element type (line 222)
        float_166760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), list_166756, float_166760)
        # Adding element type (line 222)
        float_166761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), list_166756, float_166761)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 21), list_166755, list_166756)
        # Adding element type (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_166762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        float_166763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 15), list_166762, float_166763)
        # Adding element type (line 224)
        float_166764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 15), list_166762, float_166764)
        # Adding element type (line 224)
        float_166765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 15), list_166762, float_166765)
        # Adding element type (line 224)
        float_166766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 15), list_166762, float_166766)
        # Adding element type (line 224)
        float_166767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 15), list_166762, float_166767)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 21), list_166755, list_166762)
        # Adding element type (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_166768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        float_166769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 15), list_166768, float_166769)
        # Adding element type (line 226)
        float_166770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 15), list_166768, float_166770)
        # Adding element type (line 226)
        float_166771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 15), list_166768, float_166771)
        # Adding element type (line 226)
        float_166772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 15), list_166768, float_166772)
        # Adding element type (line 226)
        float_166773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 15), list_166768, float_166773)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 21), list_166755, list_166768)
        # Adding element type (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 228)
        list_166774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 228)
        # Adding element type (line 228)
        float_166775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 15), list_166774, float_166775)
        # Adding element type (line 228)
        float_166776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 15), list_166774, float_166776)
        # Adding element type (line 228)
        float_166777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 15), list_166774, float_166777)
        # Adding element type (line 228)
        float_166778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 15), list_166774, float_166778)
        # Adding element type (line 228)
        float_166779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 15), list_166774, float_166779)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 21), list_166755, list_166774)
        # Adding element type (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_166780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        float_166781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 15), list_166780, float_166781)
        # Adding element type (line 230)
        float_166782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 15), list_166780, float_166782)
        # Adding element type (line 230)
        float_166783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 15), list_166780, float_166783)
        # Adding element type (line 230)
        float_166784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 15), list_166780, float_166784)
        # Adding element type (line 230)
        float_166785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 15), list_166780, float_166785)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 21), list_166755, list_166780)
        
        # Processing the call keyword arguments (line 222)
        kwargs_166786 = {}
        # Getting the type of 'np' (line 222)
        np_166753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 222)
        array_166754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), np_166753, 'array')
        # Calling array(args, kwargs) (line 222)
        array_call_result_166787 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), array_166754, *[list_166755], **kwargs_166786)
        
        # Processing the call keyword arguments (line 220)
        kwargs_166788 = {}
        # Getting the type of 'assert_array_almost_equal' (line 220)
        assert_array_almost_equal_166750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 220)
        assert_array_almost_equal_call_result_166789 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), assert_array_almost_equal_166750, *[cov_beta_166752, array_call_result_166787], **kwargs_166788)
        
        
        # ################# End of 'test_multi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_multi' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_166790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_166790)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_multi'
        return stypy_return_type_166790


    @norecursion
    def pearson_fcn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'pearson_fcn'
        module_type_store = module_type_store.open_function_context('pearson_fcn', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_localization', localization)
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_function_name', 'TestODR.pearson_fcn')
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_param_names_list', ['B', 'x'])
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.pearson_fcn.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.pearson_fcn', ['B', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'pearson_fcn', localization, ['B', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'pearson_fcn(...)' code ##################

        
        # Obtaining the type of the subscript
        int_166791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 17), 'int')
        # Getting the type of 'B' (line 238)
        B_166792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'B')
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___166793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 15), B_166792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 238)
        subscript_call_result_166794 = invoke(stypy.reporting.localization.Localization(__file__, 238, 15), getitem___166793, int_166791)
        
        
        # Obtaining the type of the subscript
        int_166795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 24), 'int')
        # Getting the type of 'B' (line 238)
        B_166796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'B')
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___166797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 22), B_166796, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 238)
        subscript_call_result_166798 = invoke(stypy.reporting.localization.Localization(__file__, 238, 22), getitem___166797, int_166795)
        
        # Getting the type of 'x' (line 238)
        x_166799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'x')
        # Applying the binary operator '*' (line 238)
        result_mul_166800 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 22), '*', subscript_call_result_166798, x_166799)
        
        # Applying the binary operator '+' (line 238)
        result_add_166801 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 15), '+', subscript_call_result_166794, result_mul_166800)
        
        # Assigning a type to the variable 'stypy_return_type' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'stypy_return_type', result_add_166801)
        
        # ################# End of 'pearson_fcn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'pearson_fcn' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_166802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_166802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'pearson_fcn'
        return stypy_return_type_166802


    @norecursion
    def test_pearson(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pearson'
        module_type_store = module_type_store.open_function_context('test_pearson', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.test_pearson.__dict__.__setitem__('stypy_localization', localization)
        TestODR.test_pearson.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.test_pearson.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.test_pearson.__dict__.__setitem__('stypy_function_name', 'TestODR.test_pearson')
        TestODR.test_pearson.__dict__.__setitem__('stypy_param_names_list', [])
        TestODR.test_pearson.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.test_pearson.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.test_pearson.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.test_pearson.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.test_pearson.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.test_pearson.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.test_pearson', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pearson', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pearson(...)' code ##################

        
        # Assigning a Call to a Name (line 241):
        
        # Call to array(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Obtaining an instance of the builtin type 'list' (line 241)
        list_166805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 241)
        # Adding element type (line 241)
        float_166806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166806)
        # Adding element type (line 241)
        float_166807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166807)
        # Adding element type (line 241)
        float_166808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166808)
        # Adding element type (line 241)
        float_166809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166809)
        # Adding element type (line 241)
        float_166810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166810)
        # Adding element type (line 241)
        float_166811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166811)
        # Adding element type (line 241)
        float_166812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166812)
        # Adding element type (line 241)
        float_166813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166813)
        # Adding element type (line 241)
        float_166814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166814)
        # Adding element type (line 241)
        float_166815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 23), list_166805, float_166815)
        
        # Processing the call keyword arguments (line 241)
        kwargs_166816 = {}
        # Getting the type of 'np' (line 241)
        np_166803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 14), 'np', False)
        # Obtaining the member 'array' of a type (line 241)
        array_166804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 14), np_166803, 'array')
        # Calling array(args, kwargs) (line 241)
        array_call_result_166817 = invoke(stypy.reporting.localization.Localization(__file__, 241, 14), array_166804, *[list_166805], **kwargs_166816)
        
        # Assigning a type to the variable 'p_x' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'p_x', array_call_result_166817)
        
        # Assigning a Call to a Name (line 242):
        
        # Call to array(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Obtaining an instance of the builtin type 'list' (line 242)
        list_166820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 242)
        # Adding element type (line 242)
        float_166821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166821)
        # Adding element type (line 242)
        float_166822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166822)
        # Adding element type (line 242)
        float_166823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166823)
        # Adding element type (line 242)
        float_166824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166824)
        # Adding element type (line 242)
        float_166825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166825)
        # Adding element type (line 242)
        float_166826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166826)
        # Adding element type (line 242)
        float_166827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166827)
        # Adding element type (line 242)
        float_166828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166828)
        # Adding element type (line 242)
        float_166829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166829)
        # Adding element type (line 242)
        float_166830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 23), list_166820, float_166830)
        
        # Processing the call keyword arguments (line 242)
        kwargs_166831 = {}
        # Getting the type of 'np' (line 242)
        np_166818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'np', False)
        # Obtaining the member 'array' of a type (line 242)
        array_166819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 14), np_166818, 'array')
        # Calling array(args, kwargs) (line 242)
        array_call_result_166832 = invoke(stypy.reporting.localization.Localization(__file__, 242, 14), array_166819, *[list_166820], **kwargs_166831)
        
        # Assigning a type to the variable 'p_y' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'p_y', array_call_result_166832)
        
        # Assigning a Call to a Name (line 243):
        
        # Call to array(...): (line 243)
        # Processing the call arguments (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_166835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        float_166836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166836)
        # Adding element type (line 243)
        float_166837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166837)
        # Adding element type (line 243)
        float_166838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166838)
        # Adding element type (line 243)
        float_166839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166839)
        # Adding element type (line 243)
        float_166840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166840)
        # Adding element type (line 243)
        float_166841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166841)
        # Adding element type (line 243)
        float_166842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166842)
        # Adding element type (line 243)
        float_166843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166843)
        # Adding element type (line 243)
        float_166844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166844)
        # Adding element type (line 243)
        float_166845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 24), list_166835, float_166845)
        
        # Processing the call keyword arguments (line 243)
        kwargs_166846 = {}
        # Getting the type of 'np' (line 243)
        np_166833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 243)
        array_166834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 15), np_166833, 'array')
        # Calling array(args, kwargs) (line 243)
        array_call_result_166847 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), array_166834, *[list_166835], **kwargs_166846)
        
        # Assigning a type to the variable 'p_sx' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'p_sx', array_call_result_166847)
        
        # Assigning a Call to a Name (line 244):
        
        # Call to array(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_166850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        float_166851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166851)
        # Adding element type (line 244)
        float_166852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166852)
        # Adding element type (line 244)
        float_166853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166853)
        # Adding element type (line 244)
        float_166854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166854)
        # Adding element type (line 244)
        float_166855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166855)
        # Adding element type (line 244)
        float_166856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166856)
        # Adding element type (line 244)
        float_166857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166857)
        # Adding element type (line 244)
        float_166858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166858)
        # Adding element type (line 244)
        float_166859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166859)
        # Adding element type (line 244)
        float_166860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 24), list_166850, float_166860)
        
        # Processing the call keyword arguments (line 244)
        kwargs_166861 = {}
        # Getting the type of 'np' (line 244)
        np_166848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 244)
        array_166849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 15), np_166848, 'array')
        # Calling array(args, kwargs) (line 244)
        array_call_result_166862 = invoke(stypy.reporting.localization.Localization(__file__, 244, 15), array_166849, *[list_166850], **kwargs_166861)
        
        # Assigning a type to the variable 'p_sy' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'p_sy', array_call_result_166862)
        
        # Assigning a Call to a Name (line 246):
        
        # Call to RealData(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'p_x' (line 246)
        p_x_166864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 25), 'p_x', False)
        # Getting the type of 'p_y' (line 246)
        p_y_166865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 30), 'p_y', False)
        # Processing the call keyword arguments (line 246)
        # Getting the type of 'p_sx' (line 246)
        p_sx_166866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'p_sx', False)
        keyword_166867 = p_sx_166866
        # Getting the type of 'p_sy' (line 246)
        p_sy_166868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 47), 'p_sy', False)
        keyword_166869 = p_sy_166868
        kwargs_166870 = {'sy': keyword_166869, 'sx': keyword_166867}
        # Getting the type of 'RealData' (line 246)
        RealData_166863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'RealData', False)
        # Calling RealData(args, kwargs) (line 246)
        RealData_call_result_166871 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), RealData_166863, *[p_x_166864, p_y_166865], **kwargs_166870)
        
        # Assigning a type to the variable 'p_dat' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'p_dat', RealData_call_result_166871)
        
        # Assigning a Call to a Name (line 249):
        
        # Call to RealData(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'p_y' (line 249)
        p_y_166873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'p_y', False)
        # Getting the type of 'p_x' (line 249)
        p_x_166874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 31), 'p_x', False)
        # Processing the call keyword arguments (line 249)
        # Getting the type of 'p_sy' (line 249)
        p_sy_166875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 39), 'p_sy', False)
        keyword_166876 = p_sy_166875
        # Getting the type of 'p_sx' (line 249)
        p_sx_166877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 48), 'p_sx', False)
        keyword_166878 = p_sx_166877
        kwargs_166879 = {'sy': keyword_166878, 'sx': keyword_166876}
        # Getting the type of 'RealData' (line 249)
        RealData_166872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 17), 'RealData', False)
        # Calling RealData(args, kwargs) (line 249)
        RealData_call_result_166880 = invoke(stypy.reporting.localization.Localization(__file__, 249, 17), RealData_166872, *[p_y_166873, p_x_166874], **kwargs_166879)
        
        # Assigning a type to the variable 'pr_dat' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'pr_dat', RealData_call_result_166880)
        
        # Assigning a Call to a Name (line 251):
        
        # Call to Model(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'self' (line 251)
        self_166882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 22), 'self', False)
        # Obtaining the member 'pearson_fcn' of a type (line 251)
        pearson_fcn_166883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 22), self_166882, 'pearson_fcn')
        # Processing the call keyword arguments (line 251)
        
        # Call to dict(...): (line 251)
        # Processing the call keyword arguments (line 251)
        str_166885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 55), 'str', 'Uni-linear Fit')
        keyword_166886 = str_166885
        kwargs_166887 = {'name': keyword_166886}
        # Getting the type of 'dict' (line 251)
        dict_166884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 45), 'dict', False)
        # Calling dict(args, kwargs) (line 251)
        dict_call_result_166888 = invoke(stypy.reporting.localization.Localization(__file__, 251, 45), dict_166884, *[], **kwargs_166887)
        
        keyword_166889 = dict_call_result_166888
        kwargs_166890 = {'meta': keyword_166889}
        # Getting the type of 'Model' (line 251)
        Model_166881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'Model', False)
        # Calling Model(args, kwargs) (line 251)
        Model_call_result_166891 = invoke(stypy.reporting.localization.Localization(__file__, 251, 16), Model_166881, *[pearson_fcn_166883], **kwargs_166890)
        
        # Assigning a type to the variable 'p_mod' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'p_mod', Model_call_result_166891)
        
        # Assigning a Call to a Name (line 253):
        
        # Call to ODR(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'p_dat' (line 253)
        p_dat_166893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'p_dat', False)
        # Getting the type of 'p_mod' (line 253)
        p_mod_166894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 27), 'p_mod', False)
        # Processing the call keyword arguments (line 253)
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_166895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        float_166896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 40), list_166895, float_166896)
        # Adding element type (line 253)
        float_166897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 40), list_166895, float_166897)
        
        keyword_166898 = list_166895
        kwargs_166899 = {'beta0': keyword_166898}
        # Getting the type of 'ODR' (line 253)
        ODR_166892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'ODR', False)
        # Calling ODR(args, kwargs) (line 253)
        ODR_call_result_166900 = invoke(stypy.reporting.localization.Localization(__file__, 253, 16), ODR_166892, *[p_dat_166893, p_mod_166894], **kwargs_166899)
        
        # Assigning a type to the variable 'p_odr' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'p_odr', ODR_call_result_166900)
        
        # Assigning a Call to a Name (line 254):
        
        # Call to ODR(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'pr_dat' (line 254)
        pr_dat_166902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'pr_dat', False)
        # Getting the type of 'p_mod' (line 254)
        p_mod_166903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 29), 'p_mod', False)
        # Processing the call keyword arguments (line 254)
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_166904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        float_166905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 42), list_166904, float_166905)
        # Adding element type (line 254)
        float_166906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 42), list_166904, float_166906)
        
        keyword_166907 = list_166904
        kwargs_166908 = {'beta0': keyword_166907}
        # Getting the type of 'ODR' (line 254)
        ODR_166901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'ODR', False)
        # Calling ODR(args, kwargs) (line 254)
        ODR_call_result_166909 = invoke(stypy.reporting.localization.Localization(__file__, 254, 17), ODR_166901, *[pr_dat_166902, p_mod_166903], **kwargs_166908)
        
        # Assigning a type to the variable 'pr_odr' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'pr_odr', ODR_call_result_166909)
        
        # Assigning a Call to a Name (line 256):
        
        # Call to run(...): (line 256)
        # Processing the call keyword arguments (line 256)
        kwargs_166912 = {}
        # Getting the type of 'p_odr' (line 256)
        p_odr_166910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'p_odr', False)
        # Obtaining the member 'run' of a type (line 256)
        run_166911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 14), p_odr_166910, 'run')
        # Calling run(args, kwargs) (line 256)
        run_call_result_166913 = invoke(stypy.reporting.localization.Localization(__file__, 256, 14), run_166911, *[], **kwargs_166912)
        
        # Assigning a type to the variable 'out' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'out', run_call_result_166913)
        
        # Call to assert_array_almost_equal(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'out' (line 258)
        out_166915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'out', False)
        # Obtaining the member 'beta' of a type (line 258)
        beta_166916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), out_166915, 'beta')
        
        # Call to array(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_166919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        float_166920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 21), list_166919, float_166920)
        # Adding element type (line 259)
        float_166921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 21), list_166919, float_166921)
        
        # Processing the call keyword arguments (line 259)
        kwargs_166922 = {}
        # Getting the type of 'np' (line 259)
        np_166917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 259)
        array_166918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), np_166917, 'array')
        # Calling array(args, kwargs) (line 259)
        array_call_result_166923 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), array_166918, *[list_166919], **kwargs_166922)
        
        # Processing the call keyword arguments (line 257)
        kwargs_166924 = {}
        # Getting the type of 'assert_array_almost_equal' (line 257)
        assert_array_almost_equal_166914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 257)
        assert_array_almost_equal_call_result_166925 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), assert_array_almost_equal_166914, *[beta_166916, array_call_result_166923], **kwargs_166924)
        
        
        # Call to assert_array_almost_equal(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'out' (line 262)
        out_166927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'out', False)
        # Obtaining the member 'sd_beta' of a type (line 262)
        sd_beta_166928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), out_166927, 'sd_beta')
        
        # Call to array(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Obtaining an instance of the builtin type 'list' (line 263)
        list_166931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 263)
        # Adding element type (line 263)
        float_166932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 21), list_166931, float_166932)
        # Adding element type (line 263)
        float_166933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 21), list_166931, float_166933)
        
        # Processing the call keyword arguments (line 263)
        kwargs_166934 = {}
        # Getting the type of 'np' (line 263)
        np_166929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 263)
        array_166930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), np_166929, 'array')
        # Calling array(args, kwargs) (line 263)
        array_call_result_166935 = invoke(stypy.reporting.localization.Localization(__file__, 263, 12), array_166930, *[list_166931], **kwargs_166934)
        
        # Processing the call keyword arguments (line 261)
        kwargs_166936 = {}
        # Getting the type of 'assert_array_almost_equal' (line 261)
        assert_array_almost_equal_166926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 261)
        assert_array_almost_equal_call_result_166937 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), assert_array_almost_equal_166926, *[sd_beta_166928, array_call_result_166935], **kwargs_166936)
        
        
        # Call to assert_array_almost_equal(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'out' (line 266)
        out_166939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'out', False)
        # Obtaining the member 'cov_beta' of a type (line 266)
        cov_beta_166940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), out_166939, 'cov_beta')
        
        # Call to array(...): (line 267)
        # Processing the call arguments (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_166943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_166944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        float_166945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 22), list_166944, float_166945)
        # Adding element type (line 267)
        float_166946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 22), list_166944, float_166946)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), list_166943, list_166944)
        # Adding element type (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_166947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        # Adding element type (line 268)
        float_166948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 15), list_166947, float_166948)
        # Adding element type (line 268)
        float_166949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 15), list_166947, float_166949)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 21), list_166943, list_166947)
        
        # Processing the call keyword arguments (line 267)
        kwargs_166950 = {}
        # Getting the type of 'np' (line 267)
        np_166941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 267)
        array_166942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 12), np_166941, 'array')
        # Calling array(args, kwargs) (line 267)
        array_call_result_166951 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), array_166942, *[list_166943], **kwargs_166950)
        
        # Processing the call keyword arguments (line 265)
        kwargs_166952 = {}
        # Getting the type of 'assert_array_almost_equal' (line 265)
        assert_array_almost_equal_166938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 265)
        assert_array_almost_equal_call_result_166953 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), assert_array_almost_equal_166938, *[cov_beta_166940, array_call_result_166951], **kwargs_166952)
        
        
        # Assigning a Call to a Name (line 271):
        
        # Call to run(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_166956 = {}
        # Getting the type of 'pr_odr' (line 271)
        pr_odr_166954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'pr_odr', False)
        # Obtaining the member 'run' of a type (line 271)
        run_166955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 15), pr_odr_166954, 'run')
        # Calling run(args, kwargs) (line 271)
        run_call_result_166957 = invoke(stypy.reporting.localization.Localization(__file__, 271, 15), run_166955, *[], **kwargs_166956)
        
        # Assigning a type to the variable 'rout' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'rout', run_call_result_166957)
        
        # Call to assert_array_almost_equal(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'rout' (line 273)
        rout_166959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'rout', False)
        # Obtaining the member 'beta' of a type (line 273)
        beta_166960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), rout_166959, 'beta')
        
        # Call to array(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_166963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        float_166964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 21), list_166963, float_166964)
        # Adding element type (line 274)
        float_166965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 21), list_166963, float_166965)
        
        # Processing the call keyword arguments (line 274)
        kwargs_166966 = {}
        # Getting the type of 'np' (line 274)
        np_166961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 274)
        array_166962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 12), np_166961, 'array')
        # Calling array(args, kwargs) (line 274)
        array_call_result_166967 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), array_166962, *[list_166963], **kwargs_166966)
        
        # Processing the call keyword arguments (line 272)
        kwargs_166968 = {}
        # Getting the type of 'assert_array_almost_equal' (line 272)
        assert_array_almost_equal_166958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 272)
        assert_array_almost_equal_call_result_166969 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), assert_array_almost_equal_166958, *[beta_166960, array_call_result_166967], **kwargs_166968)
        
        
        # Call to assert_array_almost_equal(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'rout' (line 277)
        rout_166971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'rout', False)
        # Obtaining the member 'sd_beta' of a type (line 277)
        sd_beta_166972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), rout_166971, 'sd_beta')
        
        # Call to array(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_166975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        float_166976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 21), list_166975, float_166976)
        # Adding element type (line 278)
        float_166977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 21), list_166975, float_166977)
        
        # Processing the call keyword arguments (line 278)
        kwargs_166978 = {}
        # Getting the type of 'np' (line 278)
        np_166973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 278)
        array_166974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), np_166973, 'array')
        # Calling array(args, kwargs) (line 278)
        array_call_result_166979 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), array_166974, *[list_166975], **kwargs_166978)
        
        # Processing the call keyword arguments (line 276)
        kwargs_166980 = {}
        # Getting the type of 'assert_array_almost_equal' (line 276)
        assert_array_almost_equal_166970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 276)
        assert_array_almost_equal_call_result_166981 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), assert_array_almost_equal_166970, *[sd_beta_166972, array_call_result_166979], **kwargs_166980)
        
        
        # Call to assert_array_almost_equal(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'rout' (line 281)
        rout_166983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'rout', False)
        # Obtaining the member 'cov_beta' of a type (line 281)
        cov_beta_166984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), rout_166983, 'cov_beta')
        
        # Call to array(...): (line 282)
        # Processing the call arguments (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_166987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_166988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        float_166989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 22), list_166988, float_166989)
        # Adding element type (line 282)
        float_166990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 22), list_166988, float_166990)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 21), list_166987, list_166988)
        # Adding element type (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_166991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        float_166992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 15), list_166991, float_166992)
        # Adding element type (line 283)
        float_166993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 15), list_166991, float_166993)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 21), list_166987, list_166991)
        
        # Processing the call keyword arguments (line 282)
        kwargs_166994 = {}
        # Getting the type of 'np' (line 282)
        np_166985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 282)
        array_166986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), np_166985, 'array')
        # Calling array(args, kwargs) (line 282)
        array_call_result_166995 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), array_166986, *[list_166987], **kwargs_166994)
        
        # Processing the call keyword arguments (line 280)
        kwargs_166996 = {}
        # Getting the type of 'assert_array_almost_equal' (line 280)
        assert_array_almost_equal_166982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 280)
        assert_array_almost_equal_call_result_166997 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), assert_array_almost_equal_166982, *[cov_beta_166984, array_call_result_166995], **kwargs_166996)
        
        
        # ################# End of 'test_pearson(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pearson' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_166998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_166998)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pearson'
        return stypy_return_type_166998


    @norecursion
    def lorentz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'lorentz'
        module_type_store = module_type_store.open_function_context('lorentz', 289, 4, False)
        # Assigning a type to the variable 'self' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.lorentz.__dict__.__setitem__('stypy_localization', localization)
        TestODR.lorentz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.lorentz.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.lorentz.__dict__.__setitem__('stypy_function_name', 'TestODR.lorentz')
        TestODR.lorentz.__dict__.__setitem__('stypy_param_names_list', ['beta', 'x'])
        TestODR.lorentz.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.lorentz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.lorentz.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.lorentz.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.lorentz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.lorentz.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.lorentz', ['beta', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'lorentz', localization, ['beta', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'lorentz(...)' code ##################

        
        # Obtaining the type of the subscript
        int_166999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 21), 'int')
        # Getting the type of 'beta' (line 290)
        beta_167000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'beta')
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___167001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 16), beta_167000, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_167002 = invoke(stypy.reporting.localization.Localization(__file__, 290, 16), getitem___167001, int_166999)
        
        
        # Obtaining the type of the subscript
        int_167003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 29), 'int')
        # Getting the type of 'beta' (line 290)
        beta_167004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'beta')
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___167005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 24), beta_167004, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_167006 = invoke(stypy.reporting.localization.Localization(__file__, 290, 24), getitem___167005, int_167003)
        
        # Applying the binary operator '*' (line 290)
        result_mul_167007 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 16), '*', subscript_call_result_167002, subscript_call_result_167006)
        
        
        # Obtaining the type of the subscript
        int_167008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 37), 'int')
        # Getting the type of 'beta' (line 290)
        beta_167009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'beta')
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___167010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 32), beta_167009, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_167011 = invoke(stypy.reporting.localization.Localization(__file__, 290, 32), getitem___167010, int_167008)
        
        # Applying the binary operator '*' (line 290)
        result_mul_167012 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 31), '*', result_mul_167007, subscript_call_result_167011)
        
        
        # Call to sqrt(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Call to power(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'x' (line 290)
        x_167017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 59), 'x', False)
        # Getting the type of 'x' (line 290)
        x_167018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 61), 'x', False)
        # Applying the binary operator '*' (line 290)
        result_mul_167019 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 59), '*', x_167017, x_167018)
        
        
        # Obtaining the type of the subscript
        int_167020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 17), 'int')
        # Getting the type of 'beta' (line 291)
        beta_167021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'beta', False)
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___167022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 12), beta_167021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_167023 = invoke(stypy.reporting.localization.Localization(__file__, 291, 12), getitem___167022, int_167020)
        
        
        # Obtaining the type of the subscript
        int_167024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 25), 'int')
        # Getting the type of 'beta' (line 291)
        beta_167025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'beta', False)
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___167026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 20), beta_167025, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_167027 = invoke(stypy.reporting.localization.Localization(__file__, 291, 20), getitem___167026, int_167024)
        
        # Applying the binary operator '*' (line 291)
        result_mul_167028 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 12), '*', subscript_call_result_167023, subscript_call_result_167027)
        
        # Applying the binary operator '-' (line 290)
        result_sub_167029 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 59), '-', result_mul_167019, result_mul_167028)
        
        float_167030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'float')
        # Processing the call keyword arguments (line 290)
        kwargs_167031 = {}
        # Getting the type of 'np' (line 290)
        np_167015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 50), 'np', False)
        # Obtaining the member 'power' of a type (line 290)
        power_167016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 50), np_167015, 'power')
        # Calling power(args, kwargs) (line 290)
        power_call_result_167032 = invoke(stypy.reporting.localization.Localization(__file__, 290, 50), power_167016, *[result_sub_167029, float_167030], **kwargs_167031)
        
        
        # Call to power(...): (line 291)
        # Processing the call arguments (line 291)
        
        # Obtaining the type of the subscript
        int_167035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 50), 'int')
        # Getting the type of 'beta' (line 291)
        beta_167036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 45), 'beta', False)
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___167037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 45), beta_167036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_167038 = invoke(stypy.reporting.localization.Localization(__file__, 291, 45), getitem___167037, int_167035)
        
        # Getting the type of 'x' (line 291)
        x_167039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 53), 'x', False)
        # Applying the binary operator '*' (line 291)
        result_mul_167040 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 45), '*', subscript_call_result_167038, x_167039)
        
        float_167041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 56), 'float')
        # Processing the call keyword arguments (line 291)
        kwargs_167042 = {}
        # Getting the type of 'np' (line 291)
        np_167033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 36), 'np', False)
        # Obtaining the member 'power' of a type (line 291)
        power_167034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 36), np_167033, 'power')
        # Calling power(args, kwargs) (line 291)
        power_call_result_167043 = invoke(stypy.reporting.localization.Localization(__file__, 291, 36), power_167034, *[result_mul_167040, float_167041], **kwargs_167042)
        
        # Applying the binary operator '+' (line 290)
        result_add_167044 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 50), '+', power_call_result_167032, power_call_result_167043)
        
        # Processing the call keyword arguments (line 290)
        kwargs_167045 = {}
        # Getting the type of 'np' (line 290)
        np_167013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 42), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 290)
        sqrt_167014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 42), np_167013, 'sqrt')
        # Calling sqrt(args, kwargs) (line 290)
        sqrt_call_result_167046 = invoke(stypy.reporting.localization.Localization(__file__, 290, 42), sqrt_167014, *[result_add_167044], **kwargs_167045)
        
        # Applying the binary operator 'div' (line 290)
        result_div_167047 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 40), 'div', result_mul_167012, sqrt_call_result_167046)
        
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', result_div_167047)
        
        # ################# End of 'lorentz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'lorentz' in the type store
        # Getting the type of 'stypy_return_type' (line 289)
        stypy_return_type_167048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_167048)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'lorentz'
        return stypy_return_type_167048


    @norecursion
    def test_lorentz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lorentz'
        module_type_store = module_type_store.open_function_context('test_lorentz', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.test_lorentz.__dict__.__setitem__('stypy_localization', localization)
        TestODR.test_lorentz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.test_lorentz.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.test_lorentz.__dict__.__setitem__('stypy_function_name', 'TestODR.test_lorentz')
        TestODR.test_lorentz.__dict__.__setitem__('stypy_param_names_list', [])
        TestODR.test_lorentz.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.test_lorentz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.test_lorentz.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.test_lorentz.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.test_lorentz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.test_lorentz.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.test_lorentz', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lorentz', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lorentz(...)' code ##################

        
        # Assigning a Call to a Name (line 294):
        
        # Call to array(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Obtaining an instance of the builtin type 'list' (line 294)
        list_167051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 294)
        # Adding element type (line 294)
        float_167052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 24), list_167051, float_167052)
        
        int_167053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 30), 'int')
        # Applying the binary operator '*' (line 294)
        result_mul_167054 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 24), '*', list_167051, int_167053)
        
        # Processing the call keyword arguments (line 294)
        kwargs_167055 = {}
        # Getting the type of 'np' (line 294)
        np_167049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 294)
        array_167050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), np_167049, 'array')
        # Calling array(args, kwargs) (line 294)
        array_call_result_167056 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), array_167050, *[result_mul_167054], **kwargs_167055)
        
        # Assigning a type to the variable 'l_sy' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'l_sy', array_call_result_167056)
        
        # Assigning a Call to a Name (line 295):
        
        # Call to array(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_167059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        float_167060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167060)
        # Adding element type (line 295)
        float_167061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167061)
        # Adding element type (line 295)
        float_167062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167062)
        # Adding element type (line 295)
        float_167063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167063)
        # Adding element type (line 295)
        float_167064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167064)
        # Adding element type (line 295)
        float_167065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167065)
        # Adding element type (line 295)
        float_167066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167066)
        # Adding element type (line 295)
        float_167067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167067)
        # Adding element type (line 295)
        float_167068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167068)
        # Adding element type (line 295)
        float_167069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167069)
        # Adding element type (line 295)
        float_167070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167070)
        # Adding element type (line 295)
        float_167071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167071)
        # Adding element type (line 295)
        float_167072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167072)
        # Adding element type (line 295)
        float_167073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167073)
        # Adding element type (line 295)
        float_167074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167074)
        # Adding element type (line 295)
        float_167075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167075)
        # Adding element type (line 295)
        float_167076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167076)
        # Adding element type (line 295)
        float_167077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_167059, float_167077)
        
        # Processing the call keyword arguments (line 295)
        kwargs_167078 = {}
        # Getting the type of 'np' (line 295)
        np_167057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 295)
        array_167058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 15), np_167057, 'array')
        # Calling array(args, kwargs) (line 295)
        array_call_result_167079 = invoke(stypy.reporting.localization.Localization(__file__, 295, 15), array_167058, *[list_167059], **kwargs_167078)
        
        # Assigning a type to the variable 'l_sx' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'l_sx', array_call_result_167079)
        
        # Assigning a Call to a Name (line 300):
        
        # Call to RealData(...): (line 300)
        # Processing the call arguments (line 300)
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_167081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        float_167082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167082)
        # Adding element type (line 301)
        float_167083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167083)
        # Adding element type (line 301)
        float_167084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167084)
        # Adding element type (line 301)
        float_167085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167085)
        # Adding element type (line 301)
        float_167086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167086)
        # Adding element type (line 301)
        float_167087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167087)
        # Adding element type (line 301)
        float_167088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167088)
        # Adding element type (line 301)
        float_167089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167089)
        # Adding element type (line 301)
        float_167090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167090)
        # Adding element type (line 301)
        float_167091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167091)
        # Adding element type (line 301)
        float_167092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167092)
        # Adding element type (line 301)
        float_167093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167093)
        # Adding element type (line 301)
        float_167094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167094)
        # Adding element type (line 301)
        float_167095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167095)
        # Adding element type (line 301)
        float_167096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167096)
        # Adding element type (line 301)
        float_167097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167097)
        # Adding element type (line 301)
        float_167098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167098)
        # Adding element type (line 301)
        float_167099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_167081, float_167099)
        
        
        # Obtaining an instance of the builtin type 'list' (line 304)
        list_167100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 304)
        # Adding element type (line 304)
        int_167101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, int_167101)
        # Adding element type (line 304)
        float_167102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167102)
        # Adding element type (line 304)
        int_167103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, int_167103)
        # Adding element type (line 304)
        int_167104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, int_167104)
        # Adding element type (line 304)
        float_167105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167105)
        # Adding element type (line 304)
        int_167106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, int_167106)
        # Adding element type (line 304)
        float_167107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167107)
        # Adding element type (line 304)
        float_167108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167108)
        # Adding element type (line 304)
        int_167109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, int_167109)
        # Adding element type (line 304)
        int_167110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 72), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, int_167110)
        # Adding element type (line 304)
        float_167111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167111)
        # Adding element type (line 304)
        int_167112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, int_167112)
        # Adding element type (line 304)
        float_167113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167113)
        # Adding element type (line 304)
        float_167114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167114)
        # Adding element type (line 304)
        int_167115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, int_167115)
        # Adding element type (line 304)
        float_167116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167116)
        # Adding element type (line 304)
        float_167117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167117)
        # Adding element type (line 304)
        float_167118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), list_167100, float_167118)
        
        # Processing the call keyword arguments (line 300)
        # Getting the type of 'l_sx' (line 306)
        l_sx_167119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'l_sx', False)
        keyword_167120 = l_sx_167119
        # Getting the type of 'l_sy' (line 307)
        l_sy_167121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'l_sy', False)
        keyword_167122 = l_sy_167121
        kwargs_167123 = {'sy': keyword_167122, 'sx': keyword_167120}
        # Getting the type of 'RealData' (line 300)
        RealData_167080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'RealData', False)
        # Calling RealData(args, kwargs) (line 300)
        RealData_call_result_167124 = invoke(stypy.reporting.localization.Localization(__file__, 300, 16), RealData_167080, *[list_167081, list_167100], **kwargs_167123)
        
        # Assigning a type to the variable 'l_dat' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'l_dat', RealData_call_result_167124)
        
        # Assigning a Call to a Name (line 309):
        
        # Call to Model(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'self' (line 309)
        self_167126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 22), 'self', False)
        # Obtaining the member 'lorentz' of a type (line 309)
        lorentz_167127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 22), self_167126, 'lorentz')
        # Processing the call keyword arguments (line 309)
        
        # Call to dict(...): (line 309)
        # Processing the call keyword arguments (line 309)
        str_167129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 51), 'str', 'Lorentz Peak')
        keyword_167130 = str_167129
        kwargs_167131 = {'name': keyword_167130}
        # Getting the type of 'dict' (line 309)
        dict_167128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 41), 'dict', False)
        # Calling dict(args, kwargs) (line 309)
        dict_call_result_167132 = invoke(stypy.reporting.localization.Localization(__file__, 309, 41), dict_167128, *[], **kwargs_167131)
        
        keyword_167133 = dict_call_result_167132
        kwargs_167134 = {'meta': keyword_167133}
        # Getting the type of 'Model' (line 309)
        Model_167125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'Model', False)
        # Calling Model(args, kwargs) (line 309)
        Model_call_result_167135 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), Model_167125, *[lorentz_167127], **kwargs_167134)
        
        # Assigning a type to the variable 'l_mod' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'l_mod', Model_call_result_167135)
        
        # Assigning a Call to a Name (line 310):
        
        # Call to ODR(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'l_dat' (line 310)
        l_dat_167137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'l_dat', False)
        # Getting the type of 'l_mod' (line 310)
        l_mod_167138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'l_mod', False)
        # Processing the call keyword arguments (line 310)
        
        # Obtaining an instance of the builtin type 'tuple' (line 310)
        tuple_167139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 310)
        # Adding element type (line 310)
        float_167140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 41), tuple_167139, float_167140)
        # Adding element type (line 310)
        float_167141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 41), tuple_167139, float_167141)
        # Adding element type (line 310)
        float_167142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 41), tuple_167139, float_167142)
        
        keyword_167143 = tuple_167139
        kwargs_167144 = {'beta0': keyword_167143}
        # Getting the type of 'ODR' (line 310)
        ODR_167136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'ODR', False)
        # Calling ODR(args, kwargs) (line 310)
        ODR_call_result_167145 = invoke(stypy.reporting.localization.Localization(__file__, 310, 16), ODR_167136, *[l_dat_167137, l_mod_167138], **kwargs_167144)
        
        # Assigning a type to the variable 'l_odr' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'l_odr', ODR_call_result_167145)
        
        # Assigning a Call to a Name (line 312):
        
        # Call to run(...): (line 312)
        # Processing the call keyword arguments (line 312)
        kwargs_167148 = {}
        # Getting the type of 'l_odr' (line 312)
        l_odr_167146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 14), 'l_odr', False)
        # Obtaining the member 'run' of a type (line 312)
        run_167147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 14), l_odr_167146, 'run')
        # Calling run(args, kwargs) (line 312)
        run_call_result_167149 = invoke(stypy.reporting.localization.Localization(__file__, 312, 14), run_167147, *[], **kwargs_167148)
        
        # Assigning a type to the variable 'out' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'out', run_call_result_167149)
        
        # Call to assert_array_almost_equal(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'out' (line 314)
        out_167151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'out', False)
        # Obtaining the member 'beta' of a type (line 314)
        beta_167152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), out_167151, 'beta')
        
        # Call to array(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_167155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        float_167156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), list_167155, float_167156)
        # Adding element type (line 315)
        float_167157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), list_167155, float_167157)
        # Adding element type (line 315)
        float_167158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 21), list_167155, float_167158)
        
        # Processing the call keyword arguments (line 315)
        kwargs_167159 = {}
        # Getting the type of 'np' (line 315)
        np_167153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 315)
        array_167154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), np_167153, 'array')
        # Calling array(args, kwargs) (line 315)
        array_call_result_167160 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), array_167154, *[list_167155], **kwargs_167159)
        
        # Processing the call keyword arguments (line 313)
        kwargs_167161 = {}
        # Getting the type of 'assert_array_almost_equal' (line 313)
        assert_array_almost_equal_167150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 313)
        assert_array_almost_equal_call_result_167162 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), assert_array_almost_equal_167150, *[beta_167152, array_call_result_167160], **kwargs_167161)
        
        
        # Call to assert_array_almost_equal(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'out' (line 319)
        out_167164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'out', False)
        # Obtaining the member 'sd_beta' of a type (line 319)
        sd_beta_167165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), out_167164, 'sd_beta')
        
        # Call to array(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Obtaining an instance of the builtin type 'list' (line 320)
        list_167168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 320)
        # Adding element type (line 320)
        float_167169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 21), list_167168, float_167169)
        # Adding element type (line 320)
        float_167170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 21), list_167168, float_167170)
        # Adding element type (line 320)
        float_167171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 21), list_167168, float_167171)
        
        # Processing the call keyword arguments (line 320)
        kwargs_167172 = {}
        # Getting the type of 'np' (line 320)
        np_167166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 320)
        array_167167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), np_167166, 'array')
        # Calling array(args, kwargs) (line 320)
        array_call_result_167173 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), array_167167, *[list_167168], **kwargs_167172)
        
        # Processing the call keyword arguments (line 318)
        kwargs_167174 = {}
        # Getting the type of 'assert_array_almost_equal' (line 318)
        assert_array_almost_equal_167163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 318)
        assert_array_almost_equal_call_result_167175 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), assert_array_almost_equal_167163, *[sd_beta_167165, array_call_result_167173], **kwargs_167174)
        
        
        # Call to assert_array_almost_equal(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'out' (line 324)
        out_167177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'out', False)
        # Obtaining the member 'cov_beta' of a type (line 324)
        cov_beta_167178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), out_167177, 'cov_beta')
        
        # Call to array(...): (line 325)
        # Processing the call arguments (line 325)
        
        # Obtaining an instance of the builtin type 'list' (line 325)
        list_167181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 325)
        # Adding element type (line 325)
        
        # Obtaining an instance of the builtin type 'list' (line 325)
        list_167182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 325)
        # Adding element type (line 325)
        float_167183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 22), list_167182, float_167183)
        # Adding element type (line 325)
        float_167184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 22), list_167182, float_167184)
        # Adding element type (line 325)
        float_167185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 22), list_167182, float_167185)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 21), list_167181, list_167182)
        # Adding element type (line 325)
        
        # Obtaining an instance of the builtin type 'list' (line 327)
        list_167186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 327)
        # Adding element type (line 327)
        float_167187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), list_167186, float_167187)
        # Adding element type (line 327)
        float_167188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), list_167186, float_167188)
        # Adding element type (line 327)
        float_167189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), list_167186, float_167189)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 21), list_167181, list_167186)
        # Adding element type (line 325)
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_167190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        # Adding element type (line 329)
        float_167191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 15), list_167190, float_167191)
        # Adding element type (line 329)
        float_167192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 15), list_167190, float_167192)
        # Adding element type (line 329)
        float_167193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 15), list_167190, float_167193)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 21), list_167181, list_167190)
        
        # Processing the call keyword arguments (line 325)
        kwargs_167194 = {}
        # Getting the type of 'np' (line 325)
        np_167179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 325)
        array_167180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), np_167179, 'array')
        # Calling array(args, kwargs) (line 325)
        array_call_result_167195 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), array_167180, *[list_167181], **kwargs_167194)
        
        # Processing the call keyword arguments (line 323)
        kwargs_167196 = {}
        # Getting the type of 'assert_array_almost_equal' (line 323)
        assert_array_almost_equal_167176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 323)
        assert_array_almost_equal_call_result_167197 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), assert_array_almost_equal_167176, *[cov_beta_167178, array_call_result_167195], **kwargs_167196)
        
        
        # ################# End of 'test_lorentz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lorentz' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_167198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_167198)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lorentz'
        return stypy_return_type_167198


    @norecursion
    def test_ticket_1253(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ticket_1253'
        module_type_store = module_type_store.open_function_context('test_ticket_1253', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_localization', localization)
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_function_name', 'TestODR.test_ticket_1253')
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_param_names_list', [])
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestODR.test_ticket_1253.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.test_ticket_1253', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ticket_1253', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ticket_1253(...)' code ##################


        @norecursion
        def linear(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'linear'
            module_type_store = module_type_store.open_function_context('linear', 334, 8, False)
            
            # Passed parameters checking function
            linear.stypy_localization = localization
            linear.stypy_type_of_self = None
            linear.stypy_type_store = module_type_store
            linear.stypy_function_name = 'linear'
            linear.stypy_param_names_list = ['c', 'x']
            linear.stypy_varargs_param_name = None
            linear.stypy_kwargs_param_name = None
            linear.stypy_call_defaults = defaults
            linear.stypy_call_varargs = varargs
            linear.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'linear', ['c', 'x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'linear', localization, ['c', 'x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'linear(...)' code ##################

            
            # Obtaining the type of the subscript
            int_167199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 21), 'int')
            # Getting the type of 'c' (line 335)
            c_167200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'c')
            # Obtaining the member '__getitem__' of a type (line 335)
            getitem___167201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 19), c_167200, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 335)
            subscript_call_result_167202 = invoke(stypy.reporting.localization.Localization(__file__, 335, 19), getitem___167201, int_167199)
            
            # Getting the type of 'x' (line 335)
            x_167203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), 'x')
            # Applying the binary operator '*' (line 335)
            result_mul_167204 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 19), '*', subscript_call_result_167202, x_167203)
            
            
            # Obtaining the type of the subscript
            int_167205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 28), 'int')
            # Getting the type of 'c' (line 335)
            c_167206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 26), 'c')
            # Obtaining the member '__getitem__' of a type (line 335)
            getitem___167207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 26), c_167206, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 335)
            subscript_call_result_167208 = invoke(stypy.reporting.localization.Localization(__file__, 335, 26), getitem___167207, int_167205)
            
            # Applying the binary operator '+' (line 335)
            result_add_167209 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 19), '+', result_mul_167204, subscript_call_result_167208)
            
            # Assigning a type to the variable 'stypy_return_type' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'stypy_return_type', result_add_167209)
            
            # ################# End of 'linear(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'linear' in the type store
            # Getting the type of 'stypy_return_type' (line 334)
            stypy_return_type_167210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_167210)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'linear'
            return stypy_return_type_167210

        # Assigning a type to the variable 'linear' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'linear', linear)
        
        # Assigning a List to a Name (line 337):
        
        # Obtaining an instance of the builtin type 'list' (line 337)
        list_167211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 337)
        # Adding element type (line 337)
        float_167212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 12), list_167211, float_167212)
        # Adding element type (line 337)
        float_167213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 12), list_167211, float_167213)
        
        # Assigning a type to the variable 'c' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'c', list_167211)
        
        # Assigning a Call to a Name (line 338):
        
        # Call to linspace(...): (line 338)
        # Processing the call arguments (line 338)
        int_167216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 24), 'int')
        int_167217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 27), 'int')
        # Processing the call keyword arguments (line 338)
        kwargs_167218 = {}
        # Getting the type of 'np' (line 338)
        np_167214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 338)
        linspace_167215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), np_167214, 'linspace')
        # Calling linspace(args, kwargs) (line 338)
        linspace_call_result_167219 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), linspace_167215, *[int_167216, int_167217], **kwargs_167218)
        
        # Assigning a type to the variable 'x' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'x', linspace_call_result_167219)
        
        # Assigning a Call to a Name (line 339):
        
        # Call to linear(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'c' (line 339)
        c_167221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'c', False)
        # Getting the type of 'x' (line 339)
        x_167222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 22), 'x', False)
        # Processing the call keyword arguments (line 339)
        kwargs_167223 = {}
        # Getting the type of 'linear' (line 339)
        linear_167220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'linear', False)
        # Calling linear(args, kwargs) (line 339)
        linear_call_result_167224 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), linear_167220, *[c_167221, x_167222], **kwargs_167223)
        
        # Assigning a type to the variable 'y' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'y', linear_call_result_167224)
        
        # Assigning a Call to a Name (line 341):
        
        # Call to Model(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'linear' (line 341)
        linear_167226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 22), 'linear', False)
        # Processing the call keyword arguments (line 341)
        kwargs_167227 = {}
        # Getting the type of 'Model' (line 341)
        Model_167225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'Model', False)
        # Calling Model(args, kwargs) (line 341)
        Model_call_result_167228 = invoke(stypy.reporting.localization.Localization(__file__, 341, 16), Model_167225, *[linear_167226], **kwargs_167227)
        
        # Assigning a type to the variable 'model' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'model', Model_call_result_167228)
        
        # Assigning a Call to a Name (line 342):
        
        # Call to Data(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'x' (line 342)
        x_167230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'x', False)
        # Getting the type of 'y' (line 342)
        y_167231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 23), 'y', False)
        # Processing the call keyword arguments (line 342)
        float_167232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 29), 'float')
        keyword_167233 = float_167232
        float_167234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 37), 'float')
        keyword_167235 = float_167234
        kwargs_167236 = {'we': keyword_167235, 'wd': keyword_167233}
        # Getting the type of 'Data' (line 342)
        Data_167229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'Data', False)
        # Calling Data(args, kwargs) (line 342)
        Data_call_result_167237 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), Data_167229, *[x_167230, y_167231], **kwargs_167236)
        
        # Assigning a type to the variable 'data' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'data', Data_call_result_167237)
        
        # Assigning a Call to a Name (line 343):
        
        # Call to ODR(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'data' (line 343)
        data_167239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'data', False)
        # Getting the type of 'model' (line 343)
        model_167240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 24), 'model', False)
        # Processing the call keyword arguments (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_167241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        float_167242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 37), list_167241, float_167242)
        # Adding element type (line 343)
        float_167243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 37), list_167241, float_167243)
        
        keyword_167244 = list_167241
        kwargs_167245 = {'beta0': keyword_167244}
        # Getting the type of 'ODR' (line 343)
        ODR_167238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'ODR', False)
        # Calling ODR(args, kwargs) (line 343)
        ODR_call_result_167246 = invoke(stypy.reporting.localization.Localization(__file__, 343, 14), ODR_167238, *[data_167239, model_167240], **kwargs_167245)
        
        # Assigning a type to the variable 'job' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'job', ODR_call_result_167246)
        
        # Assigning a Call to a Name (line 344):
        
        # Call to run(...): (line 344)
        # Processing the call keyword arguments (line 344)
        kwargs_167249 = {}
        # Getting the type of 'job' (line 344)
        job_167247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 17), 'job', False)
        # Obtaining the member 'run' of a type (line 344)
        run_167248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 17), job_167247, 'run')
        # Calling run(args, kwargs) (line 344)
        run_call_result_167250 = invoke(stypy.reporting.localization.Localization(__file__, 344, 17), run_167248, *[], **kwargs_167249)
        
        # Assigning a type to the variable 'result' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'result', run_call_result_167250)
        
        # Call to assert_equal(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'result' (line 345)
        result_167252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 21), 'result', False)
        # Obtaining the member 'info' of a type (line 345)
        info_167253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 21), result_167252, 'info')
        int_167254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 34), 'int')
        # Processing the call keyword arguments (line 345)
        kwargs_167255 = {}
        # Getting the type of 'assert_equal' (line 345)
        assert_equal_167251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 345)
        assert_equal_call_result_167256 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assert_equal_167251, *[info_167253, int_167254], **kwargs_167255)
        
        
        # ################# End of 'test_ticket_1253(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ticket_1253' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_167257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_167257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ticket_1253'
        return stypy_return_type_167257


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestODR.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestODR' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestODR', TestODR)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

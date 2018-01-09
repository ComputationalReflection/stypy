
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit test for Linear Programming via Simplex Algorithm.
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.testing import assert_, assert_allclose
8: from pytest import raises as assert_raises
9: from scipy.optimize._linprog_ip import _clean_inputs
10: from copy import deepcopy
11: 
12: 
13: def test_aliasing():
14:     c = 1
15:     A_ub = [[1]]
16:     b_ub = [1]
17:     A_eq = [[1]]
18:     b_eq = [1]
19:     bounds = (-np.inf, np.inf)
20: 
21:     c_copy = deepcopy(c)
22:     A_ub_copy = deepcopy(A_ub)
23:     b_ub_copy = deepcopy(b_ub)
24:     A_eq_copy = deepcopy(A_eq)
25:     b_eq_copy = deepcopy(b_eq)
26:     bounds_copy = deepcopy(bounds)
27: 
28:     _clean_inputs(c, A_ub, b_ub, A_eq, b_eq, bounds)
29: 
30:     assert_(c == c_copy, "c modified by _clean_inputs")
31:     assert_(A_ub == A_ub_copy, "A_ub modified by _clean_inputs")
32:     assert_(b_ub == b_ub_copy, "b_ub modified by _clean_inputs")
33:     assert_(A_eq == A_eq_copy, "A_eq modified by _clean_inputs")
34:     assert_(b_eq == b_eq_copy, "b_eq modified by _clean_inputs")
35:     assert_(bounds == bounds_copy, "bounds modified by _clean_inputs")
36: 
37: 
38: def test_aliasing2():
39:     c = np.array([1, 1])
40:     A_ub = np.array([[1, 1], [2, 2]])
41:     b_ub = np.array([[1], [1]])
42:     A_eq = np.array([[1, 1]])
43:     b_eq = np.array([1])
44:     bounds = [(-np.inf, np.inf), (None, 1)]
45: 
46:     c_copy = c.copy()
47:     A_ub_copy = A_ub.copy()
48:     b_ub_copy = b_ub.copy()
49:     A_eq_copy = A_eq.copy()
50:     b_eq_copy = b_eq.copy()
51:     bounds_copy = deepcopy(bounds)
52: 
53:     _clean_inputs(c, A_ub, b_ub, A_eq, b_eq, bounds)
54: 
55:     assert_allclose(c, c_copy, err_msg="c modified by _clean_inputs")
56:     assert_allclose(A_ub, A_ub_copy, err_msg="A_ub modified by _clean_inputs")
57:     assert_allclose(b_ub, b_ub_copy, err_msg="b_ub modified by _clean_inputs")
58:     assert_allclose(A_eq, A_eq_copy, err_msg="A_eq modified by _clean_inputs")
59:     assert_allclose(b_eq, b_eq_copy, err_msg="b_eq modified by _clean_inputs")
60:     assert_(bounds == bounds_copy, "bounds modified by _clean_inputs")
61: 
62: 
63: def test_missing_inputs():
64:     c = [1, 2]
65:     A_ub = np.array([[1, 1], [2, 2]])
66:     b_ub = np.array([1, 1])
67:     A_eq = np.array([[1, 1], [2, 2]])
68:     b_eq = np.array([1, 1])
69: 
70:     assert_raises(TypeError, _clean_inputs)
71:     assert_raises(TypeError, _clean_inputs, c=None)
72:     assert_raises(ValueError, _clean_inputs, c=c, A_ub=A_ub)
73:     assert_raises(ValueError, _clean_inputs, c=c, A_ub=A_ub, b_ub=None)
74:     assert_raises(ValueError, _clean_inputs, c=c, b_ub=b_ub)
75:     assert_raises(ValueError, _clean_inputs, c=c, A_ub=None, b_ub=b_ub)
76:     assert_raises(ValueError, _clean_inputs, c=c, A_eq=A_eq)
77:     assert_raises(ValueError, _clean_inputs, c=c, A_eq=A_eq, b_eq=None)
78:     assert_raises(ValueError, _clean_inputs, c=c, b_eq=b_eq)
79:     assert_raises(ValueError, _clean_inputs, c=c, A_eq=None, b_eq=b_eq)
80: 
81: 
82: def test_too_many_dimensions():
83:     cb = [1, 2, 3, 4]
84:     A = np.random.rand(4, 4)
85:     bad2D = [[1, 2], [3, 4]]
86:     bad3D = np.random.rand(4, 4, 4)
87:     assert_raises(ValueError, _clean_inputs, c=bad2D, A_ub=A, b_ub=cb)
88:     assert_raises(ValueError, _clean_inputs, c=cb, A_ub=bad3D, b_ub=cb)
89:     assert_raises(ValueError, _clean_inputs, c=cb, A_ub=A, b_ub=bad2D)
90:     assert_raises(ValueError, _clean_inputs, c=cb, A_eq=bad3D, b_eq=cb)
91:     assert_raises(ValueError, _clean_inputs, c=cb, A_eq=A, b_eq=bad2D)
92: 
93: 
94: def test_too_few_dimensions():
95:     bad = np.random.rand(4, 4).ravel()
96:     cb = np.random.rand(4)
97:     assert_raises(ValueError, _clean_inputs, c=cb, A_ub=bad, b_ub=cb)
98:     assert_raises(ValueError, _clean_inputs, c=cb, A_eq=bad, b_eq=cb)
99: 
100: 
101: def test_inconsistent_dimensions():
102:     m = 2
103:     n = 4
104:     c = [1, 2, 3, 4]
105: 
106:     Agood = np.random.rand(m, n)
107:     Abad = np.random.rand(m, n + 1)
108:     bgood = np.random.rand(m)
109:     bbad = np.random.rand(m + 1)
110:     boundsbad = [(0, 1)] * (n + 1)
111:     assert_raises(ValueError, _clean_inputs, c=c, A_ub=Abad, b_ub=bgood)
112:     assert_raises(ValueError, _clean_inputs, c=c, A_ub=Agood, b_ub=bbad)
113:     assert_raises(ValueError, _clean_inputs, c=c, A_eq=Abad, b_eq=bgood)
114:     assert_raises(ValueError, _clean_inputs, c=c, A_eq=Agood, b_eq=bbad)
115:     assert_raises(ValueError, _clean_inputs, c=c, bounds=boundsbad)
116: 
117: 
118: def test_type_errors():
119:     bad = "hello"
120:     c = [1, 2]
121:     A_ub = np.array([[1, 1], [2, 2]])
122:     b_ub = np.array([1, 1])
123:     A_eq = np.array([[1, 1], [2, 2]])
124:     b_eq = np.array([1, 1])
125:     bounds = [(0, 1)]
126:     assert_raises(
127:         TypeError,
128:         _clean_inputs,
129:         c=bad,
130:         A_ub=A_ub,
131:         b_ub=b_ub,
132:         A_eq=A_eq,
133:         b_eq=b_eq,
134:         bounds=bounds)
135:     assert_raises(
136:         TypeError,
137:         _clean_inputs,
138:         c=c,
139:         A_ub=bad,
140:         b_ub=b_ub,
141:         A_eq=A_eq,
142:         b_eq=b_eq,
143:         bounds=bounds)
144:     assert_raises(
145:         TypeError,
146:         _clean_inputs,
147:         c=c,
148:         A_ub=A_ub,
149:         b_ub=bad,
150:         A_eq=A_eq,
151:         b_eq=b_eq,
152:         bounds=bounds)
153:     assert_raises(
154:         TypeError,
155:         _clean_inputs,
156:         c=c,
157:         A_ub=A_ub,
158:         b_ub=b_ub,
159:         A_eq=bad,
160:         b_eq=b_eq,
161:         bounds=bounds)
162: 
163:     assert_raises(
164:         TypeError,
165:         _clean_inputs,
166:         c=c,
167:         A_ub=A_ub,
168:         b_ub=b_ub,
169:         A_eq=A_eq,
170:         b_eq=b_eq,
171:         bounds=bad)
172:     assert_raises(
173:         TypeError,
174:         _clean_inputs,
175:         c=c,
176:         A_ub=A_ub,
177:         b_ub=b_ub,
178:         A_eq=A_eq,
179:         b_eq=b_eq,
180:         bounds="hi")
181:     assert_raises(
182:         TypeError,
183:         _clean_inputs,
184:         c=c,
185:         A_ub=A_ub,
186:         b_ub=b_ub,
187:         A_eq=A_eq,
188:         b_eq=b_eq,
189:         bounds=["hi"])
190:     assert_raises(
191:         TypeError,
192:         _clean_inputs,
193:         c=c,
194:         A_ub=A_ub,
195:         b_ub=b_ub,
196:         A_eq=A_eq,
197:         b_eq=b_eq,
198:         bounds=[
199:             ("hi")])
200:     assert_raises(TypeError, _clean_inputs, c=c, A_ub=A_ub,
201:                   b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(1, "")])
202:     assert_raises(TypeError, _clean_inputs, c=c, A_ub=A_ub,
203:                   b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(1, 2), (1, "")])
204: 
205: 
206: def test_non_finite_errors():
207:     c = [1, 2]
208:     A_ub = np.array([[1, 1], [2, 2]])
209:     b_ub = np.array([1, 1])
210:     A_eq = np.array([[1, 1], [2, 2]])
211:     b_eq = np.array([1, 1])
212:     bounds = [(0, 1)]
213:     assert_raises(
214:         ValueError, _clean_inputs, c=[0, None], A_ub=A_ub, b_ub=b_ub,
215:         A_eq=A_eq, b_eq=b_eq, bounds=bounds)
216:     assert_raises(
217:         ValueError, _clean_inputs, c=[np.inf, 0], A_ub=A_ub, b_ub=b_ub,
218:         A_eq=A_eq, b_eq=b_eq, bounds=bounds)
219:     assert_raises(
220:         ValueError, _clean_inputs, c=[0, -np.inf], A_ub=A_ub, b_ub=b_ub,
221:         A_eq=A_eq, b_eq=b_eq, bounds=bounds)
222:     assert_raises(
223:         ValueError, _clean_inputs, c=[np.nan, 0], A_ub=A_ub, b_ub=b_ub,
224:         A_eq=A_eq, b_eq=b_eq, bounds=bounds)
225: 
226:     assert_raises(ValueError, _clean_inputs, c=c, A_ub=[[1, 2], [None, 1]],
227:                   b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
228:     assert_raises(
229:         ValueError,
230:         _clean_inputs,
231:         c=c,
232:         A_ub=A_ub,
233:         b_ub=[
234:             np.inf,
235:             1],
236:         A_eq=A_eq,
237:         b_eq=b_eq,
238:         bounds=bounds)
239:     assert_raises(ValueError, _clean_inputs, c=c, A_ub=A_ub, b_ub=b_ub, A_eq=[
240:                   [1, 2], [1, -np.inf]], b_eq=b_eq, bounds=bounds)
241:     assert_raises(
242:         ValueError,
243:         _clean_inputs,
244:         c=c,
245:         A_ub=A_ub,
246:         b_ub=b_ub,
247:         A_eq=A_eq,
248:         b_eq=[
249:             1,
250:             np.nan],
251:         bounds=bounds)
252: 
253: 
254: def test__clean_inputs1():
255:     c = [1, 2]
256:     A_ub = [[1, 1], [2, 2]]
257:     b_ub = [1, 1]
258:     A_eq = [[1, 1], [2, 2]]
259:     b_eq = [1, 1]
260:     bounds = None
261:     outputs = _clean_inputs(
262:         c=c,
263:         A_ub=A_ub,
264:         b_ub=b_ub,
265:         A_eq=A_eq,
266:         b_eq=b_eq,
267:         bounds=bounds)
268:     assert_allclose(outputs[0], np.array(c))
269:     assert_allclose(outputs[1], np.array(A_ub))
270:     assert_allclose(outputs[2], np.array(b_ub))
271:     assert_allclose(outputs[3], np.array(A_eq))
272:     assert_allclose(outputs[4], np.array(b_eq))
273:     assert_(outputs[5] == [(0, None)] * 2, "")
274: 
275:     assert_(outputs[0].shape == (2,), "")
276:     assert_(outputs[1].shape == (2, 2), "")
277:     assert_(outputs[2].shape == (2,), "")
278:     assert_(outputs[3].shape == (2, 2), "")
279:     assert_(outputs[4].shape == (2,), "")
280: 
281: 
282: def test__clean_inputs2():
283:     c = 1
284:     A_ub = [[1]]
285:     b_ub = 1
286:     A_eq = [[1]]
287:     b_eq = 1
288:     bounds = (0, 1)
289:     outputs = _clean_inputs(
290:         c=c,
291:         A_ub=A_ub,
292:         b_ub=b_ub,
293:         A_eq=A_eq,
294:         b_eq=b_eq,
295:         bounds=bounds)
296:     assert_allclose(outputs[0], np.array(c))
297:     assert_allclose(outputs[1], np.array(A_ub))
298:     assert_allclose(outputs[2], np.array(b_ub))
299:     assert_allclose(outputs[3], np.array(A_eq))
300:     assert_allclose(outputs[4], np.array(b_eq))
301:     assert_(outputs[5] == [(0, 1)], "")
302: 
303:     assert_(outputs[0].shape == (1,), "")
304:     assert_(outputs[1].shape == (1, 1), "")
305:     assert_(outputs[2].shape == (1,), "")
306:     assert_(outputs[3].shape == (1, 1), "")
307:     assert_(outputs[4].shape == (1,), "")
308: 
309: 
310: def test__clean_inputs3():
311:     c = [[1, 2]]
312:     A_ub = np.random.rand(2, 2)
313:     b_ub = [[1], [2]]
314:     A_eq = np.random.rand(2, 2)
315:     b_eq = [[1], [2]]
316:     bounds = [(0, 1)]
317:     outputs = _clean_inputs(
318:         c=c,
319:         A_ub=A_ub,
320:         b_ub=b_ub,
321:         A_eq=A_eq,
322:         b_eq=b_eq,
323:         bounds=bounds)
324:     assert_allclose(outputs[0], np.array([1, 2]))
325:     assert_allclose(outputs[2], np.array([1, 2]))
326:     assert_allclose(outputs[4], np.array([1, 2]))
327:     assert_(outputs[5] == [(0, 1)] * 2, "")
328: 
329:     assert_(outputs[0].shape == (2,), "")
330:     assert_(outputs[2].shape == (2,), "")
331:     assert_(outputs[4].shape == (2,), "")
332: 
333: 
334: def test_bad_bounds():
335:     c = [1, 2]
336:     assert_raises(ValueError, _clean_inputs, c=c, bounds=(1, -2))
337:     assert_raises(ValueError, _clean_inputs, c=c, bounds=[(1, -2)])
338:     assert_raises(ValueError, _clean_inputs, c=c, bounds=[(1, -2), (1, 2)])
339: 
340:     assert_raises(ValueError, _clean_inputs, c=c, bounds=(1, 2, 2))
341:     assert_raises(ValueError, _clean_inputs, c=c, bounds=[(1, 2, 2)])
342:     assert_raises(ValueError, _clean_inputs, c=c, bounds=[(1, 2), (1, 2, 2)])
343:     assert_raises(ValueError, _clean_inputs, c=c,
344:                   bounds=[(1, 2), (1, 2), (1, 2)])
345: 
346: 
347: def test_good_bounds():
348:     c = [1, 2]
349:     outputs = _clean_inputs(c=c, bounds=None)
350:     assert_(outputs[5] == [(0, None)] * 2, "")
351: 
352:     outputs = _clean_inputs(c=c, bounds=(1, 2))
353:     assert_(outputs[5] == [(1, 2)] * 2, "")
354: 
355:     outputs = _clean_inputs(c=c, bounds=[(1, 2)])
356:     assert_(outputs[5] == [(1, 2)] * 2, "")
357: 
358:     outputs = _clean_inputs(c=c, bounds=[(1, np.inf)])
359:     assert_(outputs[5] == [(1, None)] * 2, "")
360: 
361:     outputs = _clean_inputs(c=c, bounds=[(-np.inf, 1)])
362:     assert_(outputs[5] == [(None, 1)] * 2, "")
363: 
364:     outputs = _clean_inputs(c=c, bounds=[(-np.inf, np.inf), (-np.inf, np.inf)])
365:     assert_(outputs[5] == [(None, None)] * 2, "")
366: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_239607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nUnit test for Linear Programming via Simplex Algorithm.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_239608 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_239608) is not StypyTypeError):

    if (import_239608 != 'pyd_module'):
        __import__(import_239608)
        sys_modules_239609 = sys.modules[import_239608]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_239609.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_239608)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_, assert_allclose' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_239610 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_239610) is not StypyTypeError):

    if (import_239610 != 'pyd_module'):
        __import__(import_239610)
        sys_modules_239611 = sys.modules[import_239610]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_239611.module_type_store, module_type_store, ['assert_', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_239611, sys_modules_239611.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose'], [assert_, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_239610)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_239612 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_239612) is not StypyTypeError):

    if (import_239612 != 'pyd_module'):
        __import__(import_239612)
        sys_modules_239613 = sys.modules[import_239612]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_239613.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_239613, sys_modules_239613.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_239612)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.optimize._linprog_ip import _clean_inputs' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_239614 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize._linprog_ip')

if (type(import_239614) is not StypyTypeError):

    if (import_239614 != 'pyd_module'):
        __import__(import_239614)
        sys_modules_239615 = sys.modules[import_239614]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize._linprog_ip', sys_modules_239615.module_type_store, module_type_store, ['_clean_inputs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_239615, sys_modules_239615.module_type_store, module_type_store)
    else:
        from scipy.optimize._linprog_ip import _clean_inputs

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize._linprog_ip', None, module_type_store, ['_clean_inputs'], [_clean_inputs])

else:
    # Assigning a type to the variable 'scipy.optimize._linprog_ip' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize._linprog_ip', import_239614)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from copy import deepcopy' statement (line 10)
try:
    from copy import deepcopy

except:
    deepcopy = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'copy', None, module_type_store, ['deepcopy'], [deepcopy])


@norecursion
def test_aliasing(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_aliasing'
    module_type_store = module_type_store.open_function_context('test_aliasing', 13, 0, False)
    
    # Passed parameters checking function
    test_aliasing.stypy_localization = localization
    test_aliasing.stypy_type_of_self = None
    test_aliasing.stypy_type_store = module_type_store
    test_aliasing.stypy_function_name = 'test_aliasing'
    test_aliasing.stypy_param_names_list = []
    test_aliasing.stypy_varargs_param_name = None
    test_aliasing.stypy_kwargs_param_name = None
    test_aliasing.stypy_call_defaults = defaults
    test_aliasing.stypy_call_varargs = varargs
    test_aliasing.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_aliasing', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_aliasing', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_aliasing(...)' code ##################

    
    # Assigning a Num to a Name (line 14):
    int_239616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'int')
    # Assigning a type to the variable 'c' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'c', int_239616)
    
    # Assigning a List to a Name (line 15):
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_239617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_239618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    int_239619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 12), list_239618, int_239619)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 11), list_239617, list_239618)
    
    # Assigning a type to the variable 'A_ub' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'A_ub', list_239617)
    
    # Assigning a List to a Name (line 16):
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_239620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_239621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 11), list_239620, int_239621)
    
    # Assigning a type to the variable 'b_ub' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'b_ub', list_239620)
    
    # Assigning a List to a Name (line 17):
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_239622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_239623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    int_239624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), list_239623, int_239624)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 11), list_239622, list_239623)
    
    # Assigning a type to the variable 'A_eq' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'A_eq', list_239622)
    
    # Assigning a List to a Name (line 18):
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_239625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    int_239626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 11), list_239625, int_239626)
    
    # Assigning a type to the variable 'b_eq' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'b_eq', list_239625)
    
    # Assigning a Tuple to a Name (line 19):
    
    # Obtaining an instance of the builtin type 'tuple' (line 19)
    tuple_239627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 19)
    # Adding element type (line 19)
    
    # Getting the type of 'np' (line 19)
    np_239628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'np')
    # Obtaining the member 'inf' of a type (line 19)
    inf_239629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), np_239628, 'inf')
    # Applying the 'usub' unary operator (line 19)
    result___neg___239630 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 14), 'usub', inf_239629)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), tuple_239627, result___neg___239630)
    # Adding element type (line 19)
    # Getting the type of 'np' (line 19)
    np_239631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'np')
    # Obtaining the member 'inf' of a type (line 19)
    inf_239632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 23), np_239631, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), tuple_239627, inf_239632)
    
    # Assigning a type to the variable 'bounds' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'bounds', tuple_239627)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to deepcopy(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'c' (line 21)
    c_239634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'c', False)
    # Processing the call keyword arguments (line 21)
    kwargs_239635 = {}
    # Getting the type of 'deepcopy' (line 21)
    deepcopy_239633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'deepcopy', False)
    # Calling deepcopy(args, kwargs) (line 21)
    deepcopy_call_result_239636 = invoke(stypy.reporting.localization.Localization(__file__, 21, 13), deepcopy_239633, *[c_239634], **kwargs_239635)
    
    # Assigning a type to the variable 'c_copy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'c_copy', deepcopy_call_result_239636)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to deepcopy(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'A_ub' (line 22)
    A_ub_239638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'A_ub', False)
    # Processing the call keyword arguments (line 22)
    kwargs_239639 = {}
    # Getting the type of 'deepcopy' (line 22)
    deepcopy_239637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'deepcopy', False)
    # Calling deepcopy(args, kwargs) (line 22)
    deepcopy_call_result_239640 = invoke(stypy.reporting.localization.Localization(__file__, 22, 16), deepcopy_239637, *[A_ub_239638], **kwargs_239639)
    
    # Assigning a type to the variable 'A_ub_copy' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'A_ub_copy', deepcopy_call_result_239640)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to deepcopy(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'b_ub' (line 23)
    b_ub_239642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'b_ub', False)
    # Processing the call keyword arguments (line 23)
    kwargs_239643 = {}
    # Getting the type of 'deepcopy' (line 23)
    deepcopy_239641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'deepcopy', False)
    # Calling deepcopy(args, kwargs) (line 23)
    deepcopy_call_result_239644 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), deepcopy_239641, *[b_ub_239642], **kwargs_239643)
    
    # Assigning a type to the variable 'b_ub_copy' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'b_ub_copy', deepcopy_call_result_239644)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to deepcopy(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'A_eq' (line 24)
    A_eq_239646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'A_eq', False)
    # Processing the call keyword arguments (line 24)
    kwargs_239647 = {}
    # Getting the type of 'deepcopy' (line 24)
    deepcopy_239645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'deepcopy', False)
    # Calling deepcopy(args, kwargs) (line 24)
    deepcopy_call_result_239648 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), deepcopy_239645, *[A_eq_239646], **kwargs_239647)
    
    # Assigning a type to the variable 'A_eq_copy' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'A_eq_copy', deepcopy_call_result_239648)
    
    # Assigning a Call to a Name (line 25):
    
    # Call to deepcopy(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'b_eq' (line 25)
    b_eq_239650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'b_eq', False)
    # Processing the call keyword arguments (line 25)
    kwargs_239651 = {}
    # Getting the type of 'deepcopy' (line 25)
    deepcopy_239649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'deepcopy', False)
    # Calling deepcopy(args, kwargs) (line 25)
    deepcopy_call_result_239652 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), deepcopy_239649, *[b_eq_239650], **kwargs_239651)
    
    # Assigning a type to the variable 'b_eq_copy' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'b_eq_copy', deepcopy_call_result_239652)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to deepcopy(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'bounds' (line 26)
    bounds_239654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'bounds', False)
    # Processing the call keyword arguments (line 26)
    kwargs_239655 = {}
    # Getting the type of 'deepcopy' (line 26)
    deepcopy_239653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'deepcopy', False)
    # Calling deepcopy(args, kwargs) (line 26)
    deepcopy_call_result_239656 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), deepcopy_239653, *[bounds_239654], **kwargs_239655)
    
    # Assigning a type to the variable 'bounds_copy' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'bounds_copy', deepcopy_call_result_239656)
    
    # Call to _clean_inputs(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'c' (line 28)
    c_239658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'c', False)
    # Getting the type of 'A_ub' (line 28)
    A_ub_239659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'A_ub', False)
    # Getting the type of 'b_ub' (line 28)
    b_ub_239660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 27), 'b_ub', False)
    # Getting the type of 'A_eq' (line 28)
    A_eq_239661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 33), 'A_eq', False)
    # Getting the type of 'b_eq' (line 28)
    b_eq_239662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 39), 'b_eq', False)
    # Getting the type of 'bounds' (line 28)
    bounds_239663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 45), 'bounds', False)
    # Processing the call keyword arguments (line 28)
    kwargs_239664 = {}
    # Getting the type of '_clean_inputs' (line 28)
    _clean_inputs_239657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 28)
    _clean_inputs_call_result_239665 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), _clean_inputs_239657, *[c_239658, A_ub_239659, b_ub_239660, A_eq_239661, b_eq_239662, bounds_239663], **kwargs_239664)
    
    
    # Call to assert_(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Getting the type of 'c' (line 30)
    c_239667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'c', False)
    # Getting the type of 'c_copy' (line 30)
    c_copy_239668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'c_copy', False)
    # Applying the binary operator '==' (line 30)
    result_eq_239669 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), '==', c_239667, c_copy_239668)
    
    str_239670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'str', 'c modified by _clean_inputs')
    # Processing the call keyword arguments (line 30)
    kwargs_239671 = {}
    # Getting the type of 'assert_' (line 30)
    assert__239666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 30)
    assert__call_result_239672 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert__239666, *[result_eq_239669, str_239670], **kwargs_239671)
    
    
    # Call to assert_(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Getting the type of 'A_ub' (line 31)
    A_ub_239674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'A_ub', False)
    # Getting the type of 'A_ub_copy' (line 31)
    A_ub_copy_239675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'A_ub_copy', False)
    # Applying the binary operator '==' (line 31)
    result_eq_239676 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 12), '==', A_ub_239674, A_ub_copy_239675)
    
    str_239677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'str', 'A_ub modified by _clean_inputs')
    # Processing the call keyword arguments (line 31)
    kwargs_239678 = {}
    # Getting the type of 'assert_' (line 31)
    assert__239673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 31)
    assert__call_result_239679 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert__239673, *[result_eq_239676, str_239677], **kwargs_239678)
    
    
    # Call to assert_(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Getting the type of 'b_ub' (line 32)
    b_ub_239681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'b_ub', False)
    # Getting the type of 'b_ub_copy' (line 32)
    b_ub_copy_239682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'b_ub_copy', False)
    # Applying the binary operator '==' (line 32)
    result_eq_239683 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), '==', b_ub_239681, b_ub_copy_239682)
    
    str_239684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'str', 'b_ub modified by _clean_inputs')
    # Processing the call keyword arguments (line 32)
    kwargs_239685 = {}
    # Getting the type of 'assert_' (line 32)
    assert__239680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 32)
    assert__call_result_239686 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), assert__239680, *[result_eq_239683, str_239684], **kwargs_239685)
    
    
    # Call to assert_(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Getting the type of 'A_eq' (line 33)
    A_eq_239688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'A_eq', False)
    # Getting the type of 'A_eq_copy' (line 33)
    A_eq_copy_239689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'A_eq_copy', False)
    # Applying the binary operator '==' (line 33)
    result_eq_239690 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 12), '==', A_eq_239688, A_eq_copy_239689)
    
    str_239691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'str', 'A_eq modified by _clean_inputs')
    # Processing the call keyword arguments (line 33)
    kwargs_239692 = {}
    # Getting the type of 'assert_' (line 33)
    assert__239687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 33)
    assert__call_result_239693 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), assert__239687, *[result_eq_239690, str_239691], **kwargs_239692)
    
    
    # Call to assert_(...): (line 34)
    # Processing the call arguments (line 34)
    
    # Getting the type of 'b_eq' (line 34)
    b_eq_239695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'b_eq', False)
    # Getting the type of 'b_eq_copy' (line 34)
    b_eq_copy_239696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'b_eq_copy', False)
    # Applying the binary operator '==' (line 34)
    result_eq_239697 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), '==', b_eq_239695, b_eq_copy_239696)
    
    str_239698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'str', 'b_eq modified by _clean_inputs')
    # Processing the call keyword arguments (line 34)
    kwargs_239699 = {}
    # Getting the type of 'assert_' (line 34)
    assert__239694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 34)
    assert__call_result_239700 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), assert__239694, *[result_eq_239697, str_239698], **kwargs_239699)
    
    
    # Call to assert_(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Getting the type of 'bounds' (line 35)
    bounds_239702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'bounds', False)
    # Getting the type of 'bounds_copy' (line 35)
    bounds_copy_239703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'bounds_copy', False)
    # Applying the binary operator '==' (line 35)
    result_eq_239704 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 12), '==', bounds_239702, bounds_copy_239703)
    
    str_239705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 35), 'str', 'bounds modified by _clean_inputs')
    # Processing the call keyword arguments (line 35)
    kwargs_239706 = {}
    # Getting the type of 'assert_' (line 35)
    assert__239701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 35)
    assert__call_result_239707 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), assert__239701, *[result_eq_239704, str_239705], **kwargs_239706)
    
    
    # ################# End of 'test_aliasing(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_aliasing' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_239708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_239708)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_aliasing'
    return stypy_return_type_239708

# Assigning a type to the variable 'test_aliasing' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'test_aliasing', test_aliasing)

@norecursion
def test_aliasing2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_aliasing2'
    module_type_store = module_type_store.open_function_context('test_aliasing2', 38, 0, False)
    
    # Passed parameters checking function
    test_aliasing2.stypy_localization = localization
    test_aliasing2.stypy_type_of_self = None
    test_aliasing2.stypy_type_store = module_type_store
    test_aliasing2.stypy_function_name = 'test_aliasing2'
    test_aliasing2.stypy_param_names_list = []
    test_aliasing2.stypy_varargs_param_name = None
    test_aliasing2.stypy_kwargs_param_name = None
    test_aliasing2.stypy_call_defaults = defaults
    test_aliasing2.stypy_call_varargs = varargs
    test_aliasing2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_aliasing2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_aliasing2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_aliasing2(...)' code ##################

    
    # Assigning a Call to a Name (line 39):
    
    # Call to array(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_239711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    int_239712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_239711, int_239712)
    # Adding element type (line 39)
    int_239713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_239711, int_239713)
    
    # Processing the call keyword arguments (line 39)
    kwargs_239714 = {}
    # Getting the type of 'np' (line 39)
    np_239709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 39)
    array_239710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), np_239709, 'array')
    # Calling array(args, kwargs) (line 39)
    array_call_result_239715 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), array_239710, *[list_239711], **kwargs_239714)
    
    # Assigning a type to the variable 'c' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'c', array_call_result_239715)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to array(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_239718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_239719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    int_239720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 21), list_239719, int_239720)
    # Adding element type (line 40)
    int_239721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 21), list_239719, int_239721)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), list_239718, list_239719)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_239722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    int_239723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 29), list_239722, int_239723)
    # Adding element type (line 40)
    int_239724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 29), list_239722, int_239724)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), list_239718, list_239722)
    
    # Processing the call keyword arguments (line 40)
    kwargs_239725 = {}
    # Getting the type of 'np' (line 40)
    np_239716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 40)
    array_239717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 11), np_239716, 'array')
    # Calling array(args, kwargs) (line 40)
    array_call_result_239726 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), array_239717, *[list_239718], **kwargs_239725)
    
    # Assigning a type to the variable 'A_ub' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'A_ub', array_call_result_239726)
    
    # Assigning a Call to a Name (line 41):
    
    # Call to array(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_239729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_239730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    int_239731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 21), list_239730, int_239731)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), list_239729, list_239730)
    # Adding element type (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_239732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    int_239733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 26), list_239732, int_239733)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 20), list_239729, list_239732)
    
    # Processing the call keyword arguments (line 41)
    kwargs_239734 = {}
    # Getting the type of 'np' (line 41)
    np_239727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 41)
    array_239728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 11), np_239727, 'array')
    # Calling array(args, kwargs) (line 41)
    array_call_result_239735 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), array_239728, *[list_239729], **kwargs_239734)
    
    # Assigning a type to the variable 'b_ub' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'b_ub', array_call_result_239735)
    
    # Assigning a Call to a Name (line 42):
    
    # Call to array(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_239738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_239739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    int_239740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), list_239739, int_239740)
    # Adding element type (line 42)
    int_239741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), list_239739, int_239741)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 20), list_239738, list_239739)
    
    # Processing the call keyword arguments (line 42)
    kwargs_239742 = {}
    # Getting the type of 'np' (line 42)
    np_239736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 42)
    array_239737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), np_239736, 'array')
    # Calling array(args, kwargs) (line 42)
    array_call_result_239743 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), array_239737, *[list_239738], **kwargs_239742)
    
    # Assigning a type to the variable 'A_eq' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'A_eq', array_call_result_239743)
    
    # Assigning a Call to a Name (line 43):
    
    # Call to array(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_239746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    # Adding element type (line 43)
    int_239747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), list_239746, int_239747)
    
    # Processing the call keyword arguments (line 43)
    kwargs_239748 = {}
    # Getting the type of 'np' (line 43)
    np_239744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 43)
    array_239745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 11), np_239744, 'array')
    # Calling array(args, kwargs) (line 43)
    array_call_result_239749 = invoke(stypy.reporting.localization.Localization(__file__, 43, 11), array_239745, *[list_239746], **kwargs_239748)
    
    # Assigning a type to the variable 'b_eq' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'b_eq', array_call_result_239749)
    
    # Assigning a List to a Name (line 44):
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_239750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_239751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    
    # Getting the type of 'np' (line 44)
    np_239752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'np')
    # Obtaining the member 'inf' of a type (line 44)
    inf_239753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), np_239752, 'inf')
    # Applying the 'usub' unary operator (line 44)
    result___neg___239754 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 15), 'usub', inf_239753)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 15), tuple_239751, result___neg___239754)
    # Adding element type (line 44)
    # Getting the type of 'np' (line 44)
    np_239755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'np')
    # Obtaining the member 'inf' of a type (line 44)
    inf_239756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), np_239755, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 15), tuple_239751, inf_239756)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 13), list_239750, tuple_239751)
    # Adding element type (line 44)
    
    # Obtaining an instance of the builtin type 'tuple' (line 44)
    tuple_239757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'None' (line 44)
    None_239758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), tuple_239757, None_239758)
    # Adding element type (line 44)
    int_239759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), tuple_239757, int_239759)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 13), list_239750, tuple_239757)
    
    # Assigning a type to the variable 'bounds' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'bounds', list_239750)
    
    # Assigning a Call to a Name (line 46):
    
    # Call to copy(...): (line 46)
    # Processing the call keyword arguments (line 46)
    kwargs_239762 = {}
    # Getting the type of 'c' (line 46)
    c_239760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'c', False)
    # Obtaining the member 'copy' of a type (line 46)
    copy_239761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), c_239760, 'copy')
    # Calling copy(args, kwargs) (line 46)
    copy_call_result_239763 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), copy_239761, *[], **kwargs_239762)
    
    # Assigning a type to the variable 'c_copy' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'c_copy', copy_call_result_239763)
    
    # Assigning a Call to a Name (line 47):
    
    # Call to copy(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_239766 = {}
    # Getting the type of 'A_ub' (line 47)
    A_ub_239764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'A_ub', False)
    # Obtaining the member 'copy' of a type (line 47)
    copy_239765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), A_ub_239764, 'copy')
    # Calling copy(args, kwargs) (line 47)
    copy_call_result_239767 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), copy_239765, *[], **kwargs_239766)
    
    # Assigning a type to the variable 'A_ub_copy' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'A_ub_copy', copy_call_result_239767)
    
    # Assigning a Call to a Name (line 48):
    
    # Call to copy(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_239770 = {}
    # Getting the type of 'b_ub' (line 48)
    b_ub_239768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'b_ub', False)
    # Obtaining the member 'copy' of a type (line 48)
    copy_239769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), b_ub_239768, 'copy')
    # Calling copy(args, kwargs) (line 48)
    copy_call_result_239771 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), copy_239769, *[], **kwargs_239770)
    
    # Assigning a type to the variable 'b_ub_copy' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'b_ub_copy', copy_call_result_239771)
    
    # Assigning a Call to a Name (line 49):
    
    # Call to copy(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_239774 = {}
    # Getting the type of 'A_eq' (line 49)
    A_eq_239772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'A_eq', False)
    # Obtaining the member 'copy' of a type (line 49)
    copy_239773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), A_eq_239772, 'copy')
    # Calling copy(args, kwargs) (line 49)
    copy_call_result_239775 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), copy_239773, *[], **kwargs_239774)
    
    # Assigning a type to the variable 'A_eq_copy' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'A_eq_copy', copy_call_result_239775)
    
    # Assigning a Call to a Name (line 50):
    
    # Call to copy(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_239778 = {}
    # Getting the type of 'b_eq' (line 50)
    b_eq_239776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'b_eq', False)
    # Obtaining the member 'copy' of a type (line 50)
    copy_239777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 16), b_eq_239776, 'copy')
    # Calling copy(args, kwargs) (line 50)
    copy_call_result_239779 = invoke(stypy.reporting.localization.Localization(__file__, 50, 16), copy_239777, *[], **kwargs_239778)
    
    # Assigning a type to the variable 'b_eq_copy' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'b_eq_copy', copy_call_result_239779)
    
    # Assigning a Call to a Name (line 51):
    
    # Call to deepcopy(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'bounds' (line 51)
    bounds_239781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 27), 'bounds', False)
    # Processing the call keyword arguments (line 51)
    kwargs_239782 = {}
    # Getting the type of 'deepcopy' (line 51)
    deepcopy_239780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'deepcopy', False)
    # Calling deepcopy(args, kwargs) (line 51)
    deepcopy_call_result_239783 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), deepcopy_239780, *[bounds_239781], **kwargs_239782)
    
    # Assigning a type to the variable 'bounds_copy' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'bounds_copy', deepcopy_call_result_239783)
    
    # Call to _clean_inputs(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'c' (line 53)
    c_239785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'c', False)
    # Getting the type of 'A_ub' (line 53)
    A_ub_239786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'A_ub', False)
    # Getting the type of 'b_ub' (line 53)
    b_ub_239787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'b_ub', False)
    # Getting the type of 'A_eq' (line 53)
    A_eq_239788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'A_eq', False)
    # Getting the type of 'b_eq' (line 53)
    b_eq_239789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), 'b_eq', False)
    # Getting the type of 'bounds' (line 53)
    bounds_239790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 45), 'bounds', False)
    # Processing the call keyword arguments (line 53)
    kwargs_239791 = {}
    # Getting the type of '_clean_inputs' (line 53)
    _clean_inputs_239784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 53)
    _clean_inputs_call_result_239792 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), _clean_inputs_239784, *[c_239785, A_ub_239786, b_ub_239787, A_eq_239788, b_eq_239789, bounds_239790], **kwargs_239791)
    
    
    # Call to assert_allclose(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'c' (line 55)
    c_239794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'c', False)
    # Getting the type of 'c_copy' (line 55)
    c_copy_239795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'c_copy', False)
    # Processing the call keyword arguments (line 55)
    str_239796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 39), 'str', 'c modified by _clean_inputs')
    keyword_239797 = str_239796
    kwargs_239798 = {'err_msg': keyword_239797}
    # Getting the type of 'assert_allclose' (line 55)
    assert_allclose_239793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 55)
    assert_allclose_call_result_239799 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), assert_allclose_239793, *[c_239794, c_copy_239795], **kwargs_239798)
    
    
    # Call to assert_allclose(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A_ub' (line 56)
    A_ub_239801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'A_ub', False)
    # Getting the type of 'A_ub_copy' (line 56)
    A_ub_copy_239802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'A_ub_copy', False)
    # Processing the call keyword arguments (line 56)
    str_239803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 45), 'str', 'A_ub modified by _clean_inputs')
    keyword_239804 = str_239803
    kwargs_239805 = {'err_msg': keyword_239804}
    # Getting the type of 'assert_allclose' (line 56)
    assert_allclose_239800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 56)
    assert_allclose_call_result_239806 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), assert_allclose_239800, *[A_ub_239801, A_ub_copy_239802], **kwargs_239805)
    
    
    # Call to assert_allclose(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'b_ub' (line 57)
    b_ub_239808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'b_ub', False)
    # Getting the type of 'b_ub_copy' (line 57)
    b_ub_copy_239809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'b_ub_copy', False)
    # Processing the call keyword arguments (line 57)
    str_239810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 45), 'str', 'b_ub modified by _clean_inputs')
    keyword_239811 = str_239810
    kwargs_239812 = {'err_msg': keyword_239811}
    # Getting the type of 'assert_allclose' (line 57)
    assert_allclose_239807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 57)
    assert_allclose_call_result_239813 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), assert_allclose_239807, *[b_ub_239808, b_ub_copy_239809], **kwargs_239812)
    
    
    # Call to assert_allclose(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'A_eq' (line 58)
    A_eq_239815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'A_eq', False)
    # Getting the type of 'A_eq_copy' (line 58)
    A_eq_copy_239816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'A_eq_copy', False)
    # Processing the call keyword arguments (line 58)
    str_239817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 45), 'str', 'A_eq modified by _clean_inputs')
    keyword_239818 = str_239817
    kwargs_239819 = {'err_msg': keyword_239818}
    # Getting the type of 'assert_allclose' (line 58)
    assert_allclose_239814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 58)
    assert_allclose_call_result_239820 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), assert_allclose_239814, *[A_eq_239815, A_eq_copy_239816], **kwargs_239819)
    
    
    # Call to assert_allclose(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'b_eq' (line 59)
    b_eq_239822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'b_eq', False)
    # Getting the type of 'b_eq_copy' (line 59)
    b_eq_copy_239823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'b_eq_copy', False)
    # Processing the call keyword arguments (line 59)
    str_239824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 45), 'str', 'b_eq modified by _clean_inputs')
    keyword_239825 = str_239824
    kwargs_239826 = {'err_msg': keyword_239825}
    # Getting the type of 'assert_allclose' (line 59)
    assert_allclose_239821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 59)
    assert_allclose_call_result_239827 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), assert_allclose_239821, *[b_eq_239822, b_eq_copy_239823], **kwargs_239826)
    
    
    # Call to assert_(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Getting the type of 'bounds' (line 60)
    bounds_239829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'bounds', False)
    # Getting the type of 'bounds_copy' (line 60)
    bounds_copy_239830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'bounds_copy', False)
    # Applying the binary operator '==' (line 60)
    result_eq_239831 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 12), '==', bounds_239829, bounds_copy_239830)
    
    str_239832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 35), 'str', 'bounds modified by _clean_inputs')
    # Processing the call keyword arguments (line 60)
    kwargs_239833 = {}
    # Getting the type of 'assert_' (line 60)
    assert__239828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 60)
    assert__call_result_239834 = invoke(stypy.reporting.localization.Localization(__file__, 60, 4), assert__239828, *[result_eq_239831, str_239832], **kwargs_239833)
    
    
    # ################# End of 'test_aliasing2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_aliasing2' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_239835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_239835)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_aliasing2'
    return stypy_return_type_239835

# Assigning a type to the variable 'test_aliasing2' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'test_aliasing2', test_aliasing2)

@norecursion
def test_missing_inputs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_missing_inputs'
    module_type_store = module_type_store.open_function_context('test_missing_inputs', 63, 0, False)
    
    # Passed parameters checking function
    test_missing_inputs.stypy_localization = localization
    test_missing_inputs.stypy_type_of_self = None
    test_missing_inputs.stypy_type_store = module_type_store
    test_missing_inputs.stypy_function_name = 'test_missing_inputs'
    test_missing_inputs.stypy_param_names_list = []
    test_missing_inputs.stypy_varargs_param_name = None
    test_missing_inputs.stypy_kwargs_param_name = None
    test_missing_inputs.stypy_call_defaults = defaults
    test_missing_inputs.stypy_call_varargs = varargs
    test_missing_inputs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_missing_inputs', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_missing_inputs', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_missing_inputs(...)' code ##################

    
    # Assigning a List to a Name (line 64):
    
    # Obtaining an instance of the builtin type 'list' (line 64)
    list_239836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 64)
    # Adding element type (line 64)
    int_239837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 8), list_239836, int_239837)
    # Adding element type (line 64)
    int_239838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 8), list_239836, int_239838)
    
    # Assigning a type to the variable 'c' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'c', list_239836)
    
    # Assigning a Call to a Name (line 65):
    
    # Call to array(...): (line 65)
    # Processing the call arguments (line 65)
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_239841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_239842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    int_239843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 21), list_239842, int_239843)
    # Adding element type (line 65)
    int_239844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 21), list_239842, int_239844)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), list_239841, list_239842)
    # Adding element type (line 65)
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_239845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    int_239846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 29), list_239845, int_239846)
    # Adding element type (line 65)
    int_239847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 29), list_239845, int_239847)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), list_239841, list_239845)
    
    # Processing the call keyword arguments (line 65)
    kwargs_239848 = {}
    # Getting the type of 'np' (line 65)
    np_239839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 65)
    array_239840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), np_239839, 'array')
    # Calling array(args, kwargs) (line 65)
    array_call_result_239849 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), array_239840, *[list_239841], **kwargs_239848)
    
    # Assigning a type to the variable 'A_ub' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'A_ub', array_call_result_239849)
    
    # Assigning a Call to a Name (line 66):
    
    # Call to array(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Obtaining an instance of the builtin type 'list' (line 66)
    list_239852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 66)
    # Adding element type (line 66)
    int_239853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), list_239852, int_239853)
    # Adding element type (line 66)
    int_239854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), list_239852, int_239854)
    
    # Processing the call keyword arguments (line 66)
    kwargs_239855 = {}
    # Getting the type of 'np' (line 66)
    np_239850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 66)
    array_239851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), np_239850, 'array')
    # Calling array(args, kwargs) (line 66)
    array_call_result_239856 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), array_239851, *[list_239852], **kwargs_239855)
    
    # Assigning a type to the variable 'b_ub' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'b_ub', array_call_result_239856)
    
    # Assigning a Call to a Name (line 67):
    
    # Call to array(...): (line 67)
    # Processing the call arguments (line 67)
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_239859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_239860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    int_239861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 21), list_239860, int_239861)
    # Adding element type (line 67)
    int_239862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 21), list_239860, int_239862)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), list_239859, list_239860)
    # Adding element type (line 67)
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_239863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    int_239864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 29), list_239863, int_239864)
    # Adding element type (line 67)
    int_239865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 29), list_239863, int_239865)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), list_239859, list_239863)
    
    # Processing the call keyword arguments (line 67)
    kwargs_239866 = {}
    # Getting the type of 'np' (line 67)
    np_239857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 67)
    array_239858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 11), np_239857, 'array')
    # Calling array(args, kwargs) (line 67)
    array_call_result_239867 = invoke(stypy.reporting.localization.Localization(__file__, 67, 11), array_239858, *[list_239859], **kwargs_239866)
    
    # Assigning a type to the variable 'A_eq' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'A_eq', array_call_result_239867)
    
    # Assigning a Call to a Name (line 68):
    
    # Call to array(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_239870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    int_239871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 20), list_239870, int_239871)
    # Adding element type (line 68)
    int_239872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 20), list_239870, int_239872)
    
    # Processing the call keyword arguments (line 68)
    kwargs_239873 = {}
    # Getting the type of 'np' (line 68)
    np_239868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 68)
    array_239869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), np_239868, 'array')
    # Calling array(args, kwargs) (line 68)
    array_call_result_239874 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), array_239869, *[list_239870], **kwargs_239873)
    
    # Assigning a type to the variable 'b_eq' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'b_eq', array_call_result_239874)
    
    # Call to assert_raises(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'TypeError' (line 70)
    TypeError_239876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 70)
    _clean_inputs_239877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), '_clean_inputs', False)
    # Processing the call keyword arguments (line 70)
    kwargs_239878 = {}
    # Getting the type of 'assert_raises' (line 70)
    assert_raises_239875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 70)
    assert_raises_call_result_239879 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), assert_raises_239875, *[TypeError_239876, _clean_inputs_239877], **kwargs_239878)
    
    
    # Call to assert_raises(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'TypeError' (line 71)
    TypeError_239881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 71)
    _clean_inputs_239882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 29), '_clean_inputs', False)
    # Processing the call keyword arguments (line 71)
    # Getting the type of 'None' (line 71)
    None_239883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 46), 'None', False)
    keyword_239884 = None_239883
    kwargs_239885 = {'c': keyword_239884}
    # Getting the type of 'assert_raises' (line 71)
    assert_raises_239880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 71)
    assert_raises_call_result_239886 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), assert_raises_239880, *[TypeError_239881, _clean_inputs_239882], **kwargs_239885)
    
    
    # Call to assert_raises(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'ValueError' (line 72)
    ValueError_239888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 72)
    _clean_inputs_239889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 72)
    # Getting the type of 'c' (line 72)
    c_239890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 47), 'c', False)
    keyword_239891 = c_239890
    # Getting the type of 'A_ub' (line 72)
    A_ub_239892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 55), 'A_ub', False)
    keyword_239893 = A_ub_239892
    kwargs_239894 = {'c': keyword_239891, 'A_ub': keyword_239893}
    # Getting the type of 'assert_raises' (line 72)
    assert_raises_239887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 72)
    assert_raises_call_result_239895 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), assert_raises_239887, *[ValueError_239888, _clean_inputs_239889], **kwargs_239894)
    
    
    # Call to assert_raises(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'ValueError' (line 73)
    ValueError_239897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 73)
    _clean_inputs_239898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 73)
    # Getting the type of 'c' (line 73)
    c_239899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 47), 'c', False)
    keyword_239900 = c_239899
    # Getting the type of 'A_ub' (line 73)
    A_ub_239901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 55), 'A_ub', False)
    keyword_239902 = A_ub_239901
    # Getting the type of 'None' (line 73)
    None_239903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 66), 'None', False)
    keyword_239904 = None_239903
    kwargs_239905 = {'c': keyword_239900, 'b_ub': keyword_239904, 'A_ub': keyword_239902}
    # Getting the type of 'assert_raises' (line 73)
    assert_raises_239896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 73)
    assert_raises_call_result_239906 = invoke(stypy.reporting.localization.Localization(__file__, 73, 4), assert_raises_239896, *[ValueError_239897, _clean_inputs_239898], **kwargs_239905)
    
    
    # Call to assert_raises(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'ValueError' (line 74)
    ValueError_239908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 74)
    _clean_inputs_239909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 74)
    # Getting the type of 'c' (line 74)
    c_239910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 47), 'c', False)
    keyword_239911 = c_239910
    # Getting the type of 'b_ub' (line 74)
    b_ub_239912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 55), 'b_ub', False)
    keyword_239913 = b_ub_239912
    kwargs_239914 = {'c': keyword_239911, 'b_ub': keyword_239913}
    # Getting the type of 'assert_raises' (line 74)
    assert_raises_239907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 74)
    assert_raises_call_result_239915 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), assert_raises_239907, *[ValueError_239908, _clean_inputs_239909], **kwargs_239914)
    
    
    # Call to assert_raises(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'ValueError' (line 75)
    ValueError_239917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 75)
    _clean_inputs_239918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 75)
    # Getting the type of 'c' (line 75)
    c_239919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 47), 'c', False)
    keyword_239920 = c_239919
    # Getting the type of 'None' (line 75)
    None_239921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 55), 'None', False)
    keyword_239922 = None_239921
    # Getting the type of 'b_ub' (line 75)
    b_ub_239923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 66), 'b_ub', False)
    keyword_239924 = b_ub_239923
    kwargs_239925 = {'c': keyword_239920, 'b_ub': keyword_239924, 'A_ub': keyword_239922}
    # Getting the type of 'assert_raises' (line 75)
    assert_raises_239916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 75)
    assert_raises_call_result_239926 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), assert_raises_239916, *[ValueError_239917, _clean_inputs_239918], **kwargs_239925)
    
    
    # Call to assert_raises(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'ValueError' (line 76)
    ValueError_239928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 76)
    _clean_inputs_239929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 76)
    # Getting the type of 'c' (line 76)
    c_239930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 47), 'c', False)
    keyword_239931 = c_239930
    # Getting the type of 'A_eq' (line 76)
    A_eq_239932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 55), 'A_eq', False)
    keyword_239933 = A_eq_239932
    kwargs_239934 = {'c': keyword_239931, 'A_eq': keyword_239933}
    # Getting the type of 'assert_raises' (line 76)
    assert_raises_239927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 76)
    assert_raises_call_result_239935 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), assert_raises_239927, *[ValueError_239928, _clean_inputs_239929], **kwargs_239934)
    
    
    # Call to assert_raises(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'ValueError' (line 77)
    ValueError_239937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 77)
    _clean_inputs_239938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 77)
    # Getting the type of 'c' (line 77)
    c_239939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 47), 'c', False)
    keyword_239940 = c_239939
    # Getting the type of 'A_eq' (line 77)
    A_eq_239941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 55), 'A_eq', False)
    keyword_239942 = A_eq_239941
    # Getting the type of 'None' (line 77)
    None_239943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 66), 'None', False)
    keyword_239944 = None_239943
    kwargs_239945 = {'c': keyword_239940, 'A_eq': keyword_239942, 'b_eq': keyword_239944}
    # Getting the type of 'assert_raises' (line 77)
    assert_raises_239936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 77)
    assert_raises_call_result_239946 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), assert_raises_239936, *[ValueError_239937, _clean_inputs_239938], **kwargs_239945)
    
    
    # Call to assert_raises(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'ValueError' (line 78)
    ValueError_239948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 78)
    _clean_inputs_239949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 78)
    # Getting the type of 'c' (line 78)
    c_239950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 47), 'c', False)
    keyword_239951 = c_239950
    # Getting the type of 'b_eq' (line 78)
    b_eq_239952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 55), 'b_eq', False)
    keyword_239953 = b_eq_239952
    kwargs_239954 = {'c': keyword_239951, 'b_eq': keyword_239953}
    # Getting the type of 'assert_raises' (line 78)
    assert_raises_239947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 78)
    assert_raises_call_result_239955 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), assert_raises_239947, *[ValueError_239948, _clean_inputs_239949], **kwargs_239954)
    
    
    # Call to assert_raises(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'ValueError' (line 79)
    ValueError_239957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 79)
    _clean_inputs_239958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 79)
    # Getting the type of 'c' (line 79)
    c_239959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 47), 'c', False)
    keyword_239960 = c_239959
    # Getting the type of 'None' (line 79)
    None_239961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 55), 'None', False)
    keyword_239962 = None_239961
    # Getting the type of 'b_eq' (line 79)
    b_eq_239963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 66), 'b_eq', False)
    keyword_239964 = b_eq_239963
    kwargs_239965 = {'c': keyword_239960, 'A_eq': keyword_239962, 'b_eq': keyword_239964}
    # Getting the type of 'assert_raises' (line 79)
    assert_raises_239956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 79)
    assert_raises_call_result_239966 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), assert_raises_239956, *[ValueError_239957, _clean_inputs_239958], **kwargs_239965)
    
    
    # ################# End of 'test_missing_inputs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_missing_inputs' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_239967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_239967)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_missing_inputs'
    return stypy_return_type_239967

# Assigning a type to the variable 'test_missing_inputs' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'test_missing_inputs', test_missing_inputs)

@norecursion
def test_too_many_dimensions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_too_many_dimensions'
    module_type_store = module_type_store.open_function_context('test_too_many_dimensions', 82, 0, False)
    
    # Passed parameters checking function
    test_too_many_dimensions.stypy_localization = localization
    test_too_many_dimensions.stypy_type_of_self = None
    test_too_many_dimensions.stypy_type_store = module_type_store
    test_too_many_dimensions.stypy_function_name = 'test_too_many_dimensions'
    test_too_many_dimensions.stypy_param_names_list = []
    test_too_many_dimensions.stypy_varargs_param_name = None
    test_too_many_dimensions.stypy_kwargs_param_name = None
    test_too_many_dimensions.stypy_call_defaults = defaults
    test_too_many_dimensions.stypy_call_varargs = varargs
    test_too_many_dimensions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_too_many_dimensions', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_too_many_dimensions', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_too_many_dimensions(...)' code ##################

    
    # Assigning a List to a Name (line 83):
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_239968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    int_239969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 9), list_239968, int_239969)
    # Adding element type (line 83)
    int_239970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 9), list_239968, int_239970)
    # Adding element type (line 83)
    int_239971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 9), list_239968, int_239971)
    # Adding element type (line 83)
    int_239972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 9), list_239968, int_239972)
    
    # Assigning a type to the variable 'cb' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'cb', list_239968)
    
    # Assigning a Call to a Name (line 84):
    
    # Call to rand(...): (line 84)
    # Processing the call arguments (line 84)
    int_239976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'int')
    int_239977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 26), 'int')
    # Processing the call keyword arguments (line 84)
    kwargs_239978 = {}
    # Getting the type of 'np' (line 84)
    np_239973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 84)
    random_239974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), np_239973, 'random')
    # Obtaining the member 'rand' of a type (line 84)
    rand_239975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), random_239974, 'rand')
    # Calling rand(args, kwargs) (line 84)
    rand_call_result_239979 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), rand_239975, *[int_239976, int_239977], **kwargs_239978)
    
    # Assigning a type to the variable 'A' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'A', rand_call_result_239979)
    
    # Assigning a List to a Name (line 85):
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_239980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_239981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    int_239982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), list_239981, int_239982)
    # Adding element type (line 85)
    int_239983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), list_239981, int_239983)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 12), list_239980, list_239981)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_239984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    int_239985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_239984, int_239985)
    # Adding element type (line 85)
    int_239986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), list_239984, int_239986)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 12), list_239980, list_239984)
    
    # Assigning a type to the variable 'bad2D' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'bad2D', list_239980)
    
    # Assigning a Call to a Name (line 86):
    
    # Call to rand(...): (line 86)
    # Processing the call arguments (line 86)
    int_239990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'int')
    int_239991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'int')
    int_239992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 33), 'int')
    # Processing the call keyword arguments (line 86)
    kwargs_239993 = {}
    # Getting the type of 'np' (line 86)
    np_239987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 86)
    random_239988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), np_239987, 'random')
    # Obtaining the member 'rand' of a type (line 86)
    rand_239989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), random_239988, 'rand')
    # Calling rand(args, kwargs) (line 86)
    rand_call_result_239994 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), rand_239989, *[int_239990, int_239991, int_239992], **kwargs_239993)
    
    # Assigning a type to the variable 'bad3D' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'bad3D', rand_call_result_239994)
    
    # Call to assert_raises(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'ValueError' (line 87)
    ValueError_239996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 87)
    _clean_inputs_239997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 87)
    # Getting the type of 'bad2D' (line 87)
    bad2D_239998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 47), 'bad2D', False)
    keyword_239999 = bad2D_239998
    # Getting the type of 'A' (line 87)
    A_240000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 59), 'A', False)
    keyword_240001 = A_240000
    # Getting the type of 'cb' (line 87)
    cb_240002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 67), 'cb', False)
    keyword_240003 = cb_240002
    kwargs_240004 = {'c': keyword_239999, 'b_ub': keyword_240003, 'A_ub': keyword_240001}
    # Getting the type of 'assert_raises' (line 87)
    assert_raises_239995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 87)
    assert_raises_call_result_240005 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), assert_raises_239995, *[ValueError_239996, _clean_inputs_239997], **kwargs_240004)
    
    
    # Call to assert_raises(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'ValueError' (line 88)
    ValueError_240007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 88)
    _clean_inputs_240008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 88)
    # Getting the type of 'cb' (line 88)
    cb_240009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 47), 'cb', False)
    keyword_240010 = cb_240009
    # Getting the type of 'bad3D' (line 88)
    bad3D_240011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'bad3D', False)
    keyword_240012 = bad3D_240011
    # Getting the type of 'cb' (line 88)
    cb_240013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 68), 'cb', False)
    keyword_240014 = cb_240013
    kwargs_240015 = {'c': keyword_240010, 'b_ub': keyword_240014, 'A_ub': keyword_240012}
    # Getting the type of 'assert_raises' (line 88)
    assert_raises_240006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 88)
    assert_raises_call_result_240016 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), assert_raises_240006, *[ValueError_240007, _clean_inputs_240008], **kwargs_240015)
    
    
    # Call to assert_raises(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'ValueError' (line 89)
    ValueError_240018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 89)
    _clean_inputs_240019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 89)
    # Getting the type of 'cb' (line 89)
    cb_240020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 47), 'cb', False)
    keyword_240021 = cb_240020
    # Getting the type of 'A' (line 89)
    A_240022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 56), 'A', False)
    keyword_240023 = A_240022
    # Getting the type of 'bad2D' (line 89)
    bad2D_240024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 64), 'bad2D', False)
    keyword_240025 = bad2D_240024
    kwargs_240026 = {'c': keyword_240021, 'b_ub': keyword_240025, 'A_ub': keyword_240023}
    # Getting the type of 'assert_raises' (line 89)
    assert_raises_240017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 89)
    assert_raises_call_result_240027 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), assert_raises_240017, *[ValueError_240018, _clean_inputs_240019], **kwargs_240026)
    
    
    # Call to assert_raises(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'ValueError' (line 90)
    ValueError_240029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 90)
    _clean_inputs_240030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'cb' (line 90)
    cb_240031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 47), 'cb', False)
    keyword_240032 = cb_240031
    # Getting the type of 'bad3D' (line 90)
    bad3D_240033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 56), 'bad3D', False)
    keyword_240034 = bad3D_240033
    # Getting the type of 'cb' (line 90)
    cb_240035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 68), 'cb', False)
    keyword_240036 = cb_240035
    kwargs_240037 = {'c': keyword_240032, 'A_eq': keyword_240034, 'b_eq': keyword_240036}
    # Getting the type of 'assert_raises' (line 90)
    assert_raises_240028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 90)
    assert_raises_call_result_240038 = invoke(stypy.reporting.localization.Localization(__file__, 90, 4), assert_raises_240028, *[ValueError_240029, _clean_inputs_240030], **kwargs_240037)
    
    
    # Call to assert_raises(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'ValueError' (line 91)
    ValueError_240040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 91)
    _clean_inputs_240041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 91)
    # Getting the type of 'cb' (line 91)
    cb_240042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'cb', False)
    keyword_240043 = cb_240042
    # Getting the type of 'A' (line 91)
    A_240044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 56), 'A', False)
    keyword_240045 = A_240044
    # Getting the type of 'bad2D' (line 91)
    bad2D_240046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 64), 'bad2D', False)
    keyword_240047 = bad2D_240046
    kwargs_240048 = {'c': keyword_240043, 'A_eq': keyword_240045, 'b_eq': keyword_240047}
    # Getting the type of 'assert_raises' (line 91)
    assert_raises_240039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 91)
    assert_raises_call_result_240049 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), assert_raises_240039, *[ValueError_240040, _clean_inputs_240041], **kwargs_240048)
    
    
    # ################# End of 'test_too_many_dimensions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_too_many_dimensions' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_240050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_240050)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_too_many_dimensions'
    return stypy_return_type_240050

# Assigning a type to the variable 'test_too_many_dimensions' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'test_too_many_dimensions', test_too_many_dimensions)

@norecursion
def test_too_few_dimensions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_too_few_dimensions'
    module_type_store = module_type_store.open_function_context('test_too_few_dimensions', 94, 0, False)
    
    # Passed parameters checking function
    test_too_few_dimensions.stypy_localization = localization
    test_too_few_dimensions.stypy_type_of_self = None
    test_too_few_dimensions.stypy_type_store = module_type_store
    test_too_few_dimensions.stypy_function_name = 'test_too_few_dimensions'
    test_too_few_dimensions.stypy_param_names_list = []
    test_too_few_dimensions.stypy_varargs_param_name = None
    test_too_few_dimensions.stypy_kwargs_param_name = None
    test_too_few_dimensions.stypy_call_defaults = defaults
    test_too_few_dimensions.stypy_call_varargs = varargs
    test_too_few_dimensions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_too_few_dimensions', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_too_few_dimensions', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_too_few_dimensions(...)' code ##################

    
    # Assigning a Call to a Name (line 95):
    
    # Call to ravel(...): (line 95)
    # Processing the call keyword arguments (line 95)
    kwargs_240059 = {}
    
    # Call to rand(...): (line 95)
    # Processing the call arguments (line 95)
    int_240054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'int')
    int_240055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'int')
    # Processing the call keyword arguments (line 95)
    kwargs_240056 = {}
    # Getting the type of 'np' (line 95)
    np_240051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 10), 'np', False)
    # Obtaining the member 'random' of a type (line 95)
    random_240052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 10), np_240051, 'random')
    # Obtaining the member 'rand' of a type (line 95)
    rand_240053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 10), random_240052, 'rand')
    # Calling rand(args, kwargs) (line 95)
    rand_call_result_240057 = invoke(stypy.reporting.localization.Localization(__file__, 95, 10), rand_240053, *[int_240054, int_240055], **kwargs_240056)
    
    # Obtaining the member 'ravel' of a type (line 95)
    ravel_240058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 10), rand_call_result_240057, 'ravel')
    # Calling ravel(args, kwargs) (line 95)
    ravel_call_result_240060 = invoke(stypy.reporting.localization.Localization(__file__, 95, 10), ravel_240058, *[], **kwargs_240059)
    
    # Assigning a type to the variable 'bad' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'bad', ravel_call_result_240060)
    
    # Assigning a Call to a Name (line 96):
    
    # Call to rand(...): (line 96)
    # Processing the call arguments (line 96)
    int_240064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 24), 'int')
    # Processing the call keyword arguments (line 96)
    kwargs_240065 = {}
    # Getting the type of 'np' (line 96)
    np_240061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 96)
    random_240062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 9), np_240061, 'random')
    # Obtaining the member 'rand' of a type (line 96)
    rand_240063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 9), random_240062, 'rand')
    # Calling rand(args, kwargs) (line 96)
    rand_call_result_240066 = invoke(stypy.reporting.localization.Localization(__file__, 96, 9), rand_240063, *[int_240064], **kwargs_240065)
    
    # Assigning a type to the variable 'cb' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'cb', rand_call_result_240066)
    
    # Call to assert_raises(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'ValueError' (line 97)
    ValueError_240068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 97)
    _clean_inputs_240069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'cb' (line 97)
    cb_240070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 47), 'cb', False)
    keyword_240071 = cb_240070
    # Getting the type of 'bad' (line 97)
    bad_240072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 56), 'bad', False)
    keyword_240073 = bad_240072
    # Getting the type of 'cb' (line 97)
    cb_240074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 66), 'cb', False)
    keyword_240075 = cb_240074
    kwargs_240076 = {'c': keyword_240071, 'b_ub': keyword_240075, 'A_ub': keyword_240073}
    # Getting the type of 'assert_raises' (line 97)
    assert_raises_240067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 97)
    assert_raises_call_result_240077 = invoke(stypy.reporting.localization.Localization(__file__, 97, 4), assert_raises_240067, *[ValueError_240068, _clean_inputs_240069], **kwargs_240076)
    
    
    # Call to assert_raises(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'ValueError' (line 98)
    ValueError_240079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 98)
    _clean_inputs_240080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'cb' (line 98)
    cb_240081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'cb', False)
    keyword_240082 = cb_240081
    # Getting the type of 'bad' (line 98)
    bad_240083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 56), 'bad', False)
    keyword_240084 = bad_240083
    # Getting the type of 'cb' (line 98)
    cb_240085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 66), 'cb', False)
    keyword_240086 = cb_240085
    kwargs_240087 = {'c': keyword_240082, 'A_eq': keyword_240084, 'b_eq': keyword_240086}
    # Getting the type of 'assert_raises' (line 98)
    assert_raises_240078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 98)
    assert_raises_call_result_240088 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), assert_raises_240078, *[ValueError_240079, _clean_inputs_240080], **kwargs_240087)
    
    
    # ################# End of 'test_too_few_dimensions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_too_few_dimensions' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_240089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_240089)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_too_few_dimensions'
    return stypy_return_type_240089

# Assigning a type to the variable 'test_too_few_dimensions' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'test_too_few_dimensions', test_too_few_dimensions)

@norecursion
def test_inconsistent_dimensions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_inconsistent_dimensions'
    module_type_store = module_type_store.open_function_context('test_inconsistent_dimensions', 101, 0, False)
    
    # Passed parameters checking function
    test_inconsistent_dimensions.stypy_localization = localization
    test_inconsistent_dimensions.stypy_type_of_self = None
    test_inconsistent_dimensions.stypy_type_store = module_type_store
    test_inconsistent_dimensions.stypy_function_name = 'test_inconsistent_dimensions'
    test_inconsistent_dimensions.stypy_param_names_list = []
    test_inconsistent_dimensions.stypy_varargs_param_name = None
    test_inconsistent_dimensions.stypy_kwargs_param_name = None
    test_inconsistent_dimensions.stypy_call_defaults = defaults
    test_inconsistent_dimensions.stypy_call_varargs = varargs
    test_inconsistent_dimensions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_inconsistent_dimensions', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_inconsistent_dimensions', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_inconsistent_dimensions(...)' code ##################

    
    # Assigning a Num to a Name (line 102):
    int_240090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'int')
    # Assigning a type to the variable 'm' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'm', int_240090)
    
    # Assigning a Num to a Name (line 103):
    int_240091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
    # Assigning a type to the variable 'n' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'n', int_240091)
    
    # Assigning a List to a Name (line 104):
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_240092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    int_240093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), list_240092, int_240093)
    # Adding element type (line 104)
    int_240094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), list_240092, int_240094)
    # Adding element type (line 104)
    int_240095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), list_240092, int_240095)
    # Adding element type (line 104)
    int_240096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), list_240092, int_240096)
    
    # Assigning a type to the variable 'c' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'c', list_240092)
    
    # Assigning a Call to a Name (line 106):
    
    # Call to rand(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'm' (line 106)
    m_240100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'm', False)
    # Getting the type of 'n' (line 106)
    n_240101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'n', False)
    # Processing the call keyword arguments (line 106)
    kwargs_240102 = {}
    # Getting the type of 'np' (line 106)
    np_240097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 106)
    random_240098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), np_240097, 'random')
    # Obtaining the member 'rand' of a type (line 106)
    rand_240099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), random_240098, 'rand')
    # Calling rand(args, kwargs) (line 106)
    rand_call_result_240103 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), rand_240099, *[m_240100, n_240101], **kwargs_240102)
    
    # Assigning a type to the variable 'Agood' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'Agood', rand_call_result_240103)
    
    # Assigning a Call to a Name (line 107):
    
    # Call to rand(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'm' (line 107)
    m_240107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 26), 'm', False)
    # Getting the type of 'n' (line 107)
    n_240108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'n', False)
    int_240109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 33), 'int')
    # Applying the binary operator '+' (line 107)
    result_add_240110 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 29), '+', n_240108, int_240109)
    
    # Processing the call keyword arguments (line 107)
    kwargs_240111 = {}
    # Getting the type of 'np' (line 107)
    np_240104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'np', False)
    # Obtaining the member 'random' of a type (line 107)
    random_240105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), np_240104, 'random')
    # Obtaining the member 'rand' of a type (line 107)
    rand_240106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), random_240105, 'rand')
    # Calling rand(args, kwargs) (line 107)
    rand_call_result_240112 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), rand_240106, *[m_240107, result_add_240110], **kwargs_240111)
    
    # Assigning a type to the variable 'Abad' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'Abad', rand_call_result_240112)
    
    # Assigning a Call to a Name (line 108):
    
    # Call to rand(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'm' (line 108)
    m_240116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'm', False)
    # Processing the call keyword arguments (line 108)
    kwargs_240117 = {}
    # Getting the type of 'np' (line 108)
    np_240113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 108)
    random_240114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), np_240113, 'random')
    # Obtaining the member 'rand' of a type (line 108)
    rand_240115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), random_240114, 'rand')
    # Calling rand(args, kwargs) (line 108)
    rand_call_result_240118 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), rand_240115, *[m_240116], **kwargs_240117)
    
    # Assigning a type to the variable 'bgood' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'bgood', rand_call_result_240118)
    
    # Assigning a Call to a Name (line 109):
    
    # Call to rand(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'm' (line 109)
    m_240122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'm', False)
    int_240123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'int')
    # Applying the binary operator '+' (line 109)
    result_add_240124 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 26), '+', m_240122, int_240123)
    
    # Processing the call keyword arguments (line 109)
    kwargs_240125 = {}
    # Getting the type of 'np' (line 109)
    np_240119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'np', False)
    # Obtaining the member 'random' of a type (line 109)
    random_240120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), np_240119, 'random')
    # Obtaining the member 'rand' of a type (line 109)
    rand_240121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), random_240120, 'rand')
    # Calling rand(args, kwargs) (line 109)
    rand_call_result_240126 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), rand_240121, *[result_add_240124], **kwargs_240125)
    
    # Assigning a type to the variable 'bbad' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'bbad', rand_call_result_240126)
    
    # Assigning a BinOp to a Name (line 110):
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_240127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    # Adding element type (line 110)
    
    # Obtaining an instance of the builtin type 'tuple' (line 110)
    tuple_240128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 110)
    # Adding element type (line 110)
    int_240129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 18), tuple_240128, int_240129)
    # Adding element type (line 110)
    int_240130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 18), tuple_240128, int_240130)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), list_240127, tuple_240128)
    
    # Getting the type of 'n' (line 110)
    n_240131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'n')
    int_240132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 32), 'int')
    # Applying the binary operator '+' (line 110)
    result_add_240133 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 28), '+', n_240131, int_240132)
    
    # Applying the binary operator '*' (line 110)
    result_mul_240134 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 16), '*', list_240127, result_add_240133)
    
    # Assigning a type to the variable 'boundsbad' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'boundsbad', result_mul_240134)
    
    # Call to assert_raises(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'ValueError' (line 111)
    ValueError_240136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 111)
    _clean_inputs_240137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 111)
    # Getting the type of 'c' (line 111)
    c_240138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 47), 'c', False)
    keyword_240139 = c_240138
    # Getting the type of 'Abad' (line 111)
    Abad_240140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 55), 'Abad', False)
    keyword_240141 = Abad_240140
    # Getting the type of 'bgood' (line 111)
    bgood_240142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 66), 'bgood', False)
    keyword_240143 = bgood_240142
    kwargs_240144 = {'c': keyword_240139, 'b_ub': keyword_240143, 'A_ub': keyword_240141}
    # Getting the type of 'assert_raises' (line 111)
    assert_raises_240135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 111)
    assert_raises_call_result_240145 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), assert_raises_240135, *[ValueError_240136, _clean_inputs_240137], **kwargs_240144)
    
    
    # Call to assert_raises(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'ValueError' (line 112)
    ValueError_240147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 112)
    _clean_inputs_240148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 112)
    # Getting the type of 'c' (line 112)
    c_240149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 47), 'c', False)
    keyword_240150 = c_240149
    # Getting the type of 'Agood' (line 112)
    Agood_240151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 55), 'Agood', False)
    keyword_240152 = Agood_240151
    # Getting the type of 'bbad' (line 112)
    bbad_240153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 67), 'bbad', False)
    keyword_240154 = bbad_240153
    kwargs_240155 = {'c': keyword_240150, 'b_ub': keyword_240154, 'A_ub': keyword_240152}
    # Getting the type of 'assert_raises' (line 112)
    assert_raises_240146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 112)
    assert_raises_call_result_240156 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), assert_raises_240146, *[ValueError_240147, _clean_inputs_240148], **kwargs_240155)
    
    
    # Call to assert_raises(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'ValueError' (line 113)
    ValueError_240158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 113)
    _clean_inputs_240159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 113)
    # Getting the type of 'c' (line 113)
    c_240160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 47), 'c', False)
    keyword_240161 = c_240160
    # Getting the type of 'Abad' (line 113)
    Abad_240162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 55), 'Abad', False)
    keyword_240163 = Abad_240162
    # Getting the type of 'bgood' (line 113)
    bgood_240164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 66), 'bgood', False)
    keyword_240165 = bgood_240164
    kwargs_240166 = {'c': keyword_240161, 'A_eq': keyword_240163, 'b_eq': keyword_240165}
    # Getting the type of 'assert_raises' (line 113)
    assert_raises_240157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 113)
    assert_raises_call_result_240167 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), assert_raises_240157, *[ValueError_240158, _clean_inputs_240159], **kwargs_240166)
    
    
    # Call to assert_raises(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'ValueError' (line 114)
    ValueError_240169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 114)
    _clean_inputs_240170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 114)
    # Getting the type of 'c' (line 114)
    c_240171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 47), 'c', False)
    keyword_240172 = c_240171
    # Getting the type of 'Agood' (line 114)
    Agood_240173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 55), 'Agood', False)
    keyword_240174 = Agood_240173
    # Getting the type of 'bbad' (line 114)
    bbad_240175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 67), 'bbad', False)
    keyword_240176 = bbad_240175
    kwargs_240177 = {'c': keyword_240172, 'A_eq': keyword_240174, 'b_eq': keyword_240176}
    # Getting the type of 'assert_raises' (line 114)
    assert_raises_240168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 114)
    assert_raises_call_result_240178 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), assert_raises_240168, *[ValueError_240169, _clean_inputs_240170], **kwargs_240177)
    
    
    # Call to assert_raises(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'ValueError' (line 115)
    ValueError_240180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 115)
    _clean_inputs_240181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 115)
    # Getting the type of 'c' (line 115)
    c_240182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 47), 'c', False)
    keyword_240183 = c_240182
    # Getting the type of 'boundsbad' (line 115)
    boundsbad_240184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 57), 'boundsbad', False)
    keyword_240185 = boundsbad_240184
    kwargs_240186 = {'c': keyword_240183, 'bounds': keyword_240185}
    # Getting the type of 'assert_raises' (line 115)
    assert_raises_240179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 115)
    assert_raises_call_result_240187 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), assert_raises_240179, *[ValueError_240180, _clean_inputs_240181], **kwargs_240186)
    
    
    # ################# End of 'test_inconsistent_dimensions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_inconsistent_dimensions' in the type store
    # Getting the type of 'stypy_return_type' (line 101)
    stypy_return_type_240188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_240188)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_inconsistent_dimensions'
    return stypy_return_type_240188

# Assigning a type to the variable 'test_inconsistent_dimensions' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'test_inconsistent_dimensions', test_inconsistent_dimensions)

@norecursion
def test_type_errors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_type_errors'
    module_type_store = module_type_store.open_function_context('test_type_errors', 118, 0, False)
    
    # Passed parameters checking function
    test_type_errors.stypy_localization = localization
    test_type_errors.stypy_type_of_self = None
    test_type_errors.stypy_type_store = module_type_store
    test_type_errors.stypy_function_name = 'test_type_errors'
    test_type_errors.stypy_param_names_list = []
    test_type_errors.stypy_varargs_param_name = None
    test_type_errors.stypy_kwargs_param_name = None
    test_type_errors.stypy_call_defaults = defaults
    test_type_errors.stypy_call_varargs = varargs
    test_type_errors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_type_errors', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_type_errors', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_type_errors(...)' code ##################

    
    # Assigning a Str to a Name (line 119):
    str_240189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 10), 'str', 'hello')
    # Assigning a type to the variable 'bad' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'bad', str_240189)
    
    # Assigning a List to a Name (line 120):
    
    # Obtaining an instance of the builtin type 'list' (line 120)
    list_240190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 120)
    # Adding element type (line 120)
    int_240191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), list_240190, int_240191)
    # Adding element type (line 120)
    int_240192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 8), list_240190, int_240192)
    
    # Assigning a type to the variable 'c' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'c', list_240190)
    
    # Assigning a Call to a Name (line 121):
    
    # Call to array(...): (line 121)
    # Processing the call arguments (line 121)
    
    # Obtaining an instance of the builtin type 'list' (line 121)
    list_240195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 121)
    # Adding element type (line 121)
    
    # Obtaining an instance of the builtin type 'list' (line 121)
    list_240196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 121)
    # Adding element type (line 121)
    int_240197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 21), list_240196, int_240197)
    # Adding element type (line 121)
    int_240198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 21), list_240196, int_240198)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 20), list_240195, list_240196)
    # Adding element type (line 121)
    
    # Obtaining an instance of the builtin type 'list' (line 121)
    list_240199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 121)
    # Adding element type (line 121)
    int_240200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 29), list_240199, int_240200)
    # Adding element type (line 121)
    int_240201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 29), list_240199, int_240201)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 20), list_240195, list_240199)
    
    # Processing the call keyword arguments (line 121)
    kwargs_240202 = {}
    # Getting the type of 'np' (line 121)
    np_240193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 121)
    array_240194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), np_240193, 'array')
    # Calling array(args, kwargs) (line 121)
    array_call_result_240203 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), array_240194, *[list_240195], **kwargs_240202)
    
    # Assigning a type to the variable 'A_ub' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'A_ub', array_call_result_240203)
    
    # Assigning a Call to a Name (line 122):
    
    # Call to array(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Obtaining an instance of the builtin type 'list' (line 122)
    list_240206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 122)
    # Adding element type (line 122)
    int_240207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), list_240206, int_240207)
    # Adding element type (line 122)
    int_240208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), list_240206, int_240208)
    
    # Processing the call keyword arguments (line 122)
    kwargs_240209 = {}
    # Getting the type of 'np' (line 122)
    np_240204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 122)
    array_240205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 11), np_240204, 'array')
    # Calling array(args, kwargs) (line 122)
    array_call_result_240210 = invoke(stypy.reporting.localization.Localization(__file__, 122, 11), array_240205, *[list_240206], **kwargs_240209)
    
    # Assigning a type to the variable 'b_ub' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'b_ub', array_call_result_240210)
    
    # Assigning a Call to a Name (line 123):
    
    # Call to array(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Obtaining an instance of the builtin type 'list' (line 123)
    list_240213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 123)
    # Adding element type (line 123)
    
    # Obtaining an instance of the builtin type 'list' (line 123)
    list_240214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 123)
    # Adding element type (line 123)
    int_240215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 21), list_240214, int_240215)
    # Adding element type (line 123)
    int_240216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 21), list_240214, int_240216)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), list_240213, list_240214)
    # Adding element type (line 123)
    
    # Obtaining an instance of the builtin type 'list' (line 123)
    list_240217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 123)
    # Adding element type (line 123)
    int_240218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), list_240217, int_240218)
    # Adding element type (line 123)
    int_240219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), list_240217, int_240219)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), list_240213, list_240217)
    
    # Processing the call keyword arguments (line 123)
    kwargs_240220 = {}
    # Getting the type of 'np' (line 123)
    np_240211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 123)
    array_240212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), np_240211, 'array')
    # Calling array(args, kwargs) (line 123)
    array_call_result_240221 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), array_240212, *[list_240213], **kwargs_240220)
    
    # Assigning a type to the variable 'A_eq' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'A_eq', array_call_result_240221)
    
    # Assigning a Call to a Name (line 124):
    
    # Call to array(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Obtaining an instance of the builtin type 'list' (line 124)
    list_240224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 124)
    # Adding element type (line 124)
    int_240225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), list_240224, int_240225)
    # Adding element type (line 124)
    int_240226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 20), list_240224, int_240226)
    
    # Processing the call keyword arguments (line 124)
    kwargs_240227 = {}
    # Getting the type of 'np' (line 124)
    np_240222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 124)
    array_240223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 11), np_240222, 'array')
    # Calling array(args, kwargs) (line 124)
    array_call_result_240228 = invoke(stypy.reporting.localization.Localization(__file__, 124, 11), array_240223, *[list_240224], **kwargs_240227)
    
    # Assigning a type to the variable 'b_eq' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'b_eq', array_call_result_240228)
    
    # Assigning a List to a Name (line 125):
    
    # Obtaining an instance of the builtin type 'list' (line 125)
    list_240229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 125)
    # Adding element type (line 125)
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_240230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    int_240231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_240230, int_240231)
    # Adding element type (line 125)
    int_240232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_240230, int_240232)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 13), list_240229, tuple_240230)
    
    # Assigning a type to the variable 'bounds' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'bounds', list_240229)
    
    # Call to assert_raises(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'TypeError' (line 127)
    TypeError_240234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 128)
    _clean_inputs_240235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 126)
    # Getting the type of 'bad' (line 129)
    bad_240236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 10), 'bad', False)
    keyword_240237 = bad_240236
    # Getting the type of 'A_ub' (line 130)
    A_ub_240238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'A_ub', False)
    keyword_240239 = A_ub_240238
    # Getting the type of 'b_ub' (line 131)
    b_ub_240240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'b_ub', False)
    keyword_240241 = b_ub_240240
    # Getting the type of 'A_eq' (line 132)
    A_eq_240242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 13), 'A_eq', False)
    keyword_240243 = A_eq_240242
    # Getting the type of 'b_eq' (line 133)
    b_eq_240244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'b_eq', False)
    keyword_240245 = b_eq_240244
    # Getting the type of 'bounds' (line 134)
    bounds_240246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'bounds', False)
    keyword_240247 = bounds_240246
    kwargs_240248 = {'c': keyword_240237, 'A_ub': keyword_240239, 'A_eq': keyword_240243, 'bounds': keyword_240247, 'b_ub': keyword_240241, 'b_eq': keyword_240245}
    # Getting the type of 'assert_raises' (line 126)
    assert_raises_240233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 126)
    assert_raises_call_result_240249 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), assert_raises_240233, *[TypeError_240234, _clean_inputs_240235], **kwargs_240248)
    
    
    # Call to assert_raises(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'TypeError' (line 136)
    TypeError_240251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 137)
    _clean_inputs_240252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'c' (line 138)
    c_240253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 10), 'c', False)
    keyword_240254 = c_240253
    # Getting the type of 'bad' (line 139)
    bad_240255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 13), 'bad', False)
    keyword_240256 = bad_240255
    # Getting the type of 'b_ub' (line 140)
    b_ub_240257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 13), 'b_ub', False)
    keyword_240258 = b_ub_240257
    # Getting the type of 'A_eq' (line 141)
    A_eq_240259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'A_eq', False)
    keyword_240260 = A_eq_240259
    # Getting the type of 'b_eq' (line 142)
    b_eq_240261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'b_eq', False)
    keyword_240262 = b_eq_240261
    # Getting the type of 'bounds' (line 143)
    bounds_240263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'bounds', False)
    keyword_240264 = bounds_240263
    kwargs_240265 = {'c': keyword_240254, 'A_ub': keyword_240256, 'A_eq': keyword_240260, 'bounds': keyword_240264, 'b_ub': keyword_240258, 'b_eq': keyword_240262}
    # Getting the type of 'assert_raises' (line 135)
    assert_raises_240250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 135)
    assert_raises_call_result_240266 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), assert_raises_240250, *[TypeError_240251, _clean_inputs_240252], **kwargs_240265)
    
    
    # Call to assert_raises(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'TypeError' (line 145)
    TypeError_240268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 146)
    _clean_inputs_240269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 144)
    # Getting the type of 'c' (line 147)
    c_240270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 10), 'c', False)
    keyword_240271 = c_240270
    # Getting the type of 'A_ub' (line 148)
    A_ub_240272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), 'A_ub', False)
    keyword_240273 = A_ub_240272
    # Getting the type of 'bad' (line 149)
    bad_240274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 13), 'bad', False)
    keyword_240275 = bad_240274
    # Getting the type of 'A_eq' (line 150)
    A_eq_240276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'A_eq', False)
    keyword_240277 = A_eq_240276
    # Getting the type of 'b_eq' (line 151)
    b_eq_240278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'b_eq', False)
    keyword_240279 = b_eq_240278
    # Getting the type of 'bounds' (line 152)
    bounds_240280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'bounds', False)
    keyword_240281 = bounds_240280
    kwargs_240282 = {'c': keyword_240271, 'A_ub': keyword_240273, 'A_eq': keyword_240277, 'bounds': keyword_240281, 'b_ub': keyword_240275, 'b_eq': keyword_240279}
    # Getting the type of 'assert_raises' (line 144)
    assert_raises_240267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 144)
    assert_raises_call_result_240283 = invoke(stypy.reporting.localization.Localization(__file__, 144, 4), assert_raises_240267, *[TypeError_240268, _clean_inputs_240269], **kwargs_240282)
    
    
    # Call to assert_raises(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'TypeError' (line 154)
    TypeError_240285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 155)
    _clean_inputs_240286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 153)
    # Getting the type of 'c' (line 156)
    c_240287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 10), 'c', False)
    keyword_240288 = c_240287
    # Getting the type of 'A_ub' (line 157)
    A_ub_240289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'A_ub', False)
    keyword_240290 = A_ub_240289
    # Getting the type of 'b_ub' (line 158)
    b_ub_240291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 13), 'b_ub', False)
    keyword_240292 = b_ub_240291
    # Getting the type of 'bad' (line 159)
    bad_240293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'bad', False)
    keyword_240294 = bad_240293
    # Getting the type of 'b_eq' (line 160)
    b_eq_240295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'b_eq', False)
    keyword_240296 = b_eq_240295
    # Getting the type of 'bounds' (line 161)
    bounds_240297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'bounds', False)
    keyword_240298 = bounds_240297
    kwargs_240299 = {'c': keyword_240288, 'A_ub': keyword_240290, 'A_eq': keyword_240294, 'bounds': keyword_240298, 'b_ub': keyword_240292, 'b_eq': keyword_240296}
    # Getting the type of 'assert_raises' (line 153)
    assert_raises_240284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 153)
    assert_raises_call_result_240300 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), assert_raises_240284, *[TypeError_240285, _clean_inputs_240286], **kwargs_240299)
    
    
    # Call to assert_raises(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'TypeError' (line 164)
    TypeError_240302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 165)
    _clean_inputs_240303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 163)
    # Getting the type of 'c' (line 166)
    c_240304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 10), 'c', False)
    keyword_240305 = c_240304
    # Getting the type of 'A_ub' (line 167)
    A_ub_240306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 13), 'A_ub', False)
    keyword_240307 = A_ub_240306
    # Getting the type of 'b_ub' (line 168)
    b_ub_240308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 13), 'b_ub', False)
    keyword_240309 = b_ub_240308
    # Getting the type of 'A_eq' (line 169)
    A_eq_240310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'A_eq', False)
    keyword_240311 = A_eq_240310
    # Getting the type of 'b_eq' (line 170)
    b_eq_240312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 13), 'b_eq', False)
    keyword_240313 = b_eq_240312
    # Getting the type of 'bad' (line 171)
    bad_240314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'bad', False)
    keyword_240315 = bad_240314
    kwargs_240316 = {'c': keyword_240305, 'A_ub': keyword_240307, 'A_eq': keyword_240311, 'bounds': keyword_240315, 'b_ub': keyword_240309, 'b_eq': keyword_240313}
    # Getting the type of 'assert_raises' (line 163)
    assert_raises_240301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 163)
    assert_raises_call_result_240317 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), assert_raises_240301, *[TypeError_240302, _clean_inputs_240303], **kwargs_240316)
    
    
    # Call to assert_raises(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'TypeError' (line 173)
    TypeError_240319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 174)
    _clean_inputs_240320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 172)
    # Getting the type of 'c' (line 175)
    c_240321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 10), 'c', False)
    keyword_240322 = c_240321
    # Getting the type of 'A_ub' (line 176)
    A_ub_240323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'A_ub', False)
    keyword_240324 = A_ub_240323
    # Getting the type of 'b_ub' (line 177)
    b_ub_240325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'b_ub', False)
    keyword_240326 = b_ub_240325
    # Getting the type of 'A_eq' (line 178)
    A_eq_240327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 13), 'A_eq', False)
    keyword_240328 = A_eq_240327
    # Getting the type of 'b_eq' (line 179)
    b_eq_240329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'b_eq', False)
    keyword_240330 = b_eq_240329
    str_240331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 15), 'str', 'hi')
    keyword_240332 = str_240331
    kwargs_240333 = {'c': keyword_240322, 'A_ub': keyword_240324, 'A_eq': keyword_240328, 'bounds': keyword_240332, 'b_ub': keyword_240326, 'b_eq': keyword_240330}
    # Getting the type of 'assert_raises' (line 172)
    assert_raises_240318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 172)
    assert_raises_call_result_240334 = invoke(stypy.reporting.localization.Localization(__file__, 172, 4), assert_raises_240318, *[TypeError_240319, _clean_inputs_240320], **kwargs_240333)
    
    
    # Call to assert_raises(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'TypeError' (line 182)
    TypeError_240336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 183)
    _clean_inputs_240337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 181)
    # Getting the type of 'c' (line 184)
    c_240338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 10), 'c', False)
    keyword_240339 = c_240338
    # Getting the type of 'A_ub' (line 185)
    A_ub_240340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'A_ub', False)
    keyword_240341 = A_ub_240340
    # Getting the type of 'b_ub' (line 186)
    b_ub_240342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'b_ub', False)
    keyword_240343 = b_ub_240342
    # Getting the type of 'A_eq' (line 187)
    A_eq_240344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), 'A_eq', False)
    keyword_240345 = A_eq_240344
    # Getting the type of 'b_eq' (line 188)
    b_eq_240346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'b_eq', False)
    keyword_240347 = b_eq_240346
    
    # Obtaining an instance of the builtin type 'list' (line 189)
    list_240348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 189)
    # Adding element type (line 189)
    str_240349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 16), 'str', 'hi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 15), list_240348, str_240349)
    
    keyword_240350 = list_240348
    kwargs_240351 = {'c': keyword_240339, 'A_ub': keyword_240341, 'A_eq': keyword_240345, 'bounds': keyword_240350, 'b_ub': keyword_240343, 'b_eq': keyword_240347}
    # Getting the type of 'assert_raises' (line 181)
    assert_raises_240335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 181)
    assert_raises_call_result_240352 = invoke(stypy.reporting.localization.Localization(__file__, 181, 4), assert_raises_240335, *[TypeError_240336, _clean_inputs_240337], **kwargs_240351)
    
    
    # Call to assert_raises(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'TypeError' (line 191)
    TypeError_240354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 192)
    _clean_inputs_240355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 190)
    # Getting the type of 'c' (line 193)
    c_240356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 10), 'c', False)
    keyword_240357 = c_240356
    # Getting the type of 'A_ub' (line 194)
    A_ub_240358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 13), 'A_ub', False)
    keyword_240359 = A_ub_240358
    # Getting the type of 'b_ub' (line 195)
    b_ub_240360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 13), 'b_ub', False)
    keyword_240361 = b_ub_240360
    # Getting the type of 'A_eq' (line 196)
    A_eq_240362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 13), 'A_eq', False)
    keyword_240363 = A_eq_240362
    # Getting the type of 'b_eq' (line 197)
    b_eq_240364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'b_eq', False)
    keyword_240365 = b_eq_240364
    
    # Obtaining an instance of the builtin type 'list' (line 198)
    list_240366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 198)
    # Adding element type (line 198)
    str_240367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 13), 'str', 'hi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 15), list_240366, str_240367)
    
    keyword_240368 = list_240366
    kwargs_240369 = {'c': keyword_240357, 'A_ub': keyword_240359, 'A_eq': keyword_240363, 'bounds': keyword_240368, 'b_ub': keyword_240361, 'b_eq': keyword_240365}
    # Getting the type of 'assert_raises' (line 190)
    assert_raises_240353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 190)
    assert_raises_call_result_240370 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), assert_raises_240353, *[TypeError_240354, _clean_inputs_240355], **kwargs_240369)
    
    
    # Call to assert_raises(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'TypeError' (line 200)
    TypeError_240372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 200)
    _clean_inputs_240373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 29), '_clean_inputs', False)
    # Processing the call keyword arguments (line 200)
    # Getting the type of 'c' (line 200)
    c_240374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 46), 'c', False)
    keyword_240375 = c_240374
    # Getting the type of 'A_ub' (line 200)
    A_ub_240376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 54), 'A_ub', False)
    keyword_240377 = A_ub_240376
    # Getting the type of 'b_ub' (line 201)
    b_ub_240378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'b_ub', False)
    keyword_240379 = b_ub_240378
    # Getting the type of 'A_eq' (line 201)
    A_eq_240380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 34), 'A_eq', False)
    keyword_240381 = A_eq_240380
    # Getting the type of 'b_eq' (line 201)
    b_eq_240382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 45), 'b_eq', False)
    keyword_240383 = b_eq_240382
    
    # Obtaining an instance of the builtin type 'list' (line 201)
    list_240384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 58), 'list')
    # Adding type elements to the builtin type 'list' instance (line 201)
    # Adding element type (line 201)
    
    # Obtaining an instance of the builtin type 'tuple' (line 201)
    tuple_240385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 201)
    # Adding element type (line 201)
    int_240386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 60), tuple_240385, int_240386)
    # Adding element type (line 201)
    str_240387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 63), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 60), tuple_240385, str_240387)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 58), list_240384, tuple_240385)
    
    keyword_240388 = list_240384
    kwargs_240389 = {'c': keyword_240375, 'A_ub': keyword_240377, 'A_eq': keyword_240381, 'bounds': keyword_240388, 'b_ub': keyword_240379, 'b_eq': keyword_240383}
    # Getting the type of 'assert_raises' (line 200)
    assert_raises_240371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 200)
    assert_raises_call_result_240390 = invoke(stypy.reporting.localization.Localization(__file__, 200, 4), assert_raises_240371, *[TypeError_240372, _clean_inputs_240373], **kwargs_240389)
    
    
    # Call to assert_raises(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'TypeError' (line 202)
    TypeError_240392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'TypeError', False)
    # Getting the type of '_clean_inputs' (line 202)
    _clean_inputs_240393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), '_clean_inputs', False)
    # Processing the call keyword arguments (line 202)
    # Getting the type of 'c' (line 202)
    c_240394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 46), 'c', False)
    keyword_240395 = c_240394
    # Getting the type of 'A_ub' (line 202)
    A_ub_240396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 54), 'A_ub', False)
    keyword_240397 = A_ub_240396
    # Getting the type of 'b_ub' (line 203)
    b_ub_240398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'b_ub', False)
    keyword_240399 = b_ub_240398
    # Getting the type of 'A_eq' (line 203)
    A_eq_240400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 34), 'A_eq', False)
    keyword_240401 = A_eq_240400
    # Getting the type of 'b_eq' (line 203)
    b_eq_240402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 45), 'b_eq', False)
    keyword_240403 = b_eq_240402
    
    # Obtaining an instance of the builtin type 'list' (line 203)
    list_240404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 58), 'list')
    # Adding type elements to the builtin type 'list' instance (line 203)
    # Adding element type (line 203)
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_240405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    int_240406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 60), tuple_240405, int_240406)
    # Adding element type (line 203)
    int_240407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 63), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 60), tuple_240405, int_240407)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 58), list_240404, tuple_240405)
    # Adding element type (line 203)
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_240408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    int_240409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 68), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 68), tuple_240408, int_240409)
    # Adding element type (line 203)
    str_240410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 71), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 68), tuple_240408, str_240410)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 58), list_240404, tuple_240408)
    
    keyword_240411 = list_240404
    kwargs_240412 = {'c': keyword_240395, 'A_ub': keyword_240397, 'A_eq': keyword_240401, 'bounds': keyword_240411, 'b_ub': keyword_240399, 'b_eq': keyword_240403}
    # Getting the type of 'assert_raises' (line 202)
    assert_raises_240391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 202)
    assert_raises_call_result_240413 = invoke(stypy.reporting.localization.Localization(__file__, 202, 4), assert_raises_240391, *[TypeError_240392, _clean_inputs_240393], **kwargs_240412)
    
    
    # ################# End of 'test_type_errors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_type_errors' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_240414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_240414)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_type_errors'
    return stypy_return_type_240414

# Assigning a type to the variable 'test_type_errors' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'test_type_errors', test_type_errors)

@norecursion
def test_non_finite_errors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_non_finite_errors'
    module_type_store = module_type_store.open_function_context('test_non_finite_errors', 206, 0, False)
    
    # Passed parameters checking function
    test_non_finite_errors.stypy_localization = localization
    test_non_finite_errors.stypy_type_of_self = None
    test_non_finite_errors.stypy_type_store = module_type_store
    test_non_finite_errors.stypy_function_name = 'test_non_finite_errors'
    test_non_finite_errors.stypy_param_names_list = []
    test_non_finite_errors.stypy_varargs_param_name = None
    test_non_finite_errors.stypy_kwargs_param_name = None
    test_non_finite_errors.stypy_call_defaults = defaults
    test_non_finite_errors.stypy_call_varargs = varargs
    test_non_finite_errors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_non_finite_errors', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_non_finite_errors', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_non_finite_errors(...)' code ##################

    
    # Assigning a List to a Name (line 207):
    
    # Obtaining an instance of the builtin type 'list' (line 207)
    list_240415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 207)
    # Adding element type (line 207)
    int_240416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 8), list_240415, int_240416)
    # Adding element type (line 207)
    int_240417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 8), list_240415, int_240417)
    
    # Assigning a type to the variable 'c' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'c', list_240415)
    
    # Assigning a Call to a Name (line 208):
    
    # Call to array(...): (line 208)
    # Processing the call arguments (line 208)
    
    # Obtaining an instance of the builtin type 'list' (line 208)
    list_240420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 208)
    # Adding element type (line 208)
    
    # Obtaining an instance of the builtin type 'list' (line 208)
    list_240421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 208)
    # Adding element type (line 208)
    int_240422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 21), list_240421, int_240422)
    # Adding element type (line 208)
    int_240423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 21), list_240421, int_240423)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 20), list_240420, list_240421)
    # Adding element type (line 208)
    
    # Obtaining an instance of the builtin type 'list' (line 208)
    list_240424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 208)
    # Adding element type (line 208)
    int_240425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 29), list_240424, int_240425)
    # Adding element type (line 208)
    int_240426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 29), list_240424, int_240426)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 20), list_240420, list_240424)
    
    # Processing the call keyword arguments (line 208)
    kwargs_240427 = {}
    # Getting the type of 'np' (line 208)
    np_240418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 208)
    array_240419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), np_240418, 'array')
    # Calling array(args, kwargs) (line 208)
    array_call_result_240428 = invoke(stypy.reporting.localization.Localization(__file__, 208, 11), array_240419, *[list_240420], **kwargs_240427)
    
    # Assigning a type to the variable 'A_ub' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'A_ub', array_call_result_240428)
    
    # Assigning a Call to a Name (line 209):
    
    # Call to array(...): (line 209)
    # Processing the call arguments (line 209)
    
    # Obtaining an instance of the builtin type 'list' (line 209)
    list_240431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 209)
    # Adding element type (line 209)
    int_240432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 20), list_240431, int_240432)
    # Adding element type (line 209)
    int_240433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 20), list_240431, int_240433)
    
    # Processing the call keyword arguments (line 209)
    kwargs_240434 = {}
    # Getting the type of 'np' (line 209)
    np_240429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 209)
    array_240430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), np_240429, 'array')
    # Calling array(args, kwargs) (line 209)
    array_call_result_240435 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), array_240430, *[list_240431], **kwargs_240434)
    
    # Assigning a type to the variable 'b_ub' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'b_ub', array_call_result_240435)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to array(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_240438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    # Adding element type (line 210)
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_240439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    # Adding element type (line 210)
    int_240440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 21), list_240439, int_240440)
    # Adding element type (line 210)
    int_240441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 21), list_240439, int_240441)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 20), list_240438, list_240439)
    # Adding element type (line 210)
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_240442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    # Adding element type (line 210)
    int_240443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 29), list_240442, int_240443)
    # Adding element type (line 210)
    int_240444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 29), list_240442, int_240444)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 20), list_240438, list_240442)
    
    # Processing the call keyword arguments (line 210)
    kwargs_240445 = {}
    # Getting the type of 'np' (line 210)
    np_240436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 210)
    array_240437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 11), np_240436, 'array')
    # Calling array(args, kwargs) (line 210)
    array_call_result_240446 = invoke(stypy.reporting.localization.Localization(__file__, 210, 11), array_240437, *[list_240438], **kwargs_240445)
    
    # Assigning a type to the variable 'A_eq' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'A_eq', array_call_result_240446)
    
    # Assigning a Call to a Name (line 211):
    
    # Call to array(...): (line 211)
    # Processing the call arguments (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 211)
    list_240449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 211)
    # Adding element type (line 211)
    int_240450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 20), list_240449, int_240450)
    # Adding element type (line 211)
    int_240451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 20), list_240449, int_240451)
    
    # Processing the call keyword arguments (line 211)
    kwargs_240452 = {}
    # Getting the type of 'np' (line 211)
    np_240447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 211)
    array_240448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), np_240447, 'array')
    # Calling array(args, kwargs) (line 211)
    array_call_result_240453 = invoke(stypy.reporting.localization.Localization(__file__, 211, 11), array_240448, *[list_240449], **kwargs_240452)
    
    # Assigning a type to the variable 'b_eq' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'b_eq', array_call_result_240453)
    
    # Assigning a List to a Name (line 212):
    
    # Obtaining an instance of the builtin type 'list' (line 212)
    list_240454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 212)
    # Adding element type (line 212)
    
    # Obtaining an instance of the builtin type 'tuple' (line 212)
    tuple_240455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 212)
    # Adding element type (line 212)
    int_240456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 15), tuple_240455, int_240456)
    # Adding element type (line 212)
    int_240457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 15), tuple_240455, int_240457)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 13), list_240454, tuple_240455)
    
    # Assigning a type to the variable 'bounds' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'bounds', list_240454)
    
    # Call to assert_raises(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'ValueError' (line 214)
    ValueError_240459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 214)
    _clean_inputs_240460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), '_clean_inputs', False)
    # Processing the call keyword arguments (line 213)
    
    # Obtaining an instance of the builtin type 'list' (line 214)
    list_240461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 214)
    # Adding element type (line 214)
    int_240462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 37), list_240461, int_240462)
    # Adding element type (line 214)
    # Getting the type of 'None' (line 214)
    None_240463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 41), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 37), list_240461, None_240463)
    
    keyword_240464 = list_240461
    # Getting the type of 'A_ub' (line 214)
    A_ub_240465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 53), 'A_ub', False)
    keyword_240466 = A_ub_240465
    # Getting the type of 'b_ub' (line 214)
    b_ub_240467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 64), 'b_ub', False)
    keyword_240468 = b_ub_240467
    # Getting the type of 'A_eq' (line 215)
    A_eq_240469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'A_eq', False)
    keyword_240470 = A_eq_240469
    # Getting the type of 'b_eq' (line 215)
    b_eq_240471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'b_eq', False)
    keyword_240472 = b_eq_240471
    # Getting the type of 'bounds' (line 215)
    bounds_240473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 37), 'bounds', False)
    keyword_240474 = bounds_240473
    kwargs_240475 = {'c': keyword_240464, 'A_ub': keyword_240466, 'A_eq': keyword_240470, 'bounds': keyword_240474, 'b_ub': keyword_240468, 'b_eq': keyword_240472}
    # Getting the type of 'assert_raises' (line 213)
    assert_raises_240458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 213)
    assert_raises_call_result_240476 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), assert_raises_240458, *[ValueError_240459, _clean_inputs_240460], **kwargs_240475)
    
    
    # Call to assert_raises(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'ValueError' (line 217)
    ValueError_240478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 217)
    _clean_inputs_240479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), '_clean_inputs', False)
    # Processing the call keyword arguments (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 217)
    list_240480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 217)
    # Adding element type (line 217)
    # Getting the type of 'np' (line 217)
    np_240481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 38), 'np', False)
    # Obtaining the member 'inf' of a type (line 217)
    inf_240482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 38), np_240481, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 37), list_240480, inf_240482)
    # Adding element type (line 217)
    int_240483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 37), list_240480, int_240483)
    
    keyword_240484 = list_240480
    # Getting the type of 'A_ub' (line 217)
    A_ub_240485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 55), 'A_ub', False)
    keyword_240486 = A_ub_240485
    # Getting the type of 'b_ub' (line 217)
    b_ub_240487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 66), 'b_ub', False)
    keyword_240488 = b_ub_240487
    # Getting the type of 'A_eq' (line 218)
    A_eq_240489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 13), 'A_eq', False)
    keyword_240490 = A_eq_240489
    # Getting the type of 'b_eq' (line 218)
    b_eq_240491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'b_eq', False)
    keyword_240492 = b_eq_240491
    # Getting the type of 'bounds' (line 218)
    bounds_240493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 37), 'bounds', False)
    keyword_240494 = bounds_240493
    kwargs_240495 = {'c': keyword_240484, 'A_ub': keyword_240486, 'A_eq': keyword_240490, 'bounds': keyword_240494, 'b_ub': keyword_240488, 'b_eq': keyword_240492}
    # Getting the type of 'assert_raises' (line 216)
    assert_raises_240477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 216)
    assert_raises_call_result_240496 = invoke(stypy.reporting.localization.Localization(__file__, 216, 4), assert_raises_240477, *[ValueError_240478, _clean_inputs_240479], **kwargs_240495)
    
    
    # Call to assert_raises(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'ValueError' (line 220)
    ValueError_240498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 220)
    _clean_inputs_240499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), '_clean_inputs', False)
    # Processing the call keyword arguments (line 219)
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_240500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    # Adding element type (line 220)
    int_240501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 37), list_240500, int_240501)
    # Adding element type (line 220)
    
    # Getting the type of 'np' (line 220)
    np_240502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 42), 'np', False)
    # Obtaining the member 'inf' of a type (line 220)
    inf_240503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 42), np_240502, 'inf')
    # Applying the 'usub' unary operator (line 220)
    result___neg___240504 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 41), 'usub', inf_240503)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 37), list_240500, result___neg___240504)
    
    keyword_240505 = list_240500
    # Getting the type of 'A_ub' (line 220)
    A_ub_240506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 56), 'A_ub', False)
    keyword_240507 = A_ub_240506
    # Getting the type of 'b_ub' (line 220)
    b_ub_240508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 67), 'b_ub', False)
    keyword_240509 = b_ub_240508
    # Getting the type of 'A_eq' (line 221)
    A_eq_240510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 13), 'A_eq', False)
    keyword_240511 = A_eq_240510
    # Getting the type of 'b_eq' (line 221)
    b_eq_240512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'b_eq', False)
    keyword_240513 = b_eq_240512
    # Getting the type of 'bounds' (line 221)
    bounds_240514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 37), 'bounds', False)
    keyword_240515 = bounds_240514
    kwargs_240516 = {'c': keyword_240505, 'A_ub': keyword_240507, 'A_eq': keyword_240511, 'bounds': keyword_240515, 'b_ub': keyword_240509, 'b_eq': keyword_240513}
    # Getting the type of 'assert_raises' (line 219)
    assert_raises_240497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 219)
    assert_raises_call_result_240517 = invoke(stypy.reporting.localization.Localization(__file__, 219, 4), assert_raises_240497, *[ValueError_240498, _clean_inputs_240499], **kwargs_240516)
    
    
    # Call to assert_raises(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'ValueError' (line 223)
    ValueError_240519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 223)
    _clean_inputs_240520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), '_clean_inputs', False)
    # Processing the call keyword arguments (line 222)
    
    # Obtaining an instance of the builtin type 'list' (line 223)
    list_240521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 223)
    # Adding element type (line 223)
    # Getting the type of 'np' (line 223)
    np_240522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 38), 'np', False)
    # Obtaining the member 'nan' of a type (line 223)
    nan_240523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 38), np_240522, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 37), list_240521, nan_240523)
    # Adding element type (line 223)
    int_240524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 37), list_240521, int_240524)
    
    keyword_240525 = list_240521
    # Getting the type of 'A_ub' (line 223)
    A_ub_240526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 55), 'A_ub', False)
    keyword_240527 = A_ub_240526
    # Getting the type of 'b_ub' (line 223)
    b_ub_240528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 66), 'b_ub', False)
    keyword_240529 = b_ub_240528
    # Getting the type of 'A_eq' (line 224)
    A_eq_240530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 13), 'A_eq', False)
    keyword_240531 = A_eq_240530
    # Getting the type of 'b_eq' (line 224)
    b_eq_240532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'b_eq', False)
    keyword_240533 = b_eq_240532
    # Getting the type of 'bounds' (line 224)
    bounds_240534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 37), 'bounds', False)
    keyword_240535 = bounds_240534
    kwargs_240536 = {'c': keyword_240525, 'A_ub': keyword_240527, 'A_eq': keyword_240531, 'bounds': keyword_240535, 'b_ub': keyword_240529, 'b_eq': keyword_240533}
    # Getting the type of 'assert_raises' (line 222)
    assert_raises_240518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 222)
    assert_raises_call_result_240537 = invoke(stypy.reporting.localization.Localization(__file__, 222, 4), assert_raises_240518, *[ValueError_240519, _clean_inputs_240520], **kwargs_240536)
    
    
    # Call to assert_raises(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'ValueError' (line 226)
    ValueError_240539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 226)
    _clean_inputs_240540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 226)
    # Getting the type of 'c' (line 226)
    c_240541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 47), 'c', False)
    keyword_240542 = c_240541
    
    # Obtaining an instance of the builtin type 'list' (line 226)
    list_240543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 226)
    # Adding element type (line 226)
    
    # Obtaining an instance of the builtin type 'list' (line 226)
    list_240544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 226)
    # Adding element type (line 226)
    int_240545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 56), list_240544, int_240545)
    # Adding element type (line 226)
    int_240546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 60), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 56), list_240544, int_240546)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 55), list_240543, list_240544)
    # Adding element type (line 226)
    
    # Obtaining an instance of the builtin type 'list' (line 226)
    list_240547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 64), 'list')
    # Adding type elements to the builtin type 'list' instance (line 226)
    # Adding element type (line 226)
    # Getting the type of 'None' (line 226)
    None_240548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 65), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 64), list_240547, None_240548)
    # Adding element type (line 226)
    int_240549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 71), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 64), list_240547, int_240549)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 55), list_240543, list_240547)
    
    keyword_240550 = list_240543
    # Getting the type of 'b_ub' (line 227)
    b_ub_240551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'b_ub', False)
    keyword_240552 = b_ub_240551
    # Getting the type of 'A_eq' (line 227)
    A_eq_240553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'A_eq', False)
    keyword_240554 = A_eq_240553
    # Getting the type of 'b_eq' (line 227)
    b_eq_240555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 45), 'b_eq', False)
    keyword_240556 = b_eq_240555
    # Getting the type of 'bounds' (line 227)
    bounds_240557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 58), 'bounds', False)
    keyword_240558 = bounds_240557
    kwargs_240559 = {'c': keyword_240542, 'A_ub': keyword_240550, 'A_eq': keyword_240554, 'bounds': keyword_240558, 'b_ub': keyword_240552, 'b_eq': keyword_240556}
    # Getting the type of 'assert_raises' (line 226)
    assert_raises_240538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 226)
    assert_raises_call_result_240560 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), assert_raises_240538, *[ValueError_240539, _clean_inputs_240540], **kwargs_240559)
    
    
    # Call to assert_raises(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'ValueError' (line 229)
    ValueError_240562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 230)
    _clean_inputs_240563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 228)
    # Getting the type of 'c' (line 231)
    c_240564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 10), 'c', False)
    keyword_240565 = c_240564
    # Getting the type of 'A_ub' (line 232)
    A_ub_240566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 13), 'A_ub', False)
    keyword_240567 = A_ub_240566
    
    # Obtaining an instance of the builtin type 'list' (line 233)
    list_240568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 233)
    # Adding element type (line 233)
    # Getting the type of 'np' (line 234)
    np_240569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'np', False)
    # Obtaining the member 'inf' of a type (line 234)
    inf_240570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), np_240569, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 13), list_240568, inf_240570)
    # Adding element type (line 233)
    int_240571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 13), list_240568, int_240571)
    
    keyword_240572 = list_240568
    # Getting the type of 'A_eq' (line 236)
    A_eq_240573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'A_eq', False)
    keyword_240574 = A_eq_240573
    # Getting the type of 'b_eq' (line 237)
    b_eq_240575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 13), 'b_eq', False)
    keyword_240576 = b_eq_240575
    # Getting the type of 'bounds' (line 238)
    bounds_240577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'bounds', False)
    keyword_240578 = bounds_240577
    kwargs_240579 = {'c': keyword_240565, 'A_ub': keyword_240567, 'A_eq': keyword_240574, 'bounds': keyword_240578, 'b_ub': keyword_240572, 'b_eq': keyword_240576}
    # Getting the type of 'assert_raises' (line 228)
    assert_raises_240561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 228)
    assert_raises_call_result_240580 = invoke(stypy.reporting.localization.Localization(__file__, 228, 4), assert_raises_240561, *[ValueError_240562, _clean_inputs_240563], **kwargs_240579)
    
    
    # Call to assert_raises(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'ValueError' (line 239)
    ValueError_240582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 239)
    _clean_inputs_240583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 239)
    # Getting the type of 'c' (line 239)
    c_240584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 47), 'c', False)
    keyword_240585 = c_240584
    # Getting the type of 'A_ub' (line 239)
    A_ub_240586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 55), 'A_ub', False)
    keyword_240587 = A_ub_240586
    # Getting the type of 'b_ub' (line 239)
    b_ub_240588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 66), 'b_ub', False)
    keyword_240589 = b_ub_240588
    
    # Obtaining an instance of the builtin type 'list' (line 239)
    list_240590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 77), 'list')
    # Adding type elements to the builtin type 'list' instance (line 239)
    # Adding element type (line 239)
    
    # Obtaining an instance of the builtin type 'list' (line 240)
    list_240591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 240)
    # Adding element type (line 240)
    int_240592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 18), list_240591, int_240592)
    # Adding element type (line 240)
    int_240593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 18), list_240591, int_240593)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 77), list_240590, list_240591)
    # Adding element type (line 239)
    
    # Obtaining an instance of the builtin type 'list' (line 240)
    list_240594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 240)
    # Adding element type (line 240)
    int_240595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 26), list_240594, int_240595)
    # Adding element type (line 240)
    
    # Getting the type of 'np' (line 240)
    np_240596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'np', False)
    # Obtaining the member 'inf' of a type (line 240)
    inf_240597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 31), np_240596, 'inf')
    # Applying the 'usub' unary operator (line 240)
    result___neg___240598 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 30), 'usub', inf_240597)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 26), list_240594, result___neg___240598)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 77), list_240590, list_240594)
    
    keyword_240599 = list_240590
    # Getting the type of 'b_eq' (line 240)
    b_eq_240600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 46), 'b_eq', False)
    keyword_240601 = b_eq_240600
    # Getting the type of 'bounds' (line 240)
    bounds_240602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 59), 'bounds', False)
    keyword_240603 = bounds_240602
    kwargs_240604 = {'c': keyword_240585, 'A_ub': keyword_240587, 'A_eq': keyword_240599, 'bounds': keyword_240603, 'b_ub': keyword_240589, 'b_eq': keyword_240601}
    # Getting the type of 'assert_raises' (line 239)
    assert_raises_240581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 239)
    assert_raises_call_result_240605 = invoke(stypy.reporting.localization.Localization(__file__, 239, 4), assert_raises_240581, *[ValueError_240582, _clean_inputs_240583], **kwargs_240604)
    
    
    # Call to assert_raises(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'ValueError' (line 242)
    ValueError_240607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 243)
    _clean_inputs_240608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), '_clean_inputs', False)
    # Processing the call keyword arguments (line 241)
    # Getting the type of 'c' (line 244)
    c_240609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 10), 'c', False)
    keyword_240610 = c_240609
    # Getting the type of 'A_ub' (line 245)
    A_ub_240611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'A_ub', False)
    keyword_240612 = A_ub_240611
    # Getting the type of 'b_ub' (line 246)
    b_ub_240613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 13), 'b_ub', False)
    keyword_240614 = b_ub_240613
    # Getting the type of 'A_eq' (line 247)
    A_eq_240615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'A_eq', False)
    keyword_240616 = A_eq_240615
    
    # Obtaining an instance of the builtin type 'list' (line 248)
    list_240617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 248)
    # Adding element type (line 248)
    int_240618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 13), list_240617, int_240618)
    # Adding element type (line 248)
    # Getting the type of 'np' (line 250)
    np_240619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'np', False)
    # Obtaining the member 'nan' of a type (line 250)
    nan_240620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), np_240619, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 13), list_240617, nan_240620)
    
    keyword_240621 = list_240617
    # Getting the type of 'bounds' (line 251)
    bounds_240622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'bounds', False)
    keyword_240623 = bounds_240622
    kwargs_240624 = {'c': keyword_240610, 'A_ub': keyword_240612, 'A_eq': keyword_240616, 'bounds': keyword_240623, 'b_ub': keyword_240614, 'b_eq': keyword_240621}
    # Getting the type of 'assert_raises' (line 241)
    assert_raises_240606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 241)
    assert_raises_call_result_240625 = invoke(stypy.reporting.localization.Localization(__file__, 241, 4), assert_raises_240606, *[ValueError_240607, _clean_inputs_240608], **kwargs_240624)
    
    
    # ################# End of 'test_non_finite_errors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_non_finite_errors' in the type store
    # Getting the type of 'stypy_return_type' (line 206)
    stypy_return_type_240626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_240626)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_non_finite_errors'
    return stypy_return_type_240626

# Assigning a type to the variable 'test_non_finite_errors' (line 206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'test_non_finite_errors', test_non_finite_errors)

@norecursion
def test__clean_inputs1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test__clean_inputs1'
    module_type_store = module_type_store.open_function_context('test__clean_inputs1', 254, 0, False)
    
    # Passed parameters checking function
    test__clean_inputs1.stypy_localization = localization
    test__clean_inputs1.stypy_type_of_self = None
    test__clean_inputs1.stypy_type_store = module_type_store
    test__clean_inputs1.stypy_function_name = 'test__clean_inputs1'
    test__clean_inputs1.stypy_param_names_list = []
    test__clean_inputs1.stypy_varargs_param_name = None
    test__clean_inputs1.stypy_kwargs_param_name = None
    test__clean_inputs1.stypy_call_defaults = defaults
    test__clean_inputs1.stypy_call_varargs = varargs
    test__clean_inputs1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test__clean_inputs1', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test__clean_inputs1', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test__clean_inputs1(...)' code ##################

    
    # Assigning a List to a Name (line 255):
    
    # Obtaining an instance of the builtin type 'list' (line 255)
    list_240627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 255)
    # Adding element type (line 255)
    int_240628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 8), list_240627, int_240628)
    # Adding element type (line 255)
    int_240629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 8), list_240627, int_240629)
    
    # Assigning a type to the variable 'c' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'c', list_240627)
    
    # Assigning a List to a Name (line 256):
    
    # Obtaining an instance of the builtin type 'list' (line 256)
    list_240630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 256)
    # Adding element type (line 256)
    
    # Obtaining an instance of the builtin type 'list' (line 256)
    list_240631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 256)
    # Adding element type (line 256)
    int_240632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 12), list_240631, int_240632)
    # Adding element type (line 256)
    int_240633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 12), list_240631, int_240633)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 11), list_240630, list_240631)
    # Adding element type (line 256)
    
    # Obtaining an instance of the builtin type 'list' (line 256)
    list_240634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 256)
    # Adding element type (line 256)
    int_240635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 20), list_240634, int_240635)
    # Adding element type (line 256)
    int_240636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 20), list_240634, int_240636)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 11), list_240630, list_240634)
    
    # Assigning a type to the variable 'A_ub' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'A_ub', list_240630)
    
    # Assigning a List to a Name (line 257):
    
    # Obtaining an instance of the builtin type 'list' (line 257)
    list_240637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 257)
    # Adding element type (line 257)
    int_240638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 11), list_240637, int_240638)
    # Adding element type (line 257)
    int_240639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 11), list_240637, int_240639)
    
    # Assigning a type to the variable 'b_ub' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'b_ub', list_240637)
    
    # Assigning a List to a Name (line 258):
    
    # Obtaining an instance of the builtin type 'list' (line 258)
    list_240640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 258)
    # Adding element type (line 258)
    
    # Obtaining an instance of the builtin type 'list' (line 258)
    list_240641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 258)
    # Adding element type (line 258)
    int_240642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 12), list_240641, int_240642)
    # Adding element type (line 258)
    int_240643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 12), list_240641, int_240643)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 11), list_240640, list_240641)
    # Adding element type (line 258)
    
    # Obtaining an instance of the builtin type 'list' (line 258)
    list_240644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 258)
    # Adding element type (line 258)
    int_240645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 20), list_240644, int_240645)
    # Adding element type (line 258)
    int_240646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 20), list_240644, int_240646)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 11), list_240640, list_240644)
    
    # Assigning a type to the variable 'A_eq' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'A_eq', list_240640)
    
    # Assigning a List to a Name (line 259):
    
    # Obtaining an instance of the builtin type 'list' (line 259)
    list_240647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 259)
    # Adding element type (line 259)
    int_240648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), list_240647, int_240648)
    # Adding element type (line 259)
    int_240649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), list_240647, int_240649)
    
    # Assigning a type to the variable 'b_eq' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'b_eq', list_240647)
    
    # Assigning a Name to a Name (line 260):
    # Getting the type of 'None' (line 260)
    None_240650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 13), 'None')
    # Assigning a type to the variable 'bounds' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'bounds', None_240650)
    
    # Assigning a Call to a Name (line 261):
    
    # Call to _clean_inputs(...): (line 261)
    # Processing the call keyword arguments (line 261)
    # Getting the type of 'c' (line 262)
    c_240652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 10), 'c', False)
    keyword_240653 = c_240652
    # Getting the type of 'A_ub' (line 263)
    A_ub_240654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 'A_ub', False)
    keyword_240655 = A_ub_240654
    # Getting the type of 'b_ub' (line 264)
    b_ub_240656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'b_ub', False)
    keyword_240657 = b_ub_240656
    # Getting the type of 'A_eq' (line 265)
    A_eq_240658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 13), 'A_eq', False)
    keyword_240659 = A_eq_240658
    # Getting the type of 'b_eq' (line 266)
    b_eq_240660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 13), 'b_eq', False)
    keyword_240661 = b_eq_240660
    # Getting the type of 'bounds' (line 267)
    bounds_240662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'bounds', False)
    keyword_240663 = bounds_240662
    kwargs_240664 = {'c': keyword_240653, 'A_ub': keyword_240655, 'A_eq': keyword_240659, 'bounds': keyword_240663, 'b_ub': keyword_240657, 'b_eq': keyword_240661}
    # Getting the type of '_clean_inputs' (line 261)
    _clean_inputs_240651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 261)
    _clean_inputs_call_result_240665 = invoke(stypy.reporting.localization.Localization(__file__, 261, 14), _clean_inputs_240651, *[], **kwargs_240664)
    
    # Assigning a type to the variable 'outputs' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'outputs', _clean_inputs_call_result_240665)
    
    # Call to assert_allclose(...): (line 268)
    # Processing the call arguments (line 268)
    
    # Obtaining the type of the subscript
    int_240667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 28), 'int')
    # Getting the type of 'outputs' (line 268)
    outputs_240668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 268)
    getitem___240669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 20), outputs_240668, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 268)
    subscript_call_result_240670 = invoke(stypy.reporting.localization.Localization(__file__, 268, 20), getitem___240669, int_240667)
    
    
    # Call to array(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'c' (line 268)
    c_240673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 41), 'c', False)
    # Processing the call keyword arguments (line 268)
    kwargs_240674 = {}
    # Getting the type of 'np' (line 268)
    np_240671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 268)
    array_240672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 32), np_240671, 'array')
    # Calling array(args, kwargs) (line 268)
    array_call_result_240675 = invoke(stypy.reporting.localization.Localization(__file__, 268, 32), array_240672, *[c_240673], **kwargs_240674)
    
    # Processing the call keyword arguments (line 268)
    kwargs_240676 = {}
    # Getting the type of 'assert_allclose' (line 268)
    assert_allclose_240666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 268)
    assert_allclose_call_result_240677 = invoke(stypy.reporting.localization.Localization(__file__, 268, 4), assert_allclose_240666, *[subscript_call_result_240670, array_call_result_240675], **kwargs_240676)
    
    
    # Call to assert_allclose(...): (line 269)
    # Processing the call arguments (line 269)
    
    # Obtaining the type of the subscript
    int_240679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 28), 'int')
    # Getting the type of 'outputs' (line 269)
    outputs_240680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 269)
    getitem___240681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 20), outputs_240680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 269)
    subscript_call_result_240682 = invoke(stypy.reporting.localization.Localization(__file__, 269, 20), getitem___240681, int_240679)
    
    
    # Call to array(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'A_ub' (line 269)
    A_ub_240685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 41), 'A_ub', False)
    # Processing the call keyword arguments (line 269)
    kwargs_240686 = {}
    # Getting the type of 'np' (line 269)
    np_240683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 269)
    array_240684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 32), np_240683, 'array')
    # Calling array(args, kwargs) (line 269)
    array_call_result_240687 = invoke(stypy.reporting.localization.Localization(__file__, 269, 32), array_240684, *[A_ub_240685], **kwargs_240686)
    
    # Processing the call keyword arguments (line 269)
    kwargs_240688 = {}
    # Getting the type of 'assert_allclose' (line 269)
    assert_allclose_240678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 269)
    assert_allclose_call_result_240689 = invoke(stypy.reporting.localization.Localization(__file__, 269, 4), assert_allclose_240678, *[subscript_call_result_240682, array_call_result_240687], **kwargs_240688)
    
    
    # Call to assert_allclose(...): (line 270)
    # Processing the call arguments (line 270)
    
    # Obtaining the type of the subscript
    int_240691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 28), 'int')
    # Getting the type of 'outputs' (line 270)
    outputs_240692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___240693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), outputs_240692, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_240694 = invoke(stypy.reporting.localization.Localization(__file__, 270, 20), getitem___240693, int_240691)
    
    
    # Call to array(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'b_ub' (line 270)
    b_ub_240697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'b_ub', False)
    # Processing the call keyword arguments (line 270)
    kwargs_240698 = {}
    # Getting the type of 'np' (line 270)
    np_240695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 270)
    array_240696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 32), np_240695, 'array')
    # Calling array(args, kwargs) (line 270)
    array_call_result_240699 = invoke(stypy.reporting.localization.Localization(__file__, 270, 32), array_240696, *[b_ub_240697], **kwargs_240698)
    
    # Processing the call keyword arguments (line 270)
    kwargs_240700 = {}
    # Getting the type of 'assert_allclose' (line 270)
    assert_allclose_240690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 270)
    assert_allclose_call_result_240701 = invoke(stypy.reporting.localization.Localization(__file__, 270, 4), assert_allclose_240690, *[subscript_call_result_240694, array_call_result_240699], **kwargs_240700)
    
    
    # Call to assert_allclose(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Obtaining the type of the subscript
    int_240703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 28), 'int')
    # Getting the type of 'outputs' (line 271)
    outputs_240704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 271)
    getitem___240705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 20), outputs_240704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 271)
    subscript_call_result_240706 = invoke(stypy.reporting.localization.Localization(__file__, 271, 20), getitem___240705, int_240703)
    
    
    # Call to array(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'A_eq' (line 271)
    A_eq_240709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 41), 'A_eq', False)
    # Processing the call keyword arguments (line 271)
    kwargs_240710 = {}
    # Getting the type of 'np' (line 271)
    np_240707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 271)
    array_240708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 32), np_240707, 'array')
    # Calling array(args, kwargs) (line 271)
    array_call_result_240711 = invoke(stypy.reporting.localization.Localization(__file__, 271, 32), array_240708, *[A_eq_240709], **kwargs_240710)
    
    # Processing the call keyword arguments (line 271)
    kwargs_240712 = {}
    # Getting the type of 'assert_allclose' (line 271)
    assert_allclose_240702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 271)
    assert_allclose_call_result_240713 = invoke(stypy.reporting.localization.Localization(__file__, 271, 4), assert_allclose_240702, *[subscript_call_result_240706, array_call_result_240711], **kwargs_240712)
    
    
    # Call to assert_allclose(...): (line 272)
    # Processing the call arguments (line 272)
    
    # Obtaining the type of the subscript
    int_240715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 28), 'int')
    # Getting the type of 'outputs' (line 272)
    outputs_240716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___240717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 20), outputs_240716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_240718 = invoke(stypy.reporting.localization.Localization(__file__, 272, 20), getitem___240717, int_240715)
    
    
    # Call to array(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'b_eq' (line 272)
    b_eq_240721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 41), 'b_eq', False)
    # Processing the call keyword arguments (line 272)
    kwargs_240722 = {}
    # Getting the type of 'np' (line 272)
    np_240719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 272)
    array_240720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 32), np_240719, 'array')
    # Calling array(args, kwargs) (line 272)
    array_call_result_240723 = invoke(stypy.reporting.localization.Localization(__file__, 272, 32), array_240720, *[b_eq_240721], **kwargs_240722)
    
    # Processing the call keyword arguments (line 272)
    kwargs_240724 = {}
    # Getting the type of 'assert_allclose' (line 272)
    assert_allclose_240714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 272)
    assert_allclose_call_result_240725 = invoke(stypy.reporting.localization.Localization(__file__, 272, 4), assert_allclose_240714, *[subscript_call_result_240718, array_call_result_240723], **kwargs_240724)
    
    
    # Call to assert_(...): (line 273)
    # Processing the call arguments (line 273)
    
    
    # Obtaining the type of the subscript
    int_240727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 20), 'int')
    # Getting the type of 'outputs' (line 273)
    outputs_240728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 273)
    getitem___240729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), outputs_240728, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 273)
    subscript_call_result_240730 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), getitem___240729, int_240727)
    
    
    # Obtaining an instance of the builtin type 'list' (line 273)
    list_240731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 273)
    # Adding element type (line 273)
    
    # Obtaining an instance of the builtin type 'tuple' (line 273)
    tuple_240732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 273)
    # Adding element type (line 273)
    int_240733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 28), tuple_240732, int_240733)
    # Adding element type (line 273)
    # Getting the type of 'None' (line 273)
    None_240734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 31), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 28), tuple_240732, None_240734)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 26), list_240731, tuple_240732)
    
    int_240735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 40), 'int')
    # Applying the binary operator '*' (line 273)
    result_mul_240736 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 26), '*', list_240731, int_240735)
    
    # Applying the binary operator '==' (line 273)
    result_eq_240737 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 12), '==', subscript_call_result_240730, result_mul_240736)
    
    str_240738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 43), 'str', '')
    # Processing the call keyword arguments (line 273)
    kwargs_240739 = {}
    # Getting the type of 'assert_' (line 273)
    assert__240726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 273)
    assert__call_result_240740 = invoke(stypy.reporting.localization.Localization(__file__, 273, 4), assert__240726, *[result_eq_240737, str_240738], **kwargs_240739)
    
    
    # Call to assert_(...): (line 275)
    # Processing the call arguments (line 275)
    
    
    # Obtaining the type of the subscript
    int_240742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 20), 'int')
    # Getting the type of 'outputs' (line 275)
    outputs_240743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 275)
    getitem___240744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), outputs_240743, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 275)
    subscript_call_result_240745 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), getitem___240744, int_240742)
    
    # Obtaining the member 'shape' of a type (line 275)
    shape_240746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), subscript_call_result_240745, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 275)
    tuple_240747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 275)
    # Adding element type (line 275)
    int_240748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 33), tuple_240747, int_240748)
    
    # Applying the binary operator '==' (line 275)
    result_eq_240749 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 12), '==', shape_240746, tuple_240747)
    
    str_240750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 38), 'str', '')
    # Processing the call keyword arguments (line 275)
    kwargs_240751 = {}
    # Getting the type of 'assert_' (line 275)
    assert__240741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 275)
    assert__call_result_240752 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), assert__240741, *[result_eq_240749, str_240750], **kwargs_240751)
    
    
    # Call to assert_(...): (line 276)
    # Processing the call arguments (line 276)
    
    
    # Obtaining the type of the subscript
    int_240754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 20), 'int')
    # Getting the type of 'outputs' (line 276)
    outputs_240755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 276)
    getitem___240756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), outputs_240755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 276)
    subscript_call_result_240757 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), getitem___240756, int_240754)
    
    # Obtaining the member 'shape' of a type (line 276)
    shape_240758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), subscript_call_result_240757, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 276)
    tuple_240759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 276)
    # Adding element type (line 276)
    int_240760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 33), tuple_240759, int_240760)
    # Adding element type (line 276)
    int_240761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 33), tuple_240759, int_240761)
    
    # Applying the binary operator '==' (line 276)
    result_eq_240762 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 12), '==', shape_240758, tuple_240759)
    
    str_240763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 40), 'str', '')
    # Processing the call keyword arguments (line 276)
    kwargs_240764 = {}
    # Getting the type of 'assert_' (line 276)
    assert__240753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 276)
    assert__call_result_240765 = invoke(stypy.reporting.localization.Localization(__file__, 276, 4), assert__240753, *[result_eq_240762, str_240763], **kwargs_240764)
    
    
    # Call to assert_(...): (line 277)
    # Processing the call arguments (line 277)
    
    
    # Obtaining the type of the subscript
    int_240767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 20), 'int')
    # Getting the type of 'outputs' (line 277)
    outputs_240768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___240769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), outputs_240768, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_240770 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), getitem___240769, int_240767)
    
    # Obtaining the member 'shape' of a type (line 277)
    shape_240771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), subscript_call_result_240770, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 277)
    tuple_240772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 277)
    # Adding element type (line 277)
    int_240773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 33), tuple_240772, int_240773)
    
    # Applying the binary operator '==' (line 277)
    result_eq_240774 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 12), '==', shape_240771, tuple_240772)
    
    str_240775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 38), 'str', '')
    # Processing the call keyword arguments (line 277)
    kwargs_240776 = {}
    # Getting the type of 'assert_' (line 277)
    assert__240766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 277)
    assert__call_result_240777 = invoke(stypy.reporting.localization.Localization(__file__, 277, 4), assert__240766, *[result_eq_240774, str_240775], **kwargs_240776)
    
    
    # Call to assert_(...): (line 278)
    # Processing the call arguments (line 278)
    
    
    # Obtaining the type of the subscript
    int_240779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 20), 'int')
    # Getting the type of 'outputs' (line 278)
    outputs_240780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___240781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), outputs_240780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_240782 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), getitem___240781, int_240779)
    
    # Obtaining the member 'shape' of a type (line 278)
    shape_240783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), subscript_call_result_240782, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 278)
    tuple_240784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 278)
    # Adding element type (line 278)
    int_240785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 33), tuple_240784, int_240785)
    # Adding element type (line 278)
    int_240786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 33), tuple_240784, int_240786)
    
    # Applying the binary operator '==' (line 278)
    result_eq_240787 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 12), '==', shape_240783, tuple_240784)
    
    str_240788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 40), 'str', '')
    # Processing the call keyword arguments (line 278)
    kwargs_240789 = {}
    # Getting the type of 'assert_' (line 278)
    assert__240778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 278)
    assert__call_result_240790 = invoke(stypy.reporting.localization.Localization(__file__, 278, 4), assert__240778, *[result_eq_240787, str_240788], **kwargs_240789)
    
    
    # Call to assert_(...): (line 279)
    # Processing the call arguments (line 279)
    
    
    # Obtaining the type of the subscript
    int_240792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 20), 'int')
    # Getting the type of 'outputs' (line 279)
    outputs_240793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___240794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), outputs_240793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_240795 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), getitem___240794, int_240792)
    
    # Obtaining the member 'shape' of a type (line 279)
    shape_240796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), subscript_call_result_240795, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 279)
    tuple_240797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 279)
    # Adding element type (line 279)
    int_240798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 33), tuple_240797, int_240798)
    
    # Applying the binary operator '==' (line 279)
    result_eq_240799 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 12), '==', shape_240796, tuple_240797)
    
    str_240800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 38), 'str', '')
    # Processing the call keyword arguments (line 279)
    kwargs_240801 = {}
    # Getting the type of 'assert_' (line 279)
    assert__240791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 279)
    assert__call_result_240802 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), assert__240791, *[result_eq_240799, str_240800], **kwargs_240801)
    
    
    # ################# End of 'test__clean_inputs1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test__clean_inputs1' in the type store
    # Getting the type of 'stypy_return_type' (line 254)
    stypy_return_type_240803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_240803)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test__clean_inputs1'
    return stypy_return_type_240803

# Assigning a type to the variable 'test__clean_inputs1' (line 254)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'test__clean_inputs1', test__clean_inputs1)

@norecursion
def test__clean_inputs2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test__clean_inputs2'
    module_type_store = module_type_store.open_function_context('test__clean_inputs2', 282, 0, False)
    
    # Passed parameters checking function
    test__clean_inputs2.stypy_localization = localization
    test__clean_inputs2.stypy_type_of_self = None
    test__clean_inputs2.stypy_type_store = module_type_store
    test__clean_inputs2.stypy_function_name = 'test__clean_inputs2'
    test__clean_inputs2.stypy_param_names_list = []
    test__clean_inputs2.stypy_varargs_param_name = None
    test__clean_inputs2.stypy_kwargs_param_name = None
    test__clean_inputs2.stypy_call_defaults = defaults
    test__clean_inputs2.stypy_call_varargs = varargs
    test__clean_inputs2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test__clean_inputs2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test__clean_inputs2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test__clean_inputs2(...)' code ##################

    
    # Assigning a Num to a Name (line 283):
    int_240804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 8), 'int')
    # Assigning a type to the variable 'c' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'c', int_240804)
    
    # Assigning a List to a Name (line 284):
    
    # Obtaining an instance of the builtin type 'list' (line 284)
    list_240805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 284)
    # Adding element type (line 284)
    
    # Obtaining an instance of the builtin type 'list' (line 284)
    list_240806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 284)
    # Adding element type (line 284)
    int_240807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 12), list_240806, int_240807)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 11), list_240805, list_240806)
    
    # Assigning a type to the variable 'A_ub' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'A_ub', list_240805)
    
    # Assigning a Num to a Name (line 285):
    int_240808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 11), 'int')
    # Assigning a type to the variable 'b_ub' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'b_ub', int_240808)
    
    # Assigning a List to a Name (line 286):
    
    # Obtaining an instance of the builtin type 'list' (line 286)
    list_240809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 286)
    # Adding element type (line 286)
    
    # Obtaining an instance of the builtin type 'list' (line 286)
    list_240810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 286)
    # Adding element type (line 286)
    int_240811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 12), list_240810, int_240811)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 11), list_240809, list_240810)
    
    # Assigning a type to the variable 'A_eq' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'A_eq', list_240809)
    
    # Assigning a Num to a Name (line 287):
    int_240812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 11), 'int')
    # Assigning a type to the variable 'b_eq' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'b_eq', int_240812)
    
    # Assigning a Tuple to a Name (line 288):
    
    # Obtaining an instance of the builtin type 'tuple' (line 288)
    tuple_240813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 288)
    # Adding element type (line 288)
    int_240814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 14), tuple_240813, int_240814)
    # Adding element type (line 288)
    int_240815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 14), tuple_240813, int_240815)
    
    # Assigning a type to the variable 'bounds' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'bounds', tuple_240813)
    
    # Assigning a Call to a Name (line 289):
    
    # Call to _clean_inputs(...): (line 289)
    # Processing the call keyword arguments (line 289)
    # Getting the type of 'c' (line 290)
    c_240817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 10), 'c', False)
    keyword_240818 = c_240817
    # Getting the type of 'A_ub' (line 291)
    A_ub_240819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 13), 'A_ub', False)
    keyword_240820 = A_ub_240819
    # Getting the type of 'b_ub' (line 292)
    b_ub_240821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 13), 'b_ub', False)
    keyword_240822 = b_ub_240821
    # Getting the type of 'A_eq' (line 293)
    A_eq_240823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'A_eq', False)
    keyword_240824 = A_eq_240823
    # Getting the type of 'b_eq' (line 294)
    b_eq_240825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 13), 'b_eq', False)
    keyword_240826 = b_eq_240825
    # Getting the type of 'bounds' (line 295)
    bounds_240827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'bounds', False)
    keyword_240828 = bounds_240827
    kwargs_240829 = {'c': keyword_240818, 'A_ub': keyword_240820, 'A_eq': keyword_240824, 'bounds': keyword_240828, 'b_ub': keyword_240822, 'b_eq': keyword_240826}
    # Getting the type of '_clean_inputs' (line 289)
    _clean_inputs_240816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 289)
    _clean_inputs_call_result_240830 = invoke(stypy.reporting.localization.Localization(__file__, 289, 14), _clean_inputs_240816, *[], **kwargs_240829)
    
    # Assigning a type to the variable 'outputs' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'outputs', _clean_inputs_call_result_240830)
    
    # Call to assert_allclose(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Obtaining the type of the subscript
    int_240832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 28), 'int')
    # Getting the type of 'outputs' (line 296)
    outputs_240833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___240834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 20), outputs_240833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_240835 = invoke(stypy.reporting.localization.Localization(__file__, 296, 20), getitem___240834, int_240832)
    
    
    # Call to array(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'c' (line 296)
    c_240838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 41), 'c', False)
    # Processing the call keyword arguments (line 296)
    kwargs_240839 = {}
    # Getting the type of 'np' (line 296)
    np_240836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 296)
    array_240837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 32), np_240836, 'array')
    # Calling array(args, kwargs) (line 296)
    array_call_result_240840 = invoke(stypy.reporting.localization.Localization(__file__, 296, 32), array_240837, *[c_240838], **kwargs_240839)
    
    # Processing the call keyword arguments (line 296)
    kwargs_240841 = {}
    # Getting the type of 'assert_allclose' (line 296)
    assert_allclose_240831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 296)
    assert_allclose_call_result_240842 = invoke(stypy.reporting.localization.Localization(__file__, 296, 4), assert_allclose_240831, *[subscript_call_result_240835, array_call_result_240840], **kwargs_240841)
    
    
    # Call to assert_allclose(...): (line 297)
    # Processing the call arguments (line 297)
    
    # Obtaining the type of the subscript
    int_240844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 28), 'int')
    # Getting the type of 'outputs' (line 297)
    outputs_240845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 297)
    getitem___240846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 20), outputs_240845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 297)
    subscript_call_result_240847 = invoke(stypy.reporting.localization.Localization(__file__, 297, 20), getitem___240846, int_240844)
    
    
    # Call to array(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'A_ub' (line 297)
    A_ub_240850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 41), 'A_ub', False)
    # Processing the call keyword arguments (line 297)
    kwargs_240851 = {}
    # Getting the type of 'np' (line 297)
    np_240848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 297)
    array_240849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 32), np_240848, 'array')
    # Calling array(args, kwargs) (line 297)
    array_call_result_240852 = invoke(stypy.reporting.localization.Localization(__file__, 297, 32), array_240849, *[A_ub_240850], **kwargs_240851)
    
    # Processing the call keyword arguments (line 297)
    kwargs_240853 = {}
    # Getting the type of 'assert_allclose' (line 297)
    assert_allclose_240843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 297)
    assert_allclose_call_result_240854 = invoke(stypy.reporting.localization.Localization(__file__, 297, 4), assert_allclose_240843, *[subscript_call_result_240847, array_call_result_240852], **kwargs_240853)
    
    
    # Call to assert_allclose(...): (line 298)
    # Processing the call arguments (line 298)
    
    # Obtaining the type of the subscript
    int_240856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 28), 'int')
    # Getting the type of 'outputs' (line 298)
    outputs_240857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___240858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 20), outputs_240857, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_240859 = invoke(stypy.reporting.localization.Localization(__file__, 298, 20), getitem___240858, int_240856)
    
    
    # Call to array(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'b_ub' (line 298)
    b_ub_240862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'b_ub', False)
    # Processing the call keyword arguments (line 298)
    kwargs_240863 = {}
    # Getting the type of 'np' (line 298)
    np_240860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 298)
    array_240861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 32), np_240860, 'array')
    # Calling array(args, kwargs) (line 298)
    array_call_result_240864 = invoke(stypy.reporting.localization.Localization(__file__, 298, 32), array_240861, *[b_ub_240862], **kwargs_240863)
    
    # Processing the call keyword arguments (line 298)
    kwargs_240865 = {}
    # Getting the type of 'assert_allclose' (line 298)
    assert_allclose_240855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 298)
    assert_allclose_call_result_240866 = invoke(stypy.reporting.localization.Localization(__file__, 298, 4), assert_allclose_240855, *[subscript_call_result_240859, array_call_result_240864], **kwargs_240865)
    
    
    # Call to assert_allclose(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Obtaining the type of the subscript
    int_240868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 28), 'int')
    # Getting the type of 'outputs' (line 299)
    outputs_240869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___240870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 20), outputs_240869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_240871 = invoke(stypy.reporting.localization.Localization(__file__, 299, 20), getitem___240870, int_240868)
    
    
    # Call to array(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'A_eq' (line 299)
    A_eq_240874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 41), 'A_eq', False)
    # Processing the call keyword arguments (line 299)
    kwargs_240875 = {}
    # Getting the type of 'np' (line 299)
    np_240872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 299)
    array_240873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 32), np_240872, 'array')
    # Calling array(args, kwargs) (line 299)
    array_call_result_240876 = invoke(stypy.reporting.localization.Localization(__file__, 299, 32), array_240873, *[A_eq_240874], **kwargs_240875)
    
    # Processing the call keyword arguments (line 299)
    kwargs_240877 = {}
    # Getting the type of 'assert_allclose' (line 299)
    assert_allclose_240867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 299)
    assert_allclose_call_result_240878 = invoke(stypy.reporting.localization.Localization(__file__, 299, 4), assert_allclose_240867, *[subscript_call_result_240871, array_call_result_240876], **kwargs_240877)
    
    
    # Call to assert_allclose(...): (line 300)
    # Processing the call arguments (line 300)
    
    # Obtaining the type of the subscript
    int_240880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 28), 'int')
    # Getting the type of 'outputs' (line 300)
    outputs_240881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 300)
    getitem___240882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), outputs_240881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 300)
    subscript_call_result_240883 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), getitem___240882, int_240880)
    
    
    # Call to array(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'b_eq' (line 300)
    b_eq_240886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 41), 'b_eq', False)
    # Processing the call keyword arguments (line 300)
    kwargs_240887 = {}
    # Getting the type of 'np' (line 300)
    np_240884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 300)
    array_240885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 32), np_240884, 'array')
    # Calling array(args, kwargs) (line 300)
    array_call_result_240888 = invoke(stypy.reporting.localization.Localization(__file__, 300, 32), array_240885, *[b_eq_240886], **kwargs_240887)
    
    # Processing the call keyword arguments (line 300)
    kwargs_240889 = {}
    # Getting the type of 'assert_allclose' (line 300)
    assert_allclose_240879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 300)
    assert_allclose_call_result_240890 = invoke(stypy.reporting.localization.Localization(__file__, 300, 4), assert_allclose_240879, *[subscript_call_result_240883, array_call_result_240888], **kwargs_240889)
    
    
    # Call to assert_(...): (line 301)
    # Processing the call arguments (line 301)
    
    
    # Obtaining the type of the subscript
    int_240892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 20), 'int')
    # Getting the type of 'outputs' (line 301)
    outputs_240893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___240894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), outputs_240893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_240895 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), getitem___240894, int_240892)
    
    
    # Obtaining an instance of the builtin type 'list' (line 301)
    list_240896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 301)
    # Adding element type (line 301)
    
    # Obtaining an instance of the builtin type 'tuple' (line 301)
    tuple_240897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 301)
    # Adding element type (line 301)
    int_240898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 28), tuple_240897, int_240898)
    # Adding element type (line 301)
    int_240899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 28), tuple_240897, int_240899)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 26), list_240896, tuple_240897)
    
    # Applying the binary operator '==' (line 301)
    result_eq_240900 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 12), '==', subscript_call_result_240895, list_240896)
    
    str_240901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 36), 'str', '')
    # Processing the call keyword arguments (line 301)
    kwargs_240902 = {}
    # Getting the type of 'assert_' (line 301)
    assert__240891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 301)
    assert__call_result_240903 = invoke(stypy.reporting.localization.Localization(__file__, 301, 4), assert__240891, *[result_eq_240900, str_240901], **kwargs_240902)
    
    
    # Call to assert_(...): (line 303)
    # Processing the call arguments (line 303)
    
    
    # Obtaining the type of the subscript
    int_240905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 20), 'int')
    # Getting the type of 'outputs' (line 303)
    outputs_240906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___240907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 12), outputs_240906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_240908 = invoke(stypy.reporting.localization.Localization(__file__, 303, 12), getitem___240907, int_240905)
    
    # Obtaining the member 'shape' of a type (line 303)
    shape_240909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 12), subscript_call_result_240908, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 303)
    tuple_240910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 303)
    # Adding element type (line 303)
    int_240911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 33), tuple_240910, int_240911)
    
    # Applying the binary operator '==' (line 303)
    result_eq_240912 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 12), '==', shape_240909, tuple_240910)
    
    str_240913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 38), 'str', '')
    # Processing the call keyword arguments (line 303)
    kwargs_240914 = {}
    # Getting the type of 'assert_' (line 303)
    assert__240904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 303)
    assert__call_result_240915 = invoke(stypy.reporting.localization.Localization(__file__, 303, 4), assert__240904, *[result_eq_240912, str_240913], **kwargs_240914)
    
    
    # Call to assert_(...): (line 304)
    # Processing the call arguments (line 304)
    
    
    # Obtaining the type of the subscript
    int_240917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 20), 'int')
    # Getting the type of 'outputs' (line 304)
    outputs_240918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___240919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), outputs_240918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_240920 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), getitem___240919, int_240917)
    
    # Obtaining the member 'shape' of a type (line 304)
    shape_240921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), subscript_call_result_240920, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 304)
    tuple_240922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 304)
    # Adding element type (line 304)
    int_240923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 33), tuple_240922, int_240923)
    # Adding element type (line 304)
    int_240924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 33), tuple_240922, int_240924)
    
    # Applying the binary operator '==' (line 304)
    result_eq_240925 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 12), '==', shape_240921, tuple_240922)
    
    str_240926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 40), 'str', '')
    # Processing the call keyword arguments (line 304)
    kwargs_240927 = {}
    # Getting the type of 'assert_' (line 304)
    assert__240916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 304)
    assert__call_result_240928 = invoke(stypy.reporting.localization.Localization(__file__, 304, 4), assert__240916, *[result_eq_240925, str_240926], **kwargs_240927)
    
    
    # Call to assert_(...): (line 305)
    # Processing the call arguments (line 305)
    
    
    # Obtaining the type of the subscript
    int_240930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 20), 'int')
    # Getting the type of 'outputs' (line 305)
    outputs_240931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 305)
    getitem___240932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), outputs_240931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 305)
    subscript_call_result_240933 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), getitem___240932, int_240930)
    
    # Obtaining the member 'shape' of a type (line 305)
    shape_240934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), subscript_call_result_240933, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 305)
    tuple_240935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 305)
    # Adding element type (line 305)
    int_240936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 33), tuple_240935, int_240936)
    
    # Applying the binary operator '==' (line 305)
    result_eq_240937 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 12), '==', shape_240934, tuple_240935)
    
    str_240938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 38), 'str', '')
    # Processing the call keyword arguments (line 305)
    kwargs_240939 = {}
    # Getting the type of 'assert_' (line 305)
    assert__240929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 305)
    assert__call_result_240940 = invoke(stypy.reporting.localization.Localization(__file__, 305, 4), assert__240929, *[result_eq_240937, str_240938], **kwargs_240939)
    
    
    # Call to assert_(...): (line 306)
    # Processing the call arguments (line 306)
    
    
    # Obtaining the type of the subscript
    int_240942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 20), 'int')
    # Getting the type of 'outputs' (line 306)
    outputs_240943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___240944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), outputs_240943, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_240945 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), getitem___240944, int_240942)
    
    # Obtaining the member 'shape' of a type (line 306)
    shape_240946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), subscript_call_result_240945, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 306)
    tuple_240947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 306)
    # Adding element type (line 306)
    int_240948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 33), tuple_240947, int_240948)
    # Adding element type (line 306)
    int_240949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 33), tuple_240947, int_240949)
    
    # Applying the binary operator '==' (line 306)
    result_eq_240950 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 12), '==', shape_240946, tuple_240947)
    
    str_240951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 40), 'str', '')
    # Processing the call keyword arguments (line 306)
    kwargs_240952 = {}
    # Getting the type of 'assert_' (line 306)
    assert__240941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 306)
    assert__call_result_240953 = invoke(stypy.reporting.localization.Localization(__file__, 306, 4), assert__240941, *[result_eq_240950, str_240951], **kwargs_240952)
    
    
    # Call to assert_(...): (line 307)
    # Processing the call arguments (line 307)
    
    
    # Obtaining the type of the subscript
    int_240955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 20), 'int')
    # Getting the type of 'outputs' (line 307)
    outputs_240956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___240957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), outputs_240956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_240958 = invoke(stypy.reporting.localization.Localization(__file__, 307, 12), getitem___240957, int_240955)
    
    # Obtaining the member 'shape' of a type (line 307)
    shape_240959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 12), subscript_call_result_240958, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 307)
    tuple_240960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 307)
    # Adding element type (line 307)
    int_240961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 33), tuple_240960, int_240961)
    
    # Applying the binary operator '==' (line 307)
    result_eq_240962 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 12), '==', shape_240959, tuple_240960)
    
    str_240963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 38), 'str', '')
    # Processing the call keyword arguments (line 307)
    kwargs_240964 = {}
    # Getting the type of 'assert_' (line 307)
    assert__240954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 307)
    assert__call_result_240965 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), assert__240954, *[result_eq_240962, str_240963], **kwargs_240964)
    
    
    # ################# End of 'test__clean_inputs2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test__clean_inputs2' in the type store
    # Getting the type of 'stypy_return_type' (line 282)
    stypy_return_type_240966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_240966)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test__clean_inputs2'
    return stypy_return_type_240966

# Assigning a type to the variable 'test__clean_inputs2' (line 282)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'test__clean_inputs2', test__clean_inputs2)

@norecursion
def test__clean_inputs3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test__clean_inputs3'
    module_type_store = module_type_store.open_function_context('test__clean_inputs3', 310, 0, False)
    
    # Passed parameters checking function
    test__clean_inputs3.stypy_localization = localization
    test__clean_inputs3.stypy_type_of_self = None
    test__clean_inputs3.stypy_type_store = module_type_store
    test__clean_inputs3.stypy_function_name = 'test__clean_inputs3'
    test__clean_inputs3.stypy_param_names_list = []
    test__clean_inputs3.stypy_varargs_param_name = None
    test__clean_inputs3.stypy_kwargs_param_name = None
    test__clean_inputs3.stypy_call_defaults = defaults
    test__clean_inputs3.stypy_call_varargs = varargs
    test__clean_inputs3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test__clean_inputs3', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test__clean_inputs3', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test__clean_inputs3(...)' code ##################

    
    # Assigning a List to a Name (line 311):
    
    # Obtaining an instance of the builtin type 'list' (line 311)
    list_240967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 311)
    # Adding element type (line 311)
    
    # Obtaining an instance of the builtin type 'list' (line 311)
    list_240968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 311)
    # Adding element type (line 311)
    int_240969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 10), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 9), list_240968, int_240969)
    # Adding element type (line 311)
    int_240970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 9), list_240968, int_240970)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), list_240967, list_240968)
    
    # Assigning a type to the variable 'c' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'c', list_240967)
    
    # Assigning a Call to a Name (line 312):
    
    # Call to rand(...): (line 312)
    # Processing the call arguments (line 312)
    int_240974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 26), 'int')
    int_240975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 29), 'int')
    # Processing the call keyword arguments (line 312)
    kwargs_240976 = {}
    # Getting the type of 'np' (line 312)
    np_240971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'np', False)
    # Obtaining the member 'random' of a type (line 312)
    random_240972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), np_240971, 'random')
    # Obtaining the member 'rand' of a type (line 312)
    rand_240973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), random_240972, 'rand')
    # Calling rand(args, kwargs) (line 312)
    rand_call_result_240977 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), rand_240973, *[int_240974, int_240975], **kwargs_240976)
    
    # Assigning a type to the variable 'A_ub' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'A_ub', rand_call_result_240977)
    
    # Assigning a List to a Name (line 313):
    
    # Obtaining an instance of the builtin type 'list' (line 313)
    list_240978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 313)
    # Adding element type (line 313)
    
    # Obtaining an instance of the builtin type 'list' (line 313)
    list_240979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 313)
    # Adding element type (line 313)
    int_240980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 12), list_240979, int_240980)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 11), list_240978, list_240979)
    # Adding element type (line 313)
    
    # Obtaining an instance of the builtin type 'list' (line 313)
    list_240981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 313)
    # Adding element type (line 313)
    int_240982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 17), list_240981, int_240982)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 11), list_240978, list_240981)
    
    # Assigning a type to the variable 'b_ub' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'b_ub', list_240978)
    
    # Assigning a Call to a Name (line 314):
    
    # Call to rand(...): (line 314)
    # Processing the call arguments (line 314)
    int_240986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 26), 'int')
    int_240987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 29), 'int')
    # Processing the call keyword arguments (line 314)
    kwargs_240988 = {}
    # Getting the type of 'np' (line 314)
    np_240983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'np', False)
    # Obtaining the member 'random' of a type (line 314)
    random_240984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 11), np_240983, 'random')
    # Obtaining the member 'rand' of a type (line 314)
    rand_240985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 11), random_240984, 'rand')
    # Calling rand(args, kwargs) (line 314)
    rand_call_result_240989 = invoke(stypy.reporting.localization.Localization(__file__, 314, 11), rand_240985, *[int_240986, int_240987], **kwargs_240988)
    
    # Assigning a type to the variable 'A_eq' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'A_eq', rand_call_result_240989)
    
    # Assigning a List to a Name (line 315):
    
    # Obtaining an instance of the builtin type 'list' (line 315)
    list_240990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 315)
    # Adding element type (line 315)
    
    # Obtaining an instance of the builtin type 'list' (line 315)
    list_240991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 315)
    # Adding element type (line 315)
    int_240992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 12), list_240991, int_240992)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 11), list_240990, list_240991)
    # Adding element type (line 315)
    
    # Obtaining an instance of the builtin type 'list' (line 315)
    list_240993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 315)
    # Adding element type (line 315)
    int_240994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 17), list_240993, int_240994)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 11), list_240990, list_240993)
    
    # Assigning a type to the variable 'b_eq' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'b_eq', list_240990)
    
    # Assigning a List to a Name (line 316):
    
    # Obtaining an instance of the builtin type 'list' (line 316)
    list_240995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 316)
    # Adding element type (line 316)
    
    # Obtaining an instance of the builtin type 'tuple' (line 316)
    tuple_240996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 316)
    # Adding element type (line 316)
    int_240997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 15), tuple_240996, int_240997)
    # Adding element type (line 316)
    int_240998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 15), tuple_240996, int_240998)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 13), list_240995, tuple_240996)
    
    # Assigning a type to the variable 'bounds' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'bounds', list_240995)
    
    # Assigning a Call to a Name (line 317):
    
    # Call to _clean_inputs(...): (line 317)
    # Processing the call keyword arguments (line 317)
    # Getting the type of 'c' (line 318)
    c_241000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 10), 'c', False)
    keyword_241001 = c_241000
    # Getting the type of 'A_ub' (line 319)
    A_ub_241002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 13), 'A_ub', False)
    keyword_241003 = A_ub_241002
    # Getting the type of 'b_ub' (line 320)
    b_ub_241004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 13), 'b_ub', False)
    keyword_241005 = b_ub_241004
    # Getting the type of 'A_eq' (line 321)
    A_eq_241006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 13), 'A_eq', False)
    keyword_241007 = A_eq_241006
    # Getting the type of 'b_eq' (line 322)
    b_eq_241008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 13), 'b_eq', False)
    keyword_241009 = b_eq_241008
    # Getting the type of 'bounds' (line 323)
    bounds_241010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 15), 'bounds', False)
    keyword_241011 = bounds_241010
    kwargs_241012 = {'c': keyword_241001, 'A_ub': keyword_241003, 'A_eq': keyword_241007, 'bounds': keyword_241011, 'b_ub': keyword_241005, 'b_eq': keyword_241009}
    # Getting the type of '_clean_inputs' (line 317)
    _clean_inputs_240999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 317)
    _clean_inputs_call_result_241013 = invoke(stypy.reporting.localization.Localization(__file__, 317, 14), _clean_inputs_240999, *[], **kwargs_241012)
    
    # Assigning a type to the variable 'outputs' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'outputs', _clean_inputs_call_result_241013)
    
    # Call to assert_allclose(...): (line 324)
    # Processing the call arguments (line 324)
    
    # Obtaining the type of the subscript
    int_241015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 28), 'int')
    # Getting the type of 'outputs' (line 324)
    outputs_241016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 324)
    getitem___241017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 20), outputs_241016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 324)
    subscript_call_result_241018 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), getitem___241017, int_241015)
    
    
    # Call to array(...): (line 324)
    # Processing the call arguments (line 324)
    
    # Obtaining an instance of the builtin type 'list' (line 324)
    list_241021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 324)
    # Adding element type (line 324)
    int_241022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 41), list_241021, int_241022)
    # Adding element type (line 324)
    int_241023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 41), list_241021, int_241023)
    
    # Processing the call keyword arguments (line 324)
    kwargs_241024 = {}
    # Getting the type of 'np' (line 324)
    np_241019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 324)
    array_241020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 32), np_241019, 'array')
    # Calling array(args, kwargs) (line 324)
    array_call_result_241025 = invoke(stypy.reporting.localization.Localization(__file__, 324, 32), array_241020, *[list_241021], **kwargs_241024)
    
    # Processing the call keyword arguments (line 324)
    kwargs_241026 = {}
    # Getting the type of 'assert_allclose' (line 324)
    assert_allclose_241014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 324)
    assert_allclose_call_result_241027 = invoke(stypy.reporting.localization.Localization(__file__, 324, 4), assert_allclose_241014, *[subscript_call_result_241018, array_call_result_241025], **kwargs_241026)
    
    
    # Call to assert_allclose(...): (line 325)
    # Processing the call arguments (line 325)
    
    # Obtaining the type of the subscript
    int_241029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'int')
    # Getting the type of 'outputs' (line 325)
    outputs_241030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___241031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 20), outputs_241030, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_241032 = invoke(stypy.reporting.localization.Localization(__file__, 325, 20), getitem___241031, int_241029)
    
    
    # Call to array(...): (line 325)
    # Processing the call arguments (line 325)
    
    # Obtaining an instance of the builtin type 'list' (line 325)
    list_241035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 325)
    # Adding element type (line 325)
    int_241036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 41), list_241035, int_241036)
    # Adding element type (line 325)
    int_241037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 41), list_241035, int_241037)
    
    # Processing the call keyword arguments (line 325)
    kwargs_241038 = {}
    # Getting the type of 'np' (line 325)
    np_241033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 325)
    array_241034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 32), np_241033, 'array')
    # Calling array(args, kwargs) (line 325)
    array_call_result_241039 = invoke(stypy.reporting.localization.Localization(__file__, 325, 32), array_241034, *[list_241035], **kwargs_241038)
    
    # Processing the call keyword arguments (line 325)
    kwargs_241040 = {}
    # Getting the type of 'assert_allclose' (line 325)
    assert_allclose_241028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 325)
    assert_allclose_call_result_241041 = invoke(stypy.reporting.localization.Localization(__file__, 325, 4), assert_allclose_241028, *[subscript_call_result_241032, array_call_result_241039], **kwargs_241040)
    
    
    # Call to assert_allclose(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Obtaining the type of the subscript
    int_241043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 28), 'int')
    # Getting the type of 'outputs' (line 326)
    outputs_241044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 20), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 326)
    getitem___241045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 20), outputs_241044, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 326)
    subscript_call_result_241046 = invoke(stypy.reporting.localization.Localization(__file__, 326, 20), getitem___241045, int_241043)
    
    
    # Call to array(...): (line 326)
    # Processing the call arguments (line 326)
    
    # Obtaining an instance of the builtin type 'list' (line 326)
    list_241049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 326)
    # Adding element type (line 326)
    int_241050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 41), list_241049, int_241050)
    # Adding element type (line 326)
    int_241051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 41), list_241049, int_241051)
    
    # Processing the call keyword arguments (line 326)
    kwargs_241052 = {}
    # Getting the type of 'np' (line 326)
    np_241047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'np', False)
    # Obtaining the member 'array' of a type (line 326)
    array_241048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), np_241047, 'array')
    # Calling array(args, kwargs) (line 326)
    array_call_result_241053 = invoke(stypy.reporting.localization.Localization(__file__, 326, 32), array_241048, *[list_241049], **kwargs_241052)
    
    # Processing the call keyword arguments (line 326)
    kwargs_241054 = {}
    # Getting the type of 'assert_allclose' (line 326)
    assert_allclose_241042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 326)
    assert_allclose_call_result_241055 = invoke(stypy.reporting.localization.Localization(__file__, 326, 4), assert_allclose_241042, *[subscript_call_result_241046, array_call_result_241053], **kwargs_241054)
    
    
    # Call to assert_(...): (line 327)
    # Processing the call arguments (line 327)
    
    
    # Obtaining the type of the subscript
    int_241057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 20), 'int')
    # Getting the type of 'outputs' (line 327)
    outputs_241058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 327)
    getitem___241059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 12), outputs_241058, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 327)
    subscript_call_result_241060 = invoke(stypy.reporting.localization.Localization(__file__, 327, 12), getitem___241059, int_241057)
    
    
    # Obtaining an instance of the builtin type 'list' (line 327)
    list_241061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 327)
    # Adding element type (line 327)
    
    # Obtaining an instance of the builtin type 'tuple' (line 327)
    tuple_241062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 327)
    # Adding element type (line 327)
    int_241063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 28), tuple_241062, int_241063)
    # Adding element type (line 327)
    int_241064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 28), tuple_241062, int_241064)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 26), list_241061, tuple_241062)
    
    int_241065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 37), 'int')
    # Applying the binary operator '*' (line 327)
    result_mul_241066 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 26), '*', list_241061, int_241065)
    
    # Applying the binary operator '==' (line 327)
    result_eq_241067 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 12), '==', subscript_call_result_241060, result_mul_241066)
    
    str_241068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 40), 'str', '')
    # Processing the call keyword arguments (line 327)
    kwargs_241069 = {}
    # Getting the type of 'assert_' (line 327)
    assert__241056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 327)
    assert__call_result_241070 = invoke(stypy.reporting.localization.Localization(__file__, 327, 4), assert__241056, *[result_eq_241067, str_241068], **kwargs_241069)
    
    
    # Call to assert_(...): (line 329)
    # Processing the call arguments (line 329)
    
    
    # Obtaining the type of the subscript
    int_241072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 20), 'int')
    # Getting the type of 'outputs' (line 329)
    outputs_241073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___241074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 12), outputs_241073, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 329)
    subscript_call_result_241075 = invoke(stypy.reporting.localization.Localization(__file__, 329, 12), getitem___241074, int_241072)
    
    # Obtaining the member 'shape' of a type (line 329)
    shape_241076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 12), subscript_call_result_241075, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 329)
    tuple_241077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 329)
    # Adding element type (line 329)
    int_241078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 33), tuple_241077, int_241078)
    
    # Applying the binary operator '==' (line 329)
    result_eq_241079 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 12), '==', shape_241076, tuple_241077)
    
    str_241080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 38), 'str', '')
    # Processing the call keyword arguments (line 329)
    kwargs_241081 = {}
    # Getting the type of 'assert_' (line 329)
    assert__241071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 329)
    assert__call_result_241082 = invoke(stypy.reporting.localization.Localization(__file__, 329, 4), assert__241071, *[result_eq_241079, str_241080], **kwargs_241081)
    
    
    # Call to assert_(...): (line 330)
    # Processing the call arguments (line 330)
    
    
    # Obtaining the type of the subscript
    int_241084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 20), 'int')
    # Getting the type of 'outputs' (line 330)
    outputs_241085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___241086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), outputs_241085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_241087 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), getitem___241086, int_241084)
    
    # Obtaining the member 'shape' of a type (line 330)
    shape_241088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), subscript_call_result_241087, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 330)
    tuple_241089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 330)
    # Adding element type (line 330)
    int_241090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 33), tuple_241089, int_241090)
    
    # Applying the binary operator '==' (line 330)
    result_eq_241091 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 12), '==', shape_241088, tuple_241089)
    
    str_241092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 38), 'str', '')
    # Processing the call keyword arguments (line 330)
    kwargs_241093 = {}
    # Getting the type of 'assert_' (line 330)
    assert__241083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 330)
    assert__call_result_241094 = invoke(stypy.reporting.localization.Localization(__file__, 330, 4), assert__241083, *[result_eq_241091, str_241092], **kwargs_241093)
    
    
    # Call to assert_(...): (line 331)
    # Processing the call arguments (line 331)
    
    
    # Obtaining the type of the subscript
    int_241096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 20), 'int')
    # Getting the type of 'outputs' (line 331)
    outputs_241097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___241098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), outputs_241097, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_241099 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), getitem___241098, int_241096)
    
    # Obtaining the member 'shape' of a type (line 331)
    shape_241100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), subscript_call_result_241099, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 331)
    tuple_241101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 331)
    # Adding element type (line 331)
    int_241102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 33), tuple_241101, int_241102)
    
    # Applying the binary operator '==' (line 331)
    result_eq_241103 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 12), '==', shape_241100, tuple_241101)
    
    str_241104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 38), 'str', '')
    # Processing the call keyword arguments (line 331)
    kwargs_241105 = {}
    # Getting the type of 'assert_' (line 331)
    assert__241095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 331)
    assert__call_result_241106 = invoke(stypy.reporting.localization.Localization(__file__, 331, 4), assert__241095, *[result_eq_241103, str_241104], **kwargs_241105)
    
    
    # ################# End of 'test__clean_inputs3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test__clean_inputs3' in the type store
    # Getting the type of 'stypy_return_type' (line 310)
    stypy_return_type_241107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_241107)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test__clean_inputs3'
    return stypy_return_type_241107

# Assigning a type to the variable 'test__clean_inputs3' (line 310)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'test__clean_inputs3', test__clean_inputs3)

@norecursion
def test_bad_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_bad_bounds'
    module_type_store = module_type_store.open_function_context('test_bad_bounds', 334, 0, False)
    
    # Passed parameters checking function
    test_bad_bounds.stypy_localization = localization
    test_bad_bounds.stypy_type_of_self = None
    test_bad_bounds.stypy_type_store = module_type_store
    test_bad_bounds.stypy_function_name = 'test_bad_bounds'
    test_bad_bounds.stypy_param_names_list = []
    test_bad_bounds.stypy_varargs_param_name = None
    test_bad_bounds.stypy_kwargs_param_name = None
    test_bad_bounds.stypy_call_defaults = defaults
    test_bad_bounds.stypy_call_varargs = varargs
    test_bad_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_bad_bounds', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_bad_bounds', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_bad_bounds(...)' code ##################

    
    # Assigning a List to a Name (line 335):
    
    # Obtaining an instance of the builtin type 'list' (line 335)
    list_241108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 335)
    # Adding element type (line 335)
    int_241109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), list_241108, int_241109)
    # Adding element type (line 335)
    int_241110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), list_241108, int_241110)
    
    # Assigning a type to the variable 'c' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'c', list_241108)
    
    # Call to assert_raises(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'ValueError' (line 336)
    ValueError_241112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 336)
    _clean_inputs_241113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 336)
    # Getting the type of 'c' (line 336)
    c_241114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 47), 'c', False)
    keyword_241115 = c_241114
    
    # Obtaining an instance of the builtin type 'tuple' (line 336)
    tuple_241116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 58), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 336)
    # Adding element type (line 336)
    int_241117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 58), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 58), tuple_241116, int_241117)
    # Adding element type (line 336)
    int_241118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 58), tuple_241116, int_241118)
    
    keyword_241119 = tuple_241116
    kwargs_241120 = {'c': keyword_241115, 'bounds': keyword_241119}
    # Getting the type of 'assert_raises' (line 336)
    assert_raises_241111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 336)
    assert_raises_call_result_241121 = invoke(stypy.reporting.localization.Localization(__file__, 336, 4), assert_raises_241111, *[ValueError_241112, _clean_inputs_241113], **kwargs_241120)
    
    
    # Call to assert_raises(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'ValueError' (line 337)
    ValueError_241123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 337)
    _clean_inputs_241124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 337)
    # Getting the type of 'c' (line 337)
    c_241125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 47), 'c', False)
    keyword_241126 = c_241125
    
    # Obtaining an instance of the builtin type 'list' (line 337)
    list_241127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 337)
    # Adding element type (line 337)
    
    # Obtaining an instance of the builtin type 'tuple' (line 337)
    tuple_241128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 337)
    # Adding element type (line 337)
    int_241129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 59), tuple_241128, int_241129)
    # Adding element type (line 337)
    int_241130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 59), tuple_241128, int_241130)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 57), list_241127, tuple_241128)
    
    keyword_241131 = list_241127
    kwargs_241132 = {'c': keyword_241126, 'bounds': keyword_241131}
    # Getting the type of 'assert_raises' (line 337)
    assert_raises_241122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 337)
    assert_raises_call_result_241133 = invoke(stypy.reporting.localization.Localization(__file__, 337, 4), assert_raises_241122, *[ValueError_241123, _clean_inputs_241124], **kwargs_241132)
    
    
    # Call to assert_raises(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'ValueError' (line 338)
    ValueError_241135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 338)
    _clean_inputs_241136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 338)
    # Getting the type of 'c' (line 338)
    c_241137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 47), 'c', False)
    keyword_241138 = c_241137
    
    # Obtaining an instance of the builtin type 'list' (line 338)
    list_241139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 338)
    # Adding element type (line 338)
    
    # Obtaining an instance of the builtin type 'tuple' (line 338)
    tuple_241140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 338)
    # Adding element type (line 338)
    int_241141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 59), tuple_241140, int_241141)
    # Adding element type (line 338)
    int_241142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 59), tuple_241140, int_241142)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 57), list_241139, tuple_241140)
    # Adding element type (line 338)
    
    # Obtaining an instance of the builtin type 'tuple' (line 338)
    tuple_241143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 338)
    # Adding element type (line 338)
    int_241144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 68), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 68), tuple_241143, int_241144)
    # Adding element type (line 338)
    int_241145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 71), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 68), tuple_241143, int_241145)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 57), list_241139, tuple_241143)
    
    keyword_241146 = list_241139
    kwargs_241147 = {'c': keyword_241138, 'bounds': keyword_241146}
    # Getting the type of 'assert_raises' (line 338)
    assert_raises_241134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 338)
    assert_raises_call_result_241148 = invoke(stypy.reporting.localization.Localization(__file__, 338, 4), assert_raises_241134, *[ValueError_241135, _clean_inputs_241136], **kwargs_241147)
    
    
    # Call to assert_raises(...): (line 340)
    # Processing the call arguments (line 340)
    # Getting the type of 'ValueError' (line 340)
    ValueError_241150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 340)
    _clean_inputs_241151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 340)
    # Getting the type of 'c' (line 340)
    c_241152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 47), 'c', False)
    keyword_241153 = c_241152
    
    # Obtaining an instance of the builtin type 'tuple' (line 340)
    tuple_241154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 58), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 340)
    # Adding element type (line 340)
    int_241155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 58), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 58), tuple_241154, int_241155)
    # Adding element type (line 340)
    int_241156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 61), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 58), tuple_241154, int_241156)
    # Adding element type (line 340)
    int_241157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 64), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 58), tuple_241154, int_241157)
    
    keyword_241158 = tuple_241154
    kwargs_241159 = {'c': keyword_241153, 'bounds': keyword_241158}
    # Getting the type of 'assert_raises' (line 340)
    assert_raises_241149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 340)
    assert_raises_call_result_241160 = invoke(stypy.reporting.localization.Localization(__file__, 340, 4), assert_raises_241149, *[ValueError_241150, _clean_inputs_241151], **kwargs_241159)
    
    
    # Call to assert_raises(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'ValueError' (line 341)
    ValueError_241162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 341)
    _clean_inputs_241163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 341)
    # Getting the type of 'c' (line 341)
    c_241164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 47), 'c', False)
    keyword_241165 = c_241164
    
    # Obtaining an instance of the builtin type 'list' (line 341)
    list_241166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 341)
    # Adding element type (line 341)
    
    # Obtaining an instance of the builtin type 'tuple' (line 341)
    tuple_241167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 341)
    # Adding element type (line 341)
    int_241168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 59), tuple_241167, int_241168)
    # Adding element type (line 341)
    int_241169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 59), tuple_241167, int_241169)
    # Adding element type (line 341)
    int_241170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 59), tuple_241167, int_241170)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 57), list_241166, tuple_241167)
    
    keyword_241171 = list_241166
    kwargs_241172 = {'c': keyword_241165, 'bounds': keyword_241171}
    # Getting the type of 'assert_raises' (line 341)
    assert_raises_241161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 341)
    assert_raises_call_result_241173 = invoke(stypy.reporting.localization.Localization(__file__, 341, 4), assert_raises_241161, *[ValueError_241162, _clean_inputs_241163], **kwargs_241172)
    
    
    # Call to assert_raises(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'ValueError' (line 342)
    ValueError_241175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 342)
    _clean_inputs_241176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 342)
    # Getting the type of 'c' (line 342)
    c_241177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 47), 'c', False)
    keyword_241178 = c_241177
    
    # Obtaining an instance of the builtin type 'list' (line 342)
    list_241179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 342)
    # Adding element type (line 342)
    
    # Obtaining an instance of the builtin type 'tuple' (line 342)
    tuple_241180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 342)
    # Adding element type (line 342)
    int_241181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 59), tuple_241180, int_241181)
    # Adding element type (line 342)
    int_241182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 59), tuple_241180, int_241182)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 57), list_241179, tuple_241180)
    # Adding element type (line 342)
    
    # Obtaining an instance of the builtin type 'tuple' (line 342)
    tuple_241183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 342)
    # Adding element type (line 342)
    int_241184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 67), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 67), tuple_241183, int_241184)
    # Adding element type (line 342)
    int_241185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 70), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 67), tuple_241183, int_241185)
    # Adding element type (line 342)
    int_241186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 73), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 67), tuple_241183, int_241186)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 57), list_241179, tuple_241183)
    
    keyword_241187 = list_241179
    kwargs_241188 = {'c': keyword_241178, 'bounds': keyword_241187}
    # Getting the type of 'assert_raises' (line 342)
    assert_raises_241174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 342)
    assert_raises_call_result_241189 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), assert_raises_241174, *[ValueError_241175, _clean_inputs_241176], **kwargs_241188)
    
    
    # Call to assert_raises(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'ValueError' (line 343)
    ValueError_241191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'ValueError', False)
    # Getting the type of '_clean_inputs' (line 343)
    _clean_inputs_241192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 30), '_clean_inputs', False)
    # Processing the call keyword arguments (line 343)
    # Getting the type of 'c' (line 343)
    c_241193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 47), 'c', False)
    keyword_241194 = c_241193
    
    # Obtaining an instance of the builtin type 'list' (line 344)
    list_241195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 344)
    # Adding element type (line 344)
    
    # Obtaining an instance of the builtin type 'tuple' (line 344)
    tuple_241196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 344)
    # Adding element type (line 344)
    int_241197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 27), tuple_241196, int_241197)
    # Adding element type (line 344)
    int_241198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 27), tuple_241196, int_241198)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 25), list_241195, tuple_241196)
    # Adding element type (line 344)
    
    # Obtaining an instance of the builtin type 'tuple' (line 344)
    tuple_241199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 344)
    # Adding element type (line 344)
    int_241200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 35), tuple_241199, int_241200)
    # Adding element type (line 344)
    int_241201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 35), tuple_241199, int_241201)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 25), list_241195, tuple_241199)
    # Adding element type (line 344)
    
    # Obtaining an instance of the builtin type 'tuple' (line 344)
    tuple_241202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 344)
    # Adding element type (line 344)
    int_241203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 43), tuple_241202, int_241203)
    # Adding element type (line 344)
    int_241204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 43), tuple_241202, int_241204)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 25), list_241195, tuple_241202)
    
    keyword_241205 = list_241195
    kwargs_241206 = {'c': keyword_241194, 'bounds': keyword_241205}
    # Getting the type of 'assert_raises' (line 343)
    assert_raises_241190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 343)
    assert_raises_call_result_241207 = invoke(stypy.reporting.localization.Localization(__file__, 343, 4), assert_raises_241190, *[ValueError_241191, _clean_inputs_241192], **kwargs_241206)
    
    
    # ################# End of 'test_bad_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_bad_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 334)
    stypy_return_type_241208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_241208)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_bad_bounds'
    return stypy_return_type_241208

# Assigning a type to the variable 'test_bad_bounds' (line 334)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 0), 'test_bad_bounds', test_bad_bounds)

@norecursion
def test_good_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_good_bounds'
    module_type_store = module_type_store.open_function_context('test_good_bounds', 347, 0, False)
    
    # Passed parameters checking function
    test_good_bounds.stypy_localization = localization
    test_good_bounds.stypy_type_of_self = None
    test_good_bounds.stypy_type_store = module_type_store
    test_good_bounds.stypy_function_name = 'test_good_bounds'
    test_good_bounds.stypy_param_names_list = []
    test_good_bounds.stypy_varargs_param_name = None
    test_good_bounds.stypy_kwargs_param_name = None
    test_good_bounds.stypy_call_defaults = defaults
    test_good_bounds.stypy_call_varargs = varargs
    test_good_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_good_bounds', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_good_bounds', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_good_bounds(...)' code ##################

    
    # Assigning a List to a Name (line 348):
    
    # Obtaining an instance of the builtin type 'list' (line 348)
    list_241209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 348)
    # Adding element type (line 348)
    int_241210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 8), list_241209, int_241210)
    # Adding element type (line 348)
    int_241211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 8), list_241209, int_241211)
    
    # Assigning a type to the variable 'c' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'c', list_241209)
    
    # Assigning a Call to a Name (line 349):
    
    # Call to _clean_inputs(...): (line 349)
    # Processing the call keyword arguments (line 349)
    # Getting the type of 'c' (line 349)
    c_241213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 30), 'c', False)
    keyword_241214 = c_241213
    # Getting the type of 'None' (line 349)
    None_241215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 40), 'None', False)
    keyword_241216 = None_241215
    kwargs_241217 = {'c': keyword_241214, 'bounds': keyword_241216}
    # Getting the type of '_clean_inputs' (line 349)
    _clean_inputs_241212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 349)
    _clean_inputs_call_result_241218 = invoke(stypy.reporting.localization.Localization(__file__, 349, 14), _clean_inputs_241212, *[], **kwargs_241217)
    
    # Assigning a type to the variable 'outputs' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'outputs', _clean_inputs_call_result_241218)
    
    # Call to assert_(...): (line 350)
    # Processing the call arguments (line 350)
    
    
    # Obtaining the type of the subscript
    int_241220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 20), 'int')
    # Getting the type of 'outputs' (line 350)
    outputs_241221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___241222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 12), outputs_241221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_241223 = invoke(stypy.reporting.localization.Localization(__file__, 350, 12), getitem___241222, int_241220)
    
    
    # Obtaining an instance of the builtin type 'list' (line 350)
    list_241224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 350)
    # Adding element type (line 350)
    
    # Obtaining an instance of the builtin type 'tuple' (line 350)
    tuple_241225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 350)
    # Adding element type (line 350)
    int_241226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 28), tuple_241225, int_241226)
    # Adding element type (line 350)
    # Getting the type of 'None' (line 350)
    None_241227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 31), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 28), tuple_241225, None_241227)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 26), list_241224, tuple_241225)
    
    int_241228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 40), 'int')
    # Applying the binary operator '*' (line 350)
    result_mul_241229 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 26), '*', list_241224, int_241228)
    
    # Applying the binary operator '==' (line 350)
    result_eq_241230 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 12), '==', subscript_call_result_241223, result_mul_241229)
    
    str_241231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 43), 'str', '')
    # Processing the call keyword arguments (line 350)
    kwargs_241232 = {}
    # Getting the type of 'assert_' (line 350)
    assert__241219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 350)
    assert__call_result_241233 = invoke(stypy.reporting.localization.Localization(__file__, 350, 4), assert__241219, *[result_eq_241230, str_241231], **kwargs_241232)
    
    
    # Assigning a Call to a Name (line 352):
    
    # Call to _clean_inputs(...): (line 352)
    # Processing the call keyword arguments (line 352)
    # Getting the type of 'c' (line 352)
    c_241235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 30), 'c', False)
    keyword_241236 = c_241235
    
    # Obtaining an instance of the builtin type 'tuple' (line 352)
    tuple_241237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 352)
    # Adding element type (line 352)
    int_241238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 41), tuple_241237, int_241238)
    # Adding element type (line 352)
    int_241239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 41), tuple_241237, int_241239)
    
    keyword_241240 = tuple_241237
    kwargs_241241 = {'c': keyword_241236, 'bounds': keyword_241240}
    # Getting the type of '_clean_inputs' (line 352)
    _clean_inputs_241234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 352)
    _clean_inputs_call_result_241242 = invoke(stypy.reporting.localization.Localization(__file__, 352, 14), _clean_inputs_241234, *[], **kwargs_241241)
    
    # Assigning a type to the variable 'outputs' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'outputs', _clean_inputs_call_result_241242)
    
    # Call to assert_(...): (line 353)
    # Processing the call arguments (line 353)
    
    
    # Obtaining the type of the subscript
    int_241244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 20), 'int')
    # Getting the type of 'outputs' (line 353)
    outputs_241245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 353)
    getitem___241246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 12), outputs_241245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 353)
    subscript_call_result_241247 = invoke(stypy.reporting.localization.Localization(__file__, 353, 12), getitem___241246, int_241244)
    
    
    # Obtaining an instance of the builtin type 'list' (line 353)
    list_241248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 353)
    # Adding element type (line 353)
    
    # Obtaining an instance of the builtin type 'tuple' (line 353)
    tuple_241249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 353)
    # Adding element type (line 353)
    int_241250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 28), tuple_241249, int_241250)
    # Adding element type (line 353)
    int_241251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 28), tuple_241249, int_241251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 26), list_241248, tuple_241249)
    
    int_241252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 37), 'int')
    # Applying the binary operator '*' (line 353)
    result_mul_241253 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 26), '*', list_241248, int_241252)
    
    # Applying the binary operator '==' (line 353)
    result_eq_241254 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 12), '==', subscript_call_result_241247, result_mul_241253)
    
    str_241255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 40), 'str', '')
    # Processing the call keyword arguments (line 353)
    kwargs_241256 = {}
    # Getting the type of 'assert_' (line 353)
    assert__241243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 353)
    assert__call_result_241257 = invoke(stypy.reporting.localization.Localization(__file__, 353, 4), assert__241243, *[result_eq_241254, str_241255], **kwargs_241256)
    
    
    # Assigning a Call to a Name (line 355):
    
    # Call to _clean_inputs(...): (line 355)
    # Processing the call keyword arguments (line 355)
    # Getting the type of 'c' (line 355)
    c_241259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'c', False)
    keyword_241260 = c_241259
    
    # Obtaining an instance of the builtin type 'list' (line 355)
    list_241261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 355)
    # Adding element type (line 355)
    
    # Obtaining an instance of the builtin type 'tuple' (line 355)
    tuple_241262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 355)
    # Adding element type (line 355)
    int_241263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 42), tuple_241262, int_241263)
    # Adding element type (line 355)
    int_241264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 42), tuple_241262, int_241264)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 40), list_241261, tuple_241262)
    
    keyword_241265 = list_241261
    kwargs_241266 = {'c': keyword_241260, 'bounds': keyword_241265}
    # Getting the type of '_clean_inputs' (line 355)
    _clean_inputs_241258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 355)
    _clean_inputs_call_result_241267 = invoke(stypy.reporting.localization.Localization(__file__, 355, 14), _clean_inputs_241258, *[], **kwargs_241266)
    
    # Assigning a type to the variable 'outputs' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'outputs', _clean_inputs_call_result_241267)
    
    # Call to assert_(...): (line 356)
    # Processing the call arguments (line 356)
    
    
    # Obtaining the type of the subscript
    int_241269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 20), 'int')
    # Getting the type of 'outputs' (line 356)
    outputs_241270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 356)
    getitem___241271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), outputs_241270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 356)
    subscript_call_result_241272 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), getitem___241271, int_241269)
    
    
    # Obtaining an instance of the builtin type 'list' (line 356)
    list_241273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 356)
    # Adding element type (line 356)
    
    # Obtaining an instance of the builtin type 'tuple' (line 356)
    tuple_241274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 356)
    # Adding element type (line 356)
    int_241275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 28), tuple_241274, int_241275)
    # Adding element type (line 356)
    int_241276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 28), tuple_241274, int_241276)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 26), list_241273, tuple_241274)
    
    int_241277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 37), 'int')
    # Applying the binary operator '*' (line 356)
    result_mul_241278 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 26), '*', list_241273, int_241277)
    
    # Applying the binary operator '==' (line 356)
    result_eq_241279 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 12), '==', subscript_call_result_241272, result_mul_241278)
    
    str_241280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 40), 'str', '')
    # Processing the call keyword arguments (line 356)
    kwargs_241281 = {}
    # Getting the type of 'assert_' (line 356)
    assert__241268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 356)
    assert__call_result_241282 = invoke(stypy.reporting.localization.Localization(__file__, 356, 4), assert__241268, *[result_eq_241279, str_241280], **kwargs_241281)
    
    
    # Assigning a Call to a Name (line 358):
    
    # Call to _clean_inputs(...): (line 358)
    # Processing the call keyword arguments (line 358)
    # Getting the type of 'c' (line 358)
    c_241284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 30), 'c', False)
    keyword_241285 = c_241284
    
    # Obtaining an instance of the builtin type 'list' (line 358)
    list_241286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 358)
    # Adding element type (line 358)
    
    # Obtaining an instance of the builtin type 'tuple' (line 358)
    tuple_241287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 358)
    # Adding element type (line 358)
    int_241288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 42), tuple_241287, int_241288)
    # Adding element type (line 358)
    # Getting the type of 'np' (line 358)
    np_241289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 45), 'np', False)
    # Obtaining the member 'inf' of a type (line 358)
    inf_241290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 45), np_241289, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 42), tuple_241287, inf_241290)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 40), list_241286, tuple_241287)
    
    keyword_241291 = list_241286
    kwargs_241292 = {'c': keyword_241285, 'bounds': keyword_241291}
    # Getting the type of '_clean_inputs' (line 358)
    _clean_inputs_241283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 358)
    _clean_inputs_call_result_241293 = invoke(stypy.reporting.localization.Localization(__file__, 358, 14), _clean_inputs_241283, *[], **kwargs_241292)
    
    # Assigning a type to the variable 'outputs' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'outputs', _clean_inputs_call_result_241293)
    
    # Call to assert_(...): (line 359)
    # Processing the call arguments (line 359)
    
    
    # Obtaining the type of the subscript
    int_241295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 20), 'int')
    # Getting the type of 'outputs' (line 359)
    outputs_241296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___241297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), outputs_241296, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_241298 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), getitem___241297, int_241295)
    
    
    # Obtaining an instance of the builtin type 'list' (line 359)
    list_241299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 359)
    # Adding element type (line 359)
    
    # Obtaining an instance of the builtin type 'tuple' (line 359)
    tuple_241300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 359)
    # Adding element type (line 359)
    int_241301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 28), tuple_241300, int_241301)
    # Adding element type (line 359)
    # Getting the type of 'None' (line 359)
    None_241302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 31), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 28), tuple_241300, None_241302)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 26), list_241299, tuple_241300)
    
    int_241303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 40), 'int')
    # Applying the binary operator '*' (line 359)
    result_mul_241304 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 26), '*', list_241299, int_241303)
    
    # Applying the binary operator '==' (line 359)
    result_eq_241305 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 12), '==', subscript_call_result_241298, result_mul_241304)
    
    str_241306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 43), 'str', '')
    # Processing the call keyword arguments (line 359)
    kwargs_241307 = {}
    # Getting the type of 'assert_' (line 359)
    assert__241294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 359)
    assert__call_result_241308 = invoke(stypy.reporting.localization.Localization(__file__, 359, 4), assert__241294, *[result_eq_241305, str_241306], **kwargs_241307)
    
    
    # Assigning a Call to a Name (line 361):
    
    # Call to _clean_inputs(...): (line 361)
    # Processing the call keyword arguments (line 361)
    # Getting the type of 'c' (line 361)
    c_241310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 30), 'c', False)
    keyword_241311 = c_241310
    
    # Obtaining an instance of the builtin type 'list' (line 361)
    list_241312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 361)
    # Adding element type (line 361)
    
    # Obtaining an instance of the builtin type 'tuple' (line 361)
    tuple_241313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 361)
    # Adding element type (line 361)
    
    # Getting the type of 'np' (line 361)
    np_241314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 43), 'np', False)
    # Obtaining the member 'inf' of a type (line 361)
    inf_241315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 43), np_241314, 'inf')
    # Applying the 'usub' unary operator (line 361)
    result___neg___241316 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 42), 'usub', inf_241315)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 42), tuple_241313, result___neg___241316)
    # Adding element type (line 361)
    int_241317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 42), tuple_241313, int_241317)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 40), list_241312, tuple_241313)
    
    keyword_241318 = list_241312
    kwargs_241319 = {'c': keyword_241311, 'bounds': keyword_241318}
    # Getting the type of '_clean_inputs' (line 361)
    _clean_inputs_241309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 361)
    _clean_inputs_call_result_241320 = invoke(stypy.reporting.localization.Localization(__file__, 361, 14), _clean_inputs_241309, *[], **kwargs_241319)
    
    # Assigning a type to the variable 'outputs' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'outputs', _clean_inputs_call_result_241320)
    
    # Call to assert_(...): (line 362)
    # Processing the call arguments (line 362)
    
    
    # Obtaining the type of the subscript
    int_241322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 20), 'int')
    # Getting the type of 'outputs' (line 362)
    outputs_241323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___241324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), outputs_241323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_241325 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___241324, int_241322)
    
    
    # Obtaining an instance of the builtin type 'list' (line 362)
    list_241326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 362)
    # Adding element type (line 362)
    
    # Obtaining an instance of the builtin type 'tuple' (line 362)
    tuple_241327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 362)
    # Adding element type (line 362)
    # Getting the type of 'None' (line 362)
    None_241328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 28), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 28), tuple_241327, None_241328)
    # Adding element type (line 362)
    int_241329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 28), tuple_241327, int_241329)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 26), list_241326, tuple_241327)
    
    int_241330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 40), 'int')
    # Applying the binary operator '*' (line 362)
    result_mul_241331 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 26), '*', list_241326, int_241330)
    
    # Applying the binary operator '==' (line 362)
    result_eq_241332 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 12), '==', subscript_call_result_241325, result_mul_241331)
    
    str_241333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 43), 'str', '')
    # Processing the call keyword arguments (line 362)
    kwargs_241334 = {}
    # Getting the type of 'assert_' (line 362)
    assert__241321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 362)
    assert__call_result_241335 = invoke(stypy.reporting.localization.Localization(__file__, 362, 4), assert__241321, *[result_eq_241332, str_241333], **kwargs_241334)
    
    
    # Assigning a Call to a Name (line 364):
    
    # Call to _clean_inputs(...): (line 364)
    # Processing the call keyword arguments (line 364)
    # Getting the type of 'c' (line 364)
    c_241337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'c', False)
    keyword_241338 = c_241337
    
    # Obtaining an instance of the builtin type 'list' (line 364)
    list_241339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 364)
    # Adding element type (line 364)
    
    # Obtaining an instance of the builtin type 'tuple' (line 364)
    tuple_241340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 364)
    # Adding element type (line 364)
    
    # Getting the type of 'np' (line 364)
    np_241341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 43), 'np', False)
    # Obtaining the member 'inf' of a type (line 364)
    inf_241342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 43), np_241341, 'inf')
    # Applying the 'usub' unary operator (line 364)
    result___neg___241343 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 42), 'usub', inf_241342)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 42), tuple_241340, result___neg___241343)
    # Adding element type (line 364)
    # Getting the type of 'np' (line 364)
    np_241344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 51), 'np', False)
    # Obtaining the member 'inf' of a type (line 364)
    inf_241345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 51), np_241344, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 42), tuple_241340, inf_241345)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 40), list_241339, tuple_241340)
    # Adding element type (line 364)
    
    # Obtaining an instance of the builtin type 'tuple' (line 364)
    tuple_241346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 61), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 364)
    # Adding element type (line 364)
    
    # Getting the type of 'np' (line 364)
    np_241347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 62), 'np', False)
    # Obtaining the member 'inf' of a type (line 364)
    inf_241348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 62), np_241347, 'inf')
    # Applying the 'usub' unary operator (line 364)
    result___neg___241349 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 61), 'usub', inf_241348)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 61), tuple_241346, result___neg___241349)
    # Adding element type (line 364)
    # Getting the type of 'np' (line 364)
    np_241350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 70), 'np', False)
    # Obtaining the member 'inf' of a type (line 364)
    inf_241351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 70), np_241350, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 61), tuple_241346, inf_241351)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 40), list_241339, tuple_241346)
    
    keyword_241352 = list_241339
    kwargs_241353 = {'c': keyword_241338, 'bounds': keyword_241352}
    # Getting the type of '_clean_inputs' (line 364)
    _clean_inputs_241336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 14), '_clean_inputs', False)
    # Calling _clean_inputs(args, kwargs) (line 364)
    _clean_inputs_call_result_241354 = invoke(stypy.reporting.localization.Localization(__file__, 364, 14), _clean_inputs_241336, *[], **kwargs_241353)
    
    # Assigning a type to the variable 'outputs' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'outputs', _clean_inputs_call_result_241354)
    
    # Call to assert_(...): (line 365)
    # Processing the call arguments (line 365)
    
    
    # Obtaining the type of the subscript
    int_241356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 20), 'int')
    # Getting the type of 'outputs' (line 365)
    outputs_241357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'outputs', False)
    # Obtaining the member '__getitem__' of a type (line 365)
    getitem___241358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 12), outputs_241357, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 365)
    subscript_call_result_241359 = invoke(stypy.reporting.localization.Localization(__file__, 365, 12), getitem___241358, int_241356)
    
    
    # Obtaining an instance of the builtin type 'list' (line 365)
    list_241360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 365)
    # Adding element type (line 365)
    
    # Obtaining an instance of the builtin type 'tuple' (line 365)
    tuple_241361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 365)
    # Adding element type (line 365)
    # Getting the type of 'None' (line 365)
    None_241362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 28), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 28), tuple_241361, None_241362)
    # Adding element type (line 365)
    # Getting the type of 'None' (line 365)
    None_241363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 34), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 28), tuple_241361, None_241363)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 26), list_241360, tuple_241361)
    
    int_241364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 43), 'int')
    # Applying the binary operator '*' (line 365)
    result_mul_241365 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 26), '*', list_241360, int_241364)
    
    # Applying the binary operator '==' (line 365)
    result_eq_241366 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 12), '==', subscript_call_result_241359, result_mul_241365)
    
    str_241367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 46), 'str', '')
    # Processing the call keyword arguments (line 365)
    kwargs_241368 = {}
    # Getting the type of 'assert_' (line 365)
    assert__241355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 365)
    assert__call_result_241369 = invoke(stypy.reporting.localization.Localization(__file__, 365, 4), assert__241355, *[result_eq_241366, str_241367], **kwargs_241368)
    
    
    # ################# End of 'test_good_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_good_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 347)
    stypy_return_type_241370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_241370)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_good_bounds'
    return stypy_return_type_241370

# Assigning a type to the variable 'test_good_bounds' (line 347)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'test_good_bounds', test_good_bounds)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

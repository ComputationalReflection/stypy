
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Miscellaneous functions for testing masked arrays and subclasses
2: 
3: :author: Pierre Gerard-Marchant
4: :contact: pierregm_at_uga_dot_edu
5: :version: $Id: testutils.py 3529 2007-11-13 08:01:14Z jarrod.millman $
6: 
7: '''
8: from __future__ import division, absolute_import, print_function
9: 
10: import operator
11: 
12: import numpy as np
13: from numpy import ndarray, float_
14: import numpy.core.umath as umath
15: from numpy.testing import (
16:     TestCase, assert_, assert_allclose, assert_array_almost_equal_nulp,
17:     assert_raises, build_err_msg, run_module_suite,
18:     )
19: import numpy.testing.utils as utils
20: from .core import mask_or, getmask, masked_array, nomask, masked, filled
21: 
22: __all__masked = [
23:     'almost', 'approx', 'assert_almost_equal', 'assert_array_almost_equal',
24:     'assert_array_approx_equal', 'assert_array_compare',
25:     'assert_array_equal', 'assert_array_less', 'assert_close',
26:     'assert_equal', 'assert_equal_records', 'assert_mask_equal',
27:     'assert_not_equal', 'fail_if_array_equal',
28:     ]
29: 
30: # Include some normal test functions to avoid breaking other projects who
31: # have mistakenly included them from this file. SciPy is one. That is
32: # unfortunate, as some of these functions are not intended to work with
33: # masked arrays. But there was no way to tell before.
34: __some__from_testing = [
35:     'TestCase', 'assert_', 'assert_allclose',
36:     'assert_array_almost_equal_nulp', 'assert_raises', 'run_module_suite',
37:     ]
38: 
39: __all__ = __all__masked + __some__from_testing
40: 
41: 
42: def approx(a, b, fill_value=True, rtol=1e-5, atol=1e-8):
43:     '''
44:     Returns true if all components of a and b are equal to given tolerances.
45: 
46:     If fill_value is True, masked values considered equal. Otherwise,
47:     masked values are considered unequal.  The relative error rtol should
48:     be positive and << 1.0 The absolute error atol comes into play for
49:     those elements of b that are very small or zero; it says how small a
50:     must be also.
51: 
52:     '''
53:     m = mask_or(getmask(a), getmask(b))
54:     d1 = filled(a)
55:     d2 = filled(b)
56:     if d1.dtype.char == "O" or d2.dtype.char == "O":
57:         return np.equal(d1, d2).ravel()
58:     x = filled(masked_array(d1, copy=False, mask=m), fill_value).astype(float_)
59:     y = filled(masked_array(d2, copy=False, mask=m), 1).astype(float_)
60:     d = np.less_equal(umath.absolute(x - y), atol + rtol * umath.absolute(y))
61:     return d.ravel()
62: 
63: 
64: def almost(a, b, decimal=6, fill_value=True):
65:     '''
66:     Returns True if a and b are equal up to decimal places.
67: 
68:     If fill_value is True, masked values considered equal. Otherwise,
69:     masked values are considered unequal.
70: 
71:     '''
72:     m = mask_or(getmask(a), getmask(b))
73:     d1 = filled(a)
74:     d2 = filled(b)
75:     if d1.dtype.char == "O" or d2.dtype.char == "O":
76:         return np.equal(d1, d2).ravel()
77:     x = filled(masked_array(d1, copy=False, mask=m), fill_value).astype(float_)
78:     y = filled(masked_array(d2, copy=False, mask=m), 1).astype(float_)
79:     d = np.around(np.abs(x - y), decimal) <= 10.0 ** (-decimal)
80:     return d.ravel()
81: 
82: 
83: def _assert_equal_on_sequences(actual, desired, err_msg=''):
84:     '''
85:     Asserts the equality of two non-array sequences.
86: 
87:     '''
88:     assert_equal(len(actual), len(desired), err_msg)
89:     for k in range(len(desired)):
90:         assert_equal(actual[k], desired[k], 'item=%r\n%s' % (k, err_msg))
91:     return
92: 
93: 
94: def assert_equal_records(a, b):
95:     '''
96:     Asserts that two records are equal.
97: 
98:     Pretty crude for now.
99: 
100:     '''
101:     assert_equal(a.dtype, b.dtype)
102:     for f in a.dtype.names:
103:         (af, bf) = (operator.getitem(a, f), operator.getitem(b, f))
104:         if not (af is masked) and not (bf is masked):
105:             assert_equal(operator.getitem(a, f), operator.getitem(b, f))
106:     return
107: 
108: 
109: def assert_equal(actual, desired, err_msg=''):
110:     '''
111:     Asserts that two items are equal.
112: 
113:     '''
114:     # Case #1: dictionary .....
115:     if isinstance(desired, dict):
116:         if not isinstance(actual, dict):
117:             raise AssertionError(repr(type(actual)))
118:         assert_equal(len(actual), len(desired), err_msg)
119:         for k, i in desired.items():
120:             if k not in actual:
121:                 raise AssertionError("%s not in %s" % (k, actual))
122:             assert_equal(actual[k], desired[k], 'key=%r\n%s' % (k, err_msg))
123:         return
124:     # Case #2: lists .....
125:     if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
126:         return _assert_equal_on_sequences(actual, desired, err_msg='')
127:     if not (isinstance(actual, ndarray) or isinstance(desired, ndarray)):
128:         msg = build_err_msg([actual, desired], err_msg,)
129:         if not desired == actual:
130:             raise AssertionError(msg)
131:         return
132:     # Case #4. arrays or equivalent
133:     if ((actual is masked) and not (desired is masked)) or \
134:             ((desired is masked) and not (actual is masked)):
135:         msg = build_err_msg([actual, desired],
136:                             err_msg, header='', names=('x', 'y'))
137:         raise ValueError(msg)
138:     actual = np.array(actual, copy=False, subok=True)
139:     desired = np.array(desired, copy=False, subok=True)
140:     (actual_dtype, desired_dtype) = (actual.dtype, desired.dtype)
141:     if actual_dtype.char == "S" and desired_dtype.char == "S":
142:         return _assert_equal_on_sequences(actual.tolist(),
143:                                           desired.tolist(),
144:                                           err_msg='')
145:     return assert_array_equal(actual, desired, err_msg)
146: 
147: 
148: def fail_if_equal(actual, desired, err_msg='',):
149:     '''
150:     Raises an assertion error if two items are equal.
151: 
152:     '''
153:     if isinstance(desired, dict):
154:         if not isinstance(actual, dict):
155:             raise AssertionError(repr(type(actual)))
156:         fail_if_equal(len(actual), len(desired), err_msg)
157:         for k, i in desired.items():
158:             if k not in actual:
159:                 raise AssertionError(repr(k))
160:             fail_if_equal(actual[k], desired[k], 'key=%r\n%s' % (k, err_msg))
161:         return
162:     if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
163:         fail_if_equal(len(actual), len(desired), err_msg)
164:         for k in range(len(desired)):
165:             fail_if_equal(actual[k], desired[k], 'item=%r\n%s' % (k, err_msg))
166:         return
167:     if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
168:         return fail_if_array_equal(actual, desired, err_msg)
169:     msg = build_err_msg([actual, desired], err_msg)
170:     if not desired != actual:
171:         raise AssertionError(msg)
172: 
173: 
174: assert_not_equal = fail_if_equal
175: 
176: 
177: def assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True):
178:     '''
179:     Asserts that two items are almost equal.
180: 
181:     The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal).
182: 
183:     '''
184:     if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
185:         return assert_array_almost_equal(actual, desired, decimal=decimal,
186:                                          err_msg=err_msg, verbose=verbose)
187:     msg = build_err_msg([actual, desired],
188:                         err_msg=err_msg, verbose=verbose)
189:     if not round(abs(desired - actual), decimal) == 0:
190:         raise AssertionError(msg)
191: 
192: 
193: assert_close = assert_almost_equal
194: 
195: 
196: def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='',
197:                          fill_value=True):
198:     '''
199:     Asserts that comparison between two masked arrays is satisfied.
200: 
201:     The comparison is elementwise.
202: 
203:     '''
204:     # Allocate a common mask and refill
205:     m = mask_or(getmask(x), getmask(y))
206:     x = masked_array(x, copy=False, mask=m, keep_mask=False, subok=False)
207:     y = masked_array(y, copy=False, mask=m, keep_mask=False, subok=False)
208:     if ((x is masked) and not (y is masked)) or \
209:             ((y is masked) and not (x is masked)):
210:         msg = build_err_msg([x, y], err_msg=err_msg, verbose=verbose,
211:                             header=header, names=('x', 'y'))
212:         raise ValueError(msg)
213:     # OK, now run the basic tests on filled versions
214:     return utils.assert_array_compare(comparison,
215:                                       x.filled(fill_value),
216:                                       y.filled(fill_value),
217:                                       err_msg=err_msg,
218:                                       verbose=verbose, header=header)
219: 
220: 
221: def assert_array_equal(x, y, err_msg='', verbose=True):
222:     '''
223:     Checks the elementwise equality of two masked arrays.
224: 
225:     '''
226:     assert_array_compare(operator.__eq__, x, y,
227:                          err_msg=err_msg, verbose=verbose,
228:                          header='Arrays are not equal')
229: 
230: 
231: def fail_if_array_equal(x, y, err_msg='', verbose=True):
232:     '''
233:     Raises an assertion error if two masked arrays are not equal elementwise.
234: 
235:     '''
236:     def compare(x, y):
237:         return (not np.alltrue(approx(x, y)))
238:     assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
239:                          header='Arrays are not equal')
240: 
241: 
242: def assert_array_approx_equal(x, y, decimal=6, err_msg='', verbose=True):
243:     '''
244:     Checks the equality of two masked arrays, up to given number odecimals.
245: 
246:     The equality is checked elementwise.
247: 
248:     '''
249:     def compare(x, y):
250:         "Returns the result of the loose comparison between x and y)."
251:         return approx(x, y, rtol=10. ** -decimal)
252:     assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
253:                          header='Arrays are not almost equal')
254: 
255: 
256: def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
257:     '''
258:     Checks the equality of two masked arrays, up to given number odecimals.
259: 
260:     The equality is checked elementwise.
261: 
262:     '''
263:     def compare(x, y):
264:         "Returns the result of the loose comparison between x and y)."
265:         return almost(x, y, decimal)
266:     assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
267:                          header='Arrays are not almost equal')
268: 
269: 
270: def assert_array_less(x, y, err_msg='', verbose=True):
271:     '''
272:     Checks that x is smaller than y elementwise.
273: 
274:     '''
275:     assert_array_compare(operator.__lt__, x, y,
276:                          err_msg=err_msg, verbose=verbose,
277:                          header='Arrays are not less-ordered')
278: 
279: 
280: def assert_mask_equal(m1, m2, err_msg=''):
281:     '''
282:     Asserts the equality of two masks.
283: 
284:     '''
285:     if m1 is nomask:
286:         assert_(m2 is nomask)
287:     if m2 is nomask:
288:         assert_(m1 is nomask)
289:     assert_array_equal(m1, m2, err_msg=err_msg)
290: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_157058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', 'Miscellaneous functions for testing masked arrays and subclasses\n\n:author: Pierre Gerard-Marchant\n:contact: pierregm_at_uga_dot_edu\n:version: $Id: testutils.py 3529 2007-11-13 08:01:14Z jarrod.millman $\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import operator' statement (line 10)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import numpy' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_157059 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy')

if (type(import_157059) is not StypyTypeError):

    if (import_157059 != 'pyd_module'):
        __import__(import_157059)
        sys_modules_157060 = sys.modules[import_157059]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', sys_modules_157060.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy', import_157059)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy import ndarray, float_' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_157061 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_157061) is not StypyTypeError):

    if (import_157061 != 'pyd_module'):
        __import__(import_157061)
        sys_modules_157062 = sys.modules[import_157061]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', sys_modules_157062.module_type_store, module_type_store, ['ndarray', 'float_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_157062, sys_modules_157062.module_type_store, module_type_store)
    else:
        from numpy import ndarray, float_

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', None, module_type_store, ['ndarray', 'float_'], [ndarray, float_])

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_157061)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import numpy.core.umath' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_157063 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core.umath')

if (type(import_157063) is not StypyTypeError):

    if (import_157063 != 'pyd_module'):
        __import__(import_157063)
        sys_modules_157064 = sys.modules[import_157063]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'umath', sys_modules_157064.module_type_store, module_type_store)
    else:
        import numpy.core.umath as umath

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'umath', numpy.core.umath, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.umath' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core.umath', import_157063)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.testing import TestCase, assert_, assert_allclose, assert_array_almost_equal_nulp, assert_raises, build_err_msg, run_module_suite' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_157065 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.testing')

if (type(import_157065) is not StypyTypeError):

    if (import_157065 != 'pyd_module'):
        __import__(import_157065)
        sys_modules_157066 = sys.modules[import_157065]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.testing', sys_modules_157066.module_type_store, module_type_store, ['TestCase', 'assert_', 'assert_allclose', 'assert_array_almost_equal_nulp', 'assert_raises', 'build_err_msg', 'run_module_suite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_157066, sys_modules_157066.module_type_store, module_type_store)
    else:
        from numpy.testing import TestCase, assert_, assert_allclose, assert_array_almost_equal_nulp, assert_raises, build_err_msg, run_module_suite

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.testing', None, module_type_store, ['TestCase', 'assert_', 'assert_allclose', 'assert_array_almost_equal_nulp', 'assert_raises', 'build_err_msg', 'run_module_suite'], [TestCase, assert_, assert_allclose, assert_array_almost_equal_nulp, assert_raises, build_err_msg, run_module_suite])

else:
    # Assigning a type to the variable 'numpy.testing' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.testing', import_157065)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import numpy.testing.utils' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_157067 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.testing.utils')

if (type(import_157067) is not StypyTypeError):

    if (import_157067 != 'pyd_module'):
        __import__(import_157067)
        sys_modules_157068 = sys.modules[import_157067]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'utils', sys_modules_157068.module_type_store, module_type_store)
    else:
        import numpy.testing.utils as utils

        import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'utils', numpy.testing.utils, module_type_store)

else:
    # Assigning a type to the variable 'numpy.testing.utils' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.testing.utils', import_157067)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from numpy.ma.core import mask_or, getmask, masked_array, nomask, masked, filled' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_157069 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.ma.core')

if (type(import_157069) is not StypyTypeError):

    if (import_157069 != 'pyd_module'):
        __import__(import_157069)
        sys_modules_157070 = sys.modules[import_157069]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.ma.core', sys_modules_157070.module_type_store, module_type_store, ['mask_or', 'getmask', 'masked_array', 'nomask', 'masked', 'filled'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_157070, sys_modules_157070.module_type_store, module_type_store)
    else:
        from numpy.ma.core import mask_or, getmask, masked_array, nomask, masked, filled

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.ma.core', None, module_type_store, ['mask_or', 'getmask', 'masked_array', 'nomask', 'masked', 'filled'], [mask_or, getmask, masked_array, nomask, masked, filled])

else:
    # Assigning a type to the variable 'numpy.ma.core' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.ma.core', import_157069)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')


# Assigning a List to a Name (line 22):

# Assigning a List to a Name (line 22):

# Obtaining an instance of the builtin type 'list' (line 22)
list_157071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_157072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'str', 'almost')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157072)
# Adding element type (line 22)
str_157073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'str', 'approx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157073)
# Adding element type (line 22)
str_157074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'str', 'assert_almost_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157074)
# Adding element type (line 22)
str_157075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 47), 'str', 'assert_array_almost_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157075)
# Adding element type (line 22)
str_157076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'str', 'assert_array_approx_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157076)
# Adding element type (line 22)
str_157077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'str', 'assert_array_compare')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157077)
# Adding element type (line 22)
str_157078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', 'assert_array_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157078)
# Adding element type (line 22)
str_157079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', 'assert_array_less')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157079)
# Adding element type (line 22)
str_157080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 47), 'str', 'assert_close')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157080)
# Adding element type (line 22)
str_157081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', 'assert_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157081)
# Adding element type (line 22)
str_157082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'str', 'assert_equal_records')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157082)
# Adding element type (line 22)
str_157083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 44), 'str', 'assert_mask_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157083)
# Adding element type (line 22)
str_157084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'assert_not_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157084)
# Adding element type (line 22)
str_157085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'str', 'fail_if_array_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_157071, str_157085)

# Assigning a type to the variable '__all__masked' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '__all__masked', list_157071)

# Assigning a List to a Name (line 34):

# Assigning a List to a Name (line 34):

# Obtaining an instance of the builtin type 'list' (line 34)
list_157086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
str_157087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'TestCase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 23), list_157086, str_157087)
# Adding element type (line 34)
str_157088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'str', 'assert_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 23), list_157086, str_157088)
# Adding element type (line 34)
str_157089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 27), 'str', 'assert_allclose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 23), list_157086, str_157089)
# Adding element type (line 34)
str_157090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'assert_array_almost_equal_nulp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 23), list_157086, str_157090)
# Adding element type (line 34)
str_157091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 38), 'str', 'assert_raises')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 23), list_157086, str_157091)
# Adding element type (line 34)
str_157092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 55), 'str', 'run_module_suite')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 23), list_157086, str_157092)

# Assigning a type to the variable '__some__from_testing' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), '__some__from_testing', list_157086)

# Assigning a BinOp to a Name (line 39):

# Assigning a BinOp to a Name (line 39):
# Getting the type of '__all__masked' (line 39)
all__masked_157093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), '__all__masked')
# Getting the type of '__some__from_testing' (line 39)
some__from_testing_157094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), '__some__from_testing')
# Applying the binary operator '+' (line 39)
result_add_157095 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 10), '+', all__masked_157093, some__from_testing_157094)

# Assigning a type to the variable '__all__' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '__all__', result_add_157095)

@norecursion
def approx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 42)
    True_157096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'True')
    float_157097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 39), 'float')
    float_157098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'float')
    defaults = [True_157096, float_157097, float_157098]
    # Create a new context for function 'approx'
    module_type_store = module_type_store.open_function_context('approx', 42, 0, False)
    
    # Passed parameters checking function
    approx.stypy_localization = localization
    approx.stypy_type_of_self = None
    approx.stypy_type_store = module_type_store
    approx.stypy_function_name = 'approx'
    approx.stypy_param_names_list = ['a', 'b', 'fill_value', 'rtol', 'atol']
    approx.stypy_varargs_param_name = None
    approx.stypy_kwargs_param_name = None
    approx.stypy_call_defaults = defaults
    approx.stypy_call_varargs = varargs
    approx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'approx', ['a', 'b', 'fill_value', 'rtol', 'atol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'approx', localization, ['a', 'b', 'fill_value', 'rtol', 'atol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'approx(...)' code ##################

    str_157099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', '\n    Returns true if all components of a and b are equal to given tolerances.\n\n    If fill_value is True, masked values considered equal. Otherwise,\n    masked values are considered unequal.  The relative error rtol should\n    be positive and << 1.0 The absolute error atol comes into play for\n    those elements of b that are very small or zero; it says how small a\n    must be also.\n\n    ')
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to mask_or(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Call to getmask(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'a' (line 53)
    a_157102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'a', False)
    # Processing the call keyword arguments (line 53)
    kwargs_157103 = {}
    # Getting the type of 'getmask' (line 53)
    getmask_157101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'getmask', False)
    # Calling getmask(args, kwargs) (line 53)
    getmask_call_result_157104 = invoke(stypy.reporting.localization.Localization(__file__, 53, 16), getmask_157101, *[a_157102], **kwargs_157103)
    
    
    # Call to getmask(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'b' (line 53)
    b_157106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 36), 'b', False)
    # Processing the call keyword arguments (line 53)
    kwargs_157107 = {}
    # Getting the type of 'getmask' (line 53)
    getmask_157105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'getmask', False)
    # Calling getmask(args, kwargs) (line 53)
    getmask_call_result_157108 = invoke(stypy.reporting.localization.Localization(__file__, 53, 28), getmask_157105, *[b_157106], **kwargs_157107)
    
    # Processing the call keyword arguments (line 53)
    kwargs_157109 = {}
    # Getting the type of 'mask_or' (line 53)
    mask_or_157100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'mask_or', False)
    # Calling mask_or(args, kwargs) (line 53)
    mask_or_call_result_157110 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), mask_or_157100, *[getmask_call_result_157104, getmask_call_result_157108], **kwargs_157109)
    
    # Assigning a type to the variable 'm' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'm', mask_or_call_result_157110)
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to filled(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'a' (line 54)
    a_157112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'a', False)
    # Processing the call keyword arguments (line 54)
    kwargs_157113 = {}
    # Getting the type of 'filled' (line 54)
    filled_157111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 9), 'filled', False)
    # Calling filled(args, kwargs) (line 54)
    filled_call_result_157114 = invoke(stypy.reporting.localization.Localization(__file__, 54, 9), filled_157111, *[a_157112], **kwargs_157113)
    
    # Assigning a type to the variable 'd1' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'd1', filled_call_result_157114)
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to filled(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'b' (line 55)
    b_157116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'b', False)
    # Processing the call keyword arguments (line 55)
    kwargs_157117 = {}
    # Getting the type of 'filled' (line 55)
    filled_157115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'filled', False)
    # Calling filled(args, kwargs) (line 55)
    filled_call_result_157118 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), filled_157115, *[b_157116], **kwargs_157117)
    
    # Assigning a type to the variable 'd2' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'd2', filled_call_result_157118)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'd1' (line 56)
    d1_157119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 7), 'd1')
    # Obtaining the member 'dtype' of a type (line 56)
    dtype_157120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 7), d1_157119, 'dtype')
    # Obtaining the member 'char' of a type (line 56)
    char_157121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 7), dtype_157120, 'char')
    str_157122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'str', 'O')
    # Applying the binary operator '==' (line 56)
    result_eq_157123 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 7), '==', char_157121, str_157122)
    
    
    # Getting the type of 'd2' (line 56)
    d2_157124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 31), 'd2')
    # Obtaining the member 'dtype' of a type (line 56)
    dtype_157125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 31), d2_157124, 'dtype')
    # Obtaining the member 'char' of a type (line 56)
    char_157126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 31), dtype_157125, 'char')
    str_157127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 48), 'str', 'O')
    # Applying the binary operator '==' (line 56)
    result_eq_157128 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 31), '==', char_157126, str_157127)
    
    # Applying the binary operator 'or' (line 56)
    result_or_keyword_157129 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 7), 'or', result_eq_157123, result_eq_157128)
    
    # Testing the type of an if condition (line 56)
    if_condition_157130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 4), result_or_keyword_157129)
    # Assigning a type to the variable 'if_condition_157130' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'if_condition_157130', if_condition_157130)
    # SSA begins for if statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ravel(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_157138 = {}
    
    # Call to equal(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'd1' (line 57)
    d1_157133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'd1', False)
    # Getting the type of 'd2' (line 57)
    d2_157134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'd2', False)
    # Processing the call keyword arguments (line 57)
    kwargs_157135 = {}
    # Getting the type of 'np' (line 57)
    np_157131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'np', False)
    # Obtaining the member 'equal' of a type (line 57)
    equal_157132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), np_157131, 'equal')
    # Calling equal(args, kwargs) (line 57)
    equal_call_result_157136 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), equal_157132, *[d1_157133, d2_157134], **kwargs_157135)
    
    # Obtaining the member 'ravel' of a type (line 57)
    ravel_157137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), equal_call_result_157136, 'ravel')
    # Calling ravel(args, kwargs) (line 57)
    ravel_call_result_157139 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), ravel_157137, *[], **kwargs_157138)
    
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', ravel_call_result_157139)
    # SSA join for if statement (line 56)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to astype(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'float_' (line 58)
    float__157153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 72), 'float_', False)
    # Processing the call keyword arguments (line 58)
    kwargs_157154 = {}
    
    # Call to filled(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Call to masked_array(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'd1' (line 58)
    d1_157142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'd1', False)
    # Processing the call keyword arguments (line 58)
    # Getting the type of 'False' (line 58)
    False_157143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'False', False)
    keyword_157144 = False_157143
    # Getting the type of 'm' (line 58)
    m_157145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 49), 'm', False)
    keyword_157146 = m_157145
    kwargs_157147 = {'copy': keyword_157144, 'mask': keyword_157146}
    # Getting the type of 'masked_array' (line 58)
    masked_array_157141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'masked_array', False)
    # Calling masked_array(args, kwargs) (line 58)
    masked_array_call_result_157148 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), masked_array_157141, *[d1_157142], **kwargs_157147)
    
    # Getting the type of 'fill_value' (line 58)
    fill_value_157149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 53), 'fill_value', False)
    # Processing the call keyword arguments (line 58)
    kwargs_157150 = {}
    # Getting the type of 'filled' (line 58)
    filled_157140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'filled', False)
    # Calling filled(args, kwargs) (line 58)
    filled_call_result_157151 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), filled_157140, *[masked_array_call_result_157148, fill_value_157149], **kwargs_157150)
    
    # Obtaining the member 'astype' of a type (line 58)
    astype_157152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), filled_call_result_157151, 'astype')
    # Calling astype(args, kwargs) (line 58)
    astype_call_result_157155 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), astype_157152, *[float__157153], **kwargs_157154)
    
    # Assigning a type to the variable 'x' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'x', astype_call_result_157155)
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to astype(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'float_' (line 59)
    float__157169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 63), 'float_', False)
    # Processing the call keyword arguments (line 59)
    kwargs_157170 = {}
    
    # Call to filled(...): (line 59)
    # Processing the call arguments (line 59)
    
    # Call to masked_array(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'd2' (line 59)
    d2_157158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'd2', False)
    # Processing the call keyword arguments (line 59)
    # Getting the type of 'False' (line 59)
    False_157159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'False', False)
    keyword_157160 = False_157159
    # Getting the type of 'm' (line 59)
    m_157161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 49), 'm', False)
    keyword_157162 = m_157161
    kwargs_157163 = {'copy': keyword_157160, 'mask': keyword_157162}
    # Getting the type of 'masked_array' (line 59)
    masked_array_157157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'masked_array', False)
    # Calling masked_array(args, kwargs) (line 59)
    masked_array_call_result_157164 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), masked_array_157157, *[d2_157158], **kwargs_157163)
    
    int_157165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 53), 'int')
    # Processing the call keyword arguments (line 59)
    kwargs_157166 = {}
    # Getting the type of 'filled' (line 59)
    filled_157156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'filled', False)
    # Calling filled(args, kwargs) (line 59)
    filled_call_result_157167 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), filled_157156, *[masked_array_call_result_157164, int_157165], **kwargs_157166)
    
    # Obtaining the member 'astype' of a type (line 59)
    astype_157168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), filled_call_result_157167, 'astype')
    # Calling astype(args, kwargs) (line 59)
    astype_call_result_157171 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), astype_157168, *[float__157169], **kwargs_157170)
    
    # Assigning a type to the variable 'y' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'y', astype_call_result_157171)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to less_equal(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Call to absolute(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'x' (line 60)
    x_157176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'x', False)
    # Getting the type of 'y' (line 60)
    y_157177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 41), 'y', False)
    # Applying the binary operator '-' (line 60)
    result_sub_157178 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 37), '-', x_157176, y_157177)
    
    # Processing the call keyword arguments (line 60)
    kwargs_157179 = {}
    # Getting the type of 'umath' (line 60)
    umath_157174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'umath', False)
    # Obtaining the member 'absolute' of a type (line 60)
    absolute_157175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), umath_157174, 'absolute')
    # Calling absolute(args, kwargs) (line 60)
    absolute_call_result_157180 = invoke(stypy.reporting.localization.Localization(__file__, 60, 22), absolute_157175, *[result_sub_157178], **kwargs_157179)
    
    # Getting the type of 'atol' (line 60)
    atol_157181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'atol', False)
    # Getting the type of 'rtol' (line 60)
    rtol_157182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 52), 'rtol', False)
    
    # Call to absolute(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'y' (line 60)
    y_157185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 74), 'y', False)
    # Processing the call keyword arguments (line 60)
    kwargs_157186 = {}
    # Getting the type of 'umath' (line 60)
    umath_157183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 59), 'umath', False)
    # Obtaining the member 'absolute' of a type (line 60)
    absolute_157184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 59), umath_157183, 'absolute')
    # Calling absolute(args, kwargs) (line 60)
    absolute_call_result_157187 = invoke(stypy.reporting.localization.Localization(__file__, 60, 59), absolute_157184, *[y_157185], **kwargs_157186)
    
    # Applying the binary operator '*' (line 60)
    result_mul_157188 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 52), '*', rtol_157182, absolute_call_result_157187)
    
    # Applying the binary operator '+' (line 60)
    result_add_157189 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 45), '+', atol_157181, result_mul_157188)
    
    # Processing the call keyword arguments (line 60)
    kwargs_157190 = {}
    # Getting the type of 'np' (line 60)
    np_157172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'np', False)
    # Obtaining the member 'less_equal' of a type (line 60)
    less_equal_157173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), np_157172, 'less_equal')
    # Calling less_equal(args, kwargs) (line 60)
    less_equal_call_result_157191 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), less_equal_157173, *[absolute_call_result_157180, result_add_157189], **kwargs_157190)
    
    # Assigning a type to the variable 'd' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'd', less_equal_call_result_157191)
    
    # Call to ravel(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_157194 = {}
    # Getting the type of 'd' (line 61)
    d_157192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'd', False)
    # Obtaining the member 'ravel' of a type (line 61)
    ravel_157193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), d_157192, 'ravel')
    # Calling ravel(args, kwargs) (line 61)
    ravel_call_result_157195 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), ravel_157193, *[], **kwargs_157194)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', ravel_call_result_157195)
    
    # ################# End of 'approx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'approx' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_157196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'approx'
    return stypy_return_type_157196

# Assigning a type to the variable 'approx' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'approx', approx)

@norecursion
def almost(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_157197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'int')
    # Getting the type of 'True' (line 64)
    True_157198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'True')
    defaults = [int_157197, True_157198]
    # Create a new context for function 'almost'
    module_type_store = module_type_store.open_function_context('almost', 64, 0, False)
    
    # Passed parameters checking function
    almost.stypy_localization = localization
    almost.stypy_type_of_self = None
    almost.stypy_type_store = module_type_store
    almost.stypy_function_name = 'almost'
    almost.stypy_param_names_list = ['a', 'b', 'decimal', 'fill_value']
    almost.stypy_varargs_param_name = None
    almost.stypy_kwargs_param_name = None
    almost.stypy_call_defaults = defaults
    almost.stypy_call_varargs = varargs
    almost.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'almost', ['a', 'b', 'decimal', 'fill_value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'almost', localization, ['a', 'b', 'decimal', 'fill_value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'almost(...)' code ##################

    str_157199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', '\n    Returns True if a and b are equal up to decimal places.\n\n    If fill_value is True, masked values considered equal. Otherwise,\n    masked values are considered unequal.\n\n    ')
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to mask_or(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to getmask(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'a' (line 72)
    a_157202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'a', False)
    # Processing the call keyword arguments (line 72)
    kwargs_157203 = {}
    # Getting the type of 'getmask' (line 72)
    getmask_157201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'getmask', False)
    # Calling getmask(args, kwargs) (line 72)
    getmask_call_result_157204 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), getmask_157201, *[a_157202], **kwargs_157203)
    
    
    # Call to getmask(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'b' (line 72)
    b_157206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'b', False)
    # Processing the call keyword arguments (line 72)
    kwargs_157207 = {}
    # Getting the type of 'getmask' (line 72)
    getmask_157205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'getmask', False)
    # Calling getmask(args, kwargs) (line 72)
    getmask_call_result_157208 = invoke(stypy.reporting.localization.Localization(__file__, 72, 28), getmask_157205, *[b_157206], **kwargs_157207)
    
    # Processing the call keyword arguments (line 72)
    kwargs_157209 = {}
    # Getting the type of 'mask_or' (line 72)
    mask_or_157200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'mask_or', False)
    # Calling mask_or(args, kwargs) (line 72)
    mask_or_call_result_157210 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), mask_or_157200, *[getmask_call_result_157204, getmask_call_result_157208], **kwargs_157209)
    
    # Assigning a type to the variable 'm' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'm', mask_or_call_result_157210)
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to filled(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'a' (line 73)
    a_157212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'a', False)
    # Processing the call keyword arguments (line 73)
    kwargs_157213 = {}
    # Getting the type of 'filled' (line 73)
    filled_157211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 9), 'filled', False)
    # Calling filled(args, kwargs) (line 73)
    filled_call_result_157214 = invoke(stypy.reporting.localization.Localization(__file__, 73, 9), filled_157211, *[a_157212], **kwargs_157213)
    
    # Assigning a type to the variable 'd1' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'd1', filled_call_result_157214)
    
    # Assigning a Call to a Name (line 74):
    
    # Assigning a Call to a Name (line 74):
    
    # Call to filled(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'b' (line 74)
    b_157216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'b', False)
    # Processing the call keyword arguments (line 74)
    kwargs_157217 = {}
    # Getting the type of 'filled' (line 74)
    filled_157215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 9), 'filled', False)
    # Calling filled(args, kwargs) (line 74)
    filled_call_result_157218 = invoke(stypy.reporting.localization.Localization(__file__, 74, 9), filled_157215, *[b_157216], **kwargs_157217)
    
    # Assigning a type to the variable 'd2' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'd2', filled_call_result_157218)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'd1' (line 75)
    d1_157219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), 'd1')
    # Obtaining the member 'dtype' of a type (line 75)
    dtype_157220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 7), d1_157219, 'dtype')
    # Obtaining the member 'char' of a type (line 75)
    char_157221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 7), dtype_157220, 'char')
    str_157222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'str', 'O')
    # Applying the binary operator '==' (line 75)
    result_eq_157223 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), '==', char_157221, str_157222)
    
    
    # Getting the type of 'd2' (line 75)
    d2_157224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'd2')
    # Obtaining the member 'dtype' of a type (line 75)
    dtype_157225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 31), d2_157224, 'dtype')
    # Obtaining the member 'char' of a type (line 75)
    char_157226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 31), dtype_157225, 'char')
    str_157227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 48), 'str', 'O')
    # Applying the binary operator '==' (line 75)
    result_eq_157228 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 31), '==', char_157226, str_157227)
    
    # Applying the binary operator 'or' (line 75)
    result_or_keyword_157229 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'or', result_eq_157223, result_eq_157228)
    
    # Testing the type of an if condition (line 75)
    if_condition_157230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_or_keyword_157229)
    # Assigning a type to the variable 'if_condition_157230' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_157230', if_condition_157230)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ravel(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_157238 = {}
    
    # Call to equal(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'd1' (line 76)
    d1_157233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'd1', False)
    # Getting the type of 'd2' (line 76)
    d2_157234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'd2', False)
    # Processing the call keyword arguments (line 76)
    kwargs_157235 = {}
    # Getting the type of 'np' (line 76)
    np_157231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'np', False)
    # Obtaining the member 'equal' of a type (line 76)
    equal_157232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 15), np_157231, 'equal')
    # Calling equal(args, kwargs) (line 76)
    equal_call_result_157236 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), equal_157232, *[d1_157233, d2_157234], **kwargs_157235)
    
    # Obtaining the member 'ravel' of a type (line 76)
    ravel_157237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 15), equal_call_result_157236, 'ravel')
    # Calling ravel(args, kwargs) (line 76)
    ravel_call_result_157239 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), ravel_157237, *[], **kwargs_157238)
    
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', ravel_call_result_157239)
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to astype(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'float_' (line 77)
    float__157253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 72), 'float_', False)
    # Processing the call keyword arguments (line 77)
    kwargs_157254 = {}
    
    # Call to filled(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Call to masked_array(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'd1' (line 77)
    d1_157242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'd1', False)
    # Processing the call keyword arguments (line 77)
    # Getting the type of 'False' (line 77)
    False_157243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'False', False)
    keyword_157244 = False_157243
    # Getting the type of 'm' (line 77)
    m_157245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 49), 'm', False)
    keyword_157246 = m_157245
    kwargs_157247 = {'copy': keyword_157244, 'mask': keyword_157246}
    # Getting the type of 'masked_array' (line 77)
    masked_array_157241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'masked_array', False)
    # Calling masked_array(args, kwargs) (line 77)
    masked_array_call_result_157248 = invoke(stypy.reporting.localization.Localization(__file__, 77, 15), masked_array_157241, *[d1_157242], **kwargs_157247)
    
    # Getting the type of 'fill_value' (line 77)
    fill_value_157249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 53), 'fill_value', False)
    # Processing the call keyword arguments (line 77)
    kwargs_157250 = {}
    # Getting the type of 'filled' (line 77)
    filled_157240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'filled', False)
    # Calling filled(args, kwargs) (line 77)
    filled_call_result_157251 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), filled_157240, *[masked_array_call_result_157248, fill_value_157249], **kwargs_157250)
    
    # Obtaining the member 'astype' of a type (line 77)
    astype_157252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), filled_call_result_157251, 'astype')
    # Calling astype(args, kwargs) (line 77)
    astype_call_result_157255 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), astype_157252, *[float__157253], **kwargs_157254)
    
    # Assigning a type to the variable 'x' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'x', astype_call_result_157255)
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to astype(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'float_' (line 78)
    float__157269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 63), 'float_', False)
    # Processing the call keyword arguments (line 78)
    kwargs_157270 = {}
    
    # Call to filled(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Call to masked_array(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'd2' (line 78)
    d2_157258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'd2', False)
    # Processing the call keyword arguments (line 78)
    # Getting the type of 'False' (line 78)
    False_157259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'False', False)
    keyword_157260 = False_157259
    # Getting the type of 'm' (line 78)
    m_157261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 49), 'm', False)
    keyword_157262 = m_157261
    kwargs_157263 = {'copy': keyword_157260, 'mask': keyword_157262}
    # Getting the type of 'masked_array' (line 78)
    masked_array_157257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'masked_array', False)
    # Calling masked_array(args, kwargs) (line 78)
    masked_array_call_result_157264 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), masked_array_157257, *[d2_157258], **kwargs_157263)
    
    int_157265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 53), 'int')
    # Processing the call keyword arguments (line 78)
    kwargs_157266 = {}
    # Getting the type of 'filled' (line 78)
    filled_157256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'filled', False)
    # Calling filled(args, kwargs) (line 78)
    filled_call_result_157267 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), filled_157256, *[masked_array_call_result_157264, int_157265], **kwargs_157266)
    
    # Obtaining the member 'astype' of a type (line 78)
    astype_157268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), filled_call_result_157267, 'astype')
    # Calling astype(args, kwargs) (line 78)
    astype_call_result_157271 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), astype_157268, *[float__157269], **kwargs_157270)
    
    # Assigning a type to the variable 'y' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'y', astype_call_result_157271)
    
    # Assigning a Compare to a Name (line 79):
    
    # Assigning a Compare to a Name (line 79):
    
    
    # Call to around(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Call to abs(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'x' (line 79)
    x_157276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'x', False)
    # Getting the type of 'y' (line 79)
    y_157277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'y', False)
    # Applying the binary operator '-' (line 79)
    result_sub_157278 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 25), '-', x_157276, y_157277)
    
    # Processing the call keyword arguments (line 79)
    kwargs_157279 = {}
    # Getting the type of 'np' (line 79)
    np_157274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 79)
    abs_157275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 18), np_157274, 'abs')
    # Calling abs(args, kwargs) (line 79)
    abs_call_result_157280 = invoke(stypy.reporting.localization.Localization(__file__, 79, 18), abs_157275, *[result_sub_157278], **kwargs_157279)
    
    # Getting the type of 'decimal' (line 79)
    decimal_157281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'decimal', False)
    # Processing the call keyword arguments (line 79)
    kwargs_157282 = {}
    # Getting the type of 'np' (line 79)
    np_157272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'np', False)
    # Obtaining the member 'around' of a type (line 79)
    around_157273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), np_157272, 'around')
    # Calling around(args, kwargs) (line 79)
    around_call_result_157283 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), around_157273, *[abs_call_result_157280, decimal_157281], **kwargs_157282)
    
    float_157284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 45), 'float')
    
    # Getting the type of 'decimal' (line 79)
    decimal_157285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 55), 'decimal')
    # Applying the 'usub' unary operator (line 79)
    result___neg___157286 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 54), 'usub', decimal_157285)
    
    # Applying the binary operator '**' (line 79)
    result_pow_157287 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 45), '**', float_157284, result___neg___157286)
    
    # Applying the binary operator '<=' (line 79)
    result_le_157288 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 8), '<=', around_call_result_157283, result_pow_157287)
    
    # Assigning a type to the variable 'd' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'd', result_le_157288)
    
    # Call to ravel(...): (line 80)
    # Processing the call keyword arguments (line 80)
    kwargs_157291 = {}
    # Getting the type of 'd' (line 80)
    d_157289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'd', False)
    # Obtaining the member 'ravel' of a type (line 80)
    ravel_157290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), d_157289, 'ravel')
    # Calling ravel(args, kwargs) (line 80)
    ravel_call_result_157292 = invoke(stypy.reporting.localization.Localization(__file__, 80, 11), ravel_157290, *[], **kwargs_157291)
    
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type', ravel_call_result_157292)
    
    # ################# End of 'almost(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'almost' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_157293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157293)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'almost'
    return stypy_return_type_157293

# Assigning a type to the variable 'almost' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'almost', almost)

@norecursion
def _assert_equal_on_sequences(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_157294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 56), 'str', '')
    defaults = [str_157294]
    # Create a new context for function '_assert_equal_on_sequences'
    module_type_store = module_type_store.open_function_context('_assert_equal_on_sequences', 83, 0, False)
    
    # Passed parameters checking function
    _assert_equal_on_sequences.stypy_localization = localization
    _assert_equal_on_sequences.stypy_type_of_self = None
    _assert_equal_on_sequences.stypy_type_store = module_type_store
    _assert_equal_on_sequences.stypy_function_name = '_assert_equal_on_sequences'
    _assert_equal_on_sequences.stypy_param_names_list = ['actual', 'desired', 'err_msg']
    _assert_equal_on_sequences.stypy_varargs_param_name = None
    _assert_equal_on_sequences.stypy_kwargs_param_name = None
    _assert_equal_on_sequences.stypy_call_defaults = defaults
    _assert_equal_on_sequences.stypy_call_varargs = varargs
    _assert_equal_on_sequences.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_assert_equal_on_sequences', ['actual', 'desired', 'err_msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_assert_equal_on_sequences', localization, ['actual', 'desired', 'err_msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_assert_equal_on_sequences(...)' code ##################

    str_157295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'str', '\n    Asserts the equality of two non-array sequences.\n\n    ')
    
    # Call to assert_equal(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Call to len(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'actual' (line 88)
    actual_157298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'actual', False)
    # Processing the call keyword arguments (line 88)
    kwargs_157299 = {}
    # Getting the type of 'len' (line 88)
    len_157297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'len', False)
    # Calling len(args, kwargs) (line 88)
    len_call_result_157300 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), len_157297, *[actual_157298], **kwargs_157299)
    
    
    # Call to len(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'desired' (line 88)
    desired_157302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'desired', False)
    # Processing the call keyword arguments (line 88)
    kwargs_157303 = {}
    # Getting the type of 'len' (line 88)
    len_157301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'len', False)
    # Calling len(args, kwargs) (line 88)
    len_call_result_157304 = invoke(stypy.reporting.localization.Localization(__file__, 88, 30), len_157301, *[desired_157302], **kwargs_157303)
    
    # Getting the type of 'err_msg' (line 88)
    err_msg_157305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 44), 'err_msg', False)
    # Processing the call keyword arguments (line 88)
    kwargs_157306 = {}
    # Getting the type of 'assert_equal' (line 88)
    assert_equal_157296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 88)
    assert_equal_call_result_157307 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), assert_equal_157296, *[len_call_result_157300, len_call_result_157304, err_msg_157305], **kwargs_157306)
    
    
    
    # Call to range(...): (line 89)
    # Processing the call arguments (line 89)
    
    # Call to len(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'desired' (line 89)
    desired_157310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'desired', False)
    # Processing the call keyword arguments (line 89)
    kwargs_157311 = {}
    # Getting the type of 'len' (line 89)
    len_157309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'len', False)
    # Calling len(args, kwargs) (line 89)
    len_call_result_157312 = invoke(stypy.reporting.localization.Localization(__file__, 89, 19), len_157309, *[desired_157310], **kwargs_157311)
    
    # Processing the call keyword arguments (line 89)
    kwargs_157313 = {}
    # Getting the type of 'range' (line 89)
    range_157308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'range', False)
    # Calling range(args, kwargs) (line 89)
    range_call_result_157314 = invoke(stypy.reporting.localization.Localization(__file__, 89, 13), range_157308, *[len_call_result_157312], **kwargs_157313)
    
    # Testing the type of a for loop iterable (line 89)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 89, 4), range_call_result_157314)
    # Getting the type of the for loop variable (line 89)
    for_loop_var_157315 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 89, 4), range_call_result_157314)
    # Assigning a type to the variable 'k' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'k', for_loop_var_157315)
    # SSA begins for a for statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_equal(...): (line 90)
    # Processing the call arguments (line 90)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 90)
    k_157317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'k', False)
    # Getting the type of 'actual' (line 90)
    actual_157318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'actual', False)
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___157319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 21), actual_157318, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_157320 = invoke(stypy.reporting.localization.Localization(__file__, 90, 21), getitem___157319, k_157317)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 90)
    k_157321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'k', False)
    # Getting the type of 'desired' (line 90)
    desired_157322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'desired', False)
    # Obtaining the member '__getitem__' of a type (line 90)
    getitem___157323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 32), desired_157322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 90)
    subscript_call_result_157324 = invoke(stypy.reporting.localization.Localization(__file__, 90, 32), getitem___157323, k_157321)
    
    str_157325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 44), 'str', 'item=%r\n%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_157326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 61), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    # Getting the type of 'k' (line 90)
    k_157327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 61), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 61), tuple_157326, k_157327)
    # Adding element type (line 90)
    # Getting the type of 'err_msg' (line 90)
    err_msg_157328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 64), 'err_msg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 61), tuple_157326, err_msg_157328)
    
    # Applying the binary operator '%' (line 90)
    result_mod_157329 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 44), '%', str_157325, tuple_157326)
    
    # Processing the call keyword arguments (line 90)
    kwargs_157330 = {}
    # Getting the type of 'assert_equal' (line 90)
    assert_equal_157316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 90)
    assert_equal_call_result_157331 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_equal_157316, *[subscript_call_result_157320, subscript_call_result_157324, result_mod_157329], **kwargs_157330)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of '_assert_equal_on_sequences(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_assert_equal_on_sequences' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_157332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_assert_equal_on_sequences'
    return stypy_return_type_157332

# Assigning a type to the variable '_assert_equal_on_sequences' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), '_assert_equal_on_sequences', _assert_equal_on_sequences)

@norecursion
def assert_equal_records(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_equal_records'
    module_type_store = module_type_store.open_function_context('assert_equal_records', 94, 0, False)
    
    # Passed parameters checking function
    assert_equal_records.stypy_localization = localization
    assert_equal_records.stypy_type_of_self = None
    assert_equal_records.stypy_type_store = module_type_store
    assert_equal_records.stypy_function_name = 'assert_equal_records'
    assert_equal_records.stypy_param_names_list = ['a', 'b']
    assert_equal_records.stypy_varargs_param_name = None
    assert_equal_records.stypy_kwargs_param_name = None
    assert_equal_records.stypy_call_defaults = defaults
    assert_equal_records.stypy_call_varargs = varargs
    assert_equal_records.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_equal_records', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_equal_records', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_equal_records(...)' code ##################

    str_157333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'str', '\n    Asserts that two records are equal.\n\n    Pretty crude for now.\n\n    ')
    
    # Call to assert_equal(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'a' (line 101)
    a_157335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'a', False)
    # Obtaining the member 'dtype' of a type (line 101)
    dtype_157336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 17), a_157335, 'dtype')
    # Getting the type of 'b' (line 101)
    b_157337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'b', False)
    # Obtaining the member 'dtype' of a type (line 101)
    dtype_157338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 26), b_157337, 'dtype')
    # Processing the call keyword arguments (line 101)
    kwargs_157339 = {}
    # Getting the type of 'assert_equal' (line 101)
    assert_equal_157334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 101)
    assert_equal_call_result_157340 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), assert_equal_157334, *[dtype_157336, dtype_157338], **kwargs_157339)
    
    
    # Getting the type of 'a' (line 102)
    a_157341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'a')
    # Obtaining the member 'dtype' of a type (line 102)
    dtype_157342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), a_157341, 'dtype')
    # Obtaining the member 'names' of a type (line 102)
    names_157343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), dtype_157342, 'names')
    # Testing the type of a for loop iterable (line 102)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 4), names_157343)
    # Getting the type of the for loop variable (line 102)
    for_loop_var_157344 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 4), names_157343)
    # Assigning a type to the variable 'f' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'f', for_loop_var_157344)
    # SSA begins for a for statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Tuple to a Tuple (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to getitem(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'a' (line 103)
    a_157347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'a', False)
    # Getting the type of 'f' (line 103)
    f_157348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'f', False)
    # Processing the call keyword arguments (line 103)
    kwargs_157349 = {}
    # Getting the type of 'operator' (line 103)
    operator_157345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'operator', False)
    # Obtaining the member 'getitem' of a type (line 103)
    getitem_157346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), operator_157345, 'getitem')
    # Calling getitem(args, kwargs) (line 103)
    getitem_call_result_157350 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), getitem_157346, *[a_157347, f_157348], **kwargs_157349)
    
    # Assigning a type to the variable 'tuple_assignment_157054' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_assignment_157054', getitem_call_result_157350)
    
    # Assigning a Call to a Name (line 103):
    
    # Call to getitem(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'b' (line 103)
    b_157353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 61), 'b', False)
    # Getting the type of 'f' (line 103)
    f_157354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 64), 'f', False)
    # Processing the call keyword arguments (line 103)
    kwargs_157355 = {}
    # Getting the type of 'operator' (line 103)
    operator_157351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 44), 'operator', False)
    # Obtaining the member 'getitem' of a type (line 103)
    getitem_157352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 44), operator_157351, 'getitem')
    # Calling getitem(args, kwargs) (line 103)
    getitem_call_result_157356 = invoke(stypy.reporting.localization.Localization(__file__, 103, 44), getitem_157352, *[b_157353, f_157354], **kwargs_157355)
    
    # Assigning a type to the variable 'tuple_assignment_157055' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_assignment_157055', getitem_call_result_157356)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_assignment_157054' (line 103)
    tuple_assignment_157054_157357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_assignment_157054')
    # Assigning a type to the variable 'af' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 9), 'af', tuple_assignment_157054_157357)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_assignment_157055' (line 103)
    tuple_assignment_157055_157358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_assignment_157055')
    # Assigning a type to the variable 'bf' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'bf', tuple_assignment_157055_157358)
    
    
    # Evaluating a boolean operation
    
    
    # Getting the type of 'af' (line 104)
    af_157359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'af')
    # Getting the type of 'masked' (line 104)
    masked_157360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'masked')
    # Applying the binary operator 'is' (line 104)
    result_is__157361 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 16), 'is', af_157359, masked_157360)
    
    # Applying the 'not' unary operator (line 104)
    result_not__157362 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), 'not', result_is__157361)
    
    
    
    # Getting the type of 'bf' (line 104)
    bf_157363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'bf')
    # Getting the type of 'masked' (line 104)
    masked_157364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 45), 'masked')
    # Applying the binary operator 'is' (line 104)
    result_is__157365 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 39), 'is', bf_157363, masked_157364)
    
    # Applying the 'not' unary operator (line 104)
    result_not__157366 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 34), 'not', result_is__157365)
    
    # Applying the binary operator 'and' (line 104)
    result_and_keyword_157367 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), 'and', result_not__157362, result_not__157366)
    
    # Testing the type of an if condition (line 104)
    if_condition_157368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), result_and_keyword_157367)
    # Assigning a type to the variable 'if_condition_157368' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_157368', if_condition_157368)
    # SSA begins for if statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_equal(...): (line 105)
    # Processing the call arguments (line 105)
    
    # Call to getitem(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'a' (line 105)
    a_157372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'a', False)
    # Getting the type of 'f' (line 105)
    f_157373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 45), 'f', False)
    # Processing the call keyword arguments (line 105)
    kwargs_157374 = {}
    # Getting the type of 'operator' (line 105)
    operator_157370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'operator', False)
    # Obtaining the member 'getitem' of a type (line 105)
    getitem_157371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), operator_157370, 'getitem')
    # Calling getitem(args, kwargs) (line 105)
    getitem_call_result_157375 = invoke(stypy.reporting.localization.Localization(__file__, 105, 25), getitem_157371, *[a_157372, f_157373], **kwargs_157374)
    
    
    # Call to getitem(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'b' (line 105)
    b_157378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 66), 'b', False)
    # Getting the type of 'f' (line 105)
    f_157379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 69), 'f', False)
    # Processing the call keyword arguments (line 105)
    kwargs_157380 = {}
    # Getting the type of 'operator' (line 105)
    operator_157376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 49), 'operator', False)
    # Obtaining the member 'getitem' of a type (line 105)
    getitem_157377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 49), operator_157376, 'getitem')
    # Calling getitem(args, kwargs) (line 105)
    getitem_call_result_157381 = invoke(stypy.reporting.localization.Localization(__file__, 105, 49), getitem_157377, *[b_157378, f_157379], **kwargs_157380)
    
    # Processing the call keyword arguments (line 105)
    kwargs_157382 = {}
    # Getting the type of 'assert_equal' (line 105)
    assert_equal_157369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 105)
    assert_equal_call_result_157383 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), assert_equal_157369, *[getitem_call_result_157375, getitem_call_result_157381], **kwargs_157382)
    
    # SSA join for if statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'assert_equal_records(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_equal_records' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_157384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157384)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_equal_records'
    return stypy_return_type_157384

# Assigning a type to the variable 'assert_equal_records' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'assert_equal_records', assert_equal_records)

@norecursion
def assert_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_157385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 42), 'str', '')
    defaults = [str_157385]
    # Create a new context for function 'assert_equal'
    module_type_store = module_type_store.open_function_context('assert_equal', 109, 0, False)
    
    # Passed parameters checking function
    assert_equal.stypy_localization = localization
    assert_equal.stypy_type_of_self = None
    assert_equal.stypy_type_store = module_type_store
    assert_equal.stypy_function_name = 'assert_equal'
    assert_equal.stypy_param_names_list = ['actual', 'desired', 'err_msg']
    assert_equal.stypy_varargs_param_name = None
    assert_equal.stypy_kwargs_param_name = None
    assert_equal.stypy_call_defaults = defaults
    assert_equal.stypy_call_varargs = varargs
    assert_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_equal', ['actual', 'desired', 'err_msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_equal', localization, ['actual', 'desired', 'err_msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_equal(...)' code ##################

    str_157386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, (-1)), 'str', '\n    Asserts that two items are equal.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 115)
    # Getting the type of 'dict' (line 115)
    dict_157387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'dict')
    # Getting the type of 'desired' (line 115)
    desired_157388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 18), 'desired')
    
    (may_be_157389, more_types_in_union_157390) = may_be_subtype(dict_157387, desired_157388)

    if may_be_157389:

        if more_types_in_union_157390:
            # Runtime conditional SSA (line 115)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'desired' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'desired', remove_not_subtype_from_union(desired_157388, dict))
        
        # Type idiom detected: calculating its left and rigth part (line 116)
        # Getting the type of 'dict' (line 116)
        dict_157391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'dict')
        # Getting the type of 'actual' (line 116)
        actual_157392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'actual')
        
        (may_be_157393, more_types_in_union_157394) = may_not_be_subtype(dict_157391, actual_157392)

        if may_be_157393:

            if more_types_in_union_157394:
                # Runtime conditional SSA (line 116)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'actual' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'actual', remove_subtype_from_union(actual_157392, dict))
            
            # Call to AssertionError(...): (line 117)
            # Processing the call arguments (line 117)
            
            # Call to repr(...): (line 117)
            # Processing the call arguments (line 117)
            
            # Call to type(...): (line 117)
            # Processing the call arguments (line 117)
            # Getting the type of 'actual' (line 117)
            actual_157398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 43), 'actual', False)
            # Processing the call keyword arguments (line 117)
            kwargs_157399 = {}
            # Getting the type of 'type' (line 117)
            type_157397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'type', False)
            # Calling type(args, kwargs) (line 117)
            type_call_result_157400 = invoke(stypy.reporting.localization.Localization(__file__, 117, 38), type_157397, *[actual_157398], **kwargs_157399)
            
            # Processing the call keyword arguments (line 117)
            kwargs_157401 = {}
            # Getting the type of 'repr' (line 117)
            repr_157396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'repr', False)
            # Calling repr(args, kwargs) (line 117)
            repr_call_result_157402 = invoke(stypy.reporting.localization.Localization(__file__, 117, 33), repr_157396, *[type_call_result_157400], **kwargs_157401)
            
            # Processing the call keyword arguments (line 117)
            kwargs_157403 = {}
            # Getting the type of 'AssertionError' (line 117)
            AssertionError_157395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'AssertionError', False)
            # Calling AssertionError(args, kwargs) (line 117)
            AssertionError_call_result_157404 = invoke(stypy.reporting.localization.Localization(__file__, 117, 18), AssertionError_157395, *[repr_call_result_157402], **kwargs_157403)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 117, 12), AssertionError_call_result_157404, 'raise parameter', BaseException)

            if more_types_in_union_157394:
                # SSA join for if statement (line 116)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to assert_equal(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to len(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'actual' (line 118)
        actual_157407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'actual', False)
        # Processing the call keyword arguments (line 118)
        kwargs_157408 = {}
        # Getting the type of 'len' (line 118)
        len_157406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'len', False)
        # Calling len(args, kwargs) (line 118)
        len_call_result_157409 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), len_157406, *[actual_157407], **kwargs_157408)
        
        
        # Call to len(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'desired' (line 118)
        desired_157411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'desired', False)
        # Processing the call keyword arguments (line 118)
        kwargs_157412 = {}
        # Getting the type of 'len' (line 118)
        len_157410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'len', False)
        # Calling len(args, kwargs) (line 118)
        len_call_result_157413 = invoke(stypy.reporting.localization.Localization(__file__, 118, 34), len_157410, *[desired_157411], **kwargs_157412)
        
        # Getting the type of 'err_msg' (line 118)
        err_msg_157414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 48), 'err_msg', False)
        # Processing the call keyword arguments (line 118)
        kwargs_157415 = {}
        # Getting the type of 'assert_equal' (line 118)
        assert_equal_157405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 118)
        assert_equal_call_result_157416 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), assert_equal_157405, *[len_call_result_157409, len_call_result_157413, err_msg_157414], **kwargs_157415)
        
        
        
        # Call to items(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_157419 = {}
        # Getting the type of 'desired' (line 119)
        desired_157417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'desired', False)
        # Obtaining the member 'items' of a type (line 119)
        items_157418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 20), desired_157417, 'items')
        # Calling items(args, kwargs) (line 119)
        items_call_result_157420 = invoke(stypy.reporting.localization.Localization(__file__, 119, 20), items_157418, *[], **kwargs_157419)
        
        # Testing the type of a for loop iterable (line 119)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 8), items_call_result_157420)
        # Getting the type of the for loop variable (line 119)
        for_loop_var_157421 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 8), items_call_result_157420)
        # Assigning a type to the variable 'k' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 8), for_loop_var_157421))
        # Assigning a type to the variable 'i' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 8), for_loop_var_157421))
        # SSA begins for a for statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'k' (line 120)
        k_157422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'k')
        # Getting the type of 'actual' (line 120)
        actual_157423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'actual')
        # Applying the binary operator 'notin' (line 120)
        result_contains_157424 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 15), 'notin', k_157422, actual_157423)
        
        # Testing the type of an if condition (line 120)
        if_condition_157425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 12), result_contains_157424)
        # Assigning a type to the variable 'if_condition_157425' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'if_condition_157425', if_condition_157425)
        # SSA begins for if statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AssertionError(...): (line 121)
        # Processing the call arguments (line 121)
        str_157427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 37), 'str', '%s not in %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 121)
        tuple_157428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 121)
        # Adding element type (line 121)
        # Getting the type of 'k' (line 121)
        k_157429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 55), tuple_157428, k_157429)
        # Adding element type (line 121)
        # Getting the type of 'actual' (line 121)
        actual_157430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 58), 'actual', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 55), tuple_157428, actual_157430)
        
        # Applying the binary operator '%' (line 121)
        result_mod_157431 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 37), '%', str_157427, tuple_157428)
        
        # Processing the call keyword arguments (line 121)
        kwargs_157432 = {}
        # Getting the type of 'AssertionError' (line 121)
        AssertionError_157426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'AssertionError', False)
        # Calling AssertionError(args, kwargs) (line 121)
        AssertionError_call_result_157433 = invoke(stypy.reporting.localization.Localization(__file__, 121, 22), AssertionError_157426, *[result_mod_157431], **kwargs_157432)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 121, 16), AssertionError_call_result_157433, 'raise parameter', BaseException)
        # SSA join for if statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 122)
        k_157435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'k', False)
        # Getting the type of 'actual' (line 122)
        actual_157436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'actual', False)
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___157437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 25), actual_157436, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_157438 = invoke(stypy.reporting.localization.Localization(__file__, 122, 25), getitem___157437, k_157435)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 122)
        k_157439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 44), 'k', False)
        # Getting the type of 'desired' (line 122)
        desired_157440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'desired', False)
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___157441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 36), desired_157440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_157442 = invoke(stypy.reporting.localization.Localization(__file__, 122, 36), getitem___157441, k_157439)
        
        str_157443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 48), 'str', 'key=%r\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 122)
        tuple_157444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 64), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 122)
        # Adding element type (line 122)
        # Getting the type of 'k' (line 122)
        k_157445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 64), tuple_157444, k_157445)
        # Adding element type (line 122)
        # Getting the type of 'err_msg' (line 122)
        err_msg_157446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 67), 'err_msg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 64), tuple_157444, err_msg_157446)
        
        # Applying the binary operator '%' (line 122)
        result_mod_157447 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 48), '%', str_157443, tuple_157444)
        
        # Processing the call keyword arguments (line 122)
        kwargs_157448 = {}
        # Getting the type of 'assert_equal' (line 122)
        assert_equal_157434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 122)
        assert_equal_call_result_157449 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), assert_equal_157434, *[subscript_call_result_157438, subscript_call_result_157442, result_mod_157447], **kwargs_157448)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', types.NoneType)

        if more_types_in_union_157390:
            # SSA join for if statement (line 115)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'desired' (line 125)
    desired_157451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'desired', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_157452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    # Getting the type of 'list' (line 125)
    list_157453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_157452, list_157453)
    # Adding element type (line 125)
    # Getting the type of 'tuple' (line 125)
    tuple_157454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_157452, tuple_157454)
    
    # Processing the call keyword arguments (line 125)
    kwargs_157455 = {}
    # Getting the type of 'isinstance' (line 125)
    isinstance_157450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 125)
    isinstance_call_result_157456 = invoke(stypy.reporting.localization.Localization(__file__, 125, 7), isinstance_157450, *[desired_157451, tuple_157452], **kwargs_157455)
    
    
    # Call to isinstance(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'actual' (line 125)
    actual_157458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 57), 'actual', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_157459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    # Getting the type of 'list' (line 125)
    list_157460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 66), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 66), tuple_157459, list_157460)
    # Adding element type (line 125)
    # Getting the type of 'tuple' (line 125)
    tuple_157461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 72), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 66), tuple_157459, tuple_157461)
    
    # Processing the call keyword arguments (line 125)
    kwargs_157462 = {}
    # Getting the type of 'isinstance' (line 125)
    isinstance_157457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 46), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 125)
    isinstance_call_result_157463 = invoke(stypy.reporting.localization.Localization(__file__, 125, 46), isinstance_157457, *[actual_157458, tuple_157459], **kwargs_157462)
    
    # Applying the binary operator 'and' (line 125)
    result_and_keyword_157464 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 7), 'and', isinstance_call_result_157456, isinstance_call_result_157463)
    
    # Testing the type of an if condition (line 125)
    if_condition_157465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 4), result_and_keyword_157464)
    # Assigning a type to the variable 'if_condition_157465' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'if_condition_157465', if_condition_157465)
    # SSA begins for if statement (line 125)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _assert_equal_on_sequences(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'actual' (line 126)
    actual_157467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 42), 'actual', False)
    # Getting the type of 'desired' (line 126)
    desired_157468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 50), 'desired', False)
    # Processing the call keyword arguments (line 126)
    str_157469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 67), 'str', '')
    keyword_157470 = str_157469
    kwargs_157471 = {'err_msg': keyword_157470}
    # Getting the type of '_assert_equal_on_sequences' (line 126)
    _assert_equal_on_sequences_157466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), '_assert_equal_on_sequences', False)
    # Calling _assert_equal_on_sequences(args, kwargs) (line 126)
    _assert_equal_on_sequences_call_result_157472 = invoke(stypy.reporting.localization.Localization(__file__, 126, 15), _assert_equal_on_sequences_157466, *[actual_157467, desired_157468], **kwargs_157471)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', _assert_equal_on_sequences_call_result_157472)
    # SSA join for if statement (line 125)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'actual' (line 127)
    actual_157474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'actual', False)
    # Getting the type of 'ndarray' (line 127)
    ndarray_157475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'ndarray', False)
    # Processing the call keyword arguments (line 127)
    kwargs_157476 = {}
    # Getting the type of 'isinstance' (line 127)
    isinstance_157473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 127)
    isinstance_call_result_157477 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), isinstance_157473, *[actual_157474, ndarray_157475], **kwargs_157476)
    
    
    # Call to isinstance(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'desired' (line 127)
    desired_157479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 54), 'desired', False)
    # Getting the type of 'ndarray' (line 127)
    ndarray_157480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 63), 'ndarray', False)
    # Processing the call keyword arguments (line 127)
    kwargs_157481 = {}
    # Getting the type of 'isinstance' (line 127)
    isinstance_157478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 43), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 127)
    isinstance_call_result_157482 = invoke(stypy.reporting.localization.Localization(__file__, 127, 43), isinstance_157478, *[desired_157479, ndarray_157480], **kwargs_157481)
    
    # Applying the binary operator 'or' (line 127)
    result_or_keyword_157483 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 12), 'or', isinstance_call_result_157477, isinstance_call_result_157482)
    
    # Applying the 'not' unary operator (line 127)
    result_not__157484 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 7), 'not', result_or_keyword_157483)
    
    # Testing the type of an if condition (line 127)
    if_condition_157485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 4), result_not__157484)
    # Assigning a type to the variable 'if_condition_157485' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'if_condition_157485', if_condition_157485)
    # SSA begins for if statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to build_err_msg(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Obtaining an instance of the builtin type 'list' (line 128)
    list_157487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 128)
    # Adding element type (line 128)
    # Getting the type of 'actual' (line 128)
    actual_157488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'actual', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 28), list_157487, actual_157488)
    # Adding element type (line 128)
    # Getting the type of 'desired' (line 128)
    desired_157489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'desired', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 28), list_157487, desired_157489)
    
    # Getting the type of 'err_msg' (line 128)
    err_msg_157490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 47), 'err_msg', False)
    # Processing the call keyword arguments (line 128)
    kwargs_157491 = {}
    # Getting the type of 'build_err_msg' (line 128)
    build_err_msg_157486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 14), 'build_err_msg', False)
    # Calling build_err_msg(args, kwargs) (line 128)
    build_err_msg_call_result_157492 = invoke(stypy.reporting.localization.Localization(__file__, 128, 14), build_err_msg_157486, *[list_157487, err_msg_157490], **kwargs_157491)
    
    # Assigning a type to the variable 'msg' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'msg', build_err_msg_call_result_157492)
    
    
    
    # Getting the type of 'desired' (line 129)
    desired_157493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'desired')
    # Getting the type of 'actual' (line 129)
    actual_157494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'actual')
    # Applying the binary operator '==' (line 129)
    result_eq_157495 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 15), '==', desired_157493, actual_157494)
    
    # Applying the 'not' unary operator (line 129)
    result_not__157496 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 11), 'not', result_eq_157495)
    
    # Testing the type of an if condition (line 129)
    if_condition_157497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), result_not__157496)
    # Assigning a type to the variable 'if_condition_157497' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_157497', if_condition_157497)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'msg' (line 130)
    msg_157499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'msg', False)
    # Processing the call keyword arguments (line 130)
    kwargs_157500 = {}
    # Getting the type of 'AssertionError' (line 130)
    AssertionError_157498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 130)
    AssertionError_call_result_157501 = invoke(stypy.reporting.localization.Localization(__file__, 130, 18), AssertionError_157498, *[msg_157499], **kwargs_157500)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 130, 12), AssertionError_call_result_157501, 'raise parameter', BaseException)
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'actual' (line 133)
    actual_157502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 9), 'actual')
    # Getting the type of 'masked' (line 133)
    masked_157503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'masked')
    # Applying the binary operator 'is' (line 133)
    result_is__157504 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 9), 'is', actual_157502, masked_157503)
    
    
    
    # Getting the type of 'desired' (line 133)
    desired_157505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'desired')
    # Getting the type of 'masked' (line 133)
    masked_157506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 47), 'masked')
    # Applying the binary operator 'is' (line 133)
    result_is__157507 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 36), 'is', desired_157505, masked_157506)
    
    # Applying the 'not' unary operator (line 133)
    result_not__157508 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 31), 'not', result_is__157507)
    
    # Applying the binary operator 'and' (line 133)
    result_and_keyword_157509 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 8), 'and', result_is__157504, result_not__157508)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'desired' (line 134)
    desired_157510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 14), 'desired')
    # Getting the type of 'masked' (line 134)
    masked_157511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'masked')
    # Applying the binary operator 'is' (line 134)
    result_is__157512 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 14), 'is', desired_157510, masked_157511)
    
    
    
    # Getting the type of 'actual' (line 134)
    actual_157513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'actual')
    # Getting the type of 'masked' (line 134)
    masked_157514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 52), 'masked')
    # Applying the binary operator 'is' (line 134)
    result_is__157515 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 42), 'is', actual_157513, masked_157514)
    
    # Applying the 'not' unary operator (line 134)
    result_not__157516 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 37), 'not', result_is__157515)
    
    # Applying the binary operator 'and' (line 134)
    result_and_keyword_157517 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 13), 'and', result_is__157512, result_not__157516)
    
    # Applying the binary operator 'or' (line 133)
    result_or_keyword_157518 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), 'or', result_and_keyword_157509, result_and_keyword_157517)
    
    # Testing the type of an if condition (line 133)
    if_condition_157519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_or_keyword_157518)
    # Assigning a type to the variable 'if_condition_157519' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_157519', if_condition_157519)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 135):
    
    # Assigning a Call to a Name (line 135):
    
    # Call to build_err_msg(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Obtaining an instance of the builtin type 'list' (line 135)
    list_157521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 135)
    # Adding element type (line 135)
    # Getting the type of 'actual' (line 135)
    actual_157522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'actual', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 28), list_157521, actual_157522)
    # Adding element type (line 135)
    # Getting the type of 'desired' (line 135)
    desired_157523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 37), 'desired', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 28), list_157521, desired_157523)
    
    # Getting the type of 'err_msg' (line 136)
    err_msg_157524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'err_msg', False)
    # Processing the call keyword arguments (line 135)
    str_157525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 44), 'str', '')
    keyword_157526 = str_157525
    
    # Obtaining an instance of the builtin type 'tuple' (line 136)
    tuple_157527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 136)
    # Adding element type (line 136)
    str_157528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 55), 'str', 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 55), tuple_157527, str_157528)
    # Adding element type (line 136)
    str_157529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 60), 'str', 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 55), tuple_157527, str_157529)
    
    keyword_157530 = tuple_157527
    kwargs_157531 = {'header': keyword_157526, 'names': keyword_157530}
    # Getting the type of 'build_err_msg' (line 135)
    build_err_msg_157520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'build_err_msg', False)
    # Calling build_err_msg(args, kwargs) (line 135)
    build_err_msg_call_result_157532 = invoke(stypy.reporting.localization.Localization(__file__, 135, 14), build_err_msg_157520, *[list_157521, err_msg_157524], **kwargs_157531)
    
    # Assigning a type to the variable 'msg' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'msg', build_err_msg_call_result_157532)
    
    # Call to ValueError(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'msg' (line 137)
    msg_157534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'msg', False)
    # Processing the call keyword arguments (line 137)
    kwargs_157535 = {}
    # Getting the type of 'ValueError' (line 137)
    ValueError_157533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 137)
    ValueError_call_result_157536 = invoke(stypy.reporting.localization.Localization(__file__, 137, 14), ValueError_157533, *[msg_157534], **kwargs_157535)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 137, 8), ValueError_call_result_157536, 'raise parameter', BaseException)
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 138):
    
    # Assigning a Call to a Name (line 138):
    
    # Call to array(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'actual' (line 138)
    actual_157539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), 'actual', False)
    # Processing the call keyword arguments (line 138)
    # Getting the type of 'False' (line 138)
    False_157540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 35), 'False', False)
    keyword_157541 = False_157540
    # Getting the type of 'True' (line 138)
    True_157542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 48), 'True', False)
    keyword_157543 = True_157542
    kwargs_157544 = {'subok': keyword_157543, 'copy': keyword_157541}
    # Getting the type of 'np' (line 138)
    np_157537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 138)
    array_157538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 13), np_157537, 'array')
    # Calling array(args, kwargs) (line 138)
    array_call_result_157545 = invoke(stypy.reporting.localization.Localization(__file__, 138, 13), array_157538, *[actual_157539], **kwargs_157544)
    
    # Assigning a type to the variable 'actual' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'actual', array_call_result_157545)
    
    # Assigning a Call to a Name (line 139):
    
    # Assigning a Call to a Name (line 139):
    
    # Call to array(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'desired' (line 139)
    desired_157548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'desired', False)
    # Processing the call keyword arguments (line 139)
    # Getting the type of 'False' (line 139)
    False_157549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 37), 'False', False)
    keyword_157550 = False_157549
    # Getting the type of 'True' (line 139)
    True_157551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'True', False)
    keyword_157552 = True_157551
    kwargs_157553 = {'subok': keyword_157552, 'copy': keyword_157550}
    # Getting the type of 'np' (line 139)
    np_157546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 139)
    array_157547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 14), np_157546, 'array')
    # Calling array(args, kwargs) (line 139)
    array_call_result_157554 = invoke(stypy.reporting.localization.Localization(__file__, 139, 14), array_157547, *[desired_157548], **kwargs_157553)
    
    # Assigning a type to the variable 'desired' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'desired', array_call_result_157554)
    
    # Assigning a Tuple to a Tuple (line 140):
    
    # Assigning a Attribute to a Name (line 140):
    # Getting the type of 'actual' (line 140)
    actual_157555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 37), 'actual')
    # Obtaining the member 'dtype' of a type (line 140)
    dtype_157556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 37), actual_157555, 'dtype')
    # Assigning a type to the variable 'tuple_assignment_157056' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'tuple_assignment_157056', dtype_157556)
    
    # Assigning a Attribute to a Name (line 140):
    # Getting the type of 'desired' (line 140)
    desired_157557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 51), 'desired')
    # Obtaining the member 'dtype' of a type (line 140)
    dtype_157558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 51), desired_157557, 'dtype')
    # Assigning a type to the variable 'tuple_assignment_157057' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'tuple_assignment_157057', dtype_157558)
    
    # Assigning a Name to a Name (line 140):
    # Getting the type of 'tuple_assignment_157056' (line 140)
    tuple_assignment_157056_157559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'tuple_assignment_157056')
    # Assigning a type to the variable 'actual_dtype' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 5), 'actual_dtype', tuple_assignment_157056_157559)
    
    # Assigning a Name to a Name (line 140):
    # Getting the type of 'tuple_assignment_157057' (line 140)
    tuple_assignment_157057_157560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'tuple_assignment_157057')
    # Assigning a type to the variable 'desired_dtype' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'desired_dtype', tuple_assignment_157057_157560)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'actual_dtype' (line 141)
    actual_dtype_157561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 7), 'actual_dtype')
    # Obtaining the member 'char' of a type (line 141)
    char_157562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 7), actual_dtype_157561, 'char')
    str_157563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'str', 'S')
    # Applying the binary operator '==' (line 141)
    result_eq_157564 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 7), '==', char_157562, str_157563)
    
    
    # Getting the type of 'desired_dtype' (line 141)
    desired_dtype_157565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 36), 'desired_dtype')
    # Obtaining the member 'char' of a type (line 141)
    char_157566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 36), desired_dtype_157565, 'char')
    str_157567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 58), 'str', 'S')
    # Applying the binary operator '==' (line 141)
    result_eq_157568 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 36), '==', char_157566, str_157567)
    
    # Applying the binary operator 'and' (line 141)
    result_and_keyword_157569 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 7), 'and', result_eq_157564, result_eq_157568)
    
    # Testing the type of an if condition (line 141)
    if_condition_157570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 4), result_and_keyword_157569)
    # Assigning a type to the variable 'if_condition_157570' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'if_condition_157570', if_condition_157570)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _assert_equal_on_sequences(...): (line 142)
    # Processing the call arguments (line 142)
    
    # Call to tolist(...): (line 142)
    # Processing the call keyword arguments (line 142)
    kwargs_157574 = {}
    # Getting the type of 'actual' (line 142)
    actual_157572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'actual', False)
    # Obtaining the member 'tolist' of a type (line 142)
    tolist_157573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 42), actual_157572, 'tolist')
    # Calling tolist(args, kwargs) (line 142)
    tolist_call_result_157575 = invoke(stypy.reporting.localization.Localization(__file__, 142, 42), tolist_157573, *[], **kwargs_157574)
    
    
    # Call to tolist(...): (line 143)
    # Processing the call keyword arguments (line 143)
    kwargs_157578 = {}
    # Getting the type of 'desired' (line 143)
    desired_157576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'desired', False)
    # Obtaining the member 'tolist' of a type (line 143)
    tolist_157577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 42), desired_157576, 'tolist')
    # Calling tolist(args, kwargs) (line 143)
    tolist_call_result_157579 = invoke(stypy.reporting.localization.Localization(__file__, 143, 42), tolist_157577, *[], **kwargs_157578)
    
    # Processing the call keyword arguments (line 142)
    str_157580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 50), 'str', '')
    keyword_157581 = str_157580
    kwargs_157582 = {'err_msg': keyword_157581}
    # Getting the type of '_assert_equal_on_sequences' (line 142)
    _assert_equal_on_sequences_157571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), '_assert_equal_on_sequences', False)
    # Calling _assert_equal_on_sequences(args, kwargs) (line 142)
    _assert_equal_on_sequences_call_result_157583 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), _assert_equal_on_sequences_157571, *[tolist_call_result_157575, tolist_call_result_157579], **kwargs_157582)
    
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', _assert_equal_on_sequences_call_result_157583)
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_array_equal(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'actual' (line 145)
    actual_157585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), 'actual', False)
    # Getting the type of 'desired' (line 145)
    desired_157586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 38), 'desired', False)
    # Getting the type of 'err_msg' (line 145)
    err_msg_157587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 47), 'err_msg', False)
    # Processing the call keyword arguments (line 145)
    kwargs_157588 = {}
    # Getting the type of 'assert_array_equal' (line 145)
    assert_array_equal_157584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 145)
    assert_array_equal_call_result_157589 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), assert_array_equal_157584, *[actual_157585, desired_157586, err_msg_157587], **kwargs_157588)
    
    # Assigning a type to the variable 'stypy_return_type' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type', assert_array_equal_call_result_157589)
    
    # ################# End of 'assert_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_157590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_equal'
    return stypy_return_type_157590

# Assigning a type to the variable 'assert_equal' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'assert_equal', assert_equal)

@norecursion
def fail_if_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_157591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 43), 'str', '')
    defaults = [str_157591]
    # Create a new context for function 'fail_if_equal'
    module_type_store = module_type_store.open_function_context('fail_if_equal', 148, 0, False)
    
    # Passed parameters checking function
    fail_if_equal.stypy_localization = localization
    fail_if_equal.stypy_type_of_self = None
    fail_if_equal.stypy_type_store = module_type_store
    fail_if_equal.stypy_function_name = 'fail_if_equal'
    fail_if_equal.stypy_param_names_list = ['actual', 'desired', 'err_msg']
    fail_if_equal.stypy_varargs_param_name = None
    fail_if_equal.stypy_kwargs_param_name = None
    fail_if_equal.stypy_call_defaults = defaults
    fail_if_equal.stypy_call_varargs = varargs
    fail_if_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fail_if_equal', ['actual', 'desired', 'err_msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fail_if_equal', localization, ['actual', 'desired', 'err_msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fail_if_equal(...)' code ##################

    str_157592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, (-1)), 'str', '\n    Raises an assertion error if two items are equal.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 153)
    # Getting the type of 'dict' (line 153)
    dict_157593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 27), 'dict')
    # Getting the type of 'desired' (line 153)
    desired_157594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'desired')
    
    (may_be_157595, more_types_in_union_157596) = may_be_subtype(dict_157593, desired_157594)

    if may_be_157595:

        if more_types_in_union_157596:
            # Runtime conditional SSA (line 153)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'desired' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'desired', remove_not_subtype_from_union(desired_157594, dict))
        
        # Type idiom detected: calculating its left and rigth part (line 154)
        # Getting the type of 'dict' (line 154)
        dict_157597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'dict')
        # Getting the type of 'actual' (line 154)
        actual_157598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'actual')
        
        (may_be_157599, more_types_in_union_157600) = may_not_be_subtype(dict_157597, actual_157598)

        if may_be_157599:

            if more_types_in_union_157600:
                # Runtime conditional SSA (line 154)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'actual' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'actual', remove_subtype_from_union(actual_157598, dict))
            
            # Call to AssertionError(...): (line 155)
            # Processing the call arguments (line 155)
            
            # Call to repr(...): (line 155)
            # Processing the call arguments (line 155)
            
            # Call to type(...): (line 155)
            # Processing the call arguments (line 155)
            # Getting the type of 'actual' (line 155)
            actual_157604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 43), 'actual', False)
            # Processing the call keyword arguments (line 155)
            kwargs_157605 = {}
            # Getting the type of 'type' (line 155)
            type_157603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 38), 'type', False)
            # Calling type(args, kwargs) (line 155)
            type_call_result_157606 = invoke(stypy.reporting.localization.Localization(__file__, 155, 38), type_157603, *[actual_157604], **kwargs_157605)
            
            # Processing the call keyword arguments (line 155)
            kwargs_157607 = {}
            # Getting the type of 'repr' (line 155)
            repr_157602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'repr', False)
            # Calling repr(args, kwargs) (line 155)
            repr_call_result_157608 = invoke(stypy.reporting.localization.Localization(__file__, 155, 33), repr_157602, *[type_call_result_157606], **kwargs_157607)
            
            # Processing the call keyword arguments (line 155)
            kwargs_157609 = {}
            # Getting the type of 'AssertionError' (line 155)
            AssertionError_157601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'AssertionError', False)
            # Calling AssertionError(args, kwargs) (line 155)
            AssertionError_call_result_157610 = invoke(stypy.reporting.localization.Localization(__file__, 155, 18), AssertionError_157601, *[repr_call_result_157608], **kwargs_157609)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 12), AssertionError_call_result_157610, 'raise parameter', BaseException)

            if more_types_in_union_157600:
                # SSA join for if statement (line 154)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to fail_if_equal(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Call to len(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'actual' (line 156)
        actual_157613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'actual', False)
        # Processing the call keyword arguments (line 156)
        kwargs_157614 = {}
        # Getting the type of 'len' (line 156)
        len_157612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'len', False)
        # Calling len(args, kwargs) (line 156)
        len_call_result_157615 = invoke(stypy.reporting.localization.Localization(__file__, 156, 22), len_157612, *[actual_157613], **kwargs_157614)
        
        
        # Call to len(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'desired' (line 156)
        desired_157617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 39), 'desired', False)
        # Processing the call keyword arguments (line 156)
        kwargs_157618 = {}
        # Getting the type of 'len' (line 156)
        len_157616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'len', False)
        # Calling len(args, kwargs) (line 156)
        len_call_result_157619 = invoke(stypy.reporting.localization.Localization(__file__, 156, 35), len_157616, *[desired_157617], **kwargs_157618)
        
        # Getting the type of 'err_msg' (line 156)
        err_msg_157620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 49), 'err_msg', False)
        # Processing the call keyword arguments (line 156)
        kwargs_157621 = {}
        # Getting the type of 'fail_if_equal' (line 156)
        fail_if_equal_157611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'fail_if_equal', False)
        # Calling fail_if_equal(args, kwargs) (line 156)
        fail_if_equal_call_result_157622 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), fail_if_equal_157611, *[len_call_result_157615, len_call_result_157619, err_msg_157620], **kwargs_157621)
        
        
        
        # Call to items(...): (line 157)
        # Processing the call keyword arguments (line 157)
        kwargs_157625 = {}
        # Getting the type of 'desired' (line 157)
        desired_157623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'desired', False)
        # Obtaining the member 'items' of a type (line 157)
        items_157624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 20), desired_157623, 'items')
        # Calling items(args, kwargs) (line 157)
        items_call_result_157626 = invoke(stypy.reporting.localization.Localization(__file__, 157, 20), items_157624, *[], **kwargs_157625)
        
        # Testing the type of a for loop iterable (line 157)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 8), items_call_result_157626)
        # Getting the type of the for loop variable (line 157)
        for_loop_var_157627 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 8), items_call_result_157626)
        # Assigning a type to the variable 'k' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 8), for_loop_var_157627))
        # Assigning a type to the variable 'i' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 8), for_loop_var_157627))
        # SSA begins for a for statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'k' (line 158)
        k_157628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'k')
        # Getting the type of 'actual' (line 158)
        actual_157629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'actual')
        # Applying the binary operator 'notin' (line 158)
        result_contains_157630 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 15), 'notin', k_157628, actual_157629)
        
        # Testing the type of an if condition (line 158)
        if_condition_157631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 12), result_contains_157630)
        # Assigning a type to the variable 'if_condition_157631' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'if_condition_157631', if_condition_157631)
        # SSA begins for if statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AssertionError(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Call to repr(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'k' (line 159)
        k_157634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 42), 'k', False)
        # Processing the call keyword arguments (line 159)
        kwargs_157635 = {}
        # Getting the type of 'repr' (line 159)
        repr_157633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 37), 'repr', False)
        # Calling repr(args, kwargs) (line 159)
        repr_call_result_157636 = invoke(stypy.reporting.localization.Localization(__file__, 159, 37), repr_157633, *[k_157634], **kwargs_157635)
        
        # Processing the call keyword arguments (line 159)
        kwargs_157637 = {}
        # Getting the type of 'AssertionError' (line 159)
        AssertionError_157632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'AssertionError', False)
        # Calling AssertionError(args, kwargs) (line 159)
        AssertionError_call_result_157638 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), AssertionError_157632, *[repr_call_result_157636], **kwargs_157637)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 159, 16), AssertionError_call_result_157638, 'raise parameter', BaseException)
        # SSA join for if statement (line 158)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to fail_if_equal(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 160)
        k_157640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 33), 'k', False)
        # Getting the type of 'actual' (line 160)
        actual_157641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'actual', False)
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___157642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 26), actual_157641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_157643 = invoke(stypy.reporting.localization.Localization(__file__, 160, 26), getitem___157642, k_157640)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 160)
        k_157644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 45), 'k', False)
        # Getting the type of 'desired' (line 160)
        desired_157645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'desired', False)
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___157646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 37), desired_157645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_157647 = invoke(stypy.reporting.localization.Localization(__file__, 160, 37), getitem___157646, k_157644)
        
        str_157648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 49), 'str', 'key=%r\n%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_157649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 65), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        # Getting the type of 'k' (line 160)
        k_157650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 65), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 65), tuple_157649, k_157650)
        # Adding element type (line 160)
        # Getting the type of 'err_msg' (line 160)
        err_msg_157651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 68), 'err_msg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 65), tuple_157649, err_msg_157651)
        
        # Applying the binary operator '%' (line 160)
        result_mod_157652 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 49), '%', str_157648, tuple_157649)
        
        # Processing the call keyword arguments (line 160)
        kwargs_157653 = {}
        # Getting the type of 'fail_if_equal' (line 160)
        fail_if_equal_157639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'fail_if_equal', False)
        # Calling fail_if_equal(args, kwargs) (line 160)
        fail_if_equal_call_result_157654 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), fail_if_equal_157639, *[subscript_call_result_157643, subscript_call_result_157647, result_mod_157652], **kwargs_157653)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'stypy_return_type', types.NoneType)

        if more_types_in_union_157596:
            # SSA join for if statement (line 153)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'desired' (line 162)
    desired_157656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'desired', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 162)
    tuple_157657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 162)
    # Adding element type (line 162)
    # Getting the type of 'list' (line 162)
    list_157658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 28), tuple_157657, list_157658)
    # Adding element type (line 162)
    # Getting the type of 'tuple' (line 162)
    tuple_157659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 34), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 28), tuple_157657, tuple_157659)
    
    # Processing the call keyword arguments (line 162)
    kwargs_157660 = {}
    # Getting the type of 'isinstance' (line 162)
    isinstance_157655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 162)
    isinstance_call_result_157661 = invoke(stypy.reporting.localization.Localization(__file__, 162, 7), isinstance_157655, *[desired_157656, tuple_157657], **kwargs_157660)
    
    
    # Call to isinstance(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'actual' (line 162)
    actual_157663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 57), 'actual', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 162)
    tuple_157664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 162)
    # Adding element type (line 162)
    # Getting the type of 'list' (line 162)
    list_157665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 66), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 66), tuple_157664, list_157665)
    # Adding element type (line 162)
    # Getting the type of 'tuple' (line 162)
    tuple_157666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 72), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 66), tuple_157664, tuple_157666)
    
    # Processing the call keyword arguments (line 162)
    kwargs_157667 = {}
    # Getting the type of 'isinstance' (line 162)
    isinstance_157662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 46), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 162)
    isinstance_call_result_157668 = invoke(stypy.reporting.localization.Localization(__file__, 162, 46), isinstance_157662, *[actual_157663, tuple_157664], **kwargs_157667)
    
    # Applying the binary operator 'and' (line 162)
    result_and_keyword_157669 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 7), 'and', isinstance_call_result_157661, isinstance_call_result_157668)
    
    # Testing the type of an if condition (line 162)
    if_condition_157670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 4), result_and_keyword_157669)
    # Assigning a type to the variable 'if_condition_157670' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'if_condition_157670', if_condition_157670)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fail_if_equal(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Call to len(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'actual' (line 163)
    actual_157673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'actual', False)
    # Processing the call keyword arguments (line 163)
    kwargs_157674 = {}
    # Getting the type of 'len' (line 163)
    len_157672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'len', False)
    # Calling len(args, kwargs) (line 163)
    len_call_result_157675 = invoke(stypy.reporting.localization.Localization(__file__, 163, 22), len_157672, *[actual_157673], **kwargs_157674)
    
    
    # Call to len(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'desired' (line 163)
    desired_157677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 39), 'desired', False)
    # Processing the call keyword arguments (line 163)
    kwargs_157678 = {}
    # Getting the type of 'len' (line 163)
    len_157676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'len', False)
    # Calling len(args, kwargs) (line 163)
    len_call_result_157679 = invoke(stypy.reporting.localization.Localization(__file__, 163, 35), len_157676, *[desired_157677], **kwargs_157678)
    
    # Getting the type of 'err_msg' (line 163)
    err_msg_157680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 49), 'err_msg', False)
    # Processing the call keyword arguments (line 163)
    kwargs_157681 = {}
    # Getting the type of 'fail_if_equal' (line 163)
    fail_if_equal_157671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'fail_if_equal', False)
    # Calling fail_if_equal(args, kwargs) (line 163)
    fail_if_equal_call_result_157682 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), fail_if_equal_157671, *[len_call_result_157675, len_call_result_157679, err_msg_157680], **kwargs_157681)
    
    
    
    # Call to range(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Call to len(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'desired' (line 164)
    desired_157685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'desired', False)
    # Processing the call keyword arguments (line 164)
    kwargs_157686 = {}
    # Getting the type of 'len' (line 164)
    len_157684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'len', False)
    # Calling len(args, kwargs) (line 164)
    len_call_result_157687 = invoke(stypy.reporting.localization.Localization(__file__, 164, 23), len_157684, *[desired_157685], **kwargs_157686)
    
    # Processing the call keyword arguments (line 164)
    kwargs_157688 = {}
    # Getting the type of 'range' (line 164)
    range_157683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), 'range', False)
    # Calling range(args, kwargs) (line 164)
    range_call_result_157689 = invoke(stypy.reporting.localization.Localization(__file__, 164, 17), range_157683, *[len_call_result_157687], **kwargs_157688)
    
    # Testing the type of a for loop iterable (line 164)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 164, 8), range_call_result_157689)
    # Getting the type of the for loop variable (line 164)
    for_loop_var_157690 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 164, 8), range_call_result_157689)
    # Assigning a type to the variable 'k' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'k', for_loop_var_157690)
    # SSA begins for a for statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to fail_if_equal(...): (line 165)
    # Processing the call arguments (line 165)
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 165)
    k_157692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'k', False)
    # Getting the type of 'actual' (line 165)
    actual_157693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'actual', False)
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___157694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 26), actual_157693, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_157695 = invoke(stypy.reporting.localization.Localization(__file__, 165, 26), getitem___157694, k_157692)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 165)
    k_157696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 45), 'k', False)
    # Getting the type of 'desired' (line 165)
    desired_157697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'desired', False)
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___157698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 37), desired_157697, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_157699 = invoke(stypy.reporting.localization.Localization(__file__, 165, 37), getitem___157698, k_157696)
    
    str_157700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 49), 'str', 'item=%r\n%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 165)
    tuple_157701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 165)
    # Adding element type (line 165)
    # Getting the type of 'k' (line 165)
    k_157702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 66), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 66), tuple_157701, k_157702)
    # Adding element type (line 165)
    # Getting the type of 'err_msg' (line 165)
    err_msg_157703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 69), 'err_msg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 66), tuple_157701, err_msg_157703)
    
    # Applying the binary operator '%' (line 165)
    result_mod_157704 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 49), '%', str_157700, tuple_157701)
    
    # Processing the call keyword arguments (line 165)
    kwargs_157705 = {}
    # Getting the type of 'fail_if_equal' (line 165)
    fail_if_equal_157691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'fail_if_equal', False)
    # Calling fail_if_equal(args, kwargs) (line 165)
    fail_if_equal_call_result_157706 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), fail_if_equal_157691, *[subscript_call_result_157695, subscript_call_result_157699, result_mod_157704], **kwargs_157705)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'actual' (line 167)
    actual_157708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 18), 'actual', False)
    # Getting the type of 'np' (line 167)
    np_157709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 26), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 167)
    ndarray_157710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 26), np_157709, 'ndarray')
    # Processing the call keyword arguments (line 167)
    kwargs_157711 = {}
    # Getting the type of 'isinstance' (line 167)
    isinstance_157707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 167)
    isinstance_call_result_157712 = invoke(stypy.reporting.localization.Localization(__file__, 167, 7), isinstance_157707, *[actual_157708, ndarray_157710], **kwargs_157711)
    
    
    # Call to isinstance(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'desired' (line 167)
    desired_157714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 52), 'desired', False)
    # Getting the type of 'np' (line 167)
    np_157715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 61), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 167)
    ndarray_157716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 61), np_157715, 'ndarray')
    # Processing the call keyword arguments (line 167)
    kwargs_157717 = {}
    # Getting the type of 'isinstance' (line 167)
    isinstance_157713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 167)
    isinstance_call_result_157718 = invoke(stypy.reporting.localization.Localization(__file__, 167, 41), isinstance_157713, *[desired_157714, ndarray_157716], **kwargs_157717)
    
    # Applying the binary operator 'or' (line 167)
    result_or_keyword_157719 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 7), 'or', isinstance_call_result_157712, isinstance_call_result_157718)
    
    # Testing the type of an if condition (line 167)
    if_condition_157720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 4), result_or_keyword_157719)
    # Assigning a type to the variable 'if_condition_157720' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'if_condition_157720', if_condition_157720)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to fail_if_array_equal(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'actual' (line 168)
    actual_157722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'actual', False)
    # Getting the type of 'desired' (line 168)
    desired_157723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 43), 'desired', False)
    # Getting the type of 'err_msg' (line 168)
    err_msg_157724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 52), 'err_msg', False)
    # Processing the call keyword arguments (line 168)
    kwargs_157725 = {}
    # Getting the type of 'fail_if_array_equal' (line 168)
    fail_if_array_equal_157721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'fail_if_array_equal', False)
    # Calling fail_if_array_equal(args, kwargs) (line 168)
    fail_if_array_equal_call_result_157726 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), fail_if_array_equal_157721, *[actual_157722, desired_157723, err_msg_157724], **kwargs_157725)
    
    # Assigning a type to the variable 'stypy_return_type' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'stypy_return_type', fail_if_array_equal_call_result_157726)
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 169):
    
    # Assigning a Call to a Name (line 169):
    
    # Call to build_err_msg(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Obtaining an instance of the builtin type 'list' (line 169)
    list_157728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 169)
    # Adding element type (line 169)
    # Getting the type of 'actual' (line 169)
    actual_157729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'actual', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 24), list_157728, actual_157729)
    # Adding element type (line 169)
    # Getting the type of 'desired' (line 169)
    desired_157730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), 'desired', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 24), list_157728, desired_157730)
    
    # Getting the type of 'err_msg' (line 169)
    err_msg_157731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), 'err_msg', False)
    # Processing the call keyword arguments (line 169)
    kwargs_157732 = {}
    # Getting the type of 'build_err_msg' (line 169)
    build_err_msg_157727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 10), 'build_err_msg', False)
    # Calling build_err_msg(args, kwargs) (line 169)
    build_err_msg_call_result_157733 = invoke(stypy.reporting.localization.Localization(__file__, 169, 10), build_err_msg_157727, *[list_157728, err_msg_157731], **kwargs_157732)
    
    # Assigning a type to the variable 'msg' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'msg', build_err_msg_call_result_157733)
    
    
    
    # Getting the type of 'desired' (line 170)
    desired_157734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'desired')
    # Getting the type of 'actual' (line 170)
    actual_157735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'actual')
    # Applying the binary operator '!=' (line 170)
    result_ne_157736 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 11), '!=', desired_157734, actual_157735)
    
    # Applying the 'not' unary operator (line 170)
    result_not__157737 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 7), 'not', result_ne_157736)
    
    # Testing the type of an if condition (line 170)
    if_condition_157738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), result_not__157737)
    # Assigning a type to the variable 'if_condition_157738' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_157738', if_condition_157738)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'msg' (line 171)
    msg_157740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'msg', False)
    # Processing the call keyword arguments (line 171)
    kwargs_157741 = {}
    # Getting the type of 'AssertionError' (line 171)
    AssertionError_157739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 171)
    AssertionError_call_result_157742 = invoke(stypy.reporting.localization.Localization(__file__, 171, 14), AssertionError_157739, *[msg_157740], **kwargs_157741)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 171, 8), AssertionError_call_result_157742, 'raise parameter', BaseException)
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'fail_if_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fail_if_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 148)
    stypy_return_type_157743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157743)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fail_if_equal'
    return stypy_return_type_157743

# Assigning a type to the variable 'fail_if_equal' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'fail_if_equal', fail_if_equal)

# Assigning a Name to a Name (line 174):

# Assigning a Name to a Name (line 174):
# Getting the type of 'fail_if_equal' (line 174)
fail_if_equal_157744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 19), 'fail_if_equal')
# Assigning a type to the variable 'assert_not_equal' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'assert_not_equal', fail_if_equal_157744)

@norecursion
def assert_almost_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_157745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 49), 'int')
    str_157746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 60), 'str', '')
    # Getting the type of 'True' (line 177)
    True_157747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 72), 'True')
    defaults = [int_157745, str_157746, True_157747]
    # Create a new context for function 'assert_almost_equal'
    module_type_store = module_type_store.open_function_context('assert_almost_equal', 177, 0, False)
    
    # Passed parameters checking function
    assert_almost_equal.stypy_localization = localization
    assert_almost_equal.stypy_type_of_self = None
    assert_almost_equal.stypy_type_store = module_type_store
    assert_almost_equal.stypy_function_name = 'assert_almost_equal'
    assert_almost_equal.stypy_param_names_list = ['actual', 'desired', 'decimal', 'err_msg', 'verbose']
    assert_almost_equal.stypy_varargs_param_name = None
    assert_almost_equal.stypy_kwargs_param_name = None
    assert_almost_equal.stypy_call_defaults = defaults
    assert_almost_equal.stypy_call_varargs = varargs
    assert_almost_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_almost_equal', ['actual', 'desired', 'decimal', 'err_msg', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_almost_equal', localization, ['actual', 'desired', 'decimal', 'err_msg', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_almost_equal(...)' code ##################

    str_157748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, (-1)), 'str', '\n    Asserts that two items are almost equal.\n\n    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal).\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'actual' (line 184)
    actual_157750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'actual', False)
    # Getting the type of 'np' (line 184)
    np_157751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 184)
    ndarray_157752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 26), np_157751, 'ndarray')
    # Processing the call keyword arguments (line 184)
    kwargs_157753 = {}
    # Getting the type of 'isinstance' (line 184)
    isinstance_157749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 184)
    isinstance_call_result_157754 = invoke(stypy.reporting.localization.Localization(__file__, 184, 7), isinstance_157749, *[actual_157750, ndarray_157752], **kwargs_157753)
    
    
    # Call to isinstance(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'desired' (line 184)
    desired_157756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 52), 'desired', False)
    # Getting the type of 'np' (line 184)
    np_157757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 61), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 184)
    ndarray_157758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 61), np_157757, 'ndarray')
    # Processing the call keyword arguments (line 184)
    kwargs_157759 = {}
    # Getting the type of 'isinstance' (line 184)
    isinstance_157755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 41), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 184)
    isinstance_call_result_157760 = invoke(stypy.reporting.localization.Localization(__file__, 184, 41), isinstance_157755, *[desired_157756, ndarray_157758], **kwargs_157759)
    
    # Applying the binary operator 'or' (line 184)
    result_or_keyword_157761 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 7), 'or', isinstance_call_result_157754, isinstance_call_result_157760)
    
    # Testing the type of an if condition (line 184)
    if_condition_157762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 4), result_or_keyword_157761)
    # Assigning a type to the variable 'if_condition_157762' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'if_condition_157762', if_condition_157762)
    # SSA begins for if statement (line 184)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_array_almost_equal(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'actual' (line 185)
    actual_157764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 41), 'actual', False)
    # Getting the type of 'desired' (line 185)
    desired_157765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 49), 'desired', False)
    # Processing the call keyword arguments (line 185)
    # Getting the type of 'decimal' (line 185)
    decimal_157766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 66), 'decimal', False)
    keyword_157767 = decimal_157766
    # Getting the type of 'err_msg' (line 186)
    err_msg_157768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 49), 'err_msg', False)
    keyword_157769 = err_msg_157768
    # Getting the type of 'verbose' (line 186)
    verbose_157770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 66), 'verbose', False)
    keyword_157771 = verbose_157770
    kwargs_157772 = {'decimal': keyword_157767, 'err_msg': keyword_157769, 'verbose': keyword_157771}
    # Getting the type of 'assert_array_almost_equal' (line 185)
    assert_array_almost_equal_157763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'assert_array_almost_equal', False)
    # Calling assert_array_almost_equal(args, kwargs) (line 185)
    assert_array_almost_equal_call_result_157773 = invoke(stypy.reporting.localization.Localization(__file__, 185, 15), assert_array_almost_equal_157763, *[actual_157764, desired_157765], **kwargs_157772)
    
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', assert_array_almost_equal_call_result_157773)
    # SSA join for if statement (line 184)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 187):
    
    # Assigning a Call to a Name (line 187):
    
    # Call to build_err_msg(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Obtaining an instance of the builtin type 'list' (line 187)
    list_157775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 187)
    # Adding element type (line 187)
    # Getting the type of 'actual' (line 187)
    actual_157776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'actual', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), list_157775, actual_157776)
    # Adding element type (line 187)
    # Getting the type of 'desired' (line 187)
    desired_157777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'desired', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), list_157775, desired_157777)
    
    # Processing the call keyword arguments (line 187)
    # Getting the type of 'err_msg' (line 188)
    err_msg_157778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 32), 'err_msg', False)
    keyword_157779 = err_msg_157778
    # Getting the type of 'verbose' (line 188)
    verbose_157780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 49), 'verbose', False)
    keyword_157781 = verbose_157780
    kwargs_157782 = {'err_msg': keyword_157779, 'verbose': keyword_157781}
    # Getting the type of 'build_err_msg' (line 187)
    build_err_msg_157774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 10), 'build_err_msg', False)
    # Calling build_err_msg(args, kwargs) (line 187)
    build_err_msg_call_result_157783 = invoke(stypy.reporting.localization.Localization(__file__, 187, 10), build_err_msg_157774, *[list_157775], **kwargs_157782)
    
    # Assigning a type to the variable 'msg' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'msg', build_err_msg_call_result_157783)
    
    
    
    
    # Call to round(...): (line 189)
    # Processing the call arguments (line 189)
    
    # Call to abs(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'desired' (line 189)
    desired_157786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'desired', False)
    # Getting the type of 'actual' (line 189)
    actual_157787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 31), 'actual', False)
    # Applying the binary operator '-' (line 189)
    result_sub_157788 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 21), '-', desired_157786, actual_157787)
    
    # Processing the call keyword arguments (line 189)
    kwargs_157789 = {}
    # Getting the type of 'abs' (line 189)
    abs_157785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'abs', False)
    # Calling abs(args, kwargs) (line 189)
    abs_call_result_157790 = invoke(stypy.reporting.localization.Localization(__file__, 189, 17), abs_157785, *[result_sub_157788], **kwargs_157789)
    
    # Getting the type of 'decimal' (line 189)
    decimal_157791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 40), 'decimal', False)
    # Processing the call keyword arguments (line 189)
    kwargs_157792 = {}
    # Getting the type of 'round' (line 189)
    round_157784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'round', False)
    # Calling round(args, kwargs) (line 189)
    round_call_result_157793 = invoke(stypy.reporting.localization.Localization(__file__, 189, 11), round_157784, *[abs_call_result_157790, decimal_157791], **kwargs_157792)
    
    int_157794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 52), 'int')
    # Applying the binary operator '==' (line 189)
    result_eq_157795 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), '==', round_call_result_157793, int_157794)
    
    # Applying the 'not' unary operator (line 189)
    result_not__157796 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 7), 'not', result_eq_157795)
    
    # Testing the type of an if condition (line 189)
    if_condition_157797 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 4), result_not__157796)
    # Assigning a type to the variable 'if_condition_157797' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'if_condition_157797', if_condition_157797)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'msg' (line 190)
    msg_157799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 29), 'msg', False)
    # Processing the call keyword arguments (line 190)
    kwargs_157800 = {}
    # Getting the type of 'AssertionError' (line 190)
    AssertionError_157798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 190)
    AssertionError_call_result_157801 = invoke(stypy.reporting.localization.Localization(__file__, 190, 14), AssertionError_157798, *[msg_157799], **kwargs_157800)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 190, 8), AssertionError_call_result_157801, 'raise parameter', BaseException)
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'assert_almost_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_almost_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_157802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157802)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_almost_equal'
    return stypy_return_type_157802

# Assigning a type to the variable 'assert_almost_equal' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'assert_almost_equal', assert_almost_equal)

# Assigning a Name to a Name (line 193):

# Assigning a Name to a Name (line 193):
# Getting the type of 'assert_almost_equal' (line 193)
assert_almost_equal_157803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'assert_almost_equal')
# Assigning a type to the variable 'assert_close' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'assert_close', assert_almost_equal_157803)

@norecursion
def assert_array_compare(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_157804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 51), 'str', '')
    # Getting the type of 'True' (line 196)
    True_157805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 63), 'True')
    str_157806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 76), 'str', '')
    # Getting the type of 'True' (line 197)
    True_157807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 36), 'True')
    defaults = [str_157804, True_157805, str_157806, True_157807]
    # Create a new context for function 'assert_array_compare'
    module_type_store = module_type_store.open_function_context('assert_array_compare', 196, 0, False)
    
    # Passed parameters checking function
    assert_array_compare.stypy_localization = localization
    assert_array_compare.stypy_type_of_self = None
    assert_array_compare.stypy_type_store = module_type_store
    assert_array_compare.stypy_function_name = 'assert_array_compare'
    assert_array_compare.stypy_param_names_list = ['comparison', 'x', 'y', 'err_msg', 'verbose', 'header', 'fill_value']
    assert_array_compare.stypy_varargs_param_name = None
    assert_array_compare.stypy_kwargs_param_name = None
    assert_array_compare.stypy_call_defaults = defaults
    assert_array_compare.stypy_call_varargs = varargs
    assert_array_compare.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_array_compare', ['comparison', 'x', 'y', 'err_msg', 'verbose', 'header', 'fill_value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_array_compare', localization, ['comparison', 'x', 'y', 'err_msg', 'verbose', 'header', 'fill_value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_array_compare(...)' code ##################

    str_157808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, (-1)), 'str', '\n    Asserts that comparison between two masked arrays is satisfied.\n\n    The comparison is elementwise.\n\n    ')
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to mask_or(...): (line 205)
    # Processing the call arguments (line 205)
    
    # Call to getmask(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'x' (line 205)
    x_157811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'x', False)
    # Processing the call keyword arguments (line 205)
    kwargs_157812 = {}
    # Getting the type of 'getmask' (line 205)
    getmask_157810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'getmask', False)
    # Calling getmask(args, kwargs) (line 205)
    getmask_call_result_157813 = invoke(stypy.reporting.localization.Localization(__file__, 205, 16), getmask_157810, *[x_157811], **kwargs_157812)
    
    
    # Call to getmask(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'y' (line 205)
    y_157815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 36), 'y', False)
    # Processing the call keyword arguments (line 205)
    kwargs_157816 = {}
    # Getting the type of 'getmask' (line 205)
    getmask_157814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 28), 'getmask', False)
    # Calling getmask(args, kwargs) (line 205)
    getmask_call_result_157817 = invoke(stypy.reporting.localization.Localization(__file__, 205, 28), getmask_157814, *[y_157815], **kwargs_157816)
    
    # Processing the call keyword arguments (line 205)
    kwargs_157818 = {}
    # Getting the type of 'mask_or' (line 205)
    mask_or_157809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'mask_or', False)
    # Calling mask_or(args, kwargs) (line 205)
    mask_or_call_result_157819 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), mask_or_157809, *[getmask_call_result_157813, getmask_call_result_157817], **kwargs_157818)
    
    # Assigning a type to the variable 'm' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'm', mask_or_call_result_157819)
    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to masked_array(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'x' (line 206)
    x_157821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), 'x', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'False' (line 206)
    False_157822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'False', False)
    keyword_157823 = False_157822
    # Getting the type of 'm' (line 206)
    m_157824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 41), 'm', False)
    keyword_157825 = m_157824
    # Getting the type of 'False' (line 206)
    False_157826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 54), 'False', False)
    keyword_157827 = False_157826
    # Getting the type of 'False' (line 206)
    False_157828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 67), 'False', False)
    keyword_157829 = False_157828
    kwargs_157830 = {'subok': keyword_157829, 'copy': keyword_157823, 'mask': keyword_157825, 'keep_mask': keyword_157827}
    # Getting the type of 'masked_array' (line 206)
    masked_array_157820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'masked_array', False)
    # Calling masked_array(args, kwargs) (line 206)
    masked_array_call_result_157831 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), masked_array_157820, *[x_157821], **kwargs_157830)
    
    # Assigning a type to the variable 'x' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'x', masked_array_call_result_157831)
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to masked_array(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'y' (line 207)
    y_157833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), 'y', False)
    # Processing the call keyword arguments (line 207)
    # Getting the type of 'False' (line 207)
    False_157834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'False', False)
    keyword_157835 = False_157834
    # Getting the type of 'm' (line 207)
    m_157836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 41), 'm', False)
    keyword_157837 = m_157836
    # Getting the type of 'False' (line 207)
    False_157838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 54), 'False', False)
    keyword_157839 = False_157838
    # Getting the type of 'False' (line 207)
    False_157840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 67), 'False', False)
    keyword_157841 = False_157840
    kwargs_157842 = {'subok': keyword_157841, 'copy': keyword_157835, 'mask': keyword_157837, 'keep_mask': keyword_157839}
    # Getting the type of 'masked_array' (line 207)
    masked_array_157832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'masked_array', False)
    # Calling masked_array(args, kwargs) (line 207)
    masked_array_call_result_157843 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), masked_array_157832, *[y_157833], **kwargs_157842)
    
    # Assigning a type to the variable 'y' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'y', masked_array_call_result_157843)
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 208)
    x_157844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 9), 'x')
    # Getting the type of 'masked' (line 208)
    masked_157845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 14), 'masked')
    # Applying the binary operator 'is' (line 208)
    result_is__157846 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 9), 'is', x_157844, masked_157845)
    
    
    
    # Getting the type of 'y' (line 208)
    y_157847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 31), 'y')
    # Getting the type of 'masked' (line 208)
    masked_157848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 36), 'masked')
    # Applying the binary operator 'is' (line 208)
    result_is__157849 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 31), 'is', y_157847, masked_157848)
    
    # Applying the 'not' unary operator (line 208)
    result_not__157850 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 26), 'not', result_is__157849)
    
    # Applying the binary operator 'and' (line 208)
    result_and_keyword_157851 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 8), 'and', result_is__157846, result_not__157850)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'y' (line 209)
    y_157852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'y')
    # Getting the type of 'masked' (line 209)
    masked_157853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'masked')
    # Applying the binary operator 'is' (line 209)
    result_is__157854 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 14), 'is', y_157852, masked_157853)
    
    
    
    # Getting the type of 'x' (line 209)
    x_157855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'x')
    # Getting the type of 'masked' (line 209)
    masked_157856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 41), 'masked')
    # Applying the binary operator 'is' (line 209)
    result_is__157857 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 36), 'is', x_157855, masked_157856)
    
    # Applying the 'not' unary operator (line 209)
    result_not__157858 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 31), 'not', result_is__157857)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_157859 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 13), 'and', result_is__157854, result_not__157858)
    
    # Applying the binary operator 'or' (line 208)
    result_or_keyword_157860 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 7), 'or', result_and_keyword_157851, result_and_keyword_157859)
    
    # Testing the type of an if condition (line 208)
    if_condition_157861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 4), result_or_keyword_157860)
    # Assigning a type to the variable 'if_condition_157861' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'if_condition_157861', if_condition_157861)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to build_err_msg(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_157863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of 'x' (line 210)
    x_157864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 29), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 28), list_157863, x_157864)
    # Adding element type (line 210)
    # Getting the type of 'y' (line 210)
    y_157865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 28), list_157863, y_157865)
    
    # Processing the call keyword arguments (line 210)
    # Getting the type of 'err_msg' (line 210)
    err_msg_157866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 44), 'err_msg', False)
    keyword_157867 = err_msg_157866
    # Getting the type of 'verbose' (line 210)
    verbose_157868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 61), 'verbose', False)
    keyword_157869 = verbose_157868
    # Getting the type of 'header' (line 211)
    header_157870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 35), 'header', False)
    keyword_157871 = header_157870
    
    # Obtaining an instance of the builtin type 'tuple' (line 211)
    tuple_157872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 211)
    # Adding element type (line 211)
    str_157873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 50), 'str', 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 50), tuple_157872, str_157873)
    # Adding element type (line 211)
    str_157874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 55), 'str', 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 50), tuple_157872, str_157874)
    
    keyword_157875 = tuple_157872
    kwargs_157876 = {'header': keyword_157871, 'err_msg': keyword_157867, 'verbose': keyword_157869, 'names': keyword_157875}
    # Getting the type of 'build_err_msg' (line 210)
    build_err_msg_157862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'build_err_msg', False)
    # Calling build_err_msg(args, kwargs) (line 210)
    build_err_msg_call_result_157877 = invoke(stypy.reporting.localization.Localization(__file__, 210, 14), build_err_msg_157862, *[list_157863], **kwargs_157876)
    
    # Assigning a type to the variable 'msg' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'msg', build_err_msg_call_result_157877)
    
    # Call to ValueError(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'msg' (line 212)
    msg_157879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 25), 'msg', False)
    # Processing the call keyword arguments (line 212)
    kwargs_157880 = {}
    # Getting the type of 'ValueError' (line 212)
    ValueError_157878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 212)
    ValueError_call_result_157881 = invoke(stypy.reporting.localization.Localization(__file__, 212, 14), ValueError_157878, *[msg_157879], **kwargs_157880)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 212, 8), ValueError_call_result_157881, 'raise parameter', BaseException)
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_array_compare(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'comparison' (line 214)
    comparison_157884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 38), 'comparison', False)
    
    # Call to filled(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'fill_value' (line 215)
    fill_value_157887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 47), 'fill_value', False)
    # Processing the call keyword arguments (line 215)
    kwargs_157888 = {}
    # Getting the type of 'x' (line 215)
    x_157885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 38), 'x', False)
    # Obtaining the member 'filled' of a type (line 215)
    filled_157886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 38), x_157885, 'filled')
    # Calling filled(args, kwargs) (line 215)
    filled_call_result_157889 = invoke(stypy.reporting.localization.Localization(__file__, 215, 38), filled_157886, *[fill_value_157887], **kwargs_157888)
    
    
    # Call to filled(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'fill_value' (line 216)
    fill_value_157892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 47), 'fill_value', False)
    # Processing the call keyword arguments (line 216)
    kwargs_157893 = {}
    # Getting the type of 'y' (line 216)
    y_157890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 38), 'y', False)
    # Obtaining the member 'filled' of a type (line 216)
    filled_157891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 38), y_157890, 'filled')
    # Calling filled(args, kwargs) (line 216)
    filled_call_result_157894 = invoke(stypy.reporting.localization.Localization(__file__, 216, 38), filled_157891, *[fill_value_157892], **kwargs_157893)
    
    # Processing the call keyword arguments (line 214)
    # Getting the type of 'err_msg' (line 217)
    err_msg_157895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 46), 'err_msg', False)
    keyword_157896 = err_msg_157895
    # Getting the type of 'verbose' (line 218)
    verbose_157897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 46), 'verbose', False)
    keyword_157898 = verbose_157897
    # Getting the type of 'header' (line 218)
    header_157899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 62), 'header', False)
    keyword_157900 = header_157899
    kwargs_157901 = {'header': keyword_157900, 'err_msg': keyword_157896, 'verbose': keyword_157898}
    # Getting the type of 'utils' (line 214)
    utils_157882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'utils', False)
    # Obtaining the member 'assert_array_compare' of a type (line 214)
    assert_array_compare_157883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 11), utils_157882, 'assert_array_compare')
    # Calling assert_array_compare(args, kwargs) (line 214)
    assert_array_compare_call_result_157902 = invoke(stypy.reporting.localization.Localization(__file__, 214, 11), assert_array_compare_157883, *[comparison_157884, filled_call_result_157889, filled_call_result_157894], **kwargs_157901)
    
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type', assert_array_compare_call_result_157902)
    
    # ################# End of 'assert_array_compare(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_array_compare' in the type store
    # Getting the type of 'stypy_return_type' (line 196)
    stypy_return_type_157903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157903)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_array_compare'
    return stypy_return_type_157903

# Assigning a type to the variable 'assert_array_compare' (line 196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'assert_array_compare', assert_array_compare)

@norecursion
def assert_array_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_157904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 37), 'str', '')
    # Getting the type of 'True' (line 221)
    True_157905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 49), 'True')
    defaults = [str_157904, True_157905]
    # Create a new context for function 'assert_array_equal'
    module_type_store = module_type_store.open_function_context('assert_array_equal', 221, 0, False)
    
    # Passed parameters checking function
    assert_array_equal.stypy_localization = localization
    assert_array_equal.stypy_type_of_self = None
    assert_array_equal.stypy_type_store = module_type_store
    assert_array_equal.stypy_function_name = 'assert_array_equal'
    assert_array_equal.stypy_param_names_list = ['x', 'y', 'err_msg', 'verbose']
    assert_array_equal.stypy_varargs_param_name = None
    assert_array_equal.stypy_kwargs_param_name = None
    assert_array_equal.stypy_call_defaults = defaults
    assert_array_equal.stypy_call_varargs = varargs
    assert_array_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_array_equal', ['x', 'y', 'err_msg', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_array_equal', localization, ['x', 'y', 'err_msg', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_array_equal(...)' code ##################

    str_157906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, (-1)), 'str', '\n    Checks the elementwise equality of two masked arrays.\n\n    ')
    
    # Call to assert_array_compare(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'operator' (line 226)
    operator_157908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 25), 'operator', False)
    # Obtaining the member '__eq__' of a type (line 226)
    eq___157909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 25), operator_157908, '__eq__')
    # Getting the type of 'x' (line 226)
    x_157910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 42), 'x', False)
    # Getting the type of 'y' (line 226)
    y_157911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 45), 'y', False)
    # Processing the call keyword arguments (line 226)
    # Getting the type of 'err_msg' (line 227)
    err_msg_157912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'err_msg', False)
    keyword_157913 = err_msg_157912
    # Getting the type of 'verbose' (line 227)
    verbose_157914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 50), 'verbose', False)
    keyword_157915 = verbose_157914
    str_157916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 32), 'str', 'Arrays are not equal')
    keyword_157917 = str_157916
    kwargs_157918 = {'header': keyword_157917, 'err_msg': keyword_157913, 'verbose': keyword_157915}
    # Getting the type of 'assert_array_compare' (line 226)
    assert_array_compare_157907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'assert_array_compare', False)
    # Calling assert_array_compare(args, kwargs) (line 226)
    assert_array_compare_call_result_157919 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), assert_array_compare_157907, *[eq___157909, x_157910, y_157911], **kwargs_157918)
    
    
    # ################# End of 'assert_array_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_array_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 221)
    stypy_return_type_157920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157920)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_array_equal'
    return stypy_return_type_157920

# Assigning a type to the variable 'assert_array_equal' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'assert_array_equal', assert_array_equal)

@norecursion
def fail_if_array_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_157921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 38), 'str', '')
    # Getting the type of 'True' (line 231)
    True_157922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 50), 'True')
    defaults = [str_157921, True_157922]
    # Create a new context for function 'fail_if_array_equal'
    module_type_store = module_type_store.open_function_context('fail_if_array_equal', 231, 0, False)
    
    # Passed parameters checking function
    fail_if_array_equal.stypy_localization = localization
    fail_if_array_equal.stypy_type_of_self = None
    fail_if_array_equal.stypy_type_store = module_type_store
    fail_if_array_equal.stypy_function_name = 'fail_if_array_equal'
    fail_if_array_equal.stypy_param_names_list = ['x', 'y', 'err_msg', 'verbose']
    fail_if_array_equal.stypy_varargs_param_name = None
    fail_if_array_equal.stypy_kwargs_param_name = None
    fail_if_array_equal.stypy_call_defaults = defaults
    fail_if_array_equal.stypy_call_varargs = varargs
    fail_if_array_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fail_if_array_equal', ['x', 'y', 'err_msg', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fail_if_array_equal', localization, ['x', 'y', 'err_msg', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fail_if_array_equal(...)' code ##################

    str_157923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, (-1)), 'str', '\n    Raises an assertion error if two masked arrays are not equal elementwise.\n\n    ')

    @norecursion
    def compare(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'compare'
        module_type_store = module_type_store.open_function_context('compare', 236, 4, False)
        
        # Passed parameters checking function
        compare.stypy_localization = localization
        compare.stypy_type_of_self = None
        compare.stypy_type_store = module_type_store
        compare.stypy_function_name = 'compare'
        compare.stypy_param_names_list = ['x', 'y']
        compare.stypy_varargs_param_name = None
        compare.stypy_kwargs_param_name = None
        compare.stypy_call_defaults = defaults
        compare.stypy_call_varargs = varargs
        compare.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'compare', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compare', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compare(...)' code ##################

        
        
        # Call to alltrue(...): (line 237)
        # Processing the call arguments (line 237)
        
        # Call to approx(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'x' (line 237)
        x_157927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 38), 'x', False)
        # Getting the type of 'y' (line 237)
        y_157928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 41), 'y', False)
        # Processing the call keyword arguments (line 237)
        kwargs_157929 = {}
        # Getting the type of 'approx' (line 237)
        approx_157926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 31), 'approx', False)
        # Calling approx(args, kwargs) (line 237)
        approx_call_result_157930 = invoke(stypy.reporting.localization.Localization(__file__, 237, 31), approx_157926, *[x_157927, y_157928], **kwargs_157929)
        
        # Processing the call keyword arguments (line 237)
        kwargs_157931 = {}
        # Getting the type of 'np' (line 237)
        np_157924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'np', False)
        # Obtaining the member 'alltrue' of a type (line 237)
        alltrue_157925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 20), np_157924, 'alltrue')
        # Calling alltrue(args, kwargs) (line 237)
        alltrue_call_result_157932 = invoke(stypy.reporting.localization.Localization(__file__, 237, 20), alltrue_157925, *[approx_call_result_157930], **kwargs_157931)
        
        # Applying the 'not' unary operator (line 237)
        result_not__157933 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 16), 'not', alltrue_call_result_157932)
        
        # Assigning a type to the variable 'stypy_return_type' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type', result_not__157933)
        
        # ################# End of 'compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compare' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_157934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_157934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compare'
        return stypy_return_type_157934

    # Assigning a type to the variable 'compare' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'compare', compare)
    
    # Call to assert_array_compare(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'compare' (line 238)
    compare_157936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 25), 'compare', False)
    # Getting the type of 'x' (line 238)
    x_157937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 34), 'x', False)
    # Getting the type of 'y' (line 238)
    y_157938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 37), 'y', False)
    # Processing the call keyword arguments (line 238)
    # Getting the type of 'err_msg' (line 238)
    err_msg_157939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 48), 'err_msg', False)
    keyword_157940 = err_msg_157939
    # Getting the type of 'verbose' (line 238)
    verbose_157941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 65), 'verbose', False)
    keyword_157942 = verbose_157941
    str_157943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 32), 'str', 'Arrays are not equal')
    keyword_157944 = str_157943
    kwargs_157945 = {'header': keyword_157944, 'err_msg': keyword_157940, 'verbose': keyword_157942}
    # Getting the type of 'assert_array_compare' (line 238)
    assert_array_compare_157935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'assert_array_compare', False)
    # Calling assert_array_compare(args, kwargs) (line 238)
    assert_array_compare_call_result_157946 = invoke(stypy.reporting.localization.Localization(__file__, 238, 4), assert_array_compare_157935, *[compare_157936, x_157937, y_157938], **kwargs_157945)
    
    
    # ################# End of 'fail_if_array_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fail_if_array_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 231)
    stypy_return_type_157947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157947)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fail_if_array_equal'
    return stypy_return_type_157947

# Assigning a type to the variable 'fail_if_array_equal' (line 231)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'fail_if_array_equal', fail_if_array_equal)

@norecursion
def assert_array_approx_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_157948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 44), 'int')
    str_157949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 55), 'str', '')
    # Getting the type of 'True' (line 242)
    True_157950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 67), 'True')
    defaults = [int_157948, str_157949, True_157950]
    # Create a new context for function 'assert_array_approx_equal'
    module_type_store = module_type_store.open_function_context('assert_array_approx_equal', 242, 0, False)
    
    # Passed parameters checking function
    assert_array_approx_equal.stypy_localization = localization
    assert_array_approx_equal.stypy_type_of_self = None
    assert_array_approx_equal.stypy_type_store = module_type_store
    assert_array_approx_equal.stypy_function_name = 'assert_array_approx_equal'
    assert_array_approx_equal.stypy_param_names_list = ['x', 'y', 'decimal', 'err_msg', 'verbose']
    assert_array_approx_equal.stypy_varargs_param_name = None
    assert_array_approx_equal.stypy_kwargs_param_name = None
    assert_array_approx_equal.stypy_call_defaults = defaults
    assert_array_approx_equal.stypy_call_varargs = varargs
    assert_array_approx_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_array_approx_equal', ['x', 'y', 'decimal', 'err_msg', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_array_approx_equal', localization, ['x', 'y', 'decimal', 'err_msg', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_array_approx_equal(...)' code ##################

    str_157951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, (-1)), 'str', '\n    Checks the equality of two masked arrays, up to given number odecimals.\n\n    The equality is checked elementwise.\n\n    ')

    @norecursion
    def compare(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'compare'
        module_type_store = module_type_store.open_function_context('compare', 249, 4, False)
        
        # Passed parameters checking function
        compare.stypy_localization = localization
        compare.stypy_type_of_self = None
        compare.stypy_type_store = module_type_store
        compare.stypy_function_name = 'compare'
        compare.stypy_param_names_list = ['x', 'y']
        compare.stypy_varargs_param_name = None
        compare.stypy_kwargs_param_name = None
        compare.stypy_call_defaults = defaults
        compare.stypy_call_varargs = varargs
        compare.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'compare', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compare', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compare(...)' code ##################

        str_157952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 8), 'str', 'Returns the result of the loose comparison between x and y).')
        
        # Call to approx(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'x' (line 251)
        x_157954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 22), 'x', False)
        # Getting the type of 'y' (line 251)
        y_157955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 25), 'y', False)
        # Processing the call keyword arguments (line 251)
        float_157956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 33), 'float')
        
        # Getting the type of 'decimal' (line 251)
        decimal_157957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 41), 'decimal', False)
        # Applying the 'usub' unary operator (line 251)
        result___neg___157958 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 40), 'usub', decimal_157957)
        
        # Applying the binary operator '**' (line 251)
        result_pow_157959 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 33), '**', float_157956, result___neg___157958)
        
        keyword_157960 = result_pow_157959
        kwargs_157961 = {'rtol': keyword_157960}
        # Getting the type of 'approx' (line 251)
        approx_157953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'approx', False)
        # Calling approx(args, kwargs) (line 251)
        approx_call_result_157962 = invoke(stypy.reporting.localization.Localization(__file__, 251, 15), approx_157953, *[x_157954, y_157955], **kwargs_157961)
        
        # Assigning a type to the variable 'stypy_return_type' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'stypy_return_type', approx_call_result_157962)
        
        # ################# End of 'compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compare' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_157963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_157963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compare'
        return stypy_return_type_157963

    # Assigning a type to the variable 'compare' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'compare', compare)
    
    # Call to assert_array_compare(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'compare' (line 252)
    compare_157965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 25), 'compare', False)
    # Getting the type of 'x' (line 252)
    x_157966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 34), 'x', False)
    # Getting the type of 'y' (line 252)
    y_157967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 37), 'y', False)
    # Processing the call keyword arguments (line 252)
    # Getting the type of 'err_msg' (line 252)
    err_msg_157968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 48), 'err_msg', False)
    keyword_157969 = err_msg_157968
    # Getting the type of 'verbose' (line 252)
    verbose_157970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 65), 'verbose', False)
    keyword_157971 = verbose_157970
    str_157972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 32), 'str', 'Arrays are not almost equal')
    keyword_157973 = str_157972
    kwargs_157974 = {'header': keyword_157973, 'err_msg': keyword_157969, 'verbose': keyword_157971}
    # Getting the type of 'assert_array_compare' (line 252)
    assert_array_compare_157964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'assert_array_compare', False)
    # Calling assert_array_compare(args, kwargs) (line 252)
    assert_array_compare_call_result_157975 = invoke(stypy.reporting.localization.Localization(__file__, 252, 4), assert_array_compare_157964, *[compare_157965, x_157966, y_157967], **kwargs_157974)
    
    
    # ################# End of 'assert_array_approx_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_array_approx_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 242)
    stypy_return_type_157976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_157976)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_array_approx_equal'
    return stypy_return_type_157976

# Assigning a type to the variable 'assert_array_approx_equal' (line 242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'assert_array_approx_equal', assert_array_approx_equal)

@norecursion
def assert_array_almost_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_157977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 44), 'int')
    str_157978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 55), 'str', '')
    # Getting the type of 'True' (line 256)
    True_157979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 67), 'True')
    defaults = [int_157977, str_157978, True_157979]
    # Create a new context for function 'assert_array_almost_equal'
    module_type_store = module_type_store.open_function_context('assert_array_almost_equal', 256, 0, False)
    
    # Passed parameters checking function
    assert_array_almost_equal.stypy_localization = localization
    assert_array_almost_equal.stypy_type_of_self = None
    assert_array_almost_equal.stypy_type_store = module_type_store
    assert_array_almost_equal.stypy_function_name = 'assert_array_almost_equal'
    assert_array_almost_equal.stypy_param_names_list = ['x', 'y', 'decimal', 'err_msg', 'verbose']
    assert_array_almost_equal.stypy_varargs_param_name = None
    assert_array_almost_equal.stypy_kwargs_param_name = None
    assert_array_almost_equal.stypy_call_defaults = defaults
    assert_array_almost_equal.stypy_call_varargs = varargs
    assert_array_almost_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_array_almost_equal', ['x', 'y', 'decimal', 'err_msg', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_array_almost_equal', localization, ['x', 'y', 'decimal', 'err_msg', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_array_almost_equal(...)' code ##################

    str_157980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, (-1)), 'str', '\n    Checks the equality of two masked arrays, up to given number odecimals.\n\n    The equality is checked elementwise.\n\n    ')

    @norecursion
    def compare(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'compare'
        module_type_store = module_type_store.open_function_context('compare', 263, 4, False)
        
        # Passed parameters checking function
        compare.stypy_localization = localization
        compare.stypy_type_of_self = None
        compare.stypy_type_store = module_type_store
        compare.stypy_function_name = 'compare'
        compare.stypy_param_names_list = ['x', 'y']
        compare.stypy_varargs_param_name = None
        compare.stypy_kwargs_param_name = None
        compare.stypy_call_defaults = defaults
        compare.stypy_call_varargs = varargs
        compare.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'compare', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compare', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compare(...)' code ##################

        str_157981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'str', 'Returns the result of the loose comparison between x and y).')
        
        # Call to almost(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'x' (line 265)
        x_157983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'x', False)
        # Getting the type of 'y' (line 265)
        y_157984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'y', False)
        # Getting the type of 'decimal' (line 265)
        decimal_157985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 28), 'decimal', False)
        # Processing the call keyword arguments (line 265)
        kwargs_157986 = {}
        # Getting the type of 'almost' (line 265)
        almost_157982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'almost', False)
        # Calling almost(args, kwargs) (line 265)
        almost_call_result_157987 = invoke(stypy.reporting.localization.Localization(__file__, 265, 15), almost_157982, *[x_157983, y_157984, decimal_157985], **kwargs_157986)
        
        # Assigning a type to the variable 'stypy_return_type' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'stypy_return_type', almost_call_result_157987)
        
        # ################# End of 'compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compare' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_157988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_157988)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compare'
        return stypy_return_type_157988

    # Assigning a type to the variable 'compare' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'compare', compare)
    
    # Call to assert_array_compare(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'compare' (line 266)
    compare_157990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'compare', False)
    # Getting the type of 'x' (line 266)
    x_157991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 'x', False)
    # Getting the type of 'y' (line 266)
    y_157992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'y', False)
    # Processing the call keyword arguments (line 266)
    # Getting the type of 'err_msg' (line 266)
    err_msg_157993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 48), 'err_msg', False)
    keyword_157994 = err_msg_157993
    # Getting the type of 'verbose' (line 266)
    verbose_157995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 65), 'verbose', False)
    keyword_157996 = verbose_157995
    str_157997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 32), 'str', 'Arrays are not almost equal')
    keyword_157998 = str_157997
    kwargs_157999 = {'header': keyword_157998, 'err_msg': keyword_157994, 'verbose': keyword_157996}
    # Getting the type of 'assert_array_compare' (line 266)
    assert_array_compare_157989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'assert_array_compare', False)
    # Calling assert_array_compare(args, kwargs) (line 266)
    assert_array_compare_call_result_158000 = invoke(stypy.reporting.localization.Localization(__file__, 266, 4), assert_array_compare_157989, *[compare_157990, x_157991, y_157992], **kwargs_157999)
    
    
    # ################# End of 'assert_array_almost_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_array_almost_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 256)
    stypy_return_type_158001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_158001)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_array_almost_equal'
    return stypy_return_type_158001

# Assigning a type to the variable 'assert_array_almost_equal' (line 256)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'assert_array_almost_equal', assert_array_almost_equal)

@norecursion
def assert_array_less(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_158002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 36), 'str', '')
    # Getting the type of 'True' (line 270)
    True_158003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 48), 'True')
    defaults = [str_158002, True_158003]
    # Create a new context for function 'assert_array_less'
    module_type_store = module_type_store.open_function_context('assert_array_less', 270, 0, False)
    
    # Passed parameters checking function
    assert_array_less.stypy_localization = localization
    assert_array_less.stypy_type_of_self = None
    assert_array_less.stypy_type_store = module_type_store
    assert_array_less.stypy_function_name = 'assert_array_less'
    assert_array_less.stypy_param_names_list = ['x', 'y', 'err_msg', 'verbose']
    assert_array_less.stypy_varargs_param_name = None
    assert_array_less.stypy_kwargs_param_name = None
    assert_array_less.stypy_call_defaults = defaults
    assert_array_less.stypy_call_varargs = varargs
    assert_array_less.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_array_less', ['x', 'y', 'err_msg', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_array_less', localization, ['x', 'y', 'err_msg', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_array_less(...)' code ##################

    str_158004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, (-1)), 'str', '\n    Checks that x is smaller than y elementwise.\n\n    ')
    
    # Call to assert_array_compare(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'operator' (line 275)
    operator_158006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 25), 'operator', False)
    # Obtaining the member '__lt__' of a type (line 275)
    lt___158007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 25), operator_158006, '__lt__')
    # Getting the type of 'x' (line 275)
    x_158008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 42), 'x', False)
    # Getting the type of 'y' (line 275)
    y_158009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 45), 'y', False)
    # Processing the call keyword arguments (line 275)
    # Getting the type of 'err_msg' (line 276)
    err_msg_158010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 33), 'err_msg', False)
    keyword_158011 = err_msg_158010
    # Getting the type of 'verbose' (line 276)
    verbose_158012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 50), 'verbose', False)
    keyword_158013 = verbose_158012
    str_158014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'str', 'Arrays are not less-ordered')
    keyword_158015 = str_158014
    kwargs_158016 = {'header': keyword_158015, 'err_msg': keyword_158011, 'verbose': keyword_158013}
    # Getting the type of 'assert_array_compare' (line 275)
    assert_array_compare_158005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'assert_array_compare', False)
    # Calling assert_array_compare(args, kwargs) (line 275)
    assert_array_compare_call_result_158017 = invoke(stypy.reporting.localization.Localization(__file__, 275, 4), assert_array_compare_158005, *[lt___158007, x_158008, y_158009], **kwargs_158016)
    
    
    # ################# End of 'assert_array_less(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_array_less' in the type store
    # Getting the type of 'stypy_return_type' (line 270)
    stypy_return_type_158018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_158018)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_array_less'
    return stypy_return_type_158018

# Assigning a type to the variable 'assert_array_less' (line 270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 0), 'assert_array_less', assert_array_less)

@norecursion
def assert_mask_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_158019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 38), 'str', '')
    defaults = [str_158019]
    # Create a new context for function 'assert_mask_equal'
    module_type_store = module_type_store.open_function_context('assert_mask_equal', 280, 0, False)
    
    # Passed parameters checking function
    assert_mask_equal.stypy_localization = localization
    assert_mask_equal.stypy_type_of_self = None
    assert_mask_equal.stypy_type_store = module_type_store
    assert_mask_equal.stypy_function_name = 'assert_mask_equal'
    assert_mask_equal.stypy_param_names_list = ['m1', 'm2', 'err_msg']
    assert_mask_equal.stypy_varargs_param_name = None
    assert_mask_equal.stypy_kwargs_param_name = None
    assert_mask_equal.stypy_call_defaults = defaults
    assert_mask_equal.stypy_call_varargs = varargs
    assert_mask_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_mask_equal', ['m1', 'm2', 'err_msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_mask_equal', localization, ['m1', 'm2', 'err_msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_mask_equal(...)' code ##################

    str_158020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, (-1)), 'str', '\n    Asserts the equality of two masks.\n\n    ')
    
    
    # Getting the type of 'm1' (line 285)
    m1_158021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'm1')
    # Getting the type of 'nomask' (line 285)
    nomask_158022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 13), 'nomask')
    # Applying the binary operator 'is' (line 285)
    result_is__158023 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 7), 'is', m1_158021, nomask_158022)
    
    # Testing the type of an if condition (line 285)
    if_condition_158024 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 4), result_is__158023)
    # Assigning a type to the variable 'if_condition_158024' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'if_condition_158024', if_condition_158024)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_(...): (line 286)
    # Processing the call arguments (line 286)
    
    # Getting the type of 'm2' (line 286)
    m2_158026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'm2', False)
    # Getting the type of 'nomask' (line 286)
    nomask_158027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'nomask', False)
    # Applying the binary operator 'is' (line 286)
    result_is__158028 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 16), 'is', m2_158026, nomask_158027)
    
    # Processing the call keyword arguments (line 286)
    kwargs_158029 = {}
    # Getting the type of 'assert_' (line 286)
    assert__158025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 286)
    assert__call_result_158030 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), assert__158025, *[result_is__158028], **kwargs_158029)
    
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm2' (line 287)
    m2_158031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 7), 'm2')
    # Getting the type of 'nomask' (line 287)
    nomask_158032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 13), 'nomask')
    # Applying the binary operator 'is' (line 287)
    result_is__158033 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 7), 'is', m2_158031, nomask_158032)
    
    # Testing the type of an if condition (line 287)
    if_condition_158034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 4), result_is__158033)
    # Assigning a type to the variable 'if_condition_158034' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'if_condition_158034', if_condition_158034)
    # SSA begins for if statement (line 287)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_(...): (line 288)
    # Processing the call arguments (line 288)
    
    # Getting the type of 'm1' (line 288)
    m1_158036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'm1', False)
    # Getting the type of 'nomask' (line 288)
    nomask_158037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'nomask', False)
    # Applying the binary operator 'is' (line 288)
    result_is__158038 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 16), 'is', m1_158036, nomask_158037)
    
    # Processing the call keyword arguments (line 288)
    kwargs_158039 = {}
    # Getting the type of 'assert_' (line 288)
    assert__158035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 288)
    assert__call_result_158040 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), assert__158035, *[result_is__158038], **kwargs_158039)
    
    # SSA join for if statement (line 287)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_array_equal(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'm1' (line 289)
    m1_158042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'm1', False)
    # Getting the type of 'm2' (line 289)
    m2_158043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 27), 'm2', False)
    # Processing the call keyword arguments (line 289)
    # Getting the type of 'err_msg' (line 289)
    err_msg_158044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 39), 'err_msg', False)
    keyword_158045 = err_msg_158044
    kwargs_158046 = {'err_msg': keyword_158045}
    # Getting the type of 'assert_array_equal' (line 289)
    assert_array_equal_158041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 289)
    assert_array_equal_call_result_158047 = invoke(stypy.reporting.localization.Localization(__file__, 289, 4), assert_array_equal_158041, *[m1_158042, m2_158043], **kwargs_158046)
    
    
    # ################# End of 'assert_mask_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_mask_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 280)
    stypy_return_type_158048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_158048)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_mask_equal'
    return stypy_return_type_158048

# Assigning a type to the variable 'assert_mask_equal' (line 280)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), 'assert_mask_equal', assert_mask_equal)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

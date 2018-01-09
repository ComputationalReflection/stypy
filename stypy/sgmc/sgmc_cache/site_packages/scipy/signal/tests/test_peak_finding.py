
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import copy
4: 
5: import numpy as np
6: from numpy.testing import (assert_equal,
7:     assert_array_equal, assert_)
8: from scipy.signal._peak_finding import (argrelmax, argrelmin,
9:     find_peaks_cwt, _identify_ridge_lines)
10: from scipy._lib.six import xrange
11: 
12: 
13: def _gen_gaussians(center_locs, sigmas, total_length):
14:     xdata = np.arange(0, total_length).astype(float)
15:     out_data = np.zeros(total_length, dtype=float)
16:     for ind, sigma in enumerate(sigmas):
17:         tmp = (xdata - center_locs[ind]) / sigma
18:         out_data += np.exp(-(tmp**2))
19:     return out_data
20: 
21: 
22: def _gen_gaussians_even(sigmas, total_length):
23:     num_peaks = len(sigmas)
24:     delta = total_length / (num_peaks + 1)
25:     center_locs = np.linspace(delta, total_length - delta, num=num_peaks).astype(int)
26:     out_data = _gen_gaussians(center_locs, sigmas, total_length)
27:     return out_data, center_locs
28: 
29: 
30: def _gen_ridge_line(start_locs, max_locs, length, distances, gaps):
31:     '''
32:     Generate coordinates for a ridge line.
33: 
34:     Will be a series of coordinates, starting a start_loc (length 2).
35:     The maximum distance between any adjacent columns will be
36:     `max_distance`, the max distance between adjacent rows
37:     will be `map_gap'.
38: 
39:     `max_locs` should be the size of the intended matrix. The
40:     ending coordinates are guaranteed to be less than `max_locs`,
41:     although they may not approach `max_locs` at all.
42:     '''
43: 
44:     def keep_bounds(num, max_val):
45:         out = max(num, 0)
46:         out = min(out, max_val)
47:         return out
48: 
49:     gaps = copy.deepcopy(gaps)
50:     distances = copy.deepcopy(distances)
51: 
52:     locs = np.zeros([length, 2], dtype=int)
53:     locs[0, :] = start_locs
54:     total_length = max_locs[0] - start_locs[0] - sum(gaps)
55:     if total_length < length:
56:         raise ValueError('Cannot generate ridge line according to constraints')
57:     dist_int = length / len(distances) - 1
58:     gap_int = length / len(gaps) - 1
59:     for ind in xrange(1, length):
60:         nextcol = locs[ind - 1, 1]
61:         nextrow = locs[ind - 1, 0] + 1
62:         if (ind % dist_int == 0) and (len(distances) > 0):
63:             nextcol += ((-1)**ind)*distances.pop()
64:         if (ind % gap_int == 0) and (len(gaps) > 0):
65:             nextrow += gaps.pop()
66:         nextrow = keep_bounds(nextrow, max_locs[0])
67:         nextcol = keep_bounds(nextcol, max_locs[1])
68:         locs[ind, :] = [nextrow, nextcol]
69: 
70:     return [locs[:, 0], locs[:, 1]]
71: 
72: 
73: class TestRidgeLines(object):
74: 
75:     def test_empty(self):
76:         test_matr = np.zeros([20, 100])
77:         lines = _identify_ridge_lines(test_matr, 2*np.ones(20), 1)
78:         assert_(len(lines) == 0)
79: 
80:     def test_minimal(self):
81:         test_matr = np.zeros([20, 100])
82:         test_matr[0, 10] = 1
83:         lines = _identify_ridge_lines(test_matr, 2*np.ones(20), 1)
84:         assert_(len(lines) == 1)
85: 
86:         test_matr = np.zeros([20, 100])
87:         test_matr[0:2, 10] = 1
88:         lines = _identify_ridge_lines(test_matr, 2*np.ones(20), 1)
89:         assert_(len(lines) == 1)
90: 
91:     def test_single_pass(self):
92:         distances = [0, 1, 2, 5]
93:         gaps = [0, 1, 2, 0, 1]
94:         test_matr = np.zeros([20, 50]) + 1e-12
95:         length = 12
96:         line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
97:         test_matr[line[0], line[1]] = 1
98:         max_distances = max(distances)*np.ones(20)
99:         identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
100:         assert_array_equal(identified_lines, [line])
101: 
102:     def test_single_bigdist(self):
103:         distances = [0, 1, 2, 5]
104:         gaps = [0, 1, 2, 4]
105:         test_matr = np.zeros([20, 50])
106:         length = 12
107:         line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
108:         test_matr[line[0], line[1]] = 1
109:         max_dist = 3
110:         max_distances = max_dist*np.ones(20)
111:         #This should get 2 lines, since the distance is too large
112:         identified_lines = _identify_ridge_lines(test_matr, max_distances, max(gaps) + 1)
113:         assert_(len(identified_lines) == 2)
114: 
115:         for iline in identified_lines:
116:             adists = np.diff(iline[1])
117:             np.testing.assert_array_less(np.abs(adists), max_dist)
118: 
119:             agaps = np.diff(iline[0])
120:             np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)
121: 
122:     def test_single_biggap(self):
123:         distances = [0, 1, 2, 5]
124:         max_gap = 3
125:         gaps = [0, 4, 2, 1]
126:         test_matr = np.zeros([20, 50])
127:         length = 12
128:         line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
129:         test_matr[line[0], line[1]] = 1
130:         max_dist = 6
131:         max_distances = max_dist*np.ones(20)
132:         #This should get 2 lines, since the gap is too large
133:         identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
134:         assert_(len(identified_lines) == 2)
135: 
136:         for iline in identified_lines:
137:             adists = np.diff(iline[1])
138:             np.testing.assert_array_less(np.abs(adists), max_dist)
139: 
140:             agaps = np.diff(iline[0])
141:             np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)
142: 
143:     def test_single_biggaps(self):
144:         distances = [0]
145:         max_gap = 1
146:         gaps = [3, 6]
147:         test_matr = np.zeros([50, 50])
148:         length = 30
149:         line = _gen_ridge_line([0, 25], test_matr.shape, length, distances, gaps)
150:         test_matr[line[0], line[1]] = 1
151:         max_dist = 1
152:         max_distances = max_dist*np.ones(50)
153:         #This should get 3 lines, since the gaps are too large
154:         identified_lines = _identify_ridge_lines(test_matr, max_distances, max_gap)
155:         assert_(len(identified_lines) == 3)
156: 
157:         for iline in identified_lines:
158:             adists = np.diff(iline[1])
159:             np.testing.assert_array_less(np.abs(adists), max_dist)
160: 
161:             agaps = np.diff(iline[0])
162:             np.testing.assert_array_less(np.abs(agaps), max(gaps) + 0.1)
163: 
164: 
165: class TestArgrel(object):
166: 
167:     def test_empty(self):
168:         # Regression test for gh-2832.
169:         # When there are no relative extrema, make sure that
170:         # the number of empty arrays returned matches the
171:         # dimension of the input.
172: 
173:         empty_array = np.array([], dtype=int)
174: 
175:         z1 = np.zeros(5)
176: 
177:         i = argrelmin(z1)
178:         assert_equal(len(i), 1)
179:         assert_array_equal(i[0], empty_array)
180: 
181:         z2 = np.zeros((3,5))
182: 
183:         row, col = argrelmin(z2, axis=0)
184:         assert_array_equal(row, empty_array)
185:         assert_array_equal(col, empty_array)
186: 
187:         row, col = argrelmin(z2, axis=1)
188:         assert_array_equal(row, empty_array)
189:         assert_array_equal(col, empty_array)
190: 
191:     def test_basic(self):
192:         # Note: the docstrings for the argrel{min,max,extrema} functions
193:         # do not give a guarantee of the order of the indices, so we'll
194:         # sort them before testing.
195: 
196:         x = np.array([[1, 2, 2, 3, 2],
197:                       [2, 1, 2, 2, 3],
198:                       [3, 2, 1, 2, 2],
199:                       [2, 3, 2, 1, 2],
200:                       [1, 2, 3, 2, 1]])
201: 
202:         row, col = argrelmax(x, axis=0)
203:         order = np.argsort(row)
204:         assert_equal(row[order], [1, 2, 3])
205:         assert_equal(col[order], [4, 0, 1])
206: 
207:         row, col = argrelmax(x, axis=1)
208:         order = np.argsort(row)
209:         assert_equal(row[order], [0, 3, 4])
210:         assert_equal(col[order], [3, 1, 2])
211: 
212:         row, col = argrelmin(x, axis=0)
213:         order = np.argsort(row)
214:         assert_equal(row[order], [1, 2, 3])
215:         assert_equal(col[order], [1, 2, 3])
216: 
217:         row, col = argrelmin(x, axis=1)
218:         order = np.argsort(row)
219:         assert_equal(row[order], [1, 2, 3])
220:         assert_equal(col[order], [1, 2, 3])
221: 
222:     def test_highorder(self):
223:         order = 2
224:         sigmas = [1.0, 2.0, 10.0, 5.0, 15.0]
225:         test_data, act_locs = _gen_gaussians_even(sigmas, 500)
226:         test_data[act_locs + order] = test_data[act_locs]*0.99999
227:         test_data[act_locs - order] = test_data[act_locs]*0.99999
228:         rel_max_locs = argrelmax(test_data, order=order, mode='clip')[0]
229: 
230:         assert_(len(rel_max_locs) == len(act_locs))
231:         assert_((rel_max_locs == act_locs).all())
232: 
233:     def test_2d_gaussians(self):
234:         sigmas = [1.0, 2.0, 10.0]
235:         test_data, act_locs = _gen_gaussians_even(sigmas, 100)
236:         rot_factor = 20
237:         rot_range = np.arange(0, len(test_data)) - rot_factor
238:         test_data_2 = np.vstack([test_data, test_data[rot_range]])
239:         rel_max_rows, rel_max_cols = argrelmax(test_data_2, axis=1, order=1)
240: 
241:         for rw in xrange(0, test_data_2.shape[0]):
242:             inds = (rel_max_rows == rw)
243: 
244:             assert_(len(rel_max_cols[inds]) == len(act_locs))
245:             assert_((act_locs == (rel_max_cols[inds] - rot_factor*rw)).all())
246: 
247: 
248: class TestFindPeaks(object):
249: 
250:     def test_find_peaks_exact(self):
251:         '''
252:         Generate a series of gaussians and attempt to find the peak locations.
253:         '''
254:         sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
255:         num_points = 500
256:         test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
257:         widths = np.arange(0.1, max(sigmas))
258:         found_locs = find_peaks_cwt(test_data, widths, gap_thresh=2, min_snr=0,
259:                                          min_length=None)
260:         np.testing.assert_array_equal(found_locs, act_locs,
261:                         "Found maximum locations did not equal those expected")
262: 
263:     def test_find_peaks_withnoise(self):
264:         '''
265:         Verify that peak locations are (approximately) found
266:         for a series of gaussians with added noise.
267:         '''
268:         sigmas = [5.0, 3.0, 10.0, 20.0, 10.0, 50.0]
269:         num_points = 500
270:         test_data, act_locs = _gen_gaussians_even(sigmas, num_points)
271:         widths = np.arange(0.1, max(sigmas))
272:         noise_amp = 0.07
273:         np.random.seed(18181911)
274:         test_data += (np.random.rand(num_points) - 0.5)*(2*noise_amp)
275:         found_locs = find_peaks_cwt(test_data, widths, min_length=15,
276:                                          gap_thresh=1, min_snr=noise_amp / 5)
277: 
278:         np.testing.assert_equal(len(found_locs), len(act_locs), 'Different number' +
279:                                 'of peaks found than expected')
280:         diffs = np.abs(found_locs - act_locs)
281:         max_diffs = np.array(sigmas) / 5
282:         np.testing.assert_array_less(diffs, max_diffs, 'Maximum location differed' +
283:                                      'by more than %s' % (max_diffs))
284: 
285:     def test_find_peaks_nopeak(self):
286:         '''
287:         Verify that no peak is found in
288:         data that's just noise.
289:         '''
290:         noise_amp = 1.0
291:         num_points = 100
292:         np.random.seed(181819141)
293:         test_data = (np.random.rand(num_points) - 0.5)*(2*noise_amp)
294:         widths = np.arange(10, 50)
295:         found_locs = find_peaks_cwt(test_data, widths, min_snr=5, noise_perc=30)
296:         np.testing.assert_equal(len(found_locs), 0)
297: 
298: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import copy' statement (line 3)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324975 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_324975) is not StypyTypeError):

    if (import_324975 != 'pyd_module'):
        __import__(import_324975)
        sys_modules_324976 = sys.modules[import_324975]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_324976.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_324975)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_equal, assert_array_equal, assert_' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324977 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_324977) is not StypyTypeError):

    if (import_324977 != 'pyd_module'):
        __import__(import_324977)
        sys_modules_324978 = sys.modules[import_324977]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_324978.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_324978, sys_modules_324978.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_'], [assert_equal, assert_array_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_324977)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.signal._peak_finding import argrelmax, argrelmin, find_peaks_cwt, _identify_ridge_lines' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324979 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._peak_finding')

if (type(import_324979) is not StypyTypeError):

    if (import_324979 != 'pyd_module'):
        __import__(import_324979)
        sys_modules_324980 = sys.modules[import_324979]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._peak_finding', sys_modules_324980.module_type_store, module_type_store, ['argrelmax', 'argrelmin', 'find_peaks_cwt', '_identify_ridge_lines'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_324980, sys_modules_324980.module_type_store, module_type_store)
    else:
        from scipy.signal._peak_finding import argrelmax, argrelmin, find_peaks_cwt, _identify_ridge_lines

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._peak_finding', None, module_type_store, ['argrelmax', 'argrelmin', 'find_peaks_cwt', '_identify_ridge_lines'], [argrelmax, argrelmin, find_peaks_cwt, _identify_ridge_lines])

else:
    # Assigning a type to the variable 'scipy.signal._peak_finding' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._peak_finding', import_324979)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib.six import xrange' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_324981 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six')

if (type(import_324981) is not StypyTypeError):

    if (import_324981 != 'pyd_module'):
        __import__(import_324981)
        sys_modules_324982 = sys.modules[import_324981]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', sys_modules_324982.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_324982, sys_modules_324982.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib.six', import_324981)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')


@norecursion
def _gen_gaussians(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_gen_gaussians'
    module_type_store = module_type_store.open_function_context('_gen_gaussians', 13, 0, False)
    
    # Passed parameters checking function
    _gen_gaussians.stypy_localization = localization
    _gen_gaussians.stypy_type_of_self = None
    _gen_gaussians.stypy_type_store = module_type_store
    _gen_gaussians.stypy_function_name = '_gen_gaussians'
    _gen_gaussians.stypy_param_names_list = ['center_locs', 'sigmas', 'total_length']
    _gen_gaussians.stypy_varargs_param_name = None
    _gen_gaussians.stypy_kwargs_param_name = None
    _gen_gaussians.stypy_call_defaults = defaults
    _gen_gaussians.stypy_call_varargs = varargs
    _gen_gaussians.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_gen_gaussians', ['center_locs', 'sigmas', 'total_length'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_gen_gaussians', localization, ['center_locs', 'sigmas', 'total_length'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_gen_gaussians(...)' code ##################

    
    # Assigning a Call to a Name (line 14):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to astype(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'float' (line 14)
    float_324990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 46), 'float', False)
    # Processing the call keyword arguments (line 14)
    kwargs_324991 = {}
    
    # Call to arange(...): (line 14)
    # Processing the call arguments (line 14)
    int_324985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'int')
    # Getting the type of 'total_length' (line 14)
    total_length_324986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 25), 'total_length', False)
    # Processing the call keyword arguments (line 14)
    kwargs_324987 = {}
    # Getting the type of 'np' (line 14)
    np_324983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'np', False)
    # Obtaining the member 'arange' of a type (line 14)
    arange_324984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 12), np_324983, 'arange')
    # Calling arange(args, kwargs) (line 14)
    arange_call_result_324988 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), arange_324984, *[int_324985, total_length_324986], **kwargs_324987)
    
    # Obtaining the member 'astype' of a type (line 14)
    astype_324989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 12), arange_call_result_324988, 'astype')
    # Calling astype(args, kwargs) (line 14)
    astype_call_result_324992 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), astype_324989, *[float_324990], **kwargs_324991)
    
    # Assigning a type to the variable 'xdata' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'xdata', astype_call_result_324992)
    
    # Assigning a Call to a Name (line 15):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to zeros(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'total_length' (line 15)
    total_length_324995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'total_length', False)
    # Processing the call keyword arguments (line 15)
    # Getting the type of 'float' (line 15)
    float_324996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 44), 'float', False)
    keyword_324997 = float_324996
    kwargs_324998 = {'dtype': keyword_324997}
    # Getting the type of 'np' (line 15)
    np_324993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 15)
    zeros_324994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 15), np_324993, 'zeros')
    # Calling zeros(args, kwargs) (line 15)
    zeros_call_result_324999 = invoke(stypy.reporting.localization.Localization(__file__, 15, 15), zeros_324994, *[total_length_324995], **kwargs_324998)
    
    # Assigning a type to the variable 'out_data' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'out_data', zeros_call_result_324999)
    
    
    # Call to enumerate(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'sigmas' (line 16)
    sigmas_325001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 32), 'sigmas', False)
    # Processing the call keyword arguments (line 16)
    kwargs_325002 = {}
    # Getting the type of 'enumerate' (line 16)
    enumerate_325000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 16)
    enumerate_call_result_325003 = invoke(stypy.reporting.localization.Localization(__file__, 16, 22), enumerate_325000, *[sigmas_325001], **kwargs_325002)
    
    # Testing the type of a for loop iterable (line 16)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 4), enumerate_call_result_325003)
    # Getting the type of the for loop variable (line 16)
    for_loop_var_325004 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 4), enumerate_call_result_325003)
    # Assigning a type to the variable 'ind' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), for_loop_var_325004))
    # Assigning a type to the variable 'sigma' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'sigma', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), for_loop_var_325004))
    # SSA begins for a for statement (line 16)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 17):
    
    # Assigning a BinOp to a Name (line 17):
    # Getting the type of 'xdata' (line 17)
    xdata_325005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'xdata')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 17)
    ind_325006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 35), 'ind')
    # Getting the type of 'center_locs' (line 17)
    center_locs_325007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'center_locs')
    # Obtaining the member '__getitem__' of a type (line 17)
    getitem___325008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 23), center_locs_325007, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 17)
    subscript_call_result_325009 = invoke(stypy.reporting.localization.Localization(__file__, 17, 23), getitem___325008, ind_325006)
    
    # Applying the binary operator '-' (line 17)
    result_sub_325010 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 15), '-', xdata_325005, subscript_call_result_325009)
    
    # Getting the type of 'sigma' (line 17)
    sigma_325011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 43), 'sigma')
    # Applying the binary operator 'div' (line 17)
    result_div_325012 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 14), 'div', result_sub_325010, sigma_325011)
    
    # Assigning a type to the variable 'tmp' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'tmp', result_div_325012)
    
    # Getting the type of 'out_data' (line 18)
    out_data_325013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'out_data')
    
    # Call to exp(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Getting the type of 'tmp' (line 18)
    tmp_325016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 29), 'tmp', False)
    int_325017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'int')
    # Applying the binary operator '**' (line 18)
    result_pow_325018 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 29), '**', tmp_325016, int_325017)
    
    # Applying the 'usub' unary operator (line 18)
    result___neg___325019 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 27), 'usub', result_pow_325018)
    
    # Processing the call keyword arguments (line 18)
    kwargs_325020 = {}
    # Getting the type of 'np' (line 18)
    np_325014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'np', False)
    # Obtaining the member 'exp' of a type (line 18)
    exp_325015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 20), np_325014, 'exp')
    # Calling exp(args, kwargs) (line 18)
    exp_call_result_325021 = invoke(stypy.reporting.localization.Localization(__file__, 18, 20), exp_325015, *[result___neg___325019], **kwargs_325020)
    
    # Applying the binary operator '+=' (line 18)
    result_iadd_325022 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 8), '+=', out_data_325013, exp_call_result_325021)
    # Assigning a type to the variable 'out_data' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'out_data', result_iadd_325022)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'out_data' (line 19)
    out_data_325023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'out_data')
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type', out_data_325023)
    
    # ################# End of '_gen_gaussians(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_gen_gaussians' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_325024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_325024)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_gen_gaussians'
    return stypy_return_type_325024

# Assigning a type to the variable '_gen_gaussians' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_gen_gaussians', _gen_gaussians)

@norecursion
def _gen_gaussians_even(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_gen_gaussians_even'
    module_type_store = module_type_store.open_function_context('_gen_gaussians_even', 22, 0, False)
    
    # Passed parameters checking function
    _gen_gaussians_even.stypy_localization = localization
    _gen_gaussians_even.stypy_type_of_self = None
    _gen_gaussians_even.stypy_type_store = module_type_store
    _gen_gaussians_even.stypy_function_name = '_gen_gaussians_even'
    _gen_gaussians_even.stypy_param_names_list = ['sigmas', 'total_length']
    _gen_gaussians_even.stypy_varargs_param_name = None
    _gen_gaussians_even.stypy_kwargs_param_name = None
    _gen_gaussians_even.stypy_call_defaults = defaults
    _gen_gaussians_even.stypy_call_varargs = varargs
    _gen_gaussians_even.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_gen_gaussians_even', ['sigmas', 'total_length'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_gen_gaussians_even', localization, ['sigmas', 'total_length'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_gen_gaussians_even(...)' code ##################

    
    # Assigning a Call to a Name (line 23):
    
    # Assigning a Call to a Name (line 23):
    
    # Call to len(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'sigmas' (line 23)
    sigmas_325026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'sigmas', False)
    # Processing the call keyword arguments (line 23)
    kwargs_325027 = {}
    # Getting the type of 'len' (line 23)
    len_325025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'len', False)
    # Calling len(args, kwargs) (line 23)
    len_call_result_325028 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), len_325025, *[sigmas_325026], **kwargs_325027)
    
    # Assigning a type to the variable 'num_peaks' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'num_peaks', len_call_result_325028)
    
    # Assigning a BinOp to a Name (line 24):
    
    # Assigning a BinOp to a Name (line 24):
    # Getting the type of 'total_length' (line 24)
    total_length_325029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'total_length')
    # Getting the type of 'num_peaks' (line 24)
    num_peaks_325030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 28), 'num_peaks')
    int_325031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 40), 'int')
    # Applying the binary operator '+' (line 24)
    result_add_325032 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 28), '+', num_peaks_325030, int_325031)
    
    # Applying the binary operator 'div' (line 24)
    result_div_325033 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 12), 'div', total_length_325029, result_add_325032)
    
    # Assigning a type to the variable 'delta' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'delta', result_div_325033)
    
    # Assigning a Call to a Name (line 25):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to astype(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'int' (line 25)
    int_325045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 81), 'int', False)
    # Processing the call keyword arguments (line 25)
    kwargs_325046 = {}
    
    # Call to linspace(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'delta' (line 25)
    delta_325036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'delta', False)
    # Getting the type of 'total_length' (line 25)
    total_length_325037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), 'total_length', False)
    # Getting the type of 'delta' (line 25)
    delta_325038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 52), 'delta', False)
    # Applying the binary operator '-' (line 25)
    result_sub_325039 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 37), '-', total_length_325037, delta_325038)
    
    # Processing the call keyword arguments (line 25)
    # Getting the type of 'num_peaks' (line 25)
    num_peaks_325040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 63), 'num_peaks', False)
    keyword_325041 = num_peaks_325040
    kwargs_325042 = {'num': keyword_325041}
    # Getting the type of 'np' (line 25)
    np_325034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'np', False)
    # Obtaining the member 'linspace' of a type (line 25)
    linspace_325035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 18), np_325034, 'linspace')
    # Calling linspace(args, kwargs) (line 25)
    linspace_call_result_325043 = invoke(stypy.reporting.localization.Localization(__file__, 25, 18), linspace_325035, *[delta_325036, result_sub_325039], **kwargs_325042)
    
    # Obtaining the member 'astype' of a type (line 25)
    astype_325044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 18), linspace_call_result_325043, 'astype')
    # Calling astype(args, kwargs) (line 25)
    astype_call_result_325047 = invoke(stypy.reporting.localization.Localization(__file__, 25, 18), astype_325044, *[int_325045], **kwargs_325046)
    
    # Assigning a type to the variable 'center_locs' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'center_locs', astype_call_result_325047)
    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to _gen_gaussians(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'center_locs' (line 26)
    center_locs_325049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 30), 'center_locs', False)
    # Getting the type of 'sigmas' (line 26)
    sigmas_325050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 43), 'sigmas', False)
    # Getting the type of 'total_length' (line 26)
    total_length_325051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 51), 'total_length', False)
    # Processing the call keyword arguments (line 26)
    kwargs_325052 = {}
    # Getting the type of '_gen_gaussians' (line 26)
    _gen_gaussians_325048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), '_gen_gaussians', False)
    # Calling _gen_gaussians(args, kwargs) (line 26)
    _gen_gaussians_call_result_325053 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), _gen_gaussians_325048, *[center_locs_325049, sigmas_325050, total_length_325051], **kwargs_325052)
    
    # Assigning a type to the variable 'out_data' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'out_data', _gen_gaussians_call_result_325053)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_325054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    # Getting the type of 'out_data' (line 27)
    out_data_325055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'out_data')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 11), tuple_325054, out_data_325055)
    # Adding element type (line 27)
    # Getting the type of 'center_locs' (line 27)
    center_locs_325056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'center_locs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 11), tuple_325054, center_locs_325056)
    
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', tuple_325054)
    
    # ################# End of '_gen_gaussians_even(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_gen_gaussians_even' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_325057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_325057)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_gen_gaussians_even'
    return stypy_return_type_325057

# Assigning a type to the variable '_gen_gaussians_even' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_gen_gaussians_even', _gen_gaussians_even)

@norecursion
def _gen_ridge_line(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_gen_ridge_line'
    module_type_store = module_type_store.open_function_context('_gen_ridge_line', 30, 0, False)
    
    # Passed parameters checking function
    _gen_ridge_line.stypy_localization = localization
    _gen_ridge_line.stypy_type_of_self = None
    _gen_ridge_line.stypy_type_store = module_type_store
    _gen_ridge_line.stypy_function_name = '_gen_ridge_line'
    _gen_ridge_line.stypy_param_names_list = ['start_locs', 'max_locs', 'length', 'distances', 'gaps']
    _gen_ridge_line.stypy_varargs_param_name = None
    _gen_ridge_line.stypy_kwargs_param_name = None
    _gen_ridge_line.stypy_call_defaults = defaults
    _gen_ridge_line.stypy_call_varargs = varargs
    _gen_ridge_line.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_gen_ridge_line', ['start_locs', 'max_locs', 'length', 'distances', 'gaps'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_gen_ridge_line', localization, ['start_locs', 'max_locs', 'length', 'distances', 'gaps'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_gen_ridge_line(...)' code ##################

    str_325058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', "\n    Generate coordinates for a ridge line.\n\n    Will be a series of coordinates, starting a start_loc (length 2).\n    The maximum distance between any adjacent columns will be\n    `max_distance`, the max distance between adjacent rows\n    will be `map_gap'.\n\n    `max_locs` should be the size of the intended matrix. The\n    ending coordinates are guaranteed to be less than `max_locs`,\n    although they may not approach `max_locs` at all.\n    ")

    @norecursion
    def keep_bounds(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'keep_bounds'
        module_type_store = module_type_store.open_function_context('keep_bounds', 44, 4, False)
        
        # Passed parameters checking function
        keep_bounds.stypy_localization = localization
        keep_bounds.stypy_type_of_self = None
        keep_bounds.stypy_type_store = module_type_store
        keep_bounds.stypy_function_name = 'keep_bounds'
        keep_bounds.stypy_param_names_list = ['num', 'max_val']
        keep_bounds.stypy_varargs_param_name = None
        keep_bounds.stypy_kwargs_param_name = None
        keep_bounds.stypy_call_defaults = defaults
        keep_bounds.stypy_call_varargs = varargs
        keep_bounds.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'keep_bounds', ['num', 'max_val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'keep_bounds', localization, ['num', 'max_val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'keep_bounds(...)' code ##################

        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to max(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'num' (line 45)
        num_325060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'num', False)
        int_325061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_325062 = {}
        # Getting the type of 'max' (line 45)
        max_325059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'max', False)
        # Calling max(args, kwargs) (line 45)
        max_call_result_325063 = invoke(stypy.reporting.localization.Localization(__file__, 45, 14), max_325059, *[num_325060, int_325061], **kwargs_325062)
        
        # Assigning a type to the variable 'out' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'out', max_call_result_325063)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to min(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'out' (line 46)
        out_325065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'out', False)
        # Getting the type of 'max_val' (line 46)
        max_val_325066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'max_val', False)
        # Processing the call keyword arguments (line 46)
        kwargs_325067 = {}
        # Getting the type of 'min' (line 46)
        min_325064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'min', False)
        # Calling min(args, kwargs) (line 46)
        min_call_result_325068 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), min_325064, *[out_325065, max_val_325066], **kwargs_325067)
        
        # Assigning a type to the variable 'out' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'out', min_call_result_325068)
        # Getting the type of 'out' (line 47)
        out_325069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', out_325069)
        
        # ################# End of 'keep_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'keep_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_325070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325070)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'keep_bounds'
        return stypy_return_type_325070

    # Assigning a type to the variable 'keep_bounds' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'keep_bounds', keep_bounds)
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to deepcopy(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'gaps' (line 49)
    gaps_325073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'gaps', False)
    # Processing the call keyword arguments (line 49)
    kwargs_325074 = {}
    # Getting the type of 'copy' (line 49)
    copy_325071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 49)
    deepcopy_325072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), copy_325071, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 49)
    deepcopy_call_result_325075 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), deepcopy_325072, *[gaps_325073], **kwargs_325074)
    
    # Assigning a type to the variable 'gaps' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'gaps', deepcopy_call_result_325075)
    
    # Assigning a Call to a Name (line 50):
    
    # Assigning a Call to a Name (line 50):
    
    # Call to deepcopy(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'distances' (line 50)
    distances_325078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'distances', False)
    # Processing the call keyword arguments (line 50)
    kwargs_325079 = {}
    # Getting the type of 'copy' (line 50)
    copy_325076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 50)
    deepcopy_325077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 16), copy_325076, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 50)
    deepcopy_call_result_325080 = invoke(stypy.reporting.localization.Localization(__file__, 50, 16), deepcopy_325077, *[distances_325078], **kwargs_325079)
    
    # Assigning a type to the variable 'distances' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'distances', deepcopy_call_result_325080)
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to zeros(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_325083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    # Getting the type of 'length' (line 52)
    length_325084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'length', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 20), list_325083, length_325084)
    # Adding element type (line 52)
    int_325085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 20), list_325083, int_325085)
    
    # Processing the call keyword arguments (line 52)
    # Getting the type of 'int' (line 52)
    int_325086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'int', False)
    keyword_325087 = int_325086
    kwargs_325088 = {'dtype': keyword_325087}
    # Getting the type of 'np' (line 52)
    np_325081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'np', False)
    # Obtaining the member 'zeros' of a type (line 52)
    zeros_325082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), np_325081, 'zeros')
    # Calling zeros(args, kwargs) (line 52)
    zeros_call_result_325089 = invoke(stypy.reporting.localization.Localization(__file__, 52, 11), zeros_325082, *[list_325083], **kwargs_325088)
    
    # Assigning a type to the variable 'locs' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'locs', zeros_call_result_325089)
    
    # Assigning a Name to a Subscript (line 53):
    
    # Assigning a Name to a Subscript (line 53):
    # Getting the type of 'start_locs' (line 53)
    start_locs_325090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'start_locs')
    # Getting the type of 'locs' (line 53)
    locs_325091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'locs')
    int_325092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 9), 'int')
    slice_325093 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 4), None, None, None)
    # Storing an element on a container (line 53)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 4), locs_325091, ((int_325092, slice_325093), start_locs_325090))
    
    # Assigning a BinOp to a Name (line 54):
    
    # Assigning a BinOp to a Name (line 54):
    
    # Obtaining the type of the subscript
    int_325094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 28), 'int')
    # Getting the type of 'max_locs' (line 54)
    max_locs_325095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'max_locs')
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___325096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 19), max_locs_325095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_325097 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), getitem___325096, int_325094)
    
    
    # Obtaining the type of the subscript
    int_325098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'int')
    # Getting the type of 'start_locs' (line 54)
    start_locs_325099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'start_locs')
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___325100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 33), start_locs_325099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_325101 = invoke(stypy.reporting.localization.Localization(__file__, 54, 33), getitem___325100, int_325098)
    
    # Applying the binary operator '-' (line 54)
    result_sub_325102 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 19), '-', subscript_call_result_325097, subscript_call_result_325101)
    
    
    # Call to sum(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'gaps' (line 54)
    gaps_325104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'gaps', False)
    # Processing the call keyword arguments (line 54)
    kwargs_325105 = {}
    # Getting the type of 'sum' (line 54)
    sum_325103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 49), 'sum', False)
    # Calling sum(args, kwargs) (line 54)
    sum_call_result_325106 = invoke(stypy.reporting.localization.Localization(__file__, 54, 49), sum_325103, *[gaps_325104], **kwargs_325105)
    
    # Applying the binary operator '-' (line 54)
    result_sub_325107 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 47), '-', result_sub_325102, sum_call_result_325106)
    
    # Assigning a type to the variable 'total_length' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'total_length', result_sub_325107)
    
    
    # Getting the type of 'total_length' (line 55)
    total_length_325108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 7), 'total_length')
    # Getting the type of 'length' (line 55)
    length_325109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'length')
    # Applying the binary operator '<' (line 55)
    result_lt_325110 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 7), '<', total_length_325108, length_325109)
    
    # Testing the type of an if condition (line 55)
    if_condition_325111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 4), result_lt_325110)
    # Assigning a type to the variable 'if_condition_325111' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'if_condition_325111', if_condition_325111)
    # SSA begins for if statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 56)
    # Processing the call arguments (line 56)
    str_325113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'str', 'Cannot generate ridge line according to constraints')
    # Processing the call keyword arguments (line 56)
    kwargs_325114 = {}
    # Getting the type of 'ValueError' (line 56)
    ValueError_325112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 56)
    ValueError_call_result_325115 = invoke(stypy.reporting.localization.Localization(__file__, 56, 14), ValueError_325112, *[str_325113], **kwargs_325114)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 56, 8), ValueError_call_result_325115, 'raise parameter', BaseException)
    # SSA join for if statement (line 55)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 57):
    
    # Assigning a BinOp to a Name (line 57):
    # Getting the type of 'length' (line 57)
    length_325116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'length')
    
    # Call to len(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'distances' (line 57)
    distances_325118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'distances', False)
    # Processing the call keyword arguments (line 57)
    kwargs_325119 = {}
    # Getting the type of 'len' (line 57)
    len_325117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'len', False)
    # Calling len(args, kwargs) (line 57)
    len_call_result_325120 = invoke(stypy.reporting.localization.Localization(__file__, 57, 24), len_325117, *[distances_325118], **kwargs_325119)
    
    # Applying the binary operator 'div' (line 57)
    result_div_325121 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), 'div', length_325116, len_call_result_325120)
    
    int_325122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 41), 'int')
    # Applying the binary operator '-' (line 57)
    result_sub_325123 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), '-', result_div_325121, int_325122)
    
    # Assigning a type to the variable 'dist_int' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'dist_int', result_sub_325123)
    
    # Assigning a BinOp to a Name (line 58):
    
    # Assigning a BinOp to a Name (line 58):
    # Getting the type of 'length' (line 58)
    length_325124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'length')
    
    # Call to len(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'gaps' (line 58)
    gaps_325126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'gaps', False)
    # Processing the call keyword arguments (line 58)
    kwargs_325127 = {}
    # Getting the type of 'len' (line 58)
    len_325125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'len', False)
    # Calling len(args, kwargs) (line 58)
    len_call_result_325128 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), len_325125, *[gaps_325126], **kwargs_325127)
    
    # Applying the binary operator 'div' (line 58)
    result_div_325129 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 14), 'div', length_325124, len_call_result_325128)
    
    int_325130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 35), 'int')
    # Applying the binary operator '-' (line 58)
    result_sub_325131 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 14), '-', result_div_325129, int_325130)
    
    # Assigning a type to the variable 'gap_int' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'gap_int', result_sub_325131)
    
    
    # Call to xrange(...): (line 59)
    # Processing the call arguments (line 59)
    int_325133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'int')
    # Getting the type of 'length' (line 59)
    length_325134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'length', False)
    # Processing the call keyword arguments (line 59)
    kwargs_325135 = {}
    # Getting the type of 'xrange' (line 59)
    xrange_325132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'xrange', False)
    # Calling xrange(args, kwargs) (line 59)
    xrange_call_result_325136 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), xrange_325132, *[int_325133, length_325134], **kwargs_325135)
    
    # Testing the type of a for loop iterable (line 59)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 4), xrange_call_result_325136)
    # Getting the type of the for loop variable (line 59)
    for_loop_var_325137 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 4), xrange_call_result_325136)
    # Assigning a type to the variable 'ind' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'ind', for_loop_var_325137)
    # SSA begins for a for statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 60):
    
    # Assigning a Subscript to a Name (line 60):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_325138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    # Getting the type of 'ind' (line 60)
    ind_325139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'ind')
    int_325140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'int')
    # Applying the binary operator '-' (line 60)
    result_sub_325141 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), '-', ind_325139, int_325140)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_325138, result_sub_325141)
    # Adding element type (line 60)
    int_325142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), tuple_325138, int_325142)
    
    # Getting the type of 'locs' (line 60)
    locs_325143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'locs')
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___325144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 18), locs_325143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_325145 = invoke(stypy.reporting.localization.Localization(__file__, 60, 18), getitem___325144, tuple_325138)
    
    # Assigning a type to the variable 'nextcol' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'nextcol', subscript_call_result_325145)
    
    # Assigning a BinOp to a Name (line 61):
    
    # Assigning a BinOp to a Name (line 61):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_325146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    # Getting the type of 'ind' (line 61)
    ind_325147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'ind')
    int_325148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 29), 'int')
    # Applying the binary operator '-' (line 61)
    result_sub_325149 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 23), '-', ind_325147, int_325148)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), tuple_325146, result_sub_325149)
    # Adding element type (line 61)
    int_325150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), tuple_325146, int_325150)
    
    # Getting the type of 'locs' (line 61)
    locs_325151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'locs')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___325152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), locs_325151, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_325153 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), getitem___325152, tuple_325146)
    
    int_325154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 37), 'int')
    # Applying the binary operator '+' (line 61)
    result_add_325155 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 18), '+', subscript_call_result_325153, int_325154)
    
    # Assigning a type to the variable 'nextrow' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'nextrow', result_add_325155)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ind' (line 62)
    ind_325156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'ind')
    # Getting the type of 'dist_int' (line 62)
    dist_int_325157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'dist_int')
    # Applying the binary operator '%' (line 62)
    result_mod_325158 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), '%', ind_325156, dist_int_325157)
    
    int_325159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'int')
    # Applying the binary operator '==' (line 62)
    result_eq_325160 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), '==', result_mod_325158, int_325159)
    
    
    
    # Call to len(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'distances' (line 62)
    distances_325162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 42), 'distances', False)
    # Processing the call keyword arguments (line 62)
    kwargs_325163 = {}
    # Getting the type of 'len' (line 62)
    len_325161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'len', False)
    # Calling len(args, kwargs) (line 62)
    len_call_result_325164 = invoke(stypy.reporting.localization.Localization(__file__, 62, 38), len_325161, *[distances_325162], **kwargs_325163)
    
    int_325165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 55), 'int')
    # Applying the binary operator '>' (line 62)
    result_gt_325166 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 38), '>', len_call_result_325164, int_325165)
    
    # Applying the binary operator 'and' (line 62)
    result_and_keyword_325167 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), 'and', result_eq_325160, result_gt_325166)
    
    # Testing the type of an if condition (line 62)
    if_condition_325168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_and_keyword_325167)
    # Assigning a type to the variable 'if_condition_325168' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_325168', if_condition_325168)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'nextcol' (line 63)
    nextcol_325169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'nextcol')
    int_325170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'int')
    # Getting the type of 'ind' (line 63)
    ind_325171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'ind')
    # Applying the binary operator '**' (line 63)
    result_pow_325172 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 24), '**', int_325170, ind_325171)
    
    
    # Call to pop(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_325175 = {}
    # Getting the type of 'distances' (line 63)
    distances_325173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 35), 'distances', False)
    # Obtaining the member 'pop' of a type (line 63)
    pop_325174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 35), distances_325173, 'pop')
    # Calling pop(args, kwargs) (line 63)
    pop_call_result_325176 = invoke(stypy.reporting.localization.Localization(__file__, 63, 35), pop_325174, *[], **kwargs_325175)
    
    # Applying the binary operator '*' (line 63)
    result_mul_325177 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 23), '*', result_pow_325172, pop_call_result_325176)
    
    # Applying the binary operator '+=' (line 63)
    result_iadd_325178 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 12), '+=', nextcol_325169, result_mul_325177)
    # Assigning a type to the variable 'nextcol' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'nextcol', result_iadd_325178)
    
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ind' (line 64)
    ind_325179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'ind')
    # Getting the type of 'gap_int' (line 64)
    gap_int_325180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'gap_int')
    # Applying the binary operator '%' (line 64)
    result_mod_325181 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '%', ind_325179, gap_int_325180)
    
    int_325182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'int')
    # Applying the binary operator '==' (line 64)
    result_eq_325183 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '==', result_mod_325181, int_325182)
    
    
    
    # Call to len(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'gaps' (line 64)
    gaps_325185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 41), 'gaps', False)
    # Processing the call keyword arguments (line 64)
    kwargs_325186 = {}
    # Getting the type of 'len' (line 64)
    len_325184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'len', False)
    # Calling len(args, kwargs) (line 64)
    len_call_result_325187 = invoke(stypy.reporting.localization.Localization(__file__, 64, 37), len_325184, *[gaps_325185], **kwargs_325186)
    
    int_325188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 49), 'int')
    # Applying the binary operator '>' (line 64)
    result_gt_325189 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 37), '>', len_call_result_325187, int_325188)
    
    # Applying the binary operator 'and' (line 64)
    result_and_keyword_325190 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), 'and', result_eq_325183, result_gt_325189)
    
    # Testing the type of an if condition (line 64)
    if_condition_325191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_and_keyword_325190)
    # Assigning a type to the variable 'if_condition_325191' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_325191', if_condition_325191)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'nextrow' (line 65)
    nextrow_325192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'nextrow')
    
    # Call to pop(...): (line 65)
    # Processing the call keyword arguments (line 65)
    kwargs_325195 = {}
    # Getting the type of 'gaps' (line 65)
    gaps_325193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'gaps', False)
    # Obtaining the member 'pop' of a type (line 65)
    pop_325194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 23), gaps_325193, 'pop')
    # Calling pop(args, kwargs) (line 65)
    pop_call_result_325196 = invoke(stypy.reporting.localization.Localization(__file__, 65, 23), pop_325194, *[], **kwargs_325195)
    
    # Applying the binary operator '+=' (line 65)
    result_iadd_325197 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 12), '+=', nextrow_325192, pop_call_result_325196)
    # Assigning a type to the variable 'nextrow' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'nextrow', result_iadd_325197)
    
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to keep_bounds(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'nextrow' (line 66)
    nextrow_325199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'nextrow', False)
    
    # Obtaining the type of the subscript
    int_325200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 48), 'int')
    # Getting the type of 'max_locs' (line 66)
    max_locs_325201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 39), 'max_locs', False)
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___325202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 39), max_locs_325201, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_325203 = invoke(stypy.reporting.localization.Localization(__file__, 66, 39), getitem___325202, int_325200)
    
    # Processing the call keyword arguments (line 66)
    kwargs_325204 = {}
    # Getting the type of 'keep_bounds' (line 66)
    keep_bounds_325198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'keep_bounds', False)
    # Calling keep_bounds(args, kwargs) (line 66)
    keep_bounds_call_result_325205 = invoke(stypy.reporting.localization.Localization(__file__, 66, 18), keep_bounds_325198, *[nextrow_325199, subscript_call_result_325203], **kwargs_325204)
    
    # Assigning a type to the variable 'nextrow' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'nextrow', keep_bounds_call_result_325205)
    
    # Assigning a Call to a Name (line 67):
    
    # Assigning a Call to a Name (line 67):
    
    # Call to keep_bounds(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'nextcol' (line 67)
    nextcol_325207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'nextcol', False)
    
    # Obtaining the type of the subscript
    int_325208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 48), 'int')
    # Getting the type of 'max_locs' (line 67)
    max_locs_325209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'max_locs', False)
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___325210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 39), max_locs_325209, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_325211 = invoke(stypy.reporting.localization.Localization(__file__, 67, 39), getitem___325210, int_325208)
    
    # Processing the call keyword arguments (line 67)
    kwargs_325212 = {}
    # Getting the type of 'keep_bounds' (line 67)
    keep_bounds_325206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'keep_bounds', False)
    # Calling keep_bounds(args, kwargs) (line 67)
    keep_bounds_call_result_325213 = invoke(stypy.reporting.localization.Localization(__file__, 67, 18), keep_bounds_325206, *[nextcol_325207, subscript_call_result_325211], **kwargs_325212)
    
    # Assigning a type to the variable 'nextcol' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'nextcol', keep_bounds_call_result_325213)
    
    # Assigning a List to a Subscript (line 68):
    
    # Assigning a List to a Subscript (line 68):
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_325214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'nextrow' (line 68)
    nextrow_325215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'nextrow')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 23), list_325214, nextrow_325215)
    # Adding element type (line 68)
    # Getting the type of 'nextcol' (line 68)
    nextcol_325216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'nextcol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 23), list_325214, nextcol_325216)
    
    # Getting the type of 'locs' (line 68)
    locs_325217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'locs')
    # Getting the type of 'ind' (line 68)
    ind_325218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'ind')
    slice_325219 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 8), None, None, None)
    # Storing an element on a container (line 68)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 8), locs_325217, ((ind_325218, slice_325219), list_325214))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'list' (line 70)
    list_325220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 70)
    # Adding element type (line 70)
    
    # Obtaining the type of the subscript
    slice_325221 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 70, 12), None, None, None)
    int_325222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'int')
    # Getting the type of 'locs' (line 70)
    locs_325223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'locs')
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___325224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), locs_325223, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_325225 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), getitem___325224, (slice_325221, int_325222))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), list_325220, subscript_call_result_325225)
    # Adding element type (line 70)
    
    # Obtaining the type of the subscript
    slice_325226 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 70, 24), None, None, None)
    int_325227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 32), 'int')
    # Getting the type of 'locs' (line 70)
    locs_325228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'locs')
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___325229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 24), locs_325228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_325230 = invoke(stypy.reporting.localization.Localization(__file__, 70, 24), getitem___325229, (slice_325226, int_325227))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), list_325220, subscript_call_result_325230)
    
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', list_325220)
    
    # ################# End of '_gen_ridge_line(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_gen_ridge_line' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_325231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_325231)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_gen_ridge_line'
    return stypy_return_type_325231

# Assigning a type to the variable '_gen_ridge_line' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_gen_ridge_line', _gen_ridge_line)
# Declaration of the 'TestRidgeLines' class

class TestRidgeLines(object, ):

    @norecursion
    def test_empty(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty'
        module_type_store = module_type_store.open_function_context('test_empty', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_localization', localization)
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_function_name', 'TestRidgeLines.test_empty')
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_param_names_list', [])
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRidgeLines.test_empty.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRidgeLines.test_empty', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_empty', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_empty(...)' code ##################

        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to zeros(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_325234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_325235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_325234, int_325235)
        # Adding element type (line 76)
        int_325236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 29), list_325234, int_325236)
        
        # Processing the call keyword arguments (line 76)
        kwargs_325237 = {}
        # Getting the type of 'np' (line 76)
        np_325232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'np', False)
        # Obtaining the member 'zeros' of a type (line 76)
        zeros_325233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 20), np_325232, 'zeros')
        # Calling zeros(args, kwargs) (line 76)
        zeros_call_result_325238 = invoke(stypy.reporting.localization.Localization(__file__, 76, 20), zeros_325233, *[list_325234], **kwargs_325237)
        
        # Assigning a type to the variable 'test_matr' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'test_matr', zeros_call_result_325238)
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to _identify_ridge_lines(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'test_matr' (line 77)
        test_matr_325240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'test_matr', False)
        int_325241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 49), 'int')
        
        # Call to ones(...): (line 77)
        # Processing the call arguments (line 77)
        int_325244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 59), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_325245 = {}
        # Getting the type of 'np' (line 77)
        np_325242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 51), 'np', False)
        # Obtaining the member 'ones' of a type (line 77)
        ones_325243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 51), np_325242, 'ones')
        # Calling ones(args, kwargs) (line 77)
        ones_call_result_325246 = invoke(stypy.reporting.localization.Localization(__file__, 77, 51), ones_325243, *[int_325244], **kwargs_325245)
        
        # Applying the binary operator '*' (line 77)
        result_mul_325247 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 49), '*', int_325241, ones_call_result_325246)
        
        int_325248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 64), 'int')
        # Processing the call keyword arguments (line 77)
        kwargs_325249 = {}
        # Getting the type of '_identify_ridge_lines' (line 77)
        _identify_ridge_lines_325239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), '_identify_ridge_lines', False)
        # Calling _identify_ridge_lines(args, kwargs) (line 77)
        _identify_ridge_lines_call_result_325250 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), _identify_ridge_lines_325239, *[test_matr_325240, result_mul_325247, int_325248], **kwargs_325249)
        
        # Assigning a type to the variable 'lines' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'lines', _identify_ridge_lines_call_result_325250)
        
        # Call to assert_(...): (line 78)
        # Processing the call arguments (line 78)
        
        
        # Call to len(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'lines' (line 78)
        lines_325253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'lines', False)
        # Processing the call keyword arguments (line 78)
        kwargs_325254 = {}
        # Getting the type of 'len' (line 78)
        len_325252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'len', False)
        # Calling len(args, kwargs) (line 78)
        len_call_result_325255 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), len_325252, *[lines_325253], **kwargs_325254)
        
        int_325256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'int')
        # Applying the binary operator '==' (line 78)
        result_eq_325257 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '==', len_call_result_325255, int_325256)
        
        # Processing the call keyword arguments (line 78)
        kwargs_325258 = {}
        # Getting the type of 'assert_' (line 78)
        assert__325251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 78)
        assert__call_result_325259 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assert__325251, *[result_eq_325257], **kwargs_325258)
        
        
        # ################# End of 'test_empty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_325260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325260)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty'
        return stypy_return_type_325260


    @norecursion
    def test_minimal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimal'
        module_type_store = module_type_store.open_function_context('test_minimal', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_localization', localization)
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_function_name', 'TestRidgeLines.test_minimal')
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_param_names_list', [])
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRidgeLines.test_minimal.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRidgeLines.test_minimal', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimal', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimal(...)' code ##################

        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to zeros(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_325263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        int_325264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 29), list_325263, int_325264)
        # Adding element type (line 81)
        int_325265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 29), list_325263, int_325265)
        
        # Processing the call keyword arguments (line 81)
        kwargs_325266 = {}
        # Getting the type of 'np' (line 81)
        np_325261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'np', False)
        # Obtaining the member 'zeros' of a type (line 81)
        zeros_325262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 20), np_325261, 'zeros')
        # Calling zeros(args, kwargs) (line 81)
        zeros_call_result_325267 = invoke(stypy.reporting.localization.Localization(__file__, 81, 20), zeros_325262, *[list_325263], **kwargs_325266)
        
        # Assigning a type to the variable 'test_matr' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'test_matr', zeros_call_result_325267)
        
        # Assigning a Num to a Subscript (line 82):
        
        # Assigning a Num to a Subscript (line 82):
        int_325268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'int')
        # Getting the type of 'test_matr' (line 82)
        test_matr_325269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'test_matr')
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_325270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        int_325271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), tuple_325270, int_325271)
        # Adding element type (line 82)
        int_325272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), tuple_325270, int_325272)
        
        # Storing an element on a container (line 82)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), test_matr_325269, (tuple_325270, int_325268))
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to _identify_ridge_lines(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'test_matr' (line 83)
        test_matr_325274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 38), 'test_matr', False)
        int_325275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 49), 'int')
        
        # Call to ones(...): (line 83)
        # Processing the call arguments (line 83)
        int_325278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 59), 'int')
        # Processing the call keyword arguments (line 83)
        kwargs_325279 = {}
        # Getting the type of 'np' (line 83)
        np_325276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 51), 'np', False)
        # Obtaining the member 'ones' of a type (line 83)
        ones_325277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 51), np_325276, 'ones')
        # Calling ones(args, kwargs) (line 83)
        ones_call_result_325280 = invoke(stypy.reporting.localization.Localization(__file__, 83, 51), ones_325277, *[int_325278], **kwargs_325279)
        
        # Applying the binary operator '*' (line 83)
        result_mul_325281 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 49), '*', int_325275, ones_call_result_325280)
        
        int_325282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 64), 'int')
        # Processing the call keyword arguments (line 83)
        kwargs_325283 = {}
        # Getting the type of '_identify_ridge_lines' (line 83)
        _identify_ridge_lines_325273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), '_identify_ridge_lines', False)
        # Calling _identify_ridge_lines(args, kwargs) (line 83)
        _identify_ridge_lines_call_result_325284 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), _identify_ridge_lines_325273, *[test_matr_325274, result_mul_325281, int_325282], **kwargs_325283)
        
        # Assigning a type to the variable 'lines' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'lines', _identify_ridge_lines_call_result_325284)
        
        # Call to assert_(...): (line 84)
        # Processing the call arguments (line 84)
        
        
        # Call to len(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'lines' (line 84)
        lines_325287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'lines', False)
        # Processing the call keyword arguments (line 84)
        kwargs_325288 = {}
        # Getting the type of 'len' (line 84)
        len_325286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'len', False)
        # Calling len(args, kwargs) (line 84)
        len_call_result_325289 = invoke(stypy.reporting.localization.Localization(__file__, 84, 16), len_325286, *[lines_325287], **kwargs_325288)
        
        int_325290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'int')
        # Applying the binary operator '==' (line 84)
        result_eq_325291 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 16), '==', len_call_result_325289, int_325290)
        
        # Processing the call keyword arguments (line 84)
        kwargs_325292 = {}
        # Getting the type of 'assert_' (line 84)
        assert__325285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 84)
        assert__call_result_325293 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert__325285, *[result_eq_325291], **kwargs_325292)
        
        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to zeros(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_325296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        int_325297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 29), list_325296, int_325297)
        # Adding element type (line 86)
        int_325298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 29), list_325296, int_325298)
        
        # Processing the call keyword arguments (line 86)
        kwargs_325299 = {}
        # Getting the type of 'np' (line 86)
        np_325294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'np', False)
        # Obtaining the member 'zeros' of a type (line 86)
        zeros_325295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 20), np_325294, 'zeros')
        # Calling zeros(args, kwargs) (line 86)
        zeros_call_result_325300 = invoke(stypy.reporting.localization.Localization(__file__, 86, 20), zeros_325295, *[list_325296], **kwargs_325299)
        
        # Assigning a type to the variable 'test_matr' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'test_matr', zeros_call_result_325300)
        
        # Assigning a Num to a Subscript (line 87):
        
        # Assigning a Num to a Subscript (line 87):
        int_325301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'int')
        # Getting the type of 'test_matr' (line 87)
        test_matr_325302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'test_matr')
        int_325303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'int')
        int_325304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'int')
        slice_325305 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 8), int_325303, int_325304, None)
        int_325306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'int')
        # Storing an element on a container (line 87)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), test_matr_325302, ((slice_325305, int_325306), int_325301))
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to _identify_ridge_lines(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'test_matr' (line 88)
        test_matr_325308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 38), 'test_matr', False)
        int_325309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 49), 'int')
        
        # Call to ones(...): (line 88)
        # Processing the call arguments (line 88)
        int_325312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 59), 'int')
        # Processing the call keyword arguments (line 88)
        kwargs_325313 = {}
        # Getting the type of 'np' (line 88)
        np_325310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 51), 'np', False)
        # Obtaining the member 'ones' of a type (line 88)
        ones_325311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 51), np_325310, 'ones')
        # Calling ones(args, kwargs) (line 88)
        ones_call_result_325314 = invoke(stypy.reporting.localization.Localization(__file__, 88, 51), ones_325311, *[int_325312], **kwargs_325313)
        
        # Applying the binary operator '*' (line 88)
        result_mul_325315 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 49), '*', int_325309, ones_call_result_325314)
        
        int_325316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 64), 'int')
        # Processing the call keyword arguments (line 88)
        kwargs_325317 = {}
        # Getting the type of '_identify_ridge_lines' (line 88)
        _identify_ridge_lines_325307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), '_identify_ridge_lines', False)
        # Calling _identify_ridge_lines(args, kwargs) (line 88)
        _identify_ridge_lines_call_result_325318 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), _identify_ridge_lines_325307, *[test_matr_325308, result_mul_325315, int_325316], **kwargs_325317)
        
        # Assigning a type to the variable 'lines' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'lines', _identify_ridge_lines_call_result_325318)
        
        # Call to assert_(...): (line 89)
        # Processing the call arguments (line 89)
        
        
        # Call to len(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'lines' (line 89)
        lines_325321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'lines', False)
        # Processing the call keyword arguments (line 89)
        kwargs_325322 = {}
        # Getting the type of 'len' (line 89)
        len_325320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'len', False)
        # Calling len(args, kwargs) (line 89)
        len_call_result_325323 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), len_325320, *[lines_325321], **kwargs_325322)
        
        int_325324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'int')
        # Applying the binary operator '==' (line 89)
        result_eq_325325 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 16), '==', len_call_result_325323, int_325324)
        
        # Processing the call keyword arguments (line 89)
        kwargs_325326 = {}
        # Getting the type of 'assert_' (line 89)
        assert__325319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 89)
        assert__call_result_325327 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert__325319, *[result_eq_325325], **kwargs_325326)
        
        
        # ################# End of 'test_minimal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimal' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_325328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimal'
        return stypy_return_type_325328


    @norecursion
    def test_single_pass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_single_pass'
        module_type_store = module_type_store.open_function_context('test_single_pass', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_localization', localization)
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_function_name', 'TestRidgeLines.test_single_pass')
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_param_names_list', [])
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRidgeLines.test_single_pass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRidgeLines.test_single_pass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_single_pass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_single_pass(...)' code ##################

        
        # Assigning a List to a Name (line 92):
        
        # Assigning a List to a Name (line 92):
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_325329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        int_325330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 20), list_325329, int_325330)
        # Adding element type (line 92)
        int_325331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 20), list_325329, int_325331)
        # Adding element type (line 92)
        int_325332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 20), list_325329, int_325332)
        # Adding element type (line 92)
        int_325333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 20), list_325329, int_325333)
        
        # Assigning a type to the variable 'distances' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'distances', list_325329)
        
        # Assigning a List to a Name (line 93):
        
        # Assigning a List to a Name (line 93):
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_325334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        int_325335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 15), list_325334, int_325335)
        # Adding element type (line 93)
        int_325336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 15), list_325334, int_325336)
        # Adding element type (line 93)
        int_325337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 15), list_325334, int_325337)
        # Adding element type (line 93)
        int_325338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 15), list_325334, int_325338)
        # Adding element type (line 93)
        int_325339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 15), list_325334, int_325339)
        
        # Assigning a type to the variable 'gaps' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'gaps', list_325334)
        
        # Assigning a BinOp to a Name (line 94):
        
        # Assigning a BinOp to a Name (line 94):
        
        # Call to zeros(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Obtaining an instance of the builtin type 'list' (line 94)
        list_325342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 94)
        # Adding element type (line 94)
        int_325343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 29), list_325342, int_325343)
        # Adding element type (line 94)
        int_325344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 29), list_325342, int_325344)
        
        # Processing the call keyword arguments (line 94)
        kwargs_325345 = {}
        # Getting the type of 'np' (line 94)
        np_325340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'np', False)
        # Obtaining the member 'zeros' of a type (line 94)
        zeros_325341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 20), np_325340, 'zeros')
        # Calling zeros(args, kwargs) (line 94)
        zeros_call_result_325346 = invoke(stypy.reporting.localization.Localization(__file__, 94, 20), zeros_325341, *[list_325342], **kwargs_325345)
        
        float_325347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 41), 'float')
        # Applying the binary operator '+' (line 94)
        result_add_325348 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 20), '+', zeros_call_result_325346, float_325347)
        
        # Assigning a type to the variable 'test_matr' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'test_matr', result_add_325348)
        
        # Assigning a Num to a Name (line 95):
        
        # Assigning a Num to a Name (line 95):
        int_325349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 17), 'int')
        # Assigning a type to the variable 'length' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'length', int_325349)
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to _gen_ridge_line(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_325351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_325352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 31), list_325351, int_325352)
        # Adding element type (line 96)
        int_325353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 31), list_325351, int_325353)
        
        # Getting the type of 'test_matr' (line 96)
        test_matr_325354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 40), 'test_matr', False)
        # Obtaining the member 'shape' of a type (line 96)
        shape_325355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 40), test_matr_325354, 'shape')
        # Getting the type of 'length' (line 96)
        length_325356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 57), 'length', False)
        # Getting the type of 'distances' (line 96)
        distances_325357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 65), 'distances', False)
        # Getting the type of 'gaps' (line 96)
        gaps_325358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 76), 'gaps', False)
        # Processing the call keyword arguments (line 96)
        kwargs_325359 = {}
        # Getting the type of '_gen_ridge_line' (line 96)
        _gen_ridge_line_325350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), '_gen_ridge_line', False)
        # Calling _gen_ridge_line(args, kwargs) (line 96)
        _gen_ridge_line_call_result_325360 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), _gen_ridge_line_325350, *[list_325351, shape_325355, length_325356, distances_325357, gaps_325358], **kwargs_325359)
        
        # Assigning a type to the variable 'line' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'line', _gen_ridge_line_call_result_325360)
        
        # Assigning a Num to a Subscript (line 97):
        
        # Assigning a Num to a Subscript (line 97):
        int_325361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 38), 'int')
        # Getting the type of 'test_matr' (line 97)
        test_matr_325362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'test_matr')
        
        # Obtaining an instance of the builtin type 'tuple' (line 97)
        tuple_325363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 97)
        # Adding element type (line 97)
        
        # Obtaining the type of the subscript
        int_325364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'int')
        # Getting the type of 'line' (line 97)
        line_325365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'line')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___325366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 18), line_325365, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_325367 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), getitem___325366, int_325364)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 18), tuple_325363, subscript_call_result_325367)
        # Adding element type (line 97)
        
        # Obtaining the type of the subscript
        int_325368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 32), 'int')
        # Getting the type of 'line' (line 97)
        line_325369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'line')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___325370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 27), line_325369, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_325371 = invoke(stypy.reporting.localization.Localization(__file__, 97, 27), getitem___325370, int_325368)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 18), tuple_325363, subscript_call_result_325371)
        
        # Storing an element on a container (line 97)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 8), test_matr_325362, (tuple_325363, int_325361))
        
        # Assigning a BinOp to a Name (line 98):
        
        # Assigning a BinOp to a Name (line 98):
        
        # Call to max(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'distances' (line 98)
        distances_325373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'distances', False)
        # Processing the call keyword arguments (line 98)
        kwargs_325374 = {}
        # Getting the type of 'max' (line 98)
        max_325372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'max', False)
        # Calling max(args, kwargs) (line 98)
        max_call_result_325375 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), max_325372, *[distances_325373], **kwargs_325374)
        
        
        # Call to ones(...): (line 98)
        # Processing the call arguments (line 98)
        int_325378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 47), 'int')
        # Processing the call keyword arguments (line 98)
        kwargs_325379 = {}
        # Getting the type of 'np' (line 98)
        np_325376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 39), 'np', False)
        # Obtaining the member 'ones' of a type (line 98)
        ones_325377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 39), np_325376, 'ones')
        # Calling ones(args, kwargs) (line 98)
        ones_call_result_325380 = invoke(stypy.reporting.localization.Localization(__file__, 98, 39), ones_325377, *[int_325378], **kwargs_325379)
        
        # Applying the binary operator '*' (line 98)
        result_mul_325381 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 24), '*', max_call_result_325375, ones_call_result_325380)
        
        # Assigning a type to the variable 'max_distances' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'max_distances', result_mul_325381)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to _identify_ridge_lines(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'test_matr' (line 99)
        test_matr_325383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 49), 'test_matr', False)
        # Getting the type of 'max_distances' (line 99)
        max_distances_325384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 60), 'max_distances', False)
        
        # Call to max(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'gaps' (line 99)
        gaps_325386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 79), 'gaps', False)
        # Processing the call keyword arguments (line 99)
        kwargs_325387 = {}
        # Getting the type of 'max' (line 99)
        max_325385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 75), 'max', False)
        # Calling max(args, kwargs) (line 99)
        max_call_result_325388 = invoke(stypy.reporting.localization.Localization(__file__, 99, 75), max_325385, *[gaps_325386], **kwargs_325387)
        
        int_325389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 87), 'int')
        # Applying the binary operator '+' (line 99)
        result_add_325390 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 75), '+', max_call_result_325388, int_325389)
        
        # Processing the call keyword arguments (line 99)
        kwargs_325391 = {}
        # Getting the type of '_identify_ridge_lines' (line 99)
        _identify_ridge_lines_325382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), '_identify_ridge_lines', False)
        # Calling _identify_ridge_lines(args, kwargs) (line 99)
        _identify_ridge_lines_call_result_325392 = invoke(stypy.reporting.localization.Localization(__file__, 99, 27), _identify_ridge_lines_325382, *[test_matr_325383, max_distances_325384, result_add_325390], **kwargs_325391)
        
        # Assigning a type to the variable 'identified_lines' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'identified_lines', _identify_ridge_lines_call_result_325392)
        
        # Call to assert_array_equal(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'identified_lines' (line 100)
        identified_lines_325394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'identified_lines', False)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_325395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        # Getting the type of 'line' (line 100)
        line_325396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 46), 'line', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 45), list_325395, line_325396)
        
        # Processing the call keyword arguments (line 100)
        kwargs_325397 = {}
        # Getting the type of 'assert_array_equal' (line 100)
        assert_array_equal_325393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 100)
        assert_array_equal_call_result_325398 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), assert_array_equal_325393, *[identified_lines_325394, list_325395], **kwargs_325397)
        
        
        # ################# End of 'test_single_pass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_single_pass' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_325399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325399)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_single_pass'
        return stypy_return_type_325399


    @norecursion
    def test_single_bigdist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_single_bigdist'
        module_type_store = module_type_store.open_function_context('test_single_bigdist', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_localization', localization)
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_function_name', 'TestRidgeLines.test_single_bigdist')
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_param_names_list', [])
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRidgeLines.test_single_bigdist.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRidgeLines.test_single_bigdist', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_single_bigdist', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_single_bigdist(...)' code ##################

        
        # Assigning a List to a Name (line 103):
        
        # Assigning a List to a Name (line 103):
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_325400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        int_325401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), list_325400, int_325401)
        # Adding element type (line 103)
        int_325402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), list_325400, int_325402)
        # Adding element type (line 103)
        int_325403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), list_325400, int_325403)
        # Adding element type (line 103)
        int_325404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 20), list_325400, int_325404)
        
        # Assigning a type to the variable 'distances' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'distances', list_325400)
        
        # Assigning a List to a Name (line 104):
        
        # Assigning a List to a Name (line 104):
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_325405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        int_325406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_325405, int_325406)
        # Adding element type (line 104)
        int_325407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_325405, int_325407)
        # Adding element type (line 104)
        int_325408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_325405, int_325408)
        # Adding element type (line 104)
        int_325409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_325405, int_325409)
        
        # Assigning a type to the variable 'gaps' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'gaps', list_325405)
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to zeros(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_325412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        int_325413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 29), list_325412, int_325413)
        # Adding element type (line 105)
        int_325414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 29), list_325412, int_325414)
        
        # Processing the call keyword arguments (line 105)
        kwargs_325415 = {}
        # Getting the type of 'np' (line 105)
        np_325410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'np', False)
        # Obtaining the member 'zeros' of a type (line 105)
        zeros_325411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 20), np_325410, 'zeros')
        # Calling zeros(args, kwargs) (line 105)
        zeros_call_result_325416 = invoke(stypy.reporting.localization.Localization(__file__, 105, 20), zeros_325411, *[list_325412], **kwargs_325415)
        
        # Assigning a type to the variable 'test_matr' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'test_matr', zeros_call_result_325416)
        
        # Assigning a Num to a Name (line 106):
        
        # Assigning a Num to a Name (line 106):
        int_325417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 17), 'int')
        # Assigning a type to the variable 'length' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'length', int_325417)
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to _gen_ridge_line(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_325419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        int_325420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 31), list_325419, int_325420)
        # Adding element type (line 107)
        int_325421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 31), list_325419, int_325421)
        
        # Getting the type of 'test_matr' (line 107)
        test_matr_325422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 40), 'test_matr', False)
        # Obtaining the member 'shape' of a type (line 107)
        shape_325423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 40), test_matr_325422, 'shape')
        # Getting the type of 'length' (line 107)
        length_325424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 57), 'length', False)
        # Getting the type of 'distances' (line 107)
        distances_325425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 65), 'distances', False)
        # Getting the type of 'gaps' (line 107)
        gaps_325426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 76), 'gaps', False)
        # Processing the call keyword arguments (line 107)
        kwargs_325427 = {}
        # Getting the type of '_gen_ridge_line' (line 107)
        _gen_ridge_line_325418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), '_gen_ridge_line', False)
        # Calling _gen_ridge_line(args, kwargs) (line 107)
        _gen_ridge_line_call_result_325428 = invoke(stypy.reporting.localization.Localization(__file__, 107, 15), _gen_ridge_line_325418, *[list_325419, shape_325423, length_325424, distances_325425, gaps_325426], **kwargs_325427)
        
        # Assigning a type to the variable 'line' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'line', _gen_ridge_line_call_result_325428)
        
        # Assigning a Num to a Subscript (line 108):
        
        # Assigning a Num to a Subscript (line 108):
        int_325429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 38), 'int')
        # Getting the type of 'test_matr' (line 108)
        test_matr_325430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'test_matr')
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_325431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        
        # Obtaining the type of the subscript
        int_325432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'int')
        # Getting the type of 'line' (line 108)
        line_325433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'line')
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___325434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 18), line_325433, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_325435 = invoke(stypy.reporting.localization.Localization(__file__, 108, 18), getitem___325434, int_325432)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 18), tuple_325431, subscript_call_result_325435)
        # Adding element type (line 108)
        
        # Obtaining the type of the subscript
        int_325436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 32), 'int')
        # Getting the type of 'line' (line 108)
        line_325437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'line')
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___325438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), line_325437, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_325439 = invoke(stypy.reporting.localization.Localization(__file__, 108, 27), getitem___325438, int_325436)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 18), tuple_325431, subscript_call_result_325439)
        
        # Storing an element on a container (line 108)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 8), test_matr_325430, (tuple_325431, int_325429))
        
        # Assigning a Num to a Name (line 109):
        
        # Assigning a Num to a Name (line 109):
        int_325440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 19), 'int')
        # Assigning a type to the variable 'max_dist' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'max_dist', int_325440)
        
        # Assigning a BinOp to a Name (line 110):
        
        # Assigning a BinOp to a Name (line 110):
        # Getting the type of 'max_dist' (line 110)
        max_dist_325441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 24), 'max_dist')
        
        # Call to ones(...): (line 110)
        # Processing the call arguments (line 110)
        int_325444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 41), 'int')
        # Processing the call keyword arguments (line 110)
        kwargs_325445 = {}
        # Getting the type of 'np' (line 110)
        np_325442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 33), 'np', False)
        # Obtaining the member 'ones' of a type (line 110)
        ones_325443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 33), np_325442, 'ones')
        # Calling ones(args, kwargs) (line 110)
        ones_call_result_325446 = invoke(stypy.reporting.localization.Localization(__file__, 110, 33), ones_325443, *[int_325444], **kwargs_325445)
        
        # Applying the binary operator '*' (line 110)
        result_mul_325447 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 24), '*', max_dist_325441, ones_call_result_325446)
        
        # Assigning a type to the variable 'max_distances' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'max_distances', result_mul_325447)
        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to _identify_ridge_lines(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'test_matr' (line 112)
        test_matr_325449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 49), 'test_matr', False)
        # Getting the type of 'max_distances' (line 112)
        max_distances_325450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 60), 'max_distances', False)
        
        # Call to max(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'gaps' (line 112)
        gaps_325452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 79), 'gaps', False)
        # Processing the call keyword arguments (line 112)
        kwargs_325453 = {}
        # Getting the type of 'max' (line 112)
        max_325451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 75), 'max', False)
        # Calling max(args, kwargs) (line 112)
        max_call_result_325454 = invoke(stypy.reporting.localization.Localization(__file__, 112, 75), max_325451, *[gaps_325452], **kwargs_325453)
        
        int_325455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 87), 'int')
        # Applying the binary operator '+' (line 112)
        result_add_325456 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 75), '+', max_call_result_325454, int_325455)
        
        # Processing the call keyword arguments (line 112)
        kwargs_325457 = {}
        # Getting the type of '_identify_ridge_lines' (line 112)
        _identify_ridge_lines_325448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), '_identify_ridge_lines', False)
        # Calling _identify_ridge_lines(args, kwargs) (line 112)
        _identify_ridge_lines_call_result_325458 = invoke(stypy.reporting.localization.Localization(__file__, 112, 27), _identify_ridge_lines_325448, *[test_matr_325449, max_distances_325450, result_add_325456], **kwargs_325457)
        
        # Assigning a type to the variable 'identified_lines' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'identified_lines', _identify_ridge_lines_call_result_325458)
        
        # Call to assert_(...): (line 113)
        # Processing the call arguments (line 113)
        
        
        # Call to len(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'identified_lines' (line 113)
        identified_lines_325461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'identified_lines', False)
        # Processing the call keyword arguments (line 113)
        kwargs_325462 = {}
        # Getting the type of 'len' (line 113)
        len_325460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'len', False)
        # Calling len(args, kwargs) (line 113)
        len_call_result_325463 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), len_325460, *[identified_lines_325461], **kwargs_325462)
        
        int_325464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 41), 'int')
        # Applying the binary operator '==' (line 113)
        result_eq_325465 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 16), '==', len_call_result_325463, int_325464)
        
        # Processing the call keyword arguments (line 113)
        kwargs_325466 = {}
        # Getting the type of 'assert_' (line 113)
        assert__325459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 113)
        assert__call_result_325467 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert__325459, *[result_eq_325465], **kwargs_325466)
        
        
        # Getting the type of 'identified_lines' (line 115)
        identified_lines_325468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'identified_lines')
        # Testing the type of a for loop iterable (line 115)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), identified_lines_325468)
        # Getting the type of the for loop variable (line 115)
        for_loop_var_325469 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), identified_lines_325468)
        # Assigning a type to the variable 'iline' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'iline', for_loop_var_325469)
        # SSA begins for a for statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to diff(...): (line 116)
        # Processing the call arguments (line 116)
        
        # Obtaining the type of the subscript
        int_325472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 35), 'int')
        # Getting the type of 'iline' (line 116)
        iline_325473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'iline', False)
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___325474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 29), iline_325473, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_325475 = invoke(stypy.reporting.localization.Localization(__file__, 116, 29), getitem___325474, int_325472)
        
        # Processing the call keyword arguments (line 116)
        kwargs_325476 = {}
        # Getting the type of 'np' (line 116)
        np_325470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'np', False)
        # Obtaining the member 'diff' of a type (line 116)
        diff_325471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 21), np_325470, 'diff')
        # Calling diff(args, kwargs) (line 116)
        diff_call_result_325477 = invoke(stypy.reporting.localization.Localization(__file__, 116, 21), diff_325471, *[subscript_call_result_325475], **kwargs_325476)
        
        # Assigning a type to the variable 'adists' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'adists', diff_call_result_325477)
        
        # Call to assert_array_less(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Call to abs(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'adists' (line 117)
        adists_325483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 48), 'adists', False)
        # Processing the call keyword arguments (line 117)
        kwargs_325484 = {}
        # Getting the type of 'np' (line 117)
        np_325481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 41), 'np', False)
        # Obtaining the member 'abs' of a type (line 117)
        abs_325482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 41), np_325481, 'abs')
        # Calling abs(args, kwargs) (line 117)
        abs_call_result_325485 = invoke(stypy.reporting.localization.Localization(__file__, 117, 41), abs_325482, *[adists_325483], **kwargs_325484)
        
        # Getting the type of 'max_dist' (line 117)
        max_dist_325486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 57), 'max_dist', False)
        # Processing the call keyword arguments (line 117)
        kwargs_325487 = {}
        # Getting the type of 'np' (line 117)
        np_325478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'np', False)
        # Obtaining the member 'testing' of a type (line 117)
        testing_325479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), np_325478, 'testing')
        # Obtaining the member 'assert_array_less' of a type (line 117)
        assert_array_less_325480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), testing_325479, 'assert_array_less')
        # Calling assert_array_less(args, kwargs) (line 117)
        assert_array_less_call_result_325488 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), assert_array_less_325480, *[abs_call_result_325485, max_dist_325486], **kwargs_325487)
        
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to diff(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining the type of the subscript
        int_325491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 34), 'int')
        # Getting the type of 'iline' (line 119)
        iline_325492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'iline', False)
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___325493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 28), iline_325492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_325494 = invoke(stypy.reporting.localization.Localization(__file__, 119, 28), getitem___325493, int_325491)
        
        # Processing the call keyword arguments (line 119)
        kwargs_325495 = {}
        # Getting the type of 'np' (line 119)
        np_325489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'np', False)
        # Obtaining the member 'diff' of a type (line 119)
        diff_325490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 20), np_325489, 'diff')
        # Calling diff(args, kwargs) (line 119)
        diff_call_result_325496 = invoke(stypy.reporting.localization.Localization(__file__, 119, 20), diff_325490, *[subscript_call_result_325494], **kwargs_325495)
        
        # Assigning a type to the variable 'agaps' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'agaps', diff_call_result_325496)
        
        # Call to assert_array_less(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to abs(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'agaps' (line 120)
        agaps_325502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'agaps', False)
        # Processing the call keyword arguments (line 120)
        kwargs_325503 = {}
        # Getting the type of 'np' (line 120)
        np_325500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 41), 'np', False)
        # Obtaining the member 'abs' of a type (line 120)
        abs_325501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 41), np_325500, 'abs')
        # Calling abs(args, kwargs) (line 120)
        abs_call_result_325504 = invoke(stypy.reporting.localization.Localization(__file__, 120, 41), abs_325501, *[agaps_325502], **kwargs_325503)
        
        
        # Call to max(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'gaps' (line 120)
        gaps_325506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 60), 'gaps', False)
        # Processing the call keyword arguments (line 120)
        kwargs_325507 = {}
        # Getting the type of 'max' (line 120)
        max_325505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 56), 'max', False)
        # Calling max(args, kwargs) (line 120)
        max_call_result_325508 = invoke(stypy.reporting.localization.Localization(__file__, 120, 56), max_325505, *[gaps_325506], **kwargs_325507)
        
        float_325509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 68), 'float')
        # Applying the binary operator '+' (line 120)
        result_add_325510 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 56), '+', max_call_result_325508, float_325509)
        
        # Processing the call keyword arguments (line 120)
        kwargs_325511 = {}
        # Getting the type of 'np' (line 120)
        np_325497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'np', False)
        # Obtaining the member 'testing' of a type (line 120)
        testing_325498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), np_325497, 'testing')
        # Obtaining the member 'assert_array_less' of a type (line 120)
        assert_array_less_325499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), testing_325498, 'assert_array_less')
        # Calling assert_array_less(args, kwargs) (line 120)
        assert_array_less_call_result_325512 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), assert_array_less_325499, *[abs_call_result_325504, result_add_325510], **kwargs_325511)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_single_bigdist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_single_bigdist' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_325513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_single_bigdist'
        return stypy_return_type_325513


    @norecursion
    def test_single_biggap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_single_biggap'
        module_type_store = module_type_store.open_function_context('test_single_biggap', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_localization', localization)
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_function_name', 'TestRidgeLines.test_single_biggap')
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_param_names_list', [])
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRidgeLines.test_single_biggap.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRidgeLines.test_single_biggap', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_single_biggap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_single_biggap(...)' code ##################

        
        # Assigning a List to a Name (line 123):
        
        # Assigning a List to a Name (line 123):
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_325514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        int_325515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), list_325514, int_325515)
        # Adding element type (line 123)
        int_325516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), list_325514, int_325516)
        # Adding element type (line 123)
        int_325517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), list_325514, int_325517)
        # Adding element type (line 123)
        int_325518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 20), list_325514, int_325518)
        
        # Assigning a type to the variable 'distances' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'distances', list_325514)
        
        # Assigning a Num to a Name (line 124):
        
        # Assigning a Num to a Name (line 124):
        int_325519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'int')
        # Assigning a type to the variable 'max_gap' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'max_gap', int_325519)
        
        # Assigning a List to a Name (line 125):
        
        # Assigning a List to a Name (line 125):
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_325520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        int_325521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), list_325520, int_325521)
        # Adding element type (line 125)
        int_325522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), list_325520, int_325522)
        # Adding element type (line 125)
        int_325523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), list_325520, int_325523)
        # Adding element type (line 125)
        int_325524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), list_325520, int_325524)
        
        # Assigning a type to the variable 'gaps' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'gaps', list_325520)
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to zeros(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_325527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        int_325528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 29), list_325527, int_325528)
        # Adding element type (line 126)
        int_325529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 29), list_325527, int_325529)
        
        # Processing the call keyword arguments (line 126)
        kwargs_325530 = {}
        # Getting the type of 'np' (line 126)
        np_325525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 20), 'np', False)
        # Obtaining the member 'zeros' of a type (line 126)
        zeros_325526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 20), np_325525, 'zeros')
        # Calling zeros(args, kwargs) (line 126)
        zeros_call_result_325531 = invoke(stypy.reporting.localization.Localization(__file__, 126, 20), zeros_325526, *[list_325527], **kwargs_325530)
        
        # Assigning a type to the variable 'test_matr' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'test_matr', zeros_call_result_325531)
        
        # Assigning a Num to a Name (line 127):
        
        # Assigning a Num to a Name (line 127):
        int_325532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'int')
        # Assigning a type to the variable 'length' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'length', int_325532)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to _gen_ridge_line(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_325534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        int_325535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 31), list_325534, int_325535)
        # Adding element type (line 128)
        int_325536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 31), list_325534, int_325536)
        
        # Getting the type of 'test_matr' (line 128)
        test_matr_325537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'test_matr', False)
        # Obtaining the member 'shape' of a type (line 128)
        shape_325538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 40), test_matr_325537, 'shape')
        # Getting the type of 'length' (line 128)
        length_325539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 57), 'length', False)
        # Getting the type of 'distances' (line 128)
        distances_325540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 65), 'distances', False)
        # Getting the type of 'gaps' (line 128)
        gaps_325541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 76), 'gaps', False)
        # Processing the call keyword arguments (line 128)
        kwargs_325542 = {}
        # Getting the type of '_gen_ridge_line' (line 128)
        _gen_ridge_line_325533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), '_gen_ridge_line', False)
        # Calling _gen_ridge_line(args, kwargs) (line 128)
        _gen_ridge_line_call_result_325543 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), _gen_ridge_line_325533, *[list_325534, shape_325538, length_325539, distances_325540, gaps_325541], **kwargs_325542)
        
        # Assigning a type to the variable 'line' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'line', _gen_ridge_line_call_result_325543)
        
        # Assigning a Num to a Subscript (line 129):
        
        # Assigning a Num to a Subscript (line 129):
        int_325544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 38), 'int')
        # Getting the type of 'test_matr' (line 129)
        test_matr_325545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'test_matr')
        
        # Obtaining an instance of the builtin type 'tuple' (line 129)
        tuple_325546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 129)
        # Adding element type (line 129)
        
        # Obtaining the type of the subscript
        int_325547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 23), 'int')
        # Getting the type of 'line' (line 129)
        line_325548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'line')
        # Obtaining the member '__getitem__' of a type (line 129)
        getitem___325549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 18), line_325548, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 129)
        subscript_call_result_325550 = invoke(stypy.reporting.localization.Localization(__file__, 129, 18), getitem___325549, int_325547)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 18), tuple_325546, subscript_call_result_325550)
        # Adding element type (line 129)
        
        # Obtaining the type of the subscript
        int_325551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 32), 'int')
        # Getting the type of 'line' (line 129)
        line_325552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'line')
        # Obtaining the member '__getitem__' of a type (line 129)
        getitem___325553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 27), line_325552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 129)
        subscript_call_result_325554 = invoke(stypy.reporting.localization.Localization(__file__, 129, 27), getitem___325553, int_325551)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 18), tuple_325546, subscript_call_result_325554)
        
        # Storing an element on a container (line 129)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 8), test_matr_325545, (tuple_325546, int_325544))
        
        # Assigning a Num to a Name (line 130):
        
        # Assigning a Num to a Name (line 130):
        int_325555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'int')
        # Assigning a type to the variable 'max_dist' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'max_dist', int_325555)
        
        # Assigning a BinOp to a Name (line 131):
        
        # Assigning a BinOp to a Name (line 131):
        # Getting the type of 'max_dist' (line 131)
        max_dist_325556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'max_dist')
        
        # Call to ones(...): (line 131)
        # Processing the call arguments (line 131)
        int_325559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 41), 'int')
        # Processing the call keyword arguments (line 131)
        kwargs_325560 = {}
        # Getting the type of 'np' (line 131)
        np_325557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 33), 'np', False)
        # Obtaining the member 'ones' of a type (line 131)
        ones_325558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 33), np_325557, 'ones')
        # Calling ones(args, kwargs) (line 131)
        ones_call_result_325561 = invoke(stypy.reporting.localization.Localization(__file__, 131, 33), ones_325558, *[int_325559], **kwargs_325560)
        
        # Applying the binary operator '*' (line 131)
        result_mul_325562 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 24), '*', max_dist_325556, ones_call_result_325561)
        
        # Assigning a type to the variable 'max_distances' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'max_distances', result_mul_325562)
        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to _identify_ridge_lines(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'test_matr' (line 133)
        test_matr_325564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 49), 'test_matr', False)
        # Getting the type of 'max_distances' (line 133)
        max_distances_325565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 60), 'max_distances', False)
        # Getting the type of 'max_gap' (line 133)
        max_gap_325566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 75), 'max_gap', False)
        # Processing the call keyword arguments (line 133)
        kwargs_325567 = {}
        # Getting the type of '_identify_ridge_lines' (line 133)
        _identify_ridge_lines_325563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), '_identify_ridge_lines', False)
        # Calling _identify_ridge_lines(args, kwargs) (line 133)
        _identify_ridge_lines_call_result_325568 = invoke(stypy.reporting.localization.Localization(__file__, 133, 27), _identify_ridge_lines_325563, *[test_matr_325564, max_distances_325565, max_gap_325566], **kwargs_325567)
        
        # Assigning a type to the variable 'identified_lines' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'identified_lines', _identify_ridge_lines_call_result_325568)
        
        # Call to assert_(...): (line 134)
        # Processing the call arguments (line 134)
        
        
        # Call to len(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'identified_lines' (line 134)
        identified_lines_325571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'identified_lines', False)
        # Processing the call keyword arguments (line 134)
        kwargs_325572 = {}
        # Getting the type of 'len' (line 134)
        len_325570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'len', False)
        # Calling len(args, kwargs) (line 134)
        len_call_result_325573 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), len_325570, *[identified_lines_325571], **kwargs_325572)
        
        int_325574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 41), 'int')
        # Applying the binary operator '==' (line 134)
        result_eq_325575 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), '==', len_call_result_325573, int_325574)
        
        # Processing the call keyword arguments (line 134)
        kwargs_325576 = {}
        # Getting the type of 'assert_' (line 134)
        assert__325569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 134)
        assert__call_result_325577 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), assert__325569, *[result_eq_325575], **kwargs_325576)
        
        
        # Getting the type of 'identified_lines' (line 136)
        identified_lines_325578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'identified_lines')
        # Testing the type of a for loop iterable (line 136)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 136, 8), identified_lines_325578)
        # Getting the type of the for loop variable (line 136)
        for_loop_var_325579 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 136, 8), identified_lines_325578)
        # Assigning a type to the variable 'iline' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'iline', for_loop_var_325579)
        # SSA begins for a for statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to diff(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining the type of the subscript
        int_325582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 35), 'int')
        # Getting the type of 'iline' (line 137)
        iline_325583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'iline', False)
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___325584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 29), iline_325583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_325585 = invoke(stypy.reporting.localization.Localization(__file__, 137, 29), getitem___325584, int_325582)
        
        # Processing the call keyword arguments (line 137)
        kwargs_325586 = {}
        # Getting the type of 'np' (line 137)
        np_325580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'np', False)
        # Obtaining the member 'diff' of a type (line 137)
        diff_325581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 21), np_325580, 'diff')
        # Calling diff(args, kwargs) (line 137)
        diff_call_result_325587 = invoke(stypy.reporting.localization.Localization(__file__, 137, 21), diff_325581, *[subscript_call_result_325585], **kwargs_325586)
        
        # Assigning a type to the variable 'adists' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'adists', diff_call_result_325587)
        
        # Call to assert_array_less(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to abs(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'adists' (line 138)
        adists_325593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 48), 'adists', False)
        # Processing the call keyword arguments (line 138)
        kwargs_325594 = {}
        # Getting the type of 'np' (line 138)
        np_325591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 41), 'np', False)
        # Obtaining the member 'abs' of a type (line 138)
        abs_325592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 41), np_325591, 'abs')
        # Calling abs(args, kwargs) (line 138)
        abs_call_result_325595 = invoke(stypy.reporting.localization.Localization(__file__, 138, 41), abs_325592, *[adists_325593], **kwargs_325594)
        
        # Getting the type of 'max_dist' (line 138)
        max_dist_325596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 57), 'max_dist', False)
        # Processing the call keyword arguments (line 138)
        kwargs_325597 = {}
        # Getting the type of 'np' (line 138)
        np_325588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'np', False)
        # Obtaining the member 'testing' of a type (line 138)
        testing_325589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), np_325588, 'testing')
        # Obtaining the member 'assert_array_less' of a type (line 138)
        assert_array_less_325590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), testing_325589, 'assert_array_less')
        # Calling assert_array_less(args, kwargs) (line 138)
        assert_array_less_call_result_325598 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), assert_array_less_325590, *[abs_call_result_325595, max_dist_325596], **kwargs_325597)
        
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to diff(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining the type of the subscript
        int_325601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 34), 'int')
        # Getting the type of 'iline' (line 140)
        iline_325602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'iline', False)
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___325603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 28), iline_325602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_325604 = invoke(stypy.reporting.localization.Localization(__file__, 140, 28), getitem___325603, int_325601)
        
        # Processing the call keyword arguments (line 140)
        kwargs_325605 = {}
        # Getting the type of 'np' (line 140)
        np_325599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'np', False)
        # Obtaining the member 'diff' of a type (line 140)
        diff_325600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 20), np_325599, 'diff')
        # Calling diff(args, kwargs) (line 140)
        diff_call_result_325606 = invoke(stypy.reporting.localization.Localization(__file__, 140, 20), diff_325600, *[subscript_call_result_325604], **kwargs_325605)
        
        # Assigning a type to the variable 'agaps' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'agaps', diff_call_result_325606)
        
        # Call to assert_array_less(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Call to abs(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'agaps' (line 141)
        agaps_325612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 48), 'agaps', False)
        # Processing the call keyword arguments (line 141)
        kwargs_325613 = {}
        # Getting the type of 'np' (line 141)
        np_325610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'np', False)
        # Obtaining the member 'abs' of a type (line 141)
        abs_325611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 41), np_325610, 'abs')
        # Calling abs(args, kwargs) (line 141)
        abs_call_result_325614 = invoke(stypy.reporting.localization.Localization(__file__, 141, 41), abs_325611, *[agaps_325612], **kwargs_325613)
        
        
        # Call to max(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'gaps' (line 141)
        gaps_325616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 60), 'gaps', False)
        # Processing the call keyword arguments (line 141)
        kwargs_325617 = {}
        # Getting the type of 'max' (line 141)
        max_325615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 56), 'max', False)
        # Calling max(args, kwargs) (line 141)
        max_call_result_325618 = invoke(stypy.reporting.localization.Localization(__file__, 141, 56), max_325615, *[gaps_325616], **kwargs_325617)
        
        float_325619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 68), 'float')
        # Applying the binary operator '+' (line 141)
        result_add_325620 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 56), '+', max_call_result_325618, float_325619)
        
        # Processing the call keyword arguments (line 141)
        kwargs_325621 = {}
        # Getting the type of 'np' (line 141)
        np_325607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'np', False)
        # Obtaining the member 'testing' of a type (line 141)
        testing_325608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), np_325607, 'testing')
        # Obtaining the member 'assert_array_less' of a type (line 141)
        assert_array_less_325609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 12), testing_325608, 'assert_array_less')
        # Calling assert_array_less(args, kwargs) (line 141)
        assert_array_less_call_result_325622 = invoke(stypy.reporting.localization.Localization(__file__, 141, 12), assert_array_less_325609, *[abs_call_result_325614, result_add_325620], **kwargs_325621)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_single_biggap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_single_biggap' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_325623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325623)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_single_biggap'
        return stypy_return_type_325623


    @norecursion
    def test_single_biggaps(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_single_biggaps'
        module_type_store = module_type_store.open_function_context('test_single_biggaps', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_localization', localization)
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_function_name', 'TestRidgeLines.test_single_biggaps')
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_param_names_list', [])
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRidgeLines.test_single_biggaps.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRidgeLines.test_single_biggaps', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_single_biggaps', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_single_biggaps(...)' code ##################

        
        # Assigning a List to a Name (line 144):
        
        # Assigning a List to a Name (line 144):
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_325624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        int_325625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 20), list_325624, int_325625)
        
        # Assigning a type to the variable 'distances' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'distances', list_325624)
        
        # Assigning a Num to a Name (line 145):
        
        # Assigning a Num to a Name (line 145):
        int_325626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'int')
        # Assigning a type to the variable 'max_gap' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'max_gap', int_325626)
        
        # Assigning a List to a Name (line 146):
        
        # Assigning a List to a Name (line 146):
        
        # Obtaining an instance of the builtin type 'list' (line 146)
        list_325627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 146)
        # Adding element type (line 146)
        int_325628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 15), list_325627, int_325628)
        # Adding element type (line 146)
        int_325629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 15), list_325627, int_325629)
        
        # Assigning a type to the variable 'gaps' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'gaps', list_325627)
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to zeros(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_325632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        int_325633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 29), list_325632, int_325633)
        # Adding element type (line 147)
        int_325634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 29), list_325632, int_325634)
        
        # Processing the call keyword arguments (line 147)
        kwargs_325635 = {}
        # Getting the type of 'np' (line 147)
        np_325630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'np', False)
        # Obtaining the member 'zeros' of a type (line 147)
        zeros_325631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 20), np_325630, 'zeros')
        # Calling zeros(args, kwargs) (line 147)
        zeros_call_result_325636 = invoke(stypy.reporting.localization.Localization(__file__, 147, 20), zeros_325631, *[list_325632], **kwargs_325635)
        
        # Assigning a type to the variable 'test_matr' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'test_matr', zeros_call_result_325636)
        
        # Assigning a Num to a Name (line 148):
        
        # Assigning a Num to a Name (line 148):
        int_325637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 17), 'int')
        # Assigning a type to the variable 'length' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'length', int_325637)
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to _gen_ridge_line(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Obtaining an instance of the builtin type 'list' (line 149)
        list_325639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 149)
        # Adding element type (line 149)
        int_325640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 31), list_325639, int_325640)
        # Adding element type (line 149)
        int_325641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 31), list_325639, int_325641)
        
        # Getting the type of 'test_matr' (line 149)
        test_matr_325642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 40), 'test_matr', False)
        # Obtaining the member 'shape' of a type (line 149)
        shape_325643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 40), test_matr_325642, 'shape')
        # Getting the type of 'length' (line 149)
        length_325644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 57), 'length', False)
        # Getting the type of 'distances' (line 149)
        distances_325645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 65), 'distances', False)
        # Getting the type of 'gaps' (line 149)
        gaps_325646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 76), 'gaps', False)
        # Processing the call keyword arguments (line 149)
        kwargs_325647 = {}
        # Getting the type of '_gen_ridge_line' (line 149)
        _gen_ridge_line_325638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), '_gen_ridge_line', False)
        # Calling _gen_ridge_line(args, kwargs) (line 149)
        _gen_ridge_line_call_result_325648 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), _gen_ridge_line_325638, *[list_325639, shape_325643, length_325644, distances_325645, gaps_325646], **kwargs_325647)
        
        # Assigning a type to the variable 'line' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'line', _gen_ridge_line_call_result_325648)
        
        # Assigning a Num to a Subscript (line 150):
        
        # Assigning a Num to a Subscript (line 150):
        int_325649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 38), 'int')
        # Getting the type of 'test_matr' (line 150)
        test_matr_325650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'test_matr')
        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_325651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        
        # Obtaining the type of the subscript
        int_325652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 23), 'int')
        # Getting the type of 'line' (line 150)
        line_325653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'line')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___325654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 18), line_325653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_325655 = invoke(stypy.reporting.localization.Localization(__file__, 150, 18), getitem___325654, int_325652)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), tuple_325651, subscript_call_result_325655)
        # Adding element type (line 150)
        
        # Obtaining the type of the subscript
        int_325656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 32), 'int')
        # Getting the type of 'line' (line 150)
        line_325657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'line')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___325658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 27), line_325657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_325659 = invoke(stypy.reporting.localization.Localization(__file__, 150, 27), getitem___325658, int_325656)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 18), tuple_325651, subscript_call_result_325659)
        
        # Storing an element on a container (line 150)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 8), test_matr_325650, (tuple_325651, int_325649))
        
        # Assigning a Num to a Name (line 151):
        
        # Assigning a Num to a Name (line 151):
        int_325660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 19), 'int')
        # Assigning a type to the variable 'max_dist' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'max_dist', int_325660)
        
        # Assigning a BinOp to a Name (line 152):
        
        # Assigning a BinOp to a Name (line 152):
        # Getting the type of 'max_dist' (line 152)
        max_dist_325661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'max_dist')
        
        # Call to ones(...): (line 152)
        # Processing the call arguments (line 152)
        int_325664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 41), 'int')
        # Processing the call keyword arguments (line 152)
        kwargs_325665 = {}
        # Getting the type of 'np' (line 152)
        np_325662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 33), 'np', False)
        # Obtaining the member 'ones' of a type (line 152)
        ones_325663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 33), np_325662, 'ones')
        # Calling ones(args, kwargs) (line 152)
        ones_call_result_325666 = invoke(stypy.reporting.localization.Localization(__file__, 152, 33), ones_325663, *[int_325664], **kwargs_325665)
        
        # Applying the binary operator '*' (line 152)
        result_mul_325667 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 24), '*', max_dist_325661, ones_call_result_325666)
        
        # Assigning a type to the variable 'max_distances' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'max_distances', result_mul_325667)
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to _identify_ridge_lines(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'test_matr' (line 154)
        test_matr_325669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'test_matr', False)
        # Getting the type of 'max_distances' (line 154)
        max_distances_325670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 60), 'max_distances', False)
        # Getting the type of 'max_gap' (line 154)
        max_gap_325671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 75), 'max_gap', False)
        # Processing the call keyword arguments (line 154)
        kwargs_325672 = {}
        # Getting the type of '_identify_ridge_lines' (line 154)
        _identify_ridge_lines_325668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), '_identify_ridge_lines', False)
        # Calling _identify_ridge_lines(args, kwargs) (line 154)
        _identify_ridge_lines_call_result_325673 = invoke(stypy.reporting.localization.Localization(__file__, 154, 27), _identify_ridge_lines_325668, *[test_matr_325669, max_distances_325670, max_gap_325671], **kwargs_325672)
        
        # Assigning a type to the variable 'identified_lines' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'identified_lines', _identify_ridge_lines_call_result_325673)
        
        # Call to assert_(...): (line 155)
        # Processing the call arguments (line 155)
        
        
        # Call to len(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'identified_lines' (line 155)
        identified_lines_325676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'identified_lines', False)
        # Processing the call keyword arguments (line 155)
        kwargs_325677 = {}
        # Getting the type of 'len' (line 155)
        len_325675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'len', False)
        # Calling len(args, kwargs) (line 155)
        len_call_result_325678 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), len_325675, *[identified_lines_325676], **kwargs_325677)
        
        int_325679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 41), 'int')
        # Applying the binary operator '==' (line 155)
        result_eq_325680 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 16), '==', len_call_result_325678, int_325679)
        
        # Processing the call keyword arguments (line 155)
        kwargs_325681 = {}
        # Getting the type of 'assert_' (line 155)
        assert__325674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 155)
        assert__call_result_325682 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), assert__325674, *[result_eq_325680], **kwargs_325681)
        
        
        # Getting the type of 'identified_lines' (line 157)
        identified_lines_325683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'identified_lines')
        # Testing the type of a for loop iterable (line 157)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 8), identified_lines_325683)
        # Getting the type of the for loop variable (line 157)
        for_loop_var_325684 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 8), identified_lines_325683)
        # Assigning a type to the variable 'iline' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'iline', for_loop_var_325684)
        # SSA begins for a for statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to diff(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Obtaining the type of the subscript
        int_325687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 35), 'int')
        # Getting the type of 'iline' (line 158)
        iline_325688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'iline', False)
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___325689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 29), iline_325688, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_325690 = invoke(stypy.reporting.localization.Localization(__file__, 158, 29), getitem___325689, int_325687)
        
        # Processing the call keyword arguments (line 158)
        kwargs_325691 = {}
        # Getting the type of 'np' (line 158)
        np_325685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'np', False)
        # Obtaining the member 'diff' of a type (line 158)
        diff_325686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 21), np_325685, 'diff')
        # Calling diff(args, kwargs) (line 158)
        diff_call_result_325692 = invoke(stypy.reporting.localization.Localization(__file__, 158, 21), diff_325686, *[subscript_call_result_325690], **kwargs_325691)
        
        # Assigning a type to the variable 'adists' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'adists', diff_call_result_325692)
        
        # Call to assert_array_less(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Call to abs(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'adists' (line 159)
        adists_325698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 48), 'adists', False)
        # Processing the call keyword arguments (line 159)
        kwargs_325699 = {}
        # Getting the type of 'np' (line 159)
        np_325696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 41), 'np', False)
        # Obtaining the member 'abs' of a type (line 159)
        abs_325697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 41), np_325696, 'abs')
        # Calling abs(args, kwargs) (line 159)
        abs_call_result_325700 = invoke(stypy.reporting.localization.Localization(__file__, 159, 41), abs_325697, *[adists_325698], **kwargs_325699)
        
        # Getting the type of 'max_dist' (line 159)
        max_dist_325701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 57), 'max_dist', False)
        # Processing the call keyword arguments (line 159)
        kwargs_325702 = {}
        # Getting the type of 'np' (line 159)
        np_325693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'np', False)
        # Obtaining the member 'testing' of a type (line 159)
        testing_325694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), np_325693, 'testing')
        # Obtaining the member 'assert_array_less' of a type (line 159)
        assert_array_less_325695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), testing_325694, 'assert_array_less')
        # Calling assert_array_less(args, kwargs) (line 159)
        assert_array_less_call_result_325703 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), assert_array_less_325695, *[abs_call_result_325700, max_dist_325701], **kwargs_325702)
        
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to diff(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Obtaining the type of the subscript
        int_325706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 34), 'int')
        # Getting the type of 'iline' (line 161)
        iline_325707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'iline', False)
        # Obtaining the member '__getitem__' of a type (line 161)
        getitem___325708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 28), iline_325707, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 161)
        subscript_call_result_325709 = invoke(stypy.reporting.localization.Localization(__file__, 161, 28), getitem___325708, int_325706)
        
        # Processing the call keyword arguments (line 161)
        kwargs_325710 = {}
        # Getting the type of 'np' (line 161)
        np_325704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'np', False)
        # Obtaining the member 'diff' of a type (line 161)
        diff_325705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 20), np_325704, 'diff')
        # Calling diff(args, kwargs) (line 161)
        diff_call_result_325711 = invoke(stypy.reporting.localization.Localization(__file__, 161, 20), diff_325705, *[subscript_call_result_325709], **kwargs_325710)
        
        # Assigning a type to the variable 'agaps' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'agaps', diff_call_result_325711)
        
        # Call to assert_array_less(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Call to abs(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'agaps' (line 162)
        agaps_325717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 48), 'agaps', False)
        # Processing the call keyword arguments (line 162)
        kwargs_325718 = {}
        # Getting the type of 'np' (line 162)
        np_325715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'np', False)
        # Obtaining the member 'abs' of a type (line 162)
        abs_325716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 41), np_325715, 'abs')
        # Calling abs(args, kwargs) (line 162)
        abs_call_result_325719 = invoke(stypy.reporting.localization.Localization(__file__, 162, 41), abs_325716, *[agaps_325717], **kwargs_325718)
        
        
        # Call to max(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'gaps' (line 162)
        gaps_325721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 60), 'gaps', False)
        # Processing the call keyword arguments (line 162)
        kwargs_325722 = {}
        # Getting the type of 'max' (line 162)
        max_325720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 56), 'max', False)
        # Calling max(args, kwargs) (line 162)
        max_call_result_325723 = invoke(stypy.reporting.localization.Localization(__file__, 162, 56), max_325720, *[gaps_325721], **kwargs_325722)
        
        float_325724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 68), 'float')
        # Applying the binary operator '+' (line 162)
        result_add_325725 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 56), '+', max_call_result_325723, float_325724)
        
        # Processing the call keyword arguments (line 162)
        kwargs_325726 = {}
        # Getting the type of 'np' (line 162)
        np_325712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'np', False)
        # Obtaining the member 'testing' of a type (line 162)
        testing_325713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), np_325712, 'testing')
        # Obtaining the member 'assert_array_less' of a type (line 162)
        assert_array_less_325714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), testing_325713, 'assert_array_less')
        # Calling assert_array_less(args, kwargs) (line 162)
        assert_array_less_call_result_325727 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), assert_array_less_325714, *[abs_call_result_325719, result_add_325725], **kwargs_325726)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_single_biggaps(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_single_biggaps' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_325728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325728)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_single_biggaps'
        return stypy_return_type_325728


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 73, 0, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRidgeLines.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestRidgeLines' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'TestRidgeLines', TestRidgeLines)
# Declaration of the 'TestArgrel' class

class TestArgrel(object, ):

    @norecursion
    def test_empty(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty'
        module_type_store = module_type_store.open_function_context('test_empty', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArgrel.test_empty.__dict__.__setitem__('stypy_localization', localization)
        TestArgrel.test_empty.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArgrel.test_empty.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArgrel.test_empty.__dict__.__setitem__('stypy_function_name', 'TestArgrel.test_empty')
        TestArgrel.test_empty.__dict__.__setitem__('stypy_param_names_list', [])
        TestArgrel.test_empty.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArgrel.test_empty.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArgrel.test_empty.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArgrel.test_empty.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArgrel.test_empty.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArgrel.test_empty.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArgrel.test_empty', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_empty', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_empty(...)' code ##################

        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to array(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Obtaining an instance of the builtin type 'list' (line 173)
        list_325731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 173)
        
        # Processing the call keyword arguments (line 173)
        # Getting the type of 'int' (line 173)
        int_325732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 41), 'int', False)
        keyword_325733 = int_325732
        kwargs_325734 = {'dtype': keyword_325733}
        # Getting the type of 'np' (line 173)
        np_325729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'np', False)
        # Obtaining the member 'array' of a type (line 173)
        array_325730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 22), np_325729, 'array')
        # Calling array(args, kwargs) (line 173)
        array_call_result_325735 = invoke(stypy.reporting.localization.Localization(__file__, 173, 22), array_325730, *[list_325731], **kwargs_325734)
        
        # Assigning a type to the variable 'empty_array' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'empty_array', array_call_result_325735)
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to zeros(...): (line 175)
        # Processing the call arguments (line 175)
        int_325738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 22), 'int')
        # Processing the call keyword arguments (line 175)
        kwargs_325739 = {}
        # Getting the type of 'np' (line 175)
        np_325736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), 'np', False)
        # Obtaining the member 'zeros' of a type (line 175)
        zeros_325737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 13), np_325736, 'zeros')
        # Calling zeros(args, kwargs) (line 175)
        zeros_call_result_325740 = invoke(stypy.reporting.localization.Localization(__file__, 175, 13), zeros_325737, *[int_325738], **kwargs_325739)
        
        # Assigning a type to the variable 'z1' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'z1', zeros_call_result_325740)
        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to argrelmin(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'z1' (line 177)
        z1_325742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 22), 'z1', False)
        # Processing the call keyword arguments (line 177)
        kwargs_325743 = {}
        # Getting the type of 'argrelmin' (line 177)
        argrelmin_325741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 177)
        argrelmin_call_result_325744 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), argrelmin_325741, *[z1_325742], **kwargs_325743)
        
        # Assigning a type to the variable 'i' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'i', argrelmin_call_result_325744)
        
        # Call to assert_equal(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Call to len(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'i' (line 178)
        i_325747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'i', False)
        # Processing the call keyword arguments (line 178)
        kwargs_325748 = {}
        # Getting the type of 'len' (line 178)
        len_325746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'len', False)
        # Calling len(args, kwargs) (line 178)
        len_call_result_325749 = invoke(stypy.reporting.localization.Localization(__file__, 178, 21), len_325746, *[i_325747], **kwargs_325748)
        
        int_325750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 29), 'int')
        # Processing the call keyword arguments (line 178)
        kwargs_325751 = {}
        # Getting the type of 'assert_equal' (line 178)
        assert_equal_325745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 178)
        assert_equal_call_result_325752 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), assert_equal_325745, *[len_call_result_325749, int_325750], **kwargs_325751)
        
        
        # Call to assert_array_equal(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Obtaining the type of the subscript
        int_325754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'int')
        # Getting the type of 'i' (line 179)
        i_325755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'i', False)
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___325756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 27), i_325755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_325757 = invoke(stypy.reporting.localization.Localization(__file__, 179, 27), getitem___325756, int_325754)
        
        # Getting the type of 'empty_array' (line 179)
        empty_array_325758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 33), 'empty_array', False)
        # Processing the call keyword arguments (line 179)
        kwargs_325759 = {}
        # Getting the type of 'assert_array_equal' (line 179)
        assert_array_equal_325753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 179)
        assert_array_equal_call_result_325760 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), assert_array_equal_325753, *[subscript_call_result_325757, empty_array_325758], **kwargs_325759)
        
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to zeros(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_325763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        # Adding element type (line 181)
        int_325764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 23), tuple_325763, int_325764)
        # Adding element type (line 181)
        int_325765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 23), tuple_325763, int_325765)
        
        # Processing the call keyword arguments (line 181)
        kwargs_325766 = {}
        # Getting the type of 'np' (line 181)
        np_325761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'np', False)
        # Obtaining the member 'zeros' of a type (line 181)
        zeros_325762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 13), np_325761, 'zeros')
        # Calling zeros(args, kwargs) (line 181)
        zeros_call_result_325767 = invoke(stypy.reporting.localization.Localization(__file__, 181, 13), zeros_325762, *[tuple_325763], **kwargs_325766)
        
        # Assigning a type to the variable 'z2' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'z2', zeros_call_result_325767)
        
        # Assigning a Call to a Tuple (line 183):
        
        # Assigning a Subscript to a Name (line 183):
        
        # Obtaining the type of the subscript
        int_325768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 8), 'int')
        
        # Call to argrelmin(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'z2' (line 183)
        z2_325770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 29), 'z2', False)
        # Processing the call keyword arguments (line 183)
        int_325771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 38), 'int')
        keyword_325772 = int_325771
        kwargs_325773 = {'axis': keyword_325772}
        # Getting the type of 'argrelmin' (line 183)
        argrelmin_325769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 183)
        argrelmin_call_result_325774 = invoke(stypy.reporting.localization.Localization(__file__, 183, 19), argrelmin_325769, *[z2_325770], **kwargs_325773)
        
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___325775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), argrelmin_call_result_325774, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_325776 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), getitem___325775, int_325768)
        
        # Assigning a type to the variable 'tuple_var_assignment_324953' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_var_assignment_324953', subscript_call_result_325776)
        
        # Assigning a Subscript to a Name (line 183):
        
        # Obtaining the type of the subscript
        int_325777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 8), 'int')
        
        # Call to argrelmin(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'z2' (line 183)
        z2_325779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 29), 'z2', False)
        # Processing the call keyword arguments (line 183)
        int_325780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 38), 'int')
        keyword_325781 = int_325780
        kwargs_325782 = {'axis': keyword_325781}
        # Getting the type of 'argrelmin' (line 183)
        argrelmin_325778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 183)
        argrelmin_call_result_325783 = invoke(stypy.reporting.localization.Localization(__file__, 183, 19), argrelmin_325778, *[z2_325779], **kwargs_325782)
        
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___325784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), argrelmin_call_result_325783, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_325785 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), getitem___325784, int_325777)
        
        # Assigning a type to the variable 'tuple_var_assignment_324954' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_var_assignment_324954', subscript_call_result_325785)
        
        # Assigning a Name to a Name (line 183):
        # Getting the type of 'tuple_var_assignment_324953' (line 183)
        tuple_var_assignment_324953_325786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_var_assignment_324953')
        # Assigning a type to the variable 'row' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'row', tuple_var_assignment_324953_325786)
        
        # Assigning a Name to a Name (line 183):
        # Getting the type of 'tuple_var_assignment_324954' (line 183)
        tuple_var_assignment_324954_325787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_var_assignment_324954')
        # Assigning a type to the variable 'col' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 13), 'col', tuple_var_assignment_324954_325787)
        
        # Call to assert_array_equal(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'row' (line 184)
        row_325789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'row', False)
        # Getting the type of 'empty_array' (line 184)
        empty_array_325790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 32), 'empty_array', False)
        # Processing the call keyword arguments (line 184)
        kwargs_325791 = {}
        # Getting the type of 'assert_array_equal' (line 184)
        assert_array_equal_325788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 184)
        assert_array_equal_call_result_325792 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), assert_array_equal_325788, *[row_325789, empty_array_325790], **kwargs_325791)
        
        
        # Call to assert_array_equal(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'col' (line 185)
        col_325794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'col', False)
        # Getting the type of 'empty_array' (line 185)
        empty_array_325795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 32), 'empty_array', False)
        # Processing the call keyword arguments (line 185)
        kwargs_325796 = {}
        # Getting the type of 'assert_array_equal' (line 185)
        assert_array_equal_325793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 185)
        assert_array_equal_call_result_325797 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert_array_equal_325793, *[col_325794, empty_array_325795], **kwargs_325796)
        
        
        # Assigning a Call to a Tuple (line 187):
        
        # Assigning a Subscript to a Name (line 187):
        
        # Obtaining the type of the subscript
        int_325798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 8), 'int')
        
        # Call to argrelmin(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'z2' (line 187)
        z2_325800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 29), 'z2', False)
        # Processing the call keyword arguments (line 187)
        int_325801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 38), 'int')
        keyword_325802 = int_325801
        kwargs_325803 = {'axis': keyword_325802}
        # Getting the type of 'argrelmin' (line 187)
        argrelmin_325799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 187)
        argrelmin_call_result_325804 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), argrelmin_325799, *[z2_325800], **kwargs_325803)
        
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___325805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), argrelmin_call_result_325804, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_325806 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), getitem___325805, int_325798)
        
        # Assigning a type to the variable 'tuple_var_assignment_324955' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tuple_var_assignment_324955', subscript_call_result_325806)
        
        # Assigning a Subscript to a Name (line 187):
        
        # Obtaining the type of the subscript
        int_325807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 8), 'int')
        
        # Call to argrelmin(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'z2' (line 187)
        z2_325809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 29), 'z2', False)
        # Processing the call keyword arguments (line 187)
        int_325810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 38), 'int')
        keyword_325811 = int_325810
        kwargs_325812 = {'axis': keyword_325811}
        # Getting the type of 'argrelmin' (line 187)
        argrelmin_325808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 187)
        argrelmin_call_result_325813 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), argrelmin_325808, *[z2_325809], **kwargs_325812)
        
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___325814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), argrelmin_call_result_325813, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_325815 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), getitem___325814, int_325807)
        
        # Assigning a type to the variable 'tuple_var_assignment_324956' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tuple_var_assignment_324956', subscript_call_result_325815)
        
        # Assigning a Name to a Name (line 187):
        # Getting the type of 'tuple_var_assignment_324955' (line 187)
        tuple_var_assignment_324955_325816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tuple_var_assignment_324955')
        # Assigning a type to the variable 'row' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'row', tuple_var_assignment_324955_325816)
        
        # Assigning a Name to a Name (line 187):
        # Getting the type of 'tuple_var_assignment_324956' (line 187)
        tuple_var_assignment_324956_325817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'tuple_var_assignment_324956')
        # Assigning a type to the variable 'col' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), 'col', tuple_var_assignment_324956_325817)
        
        # Call to assert_array_equal(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'row' (line 188)
        row_325819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 27), 'row', False)
        # Getting the type of 'empty_array' (line 188)
        empty_array_325820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 32), 'empty_array', False)
        # Processing the call keyword arguments (line 188)
        kwargs_325821 = {}
        # Getting the type of 'assert_array_equal' (line 188)
        assert_array_equal_325818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 188)
        assert_array_equal_call_result_325822 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), assert_array_equal_325818, *[row_325819, empty_array_325820], **kwargs_325821)
        
        
        # Call to assert_array_equal(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'col' (line 189)
        col_325824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 27), 'col', False)
        # Getting the type of 'empty_array' (line 189)
        empty_array_325825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'empty_array', False)
        # Processing the call keyword arguments (line 189)
        kwargs_325826 = {}
        # Getting the type of 'assert_array_equal' (line 189)
        assert_array_equal_325823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 189)
        assert_array_equal_call_result_325827 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assert_array_equal_325823, *[col_325824, empty_array_325825], **kwargs_325826)
        
        
        # ################# End of 'test_empty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_325828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_325828)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty'
        return stypy_return_type_325828


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArgrel.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestArgrel.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArgrel.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArgrel.test_basic.__dict__.__setitem__('stypy_function_name', 'TestArgrel.test_basic')
        TestArgrel.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestArgrel.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArgrel.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArgrel.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArgrel.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArgrel.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArgrel.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArgrel.test_basic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_basic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_basic(...)' code ##################

        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to array(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_325831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_325832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        int_325833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 22), list_325832, int_325833)
        # Adding element type (line 196)
        int_325834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 22), list_325832, int_325834)
        # Adding element type (line 196)
        int_325835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 22), list_325832, int_325835)
        # Adding element type (line 196)
        int_325836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 22), list_325832, int_325836)
        # Adding element type (line 196)
        int_325837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 22), list_325832, int_325837)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 21), list_325831, list_325832)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_325838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        int_325839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 22), list_325838, int_325839)
        # Adding element type (line 197)
        int_325840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 22), list_325838, int_325840)
        # Adding element type (line 197)
        int_325841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 22), list_325838, int_325841)
        # Adding element type (line 197)
        int_325842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 22), list_325838, int_325842)
        # Adding element type (line 197)
        int_325843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 22), list_325838, int_325843)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 21), list_325831, list_325838)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 198)
        list_325844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 198)
        # Adding element type (line 198)
        int_325845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 22), list_325844, int_325845)
        # Adding element type (line 198)
        int_325846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 22), list_325844, int_325846)
        # Adding element type (line 198)
        int_325847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 22), list_325844, int_325847)
        # Adding element type (line 198)
        int_325848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 22), list_325844, int_325848)
        # Adding element type (line 198)
        int_325849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 22), list_325844, int_325849)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 21), list_325831, list_325844)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_325850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        int_325851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), list_325850, int_325851)
        # Adding element type (line 199)
        int_325852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), list_325850, int_325852)
        # Adding element type (line 199)
        int_325853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), list_325850, int_325853)
        # Adding element type (line 199)
        int_325854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), list_325850, int_325854)
        # Adding element type (line 199)
        int_325855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), list_325850, int_325855)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 21), list_325831, list_325850)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_325856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        int_325857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 22), list_325856, int_325857)
        # Adding element type (line 200)
        int_325858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 22), list_325856, int_325858)
        # Adding element type (line 200)
        int_325859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 22), list_325856, int_325859)
        # Adding element type (line 200)
        int_325860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 22), list_325856, int_325860)
        # Adding element type (line 200)
        int_325861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 22), list_325856, int_325861)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 21), list_325831, list_325856)
        
        # Processing the call keyword arguments (line 196)
        kwargs_325862 = {}
        # Getting the type of 'np' (line 196)
        np_325829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 196)
        array_325830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), np_325829, 'array')
        # Calling array(args, kwargs) (line 196)
        array_call_result_325863 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), array_325830, *[list_325831], **kwargs_325862)
        
        # Assigning a type to the variable 'x' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'x', array_call_result_325863)
        
        # Assigning a Call to a Tuple (line 202):
        
        # Assigning a Subscript to a Name (line 202):
        
        # Obtaining the type of the subscript
        int_325864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 8), 'int')
        
        # Call to argrelmax(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'x' (line 202)
        x_325866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'x', False)
        # Processing the call keyword arguments (line 202)
        int_325867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 37), 'int')
        keyword_325868 = int_325867
        kwargs_325869 = {'axis': keyword_325868}
        # Getting the type of 'argrelmax' (line 202)
        argrelmax_325865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'argrelmax', False)
        # Calling argrelmax(args, kwargs) (line 202)
        argrelmax_call_result_325870 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), argrelmax_325865, *[x_325866], **kwargs_325869)
        
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___325871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), argrelmax_call_result_325870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_325872 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), getitem___325871, int_325864)
        
        # Assigning a type to the variable 'tuple_var_assignment_324957' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_324957', subscript_call_result_325872)
        
        # Assigning a Subscript to a Name (line 202):
        
        # Obtaining the type of the subscript
        int_325873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 8), 'int')
        
        # Call to argrelmax(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'x' (line 202)
        x_325875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'x', False)
        # Processing the call keyword arguments (line 202)
        int_325876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 37), 'int')
        keyword_325877 = int_325876
        kwargs_325878 = {'axis': keyword_325877}
        # Getting the type of 'argrelmax' (line 202)
        argrelmax_325874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'argrelmax', False)
        # Calling argrelmax(args, kwargs) (line 202)
        argrelmax_call_result_325879 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), argrelmax_325874, *[x_325875], **kwargs_325878)
        
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___325880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), argrelmax_call_result_325879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_325881 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), getitem___325880, int_325873)
        
        # Assigning a type to the variable 'tuple_var_assignment_324958' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_324958', subscript_call_result_325881)
        
        # Assigning a Name to a Name (line 202):
        # Getting the type of 'tuple_var_assignment_324957' (line 202)
        tuple_var_assignment_324957_325882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_324957')
        # Assigning a type to the variable 'row' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'row', tuple_var_assignment_324957_325882)
        
        # Assigning a Name to a Name (line 202):
        # Getting the type of 'tuple_var_assignment_324958' (line 202)
        tuple_var_assignment_324958_325883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_324958')
        # Assigning a type to the variable 'col' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 13), 'col', tuple_var_assignment_324958_325883)
        
        # Assigning a Call to a Name (line 203):
        
        # Assigning a Call to a Name (line 203):
        
        # Call to argsort(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'row' (line 203)
        row_325886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'row', False)
        # Processing the call keyword arguments (line 203)
        kwargs_325887 = {}
        # Getting the type of 'np' (line 203)
        np_325884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'np', False)
        # Obtaining the member 'argsort' of a type (line 203)
        argsort_325885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), np_325884, 'argsort')
        # Calling argsort(args, kwargs) (line 203)
        argsort_call_result_325888 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), argsort_325885, *[row_325886], **kwargs_325887)
        
        # Assigning a type to the variable 'order' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'order', argsort_call_result_325888)
        
        # Call to assert_equal(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 204)
        order_325890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 25), 'order', False)
        # Getting the type of 'row' (line 204)
        row_325891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'row', False)
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___325892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), row_325891, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_325893 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), getitem___325892, order_325890)
        
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_325894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        int_325895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 33), list_325894, int_325895)
        # Adding element type (line 204)
        int_325896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 33), list_325894, int_325896)
        # Adding element type (line 204)
        int_325897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 33), list_325894, int_325897)
        
        # Processing the call keyword arguments (line 204)
        kwargs_325898 = {}
        # Getting the type of 'assert_equal' (line 204)
        assert_equal_325889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 204)
        assert_equal_call_result_325899 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), assert_equal_325889, *[subscript_call_result_325893, list_325894], **kwargs_325898)
        
        
        # Call to assert_equal(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 205)
        order_325901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 25), 'order', False)
        # Getting the type of 'col' (line 205)
        col_325902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'col', False)
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___325903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 21), col_325902, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_325904 = invoke(stypy.reporting.localization.Localization(__file__, 205, 21), getitem___325903, order_325901)
        
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_325905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        int_325906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 33), list_325905, int_325906)
        # Adding element type (line 205)
        int_325907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 33), list_325905, int_325907)
        # Adding element type (line 205)
        int_325908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 33), list_325905, int_325908)
        
        # Processing the call keyword arguments (line 205)
        kwargs_325909 = {}
        # Getting the type of 'assert_equal' (line 205)
        assert_equal_325900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 205)
        assert_equal_call_result_325910 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), assert_equal_325900, *[subscript_call_result_325904, list_325905], **kwargs_325909)
        
        
        # Assigning a Call to a Tuple (line 207):
        
        # Assigning a Subscript to a Name (line 207):
        
        # Obtaining the type of the subscript
        int_325911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'int')
        
        # Call to argrelmax(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'x' (line 207)
        x_325913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'x', False)
        # Processing the call keyword arguments (line 207)
        int_325914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 37), 'int')
        keyword_325915 = int_325914
        kwargs_325916 = {'axis': keyword_325915}
        # Getting the type of 'argrelmax' (line 207)
        argrelmax_325912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'argrelmax', False)
        # Calling argrelmax(args, kwargs) (line 207)
        argrelmax_call_result_325917 = invoke(stypy.reporting.localization.Localization(__file__, 207, 19), argrelmax_325912, *[x_325913], **kwargs_325916)
        
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___325918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), argrelmax_call_result_325917, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_325919 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), getitem___325918, int_325911)
        
        # Assigning a type to the variable 'tuple_var_assignment_324959' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_324959', subscript_call_result_325919)
        
        # Assigning a Subscript to a Name (line 207):
        
        # Obtaining the type of the subscript
        int_325920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 8), 'int')
        
        # Call to argrelmax(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'x' (line 207)
        x_325922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'x', False)
        # Processing the call keyword arguments (line 207)
        int_325923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 37), 'int')
        keyword_325924 = int_325923
        kwargs_325925 = {'axis': keyword_325924}
        # Getting the type of 'argrelmax' (line 207)
        argrelmax_325921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'argrelmax', False)
        # Calling argrelmax(args, kwargs) (line 207)
        argrelmax_call_result_325926 = invoke(stypy.reporting.localization.Localization(__file__, 207, 19), argrelmax_325921, *[x_325922], **kwargs_325925)
        
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___325927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), argrelmax_call_result_325926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_325928 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), getitem___325927, int_325920)
        
        # Assigning a type to the variable 'tuple_var_assignment_324960' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_324960', subscript_call_result_325928)
        
        # Assigning a Name to a Name (line 207):
        # Getting the type of 'tuple_var_assignment_324959' (line 207)
        tuple_var_assignment_324959_325929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_324959')
        # Assigning a type to the variable 'row' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'row', tuple_var_assignment_324959_325929)
        
        # Assigning a Name to a Name (line 207):
        # Getting the type of 'tuple_var_assignment_324960' (line 207)
        tuple_var_assignment_324960_325930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'tuple_var_assignment_324960')
        # Assigning a type to the variable 'col' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'col', tuple_var_assignment_324960_325930)
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to argsort(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'row' (line 208)
        row_325933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'row', False)
        # Processing the call keyword arguments (line 208)
        kwargs_325934 = {}
        # Getting the type of 'np' (line 208)
        np_325931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'np', False)
        # Obtaining the member 'argsort' of a type (line 208)
        argsort_325932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), np_325931, 'argsort')
        # Calling argsort(args, kwargs) (line 208)
        argsort_call_result_325935 = invoke(stypy.reporting.localization.Localization(__file__, 208, 16), argsort_325932, *[row_325933], **kwargs_325934)
        
        # Assigning a type to the variable 'order' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'order', argsort_call_result_325935)
        
        # Call to assert_equal(...): (line 209)
        # Processing the call arguments (line 209)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 209)
        order_325937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 25), 'order', False)
        # Getting the type of 'row' (line 209)
        row_325938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'row', False)
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___325939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 21), row_325938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_325940 = invoke(stypy.reporting.localization.Localization(__file__, 209, 21), getitem___325939, order_325937)
        
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_325941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        int_325942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 33), list_325941, int_325942)
        # Adding element type (line 209)
        int_325943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 33), list_325941, int_325943)
        # Adding element type (line 209)
        int_325944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 33), list_325941, int_325944)
        
        # Processing the call keyword arguments (line 209)
        kwargs_325945 = {}
        # Getting the type of 'assert_equal' (line 209)
        assert_equal_325936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 209)
        assert_equal_call_result_325946 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), assert_equal_325936, *[subscript_call_result_325940, list_325941], **kwargs_325945)
        
        
        # Call to assert_equal(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 210)
        order_325948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'order', False)
        # Getting the type of 'col' (line 210)
        col_325949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'col', False)
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___325950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 21), col_325949, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_325951 = invoke(stypy.reporting.localization.Localization(__file__, 210, 21), getitem___325950, order_325948)
        
        
        # Obtaining an instance of the builtin type 'list' (line 210)
        list_325952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 210)
        # Adding element type (line 210)
        int_325953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 33), list_325952, int_325953)
        # Adding element type (line 210)
        int_325954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 33), list_325952, int_325954)
        # Adding element type (line 210)
        int_325955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 33), list_325952, int_325955)
        
        # Processing the call keyword arguments (line 210)
        kwargs_325956 = {}
        # Getting the type of 'assert_equal' (line 210)
        assert_equal_325947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 210)
        assert_equal_call_result_325957 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), assert_equal_325947, *[subscript_call_result_325951, list_325952], **kwargs_325956)
        
        
        # Assigning a Call to a Tuple (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_325958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
        
        # Call to argrelmin(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'x' (line 212)
        x_325960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'x', False)
        # Processing the call keyword arguments (line 212)
        int_325961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 37), 'int')
        keyword_325962 = int_325961
        kwargs_325963 = {'axis': keyword_325962}
        # Getting the type of 'argrelmin' (line 212)
        argrelmin_325959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 212)
        argrelmin_call_result_325964 = invoke(stypy.reporting.localization.Localization(__file__, 212, 19), argrelmin_325959, *[x_325960], **kwargs_325963)
        
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___325965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), argrelmin_call_result_325964, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_325966 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___325965, int_325958)
        
        # Assigning a type to the variable 'tuple_var_assignment_324961' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_324961', subscript_call_result_325966)
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_325967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
        
        # Call to argrelmin(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'x' (line 212)
        x_325969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'x', False)
        # Processing the call keyword arguments (line 212)
        int_325970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 37), 'int')
        keyword_325971 = int_325970
        kwargs_325972 = {'axis': keyword_325971}
        # Getting the type of 'argrelmin' (line 212)
        argrelmin_325968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 212)
        argrelmin_call_result_325973 = invoke(stypy.reporting.localization.Localization(__file__, 212, 19), argrelmin_325968, *[x_325969], **kwargs_325972)
        
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___325974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), argrelmin_call_result_325973, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_325975 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___325974, int_325967)
        
        # Assigning a type to the variable 'tuple_var_assignment_324962' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_324962', subscript_call_result_325975)
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'tuple_var_assignment_324961' (line 212)
        tuple_var_assignment_324961_325976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_324961')
        # Assigning a type to the variable 'row' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'row', tuple_var_assignment_324961_325976)
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'tuple_var_assignment_324962' (line 212)
        tuple_var_assignment_324962_325977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_324962')
        # Assigning a type to the variable 'col' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'col', tuple_var_assignment_324962_325977)
        
        # Assigning a Call to a Name (line 213):
        
        # Assigning a Call to a Name (line 213):
        
        # Call to argsort(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'row' (line 213)
        row_325980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'row', False)
        # Processing the call keyword arguments (line 213)
        kwargs_325981 = {}
        # Getting the type of 'np' (line 213)
        np_325978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'np', False)
        # Obtaining the member 'argsort' of a type (line 213)
        argsort_325979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), np_325978, 'argsort')
        # Calling argsort(args, kwargs) (line 213)
        argsort_call_result_325982 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), argsort_325979, *[row_325980], **kwargs_325981)
        
        # Assigning a type to the variable 'order' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'order', argsort_call_result_325982)
        
        # Call to assert_equal(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 214)
        order_325984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 25), 'order', False)
        # Getting the type of 'row' (line 214)
        row_325985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'row', False)
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___325986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), row_325985, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_325987 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), getitem___325986, order_325984)
        
        
        # Obtaining an instance of the builtin type 'list' (line 214)
        list_325988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 214)
        # Adding element type (line 214)
        int_325989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 33), list_325988, int_325989)
        # Adding element type (line 214)
        int_325990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 33), list_325988, int_325990)
        # Adding element type (line 214)
        int_325991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 33), list_325988, int_325991)
        
        # Processing the call keyword arguments (line 214)
        kwargs_325992 = {}
        # Getting the type of 'assert_equal' (line 214)
        assert_equal_325983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 214)
        assert_equal_call_result_325993 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assert_equal_325983, *[subscript_call_result_325987, list_325988], **kwargs_325992)
        
        
        # Call to assert_equal(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 215)
        order_325995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 25), 'order', False)
        # Getting the type of 'col' (line 215)
        col_325996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'col', False)
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___325997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 21), col_325996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_325998 = invoke(stypy.reporting.localization.Localization(__file__, 215, 21), getitem___325997, order_325995)
        
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_325999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        # Adding element type (line 215)
        int_326000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 33), list_325999, int_326000)
        # Adding element type (line 215)
        int_326001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 33), list_325999, int_326001)
        # Adding element type (line 215)
        int_326002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 33), list_325999, int_326002)
        
        # Processing the call keyword arguments (line 215)
        kwargs_326003 = {}
        # Getting the type of 'assert_equal' (line 215)
        assert_equal_325994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 215)
        assert_equal_call_result_326004 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), assert_equal_325994, *[subscript_call_result_325998, list_325999], **kwargs_326003)
        
        
        # Assigning a Call to a Tuple (line 217):
        
        # Assigning a Subscript to a Name (line 217):
        
        # Obtaining the type of the subscript
        int_326005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'int')
        
        # Call to argrelmin(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'x' (line 217)
        x_326007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'x', False)
        # Processing the call keyword arguments (line 217)
        int_326008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 37), 'int')
        keyword_326009 = int_326008
        kwargs_326010 = {'axis': keyword_326009}
        # Getting the type of 'argrelmin' (line 217)
        argrelmin_326006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 217)
        argrelmin_call_result_326011 = invoke(stypy.reporting.localization.Localization(__file__, 217, 19), argrelmin_326006, *[x_326007], **kwargs_326010)
        
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___326012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), argrelmin_call_result_326011, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_326013 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), getitem___326012, int_326005)
        
        # Assigning a type to the variable 'tuple_var_assignment_324963' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_324963', subscript_call_result_326013)
        
        # Assigning a Subscript to a Name (line 217):
        
        # Obtaining the type of the subscript
        int_326014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'int')
        
        # Call to argrelmin(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'x' (line 217)
        x_326016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'x', False)
        # Processing the call keyword arguments (line 217)
        int_326017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 37), 'int')
        keyword_326018 = int_326017
        kwargs_326019 = {'axis': keyword_326018}
        # Getting the type of 'argrelmin' (line 217)
        argrelmin_326015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'argrelmin', False)
        # Calling argrelmin(args, kwargs) (line 217)
        argrelmin_call_result_326020 = invoke(stypy.reporting.localization.Localization(__file__, 217, 19), argrelmin_326015, *[x_326016], **kwargs_326019)
        
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___326021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), argrelmin_call_result_326020, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_326022 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), getitem___326021, int_326014)
        
        # Assigning a type to the variable 'tuple_var_assignment_324964' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_324964', subscript_call_result_326022)
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'tuple_var_assignment_324963' (line 217)
        tuple_var_assignment_324963_326023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_324963')
        # Assigning a type to the variable 'row' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'row', tuple_var_assignment_324963_326023)
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'tuple_var_assignment_324964' (line 217)
        tuple_var_assignment_324964_326024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_324964')
        # Assigning a type to the variable 'col' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'col', tuple_var_assignment_324964_326024)
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to argsort(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'row' (line 218)
        row_326027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'row', False)
        # Processing the call keyword arguments (line 218)
        kwargs_326028 = {}
        # Getting the type of 'np' (line 218)
        np_326025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'np', False)
        # Obtaining the member 'argsort' of a type (line 218)
        argsort_326026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 16), np_326025, 'argsort')
        # Calling argsort(args, kwargs) (line 218)
        argsort_call_result_326029 = invoke(stypy.reporting.localization.Localization(__file__, 218, 16), argsort_326026, *[row_326027], **kwargs_326028)
        
        # Assigning a type to the variable 'order' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'order', argsort_call_result_326029)
        
        # Call to assert_equal(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 219)
        order_326031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'order', False)
        # Getting the type of 'row' (line 219)
        row_326032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'row', False)
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___326033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 21), row_326032, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_326034 = invoke(stypy.reporting.localization.Localization(__file__, 219, 21), getitem___326033, order_326031)
        
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_326035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        int_326036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 33), list_326035, int_326036)
        # Adding element type (line 219)
        int_326037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 33), list_326035, int_326037)
        # Adding element type (line 219)
        int_326038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 33), list_326035, int_326038)
        
        # Processing the call keyword arguments (line 219)
        kwargs_326039 = {}
        # Getting the type of 'assert_equal' (line 219)
        assert_equal_326030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 219)
        assert_equal_call_result_326040 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), assert_equal_326030, *[subscript_call_result_326034, list_326035], **kwargs_326039)
        
        
        # Call to assert_equal(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 220)
        order_326042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'order', False)
        # Getting the type of 'col' (line 220)
        col_326043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'col', False)
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___326044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), col_326043, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_326045 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), getitem___326044, order_326042)
        
        
        # Obtaining an instance of the builtin type 'list' (line 220)
        list_326046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 220)
        # Adding element type (line 220)
        int_326047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 33), list_326046, int_326047)
        # Adding element type (line 220)
        int_326048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 33), list_326046, int_326048)
        # Adding element type (line 220)
        int_326049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 33), list_326046, int_326049)
        
        # Processing the call keyword arguments (line 220)
        kwargs_326050 = {}
        # Getting the type of 'assert_equal' (line 220)
        assert_equal_326041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 220)
        assert_equal_call_result_326051 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), assert_equal_326041, *[subscript_call_result_326045, list_326046], **kwargs_326050)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_326052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_326052)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_326052


    @norecursion
    def test_highorder(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_highorder'
        module_type_store = module_type_store.open_function_context('test_highorder', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_localization', localization)
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_function_name', 'TestArgrel.test_highorder')
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_param_names_list', [])
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArgrel.test_highorder.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArgrel.test_highorder', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_highorder', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_highorder(...)' code ##################

        
        # Assigning a Num to a Name (line 223):
        
        # Assigning a Num to a Name (line 223):
        int_326053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 16), 'int')
        # Assigning a type to the variable 'order' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'order', int_326053)
        
        # Assigning a List to a Name (line 224):
        
        # Assigning a List to a Name (line 224):
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_326054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        float_326055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 17), list_326054, float_326055)
        # Adding element type (line 224)
        float_326056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 17), list_326054, float_326056)
        # Adding element type (line 224)
        float_326057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 17), list_326054, float_326057)
        # Adding element type (line 224)
        float_326058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 17), list_326054, float_326058)
        # Adding element type (line 224)
        float_326059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 17), list_326054, float_326059)
        
        # Assigning a type to the variable 'sigmas' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'sigmas', list_326054)
        
        # Assigning a Call to a Tuple (line 225):
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_326060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        
        # Call to _gen_gaussians_even(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'sigmas' (line 225)
        sigmas_326062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 50), 'sigmas', False)
        int_326063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 58), 'int')
        # Processing the call keyword arguments (line 225)
        kwargs_326064 = {}
        # Getting the type of '_gen_gaussians_even' (line 225)
        _gen_gaussians_even_326061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 30), '_gen_gaussians_even', False)
        # Calling _gen_gaussians_even(args, kwargs) (line 225)
        _gen_gaussians_even_call_result_326065 = invoke(stypy.reporting.localization.Localization(__file__, 225, 30), _gen_gaussians_even_326061, *[sigmas_326062, int_326063], **kwargs_326064)
        
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___326066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), _gen_gaussians_even_call_result_326065, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_326067 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___326066, int_326060)
        
        # Assigning a type to the variable 'tuple_var_assignment_324965' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_324965', subscript_call_result_326067)
        
        # Assigning a Subscript to a Name (line 225):
        
        # Obtaining the type of the subscript
        int_326068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        
        # Call to _gen_gaussians_even(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'sigmas' (line 225)
        sigmas_326070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 50), 'sigmas', False)
        int_326071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 58), 'int')
        # Processing the call keyword arguments (line 225)
        kwargs_326072 = {}
        # Getting the type of '_gen_gaussians_even' (line 225)
        _gen_gaussians_even_326069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 30), '_gen_gaussians_even', False)
        # Calling _gen_gaussians_even(args, kwargs) (line 225)
        _gen_gaussians_even_call_result_326073 = invoke(stypy.reporting.localization.Localization(__file__, 225, 30), _gen_gaussians_even_326069, *[sigmas_326070, int_326071], **kwargs_326072)
        
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___326074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), _gen_gaussians_even_call_result_326073, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_326075 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), getitem___326074, int_326068)
        
        # Assigning a type to the variable 'tuple_var_assignment_324966' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_324966', subscript_call_result_326075)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_324965' (line 225)
        tuple_var_assignment_324965_326076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_324965')
        # Assigning a type to the variable 'test_data' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'test_data', tuple_var_assignment_324965_326076)
        
        # Assigning a Name to a Name (line 225):
        # Getting the type of 'tuple_var_assignment_324966' (line 225)
        tuple_var_assignment_324966_326077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'tuple_var_assignment_324966')
        # Assigning a type to the variable 'act_locs' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'act_locs', tuple_var_assignment_324966_326077)
        
        # Assigning a BinOp to a Subscript (line 226):
        
        # Assigning a BinOp to a Subscript (line 226):
        
        # Obtaining the type of the subscript
        # Getting the type of 'act_locs' (line 226)
        act_locs_326078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 48), 'act_locs')
        # Getting the type of 'test_data' (line 226)
        test_data_326079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 38), 'test_data')
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___326080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 38), test_data_326079, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_326081 = invoke(stypy.reporting.localization.Localization(__file__, 226, 38), getitem___326080, act_locs_326078)
        
        float_326082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 58), 'float')
        # Applying the binary operator '*' (line 226)
        result_mul_326083 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 38), '*', subscript_call_result_326081, float_326082)
        
        # Getting the type of 'test_data' (line 226)
        test_data_326084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'test_data')
        # Getting the type of 'act_locs' (line 226)
        act_locs_326085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'act_locs')
        # Getting the type of 'order' (line 226)
        order_326086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'order')
        # Applying the binary operator '+' (line 226)
        result_add_326087 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 18), '+', act_locs_326085, order_326086)
        
        # Storing an element on a container (line 226)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 8), test_data_326084, (result_add_326087, result_mul_326083))
        
        # Assigning a BinOp to a Subscript (line 227):
        
        # Assigning a BinOp to a Subscript (line 227):
        
        # Obtaining the type of the subscript
        # Getting the type of 'act_locs' (line 227)
        act_locs_326088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 48), 'act_locs')
        # Getting the type of 'test_data' (line 227)
        test_data_326089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 38), 'test_data')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___326090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 38), test_data_326089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_326091 = invoke(stypy.reporting.localization.Localization(__file__, 227, 38), getitem___326090, act_locs_326088)
        
        float_326092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 58), 'float')
        # Applying the binary operator '*' (line 227)
        result_mul_326093 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 38), '*', subscript_call_result_326091, float_326092)
        
        # Getting the type of 'test_data' (line 227)
        test_data_326094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'test_data')
        # Getting the type of 'act_locs' (line 227)
        act_locs_326095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 18), 'act_locs')
        # Getting the type of 'order' (line 227)
        order_326096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'order')
        # Applying the binary operator '-' (line 227)
        result_sub_326097 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 18), '-', act_locs_326095, order_326096)
        
        # Storing an element on a container (line 227)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 8), test_data_326094, (result_sub_326097, result_mul_326093))
        
        # Assigning a Subscript to a Name (line 228):
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_326098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 70), 'int')
        
        # Call to argrelmax(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'test_data' (line 228)
        test_data_326100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 33), 'test_data', False)
        # Processing the call keyword arguments (line 228)
        # Getting the type of 'order' (line 228)
        order_326101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'order', False)
        keyword_326102 = order_326101
        str_326103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 62), 'str', 'clip')
        keyword_326104 = str_326103
        kwargs_326105 = {'order': keyword_326102, 'mode': keyword_326104}
        # Getting the type of 'argrelmax' (line 228)
        argrelmax_326099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 23), 'argrelmax', False)
        # Calling argrelmax(args, kwargs) (line 228)
        argrelmax_call_result_326106 = invoke(stypy.reporting.localization.Localization(__file__, 228, 23), argrelmax_326099, *[test_data_326100], **kwargs_326105)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___326107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 23), argrelmax_call_result_326106, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_326108 = invoke(stypy.reporting.localization.Localization(__file__, 228, 23), getitem___326107, int_326098)
        
        # Assigning a type to the variable 'rel_max_locs' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'rel_max_locs', subscript_call_result_326108)
        
        # Call to assert_(...): (line 230)
        # Processing the call arguments (line 230)
        
        
        # Call to len(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'rel_max_locs' (line 230)
        rel_max_locs_326111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'rel_max_locs', False)
        # Processing the call keyword arguments (line 230)
        kwargs_326112 = {}
        # Getting the type of 'len' (line 230)
        len_326110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'len', False)
        # Calling len(args, kwargs) (line 230)
        len_call_result_326113 = invoke(stypy.reporting.localization.Localization(__file__, 230, 16), len_326110, *[rel_max_locs_326111], **kwargs_326112)
        
        
        # Call to len(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'act_locs' (line 230)
        act_locs_326115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 41), 'act_locs', False)
        # Processing the call keyword arguments (line 230)
        kwargs_326116 = {}
        # Getting the type of 'len' (line 230)
        len_326114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 37), 'len', False)
        # Calling len(args, kwargs) (line 230)
        len_call_result_326117 = invoke(stypy.reporting.localization.Localization(__file__, 230, 37), len_326114, *[act_locs_326115], **kwargs_326116)
        
        # Applying the binary operator '==' (line 230)
        result_eq_326118 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 16), '==', len_call_result_326113, len_call_result_326117)
        
        # Processing the call keyword arguments (line 230)
        kwargs_326119 = {}
        # Getting the type of 'assert_' (line 230)
        assert__326109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 230)
        assert__call_result_326120 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assert__326109, *[result_eq_326118], **kwargs_326119)
        
        
        # Call to assert_(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Call to all(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_326126 = {}
        
        # Getting the type of 'rel_max_locs' (line 231)
        rel_max_locs_326122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'rel_max_locs', False)
        # Getting the type of 'act_locs' (line 231)
        act_locs_326123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 33), 'act_locs', False)
        # Applying the binary operator '==' (line 231)
        result_eq_326124 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 17), '==', rel_max_locs_326122, act_locs_326123)
        
        # Obtaining the member 'all' of a type (line 231)
        all_326125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 17), result_eq_326124, 'all')
        # Calling all(args, kwargs) (line 231)
        all_call_result_326127 = invoke(stypy.reporting.localization.Localization(__file__, 231, 17), all_326125, *[], **kwargs_326126)
        
        # Processing the call keyword arguments (line 231)
        kwargs_326128 = {}
        # Getting the type of 'assert_' (line 231)
        assert__326121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 231)
        assert__call_result_326129 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), assert__326121, *[all_call_result_326127], **kwargs_326128)
        
        
        # ################# End of 'test_highorder(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_highorder' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_326130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_326130)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_highorder'
        return stypy_return_type_326130


    @norecursion
    def test_2d_gaussians(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2d_gaussians'
        module_type_store = module_type_store.open_function_context('test_2d_gaussians', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_localization', localization)
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_function_name', 'TestArgrel.test_2d_gaussians')
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_param_names_list', [])
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArgrel.test_2d_gaussians.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArgrel.test_2d_gaussians', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2d_gaussians', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2d_gaussians(...)' code ##################

        
        # Assigning a List to a Name (line 234):
        
        # Assigning a List to a Name (line 234):
        
        # Obtaining an instance of the builtin type 'list' (line 234)
        list_326131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 234)
        # Adding element type (line 234)
        float_326132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 17), list_326131, float_326132)
        # Adding element type (line 234)
        float_326133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 17), list_326131, float_326133)
        # Adding element type (line 234)
        float_326134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 17), list_326131, float_326134)
        
        # Assigning a type to the variable 'sigmas' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'sigmas', list_326131)
        
        # Assigning a Call to a Tuple (line 235):
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        int_326135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'int')
        
        # Call to _gen_gaussians_even(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'sigmas' (line 235)
        sigmas_326137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 50), 'sigmas', False)
        int_326138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 58), 'int')
        # Processing the call keyword arguments (line 235)
        kwargs_326139 = {}
        # Getting the type of '_gen_gaussians_even' (line 235)
        _gen_gaussians_even_326136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), '_gen_gaussians_even', False)
        # Calling _gen_gaussians_even(args, kwargs) (line 235)
        _gen_gaussians_even_call_result_326140 = invoke(stypy.reporting.localization.Localization(__file__, 235, 30), _gen_gaussians_even_326136, *[sigmas_326137, int_326138], **kwargs_326139)
        
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___326141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), _gen_gaussians_even_call_result_326140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_326142 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), getitem___326141, int_326135)
        
        # Assigning a type to the variable 'tuple_var_assignment_324967' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_324967', subscript_call_result_326142)
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        int_326143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'int')
        
        # Call to _gen_gaussians_even(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'sigmas' (line 235)
        sigmas_326145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 50), 'sigmas', False)
        int_326146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 58), 'int')
        # Processing the call keyword arguments (line 235)
        kwargs_326147 = {}
        # Getting the type of '_gen_gaussians_even' (line 235)
        _gen_gaussians_even_326144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), '_gen_gaussians_even', False)
        # Calling _gen_gaussians_even(args, kwargs) (line 235)
        _gen_gaussians_even_call_result_326148 = invoke(stypy.reporting.localization.Localization(__file__, 235, 30), _gen_gaussians_even_326144, *[sigmas_326145, int_326146], **kwargs_326147)
        
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___326149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), _gen_gaussians_even_call_result_326148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_326150 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), getitem___326149, int_326143)
        
        # Assigning a type to the variable 'tuple_var_assignment_324968' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_324968', subscript_call_result_326150)
        
        # Assigning a Name to a Name (line 235):
        # Getting the type of 'tuple_var_assignment_324967' (line 235)
        tuple_var_assignment_324967_326151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_324967')
        # Assigning a type to the variable 'test_data' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'test_data', tuple_var_assignment_324967_326151)
        
        # Assigning a Name to a Name (line 235):
        # Getting the type of 'tuple_var_assignment_324968' (line 235)
        tuple_var_assignment_324968_326152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_324968')
        # Assigning a type to the variable 'act_locs' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'act_locs', tuple_var_assignment_324968_326152)
        
        # Assigning a Num to a Name (line 236):
        
        # Assigning a Num to a Name (line 236):
        int_326153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 21), 'int')
        # Assigning a type to the variable 'rot_factor' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'rot_factor', int_326153)
        
        # Assigning a BinOp to a Name (line 237):
        
        # Assigning a BinOp to a Name (line 237):
        
        # Call to arange(...): (line 237)
        # Processing the call arguments (line 237)
        int_326156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 30), 'int')
        
        # Call to len(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'test_data' (line 237)
        test_data_326158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 37), 'test_data', False)
        # Processing the call keyword arguments (line 237)
        kwargs_326159 = {}
        # Getting the type of 'len' (line 237)
        len_326157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'len', False)
        # Calling len(args, kwargs) (line 237)
        len_call_result_326160 = invoke(stypy.reporting.localization.Localization(__file__, 237, 33), len_326157, *[test_data_326158], **kwargs_326159)
        
        # Processing the call keyword arguments (line 237)
        kwargs_326161 = {}
        # Getting the type of 'np' (line 237)
        np_326154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'np', False)
        # Obtaining the member 'arange' of a type (line 237)
        arange_326155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 20), np_326154, 'arange')
        # Calling arange(args, kwargs) (line 237)
        arange_call_result_326162 = invoke(stypy.reporting.localization.Localization(__file__, 237, 20), arange_326155, *[int_326156, len_call_result_326160], **kwargs_326161)
        
        # Getting the type of 'rot_factor' (line 237)
        rot_factor_326163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 51), 'rot_factor')
        # Applying the binary operator '-' (line 237)
        result_sub_326164 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 20), '-', arange_call_result_326162, rot_factor_326163)
        
        # Assigning a type to the variable 'rot_range' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'rot_range', result_sub_326164)
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to vstack(...): (line 238)
        # Processing the call arguments (line 238)
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_326167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        # Getting the type of 'test_data' (line 238)
        test_data_326168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'test_data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 32), list_326167, test_data_326168)
        # Adding element type (line 238)
        
        # Obtaining the type of the subscript
        # Getting the type of 'rot_range' (line 238)
        rot_range_326169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 54), 'rot_range', False)
        # Getting the type of 'test_data' (line 238)
        test_data_326170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 44), 'test_data', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___326171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 44), test_data_326170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 238)
        subscript_call_result_326172 = invoke(stypy.reporting.localization.Localization(__file__, 238, 44), getitem___326171, rot_range_326169)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 32), list_326167, subscript_call_result_326172)
        
        # Processing the call keyword arguments (line 238)
        kwargs_326173 = {}
        # Getting the type of 'np' (line 238)
        np_326165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'np', False)
        # Obtaining the member 'vstack' of a type (line 238)
        vstack_326166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 22), np_326165, 'vstack')
        # Calling vstack(args, kwargs) (line 238)
        vstack_call_result_326174 = invoke(stypy.reporting.localization.Localization(__file__, 238, 22), vstack_326166, *[list_326167], **kwargs_326173)
        
        # Assigning a type to the variable 'test_data_2' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'test_data_2', vstack_call_result_326174)
        
        # Assigning a Call to a Tuple (line 239):
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_326175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 8), 'int')
        
        # Call to argrelmax(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'test_data_2' (line 239)
        test_data_2_326177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 47), 'test_data_2', False)
        # Processing the call keyword arguments (line 239)
        int_326178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 65), 'int')
        keyword_326179 = int_326178
        int_326180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 74), 'int')
        keyword_326181 = int_326180
        kwargs_326182 = {'order': keyword_326181, 'axis': keyword_326179}
        # Getting the type of 'argrelmax' (line 239)
        argrelmax_326176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 37), 'argrelmax', False)
        # Calling argrelmax(args, kwargs) (line 239)
        argrelmax_call_result_326183 = invoke(stypy.reporting.localization.Localization(__file__, 239, 37), argrelmax_326176, *[test_data_2_326177], **kwargs_326182)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___326184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), argrelmax_call_result_326183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_326185 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), getitem___326184, int_326175)
        
        # Assigning a type to the variable 'tuple_var_assignment_324969' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_324969', subscript_call_result_326185)
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_326186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 8), 'int')
        
        # Call to argrelmax(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'test_data_2' (line 239)
        test_data_2_326188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 47), 'test_data_2', False)
        # Processing the call keyword arguments (line 239)
        int_326189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 65), 'int')
        keyword_326190 = int_326189
        int_326191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 74), 'int')
        keyword_326192 = int_326191
        kwargs_326193 = {'order': keyword_326192, 'axis': keyword_326190}
        # Getting the type of 'argrelmax' (line 239)
        argrelmax_326187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 37), 'argrelmax', False)
        # Calling argrelmax(args, kwargs) (line 239)
        argrelmax_call_result_326194 = invoke(stypy.reporting.localization.Localization(__file__, 239, 37), argrelmax_326187, *[test_data_2_326188], **kwargs_326193)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___326195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), argrelmax_call_result_326194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_326196 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), getitem___326195, int_326186)
        
        # Assigning a type to the variable 'tuple_var_assignment_324970' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_324970', subscript_call_result_326196)
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'tuple_var_assignment_324969' (line 239)
        tuple_var_assignment_324969_326197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_324969')
        # Assigning a type to the variable 'rel_max_rows' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'rel_max_rows', tuple_var_assignment_324969_326197)
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'tuple_var_assignment_324970' (line 239)
        tuple_var_assignment_324970_326198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_324970')
        # Assigning a type to the variable 'rel_max_cols' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'rel_max_cols', tuple_var_assignment_324970_326198)
        
        
        # Call to xrange(...): (line 241)
        # Processing the call arguments (line 241)
        int_326200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 25), 'int')
        
        # Obtaining the type of the subscript
        int_326201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 46), 'int')
        # Getting the type of 'test_data_2' (line 241)
        test_data_2_326202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 28), 'test_data_2', False)
        # Obtaining the member 'shape' of a type (line 241)
        shape_326203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 28), test_data_2_326202, 'shape')
        # Obtaining the member '__getitem__' of a type (line 241)
        getitem___326204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 28), shape_326203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 241)
        subscript_call_result_326205 = invoke(stypy.reporting.localization.Localization(__file__, 241, 28), getitem___326204, int_326201)
        
        # Processing the call keyword arguments (line 241)
        kwargs_326206 = {}
        # Getting the type of 'xrange' (line 241)
        xrange_326199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'xrange', False)
        # Calling xrange(args, kwargs) (line 241)
        xrange_call_result_326207 = invoke(stypy.reporting.localization.Localization(__file__, 241, 18), xrange_326199, *[int_326200, subscript_call_result_326205], **kwargs_326206)
        
        # Testing the type of a for loop iterable (line 241)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 241, 8), xrange_call_result_326207)
        # Getting the type of the for loop variable (line 241)
        for_loop_var_326208 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 241, 8), xrange_call_result_326207)
        # Assigning a type to the variable 'rw' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'rw', for_loop_var_326208)
        # SSA begins for a for statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Compare to a Name (line 242):
        
        # Assigning a Compare to a Name (line 242):
        
        # Getting the type of 'rel_max_rows' (line 242)
        rel_max_rows_326209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'rel_max_rows')
        # Getting the type of 'rw' (line 242)
        rw_326210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 36), 'rw')
        # Applying the binary operator '==' (line 242)
        result_eq_326211 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 20), '==', rel_max_rows_326209, rw_326210)
        
        # Assigning a type to the variable 'inds' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'inds', result_eq_326211)
        
        # Call to assert_(...): (line 244)
        # Processing the call arguments (line 244)
        
        
        # Call to len(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Obtaining the type of the subscript
        # Getting the type of 'inds' (line 244)
        inds_326214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 37), 'inds', False)
        # Getting the type of 'rel_max_cols' (line 244)
        rel_max_cols_326215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'rel_max_cols', False)
        # Obtaining the member '__getitem__' of a type (line 244)
        getitem___326216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), rel_max_cols_326215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 244)
        subscript_call_result_326217 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), getitem___326216, inds_326214)
        
        # Processing the call keyword arguments (line 244)
        kwargs_326218 = {}
        # Getting the type of 'len' (line 244)
        len_326213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'len', False)
        # Calling len(args, kwargs) (line 244)
        len_call_result_326219 = invoke(stypy.reporting.localization.Localization(__file__, 244, 20), len_326213, *[subscript_call_result_326217], **kwargs_326218)
        
        
        # Call to len(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'act_locs' (line 244)
        act_locs_326221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 51), 'act_locs', False)
        # Processing the call keyword arguments (line 244)
        kwargs_326222 = {}
        # Getting the type of 'len' (line 244)
        len_326220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 47), 'len', False)
        # Calling len(args, kwargs) (line 244)
        len_call_result_326223 = invoke(stypy.reporting.localization.Localization(__file__, 244, 47), len_326220, *[act_locs_326221], **kwargs_326222)
        
        # Applying the binary operator '==' (line 244)
        result_eq_326224 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 20), '==', len_call_result_326219, len_call_result_326223)
        
        # Processing the call keyword arguments (line 244)
        kwargs_326225 = {}
        # Getting the type of 'assert_' (line 244)
        assert__326212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 244)
        assert__call_result_326226 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), assert__326212, *[result_eq_326224], **kwargs_326225)
        
        
        # Call to assert_(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Call to all(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_326239 = {}
        
        # Getting the type of 'act_locs' (line 245)
        act_locs_326228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'act_locs', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'inds' (line 245)
        inds_326229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 47), 'inds', False)
        # Getting the type of 'rel_max_cols' (line 245)
        rel_max_cols_326230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 34), 'rel_max_cols', False)
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___326231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 34), rel_max_cols_326230, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_326232 = invoke(stypy.reporting.localization.Localization(__file__, 245, 34), getitem___326231, inds_326229)
        
        # Getting the type of 'rot_factor' (line 245)
        rot_factor_326233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 55), 'rot_factor', False)
        # Getting the type of 'rw' (line 245)
        rw_326234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 66), 'rw', False)
        # Applying the binary operator '*' (line 245)
        result_mul_326235 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 55), '*', rot_factor_326233, rw_326234)
        
        # Applying the binary operator '-' (line 245)
        result_sub_326236 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 34), '-', subscript_call_result_326232, result_mul_326235)
        
        # Applying the binary operator '==' (line 245)
        result_eq_326237 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 21), '==', act_locs_326228, result_sub_326236)
        
        # Obtaining the member 'all' of a type (line 245)
        all_326238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 21), result_eq_326237, 'all')
        # Calling all(args, kwargs) (line 245)
        all_call_result_326240 = invoke(stypy.reporting.localization.Localization(__file__, 245, 21), all_326238, *[], **kwargs_326239)
        
        # Processing the call keyword arguments (line 245)
        kwargs_326241 = {}
        # Getting the type of 'assert_' (line 245)
        assert__326227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 245)
        assert__call_result_326242 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), assert__326227, *[all_call_result_326240], **kwargs_326241)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_2d_gaussians(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2d_gaussians' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_326243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_326243)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2d_gaussians'
        return stypy_return_type_326243


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 165, 0, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArgrel.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestArgrel' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'TestArgrel', TestArgrel)
# Declaration of the 'TestFindPeaks' class

class TestFindPeaks(object, ):

    @norecursion
    def test_find_peaks_exact(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_find_peaks_exact'
        module_type_store = module_type_store.open_function_context('test_find_peaks_exact', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_localization', localization)
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_function_name', 'TestFindPeaks.test_find_peaks_exact')
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_param_names_list', [])
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFindPeaks.test_find_peaks_exact.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFindPeaks.test_find_peaks_exact', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_find_peaks_exact', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_find_peaks_exact(...)' code ##################

        str_326244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', '\n        Generate a series of gaussians and attempt to find the peak locations.\n        ')
        
        # Assigning a List to a Name (line 254):
        
        # Assigning a List to a Name (line 254):
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_326245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        float_326246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_326245, float_326246)
        # Adding element type (line 254)
        float_326247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_326245, float_326247)
        # Adding element type (line 254)
        float_326248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_326245, float_326248)
        # Adding element type (line 254)
        float_326249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_326245, float_326249)
        # Adding element type (line 254)
        float_326250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_326245, float_326250)
        # Adding element type (line 254)
        float_326251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_326245, float_326251)
        
        # Assigning a type to the variable 'sigmas' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'sigmas', list_326245)
        
        # Assigning a Num to a Name (line 255):
        
        # Assigning a Num to a Name (line 255):
        int_326252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 21), 'int')
        # Assigning a type to the variable 'num_points' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'num_points', int_326252)
        
        # Assigning a Call to a Tuple (line 256):
        
        # Assigning a Subscript to a Name (line 256):
        
        # Obtaining the type of the subscript
        int_326253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 8), 'int')
        
        # Call to _gen_gaussians_even(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'sigmas' (line 256)
        sigmas_326255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 50), 'sigmas', False)
        # Getting the type of 'num_points' (line 256)
        num_points_326256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 58), 'num_points', False)
        # Processing the call keyword arguments (line 256)
        kwargs_326257 = {}
        # Getting the type of '_gen_gaussians_even' (line 256)
        _gen_gaussians_even_326254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 30), '_gen_gaussians_even', False)
        # Calling _gen_gaussians_even(args, kwargs) (line 256)
        _gen_gaussians_even_call_result_326258 = invoke(stypy.reporting.localization.Localization(__file__, 256, 30), _gen_gaussians_even_326254, *[sigmas_326255, num_points_326256], **kwargs_326257)
        
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___326259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), _gen_gaussians_even_call_result_326258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_326260 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), getitem___326259, int_326253)
        
        # Assigning a type to the variable 'tuple_var_assignment_324971' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tuple_var_assignment_324971', subscript_call_result_326260)
        
        # Assigning a Subscript to a Name (line 256):
        
        # Obtaining the type of the subscript
        int_326261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 8), 'int')
        
        # Call to _gen_gaussians_even(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'sigmas' (line 256)
        sigmas_326263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 50), 'sigmas', False)
        # Getting the type of 'num_points' (line 256)
        num_points_326264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 58), 'num_points', False)
        # Processing the call keyword arguments (line 256)
        kwargs_326265 = {}
        # Getting the type of '_gen_gaussians_even' (line 256)
        _gen_gaussians_even_326262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 30), '_gen_gaussians_even', False)
        # Calling _gen_gaussians_even(args, kwargs) (line 256)
        _gen_gaussians_even_call_result_326266 = invoke(stypy.reporting.localization.Localization(__file__, 256, 30), _gen_gaussians_even_326262, *[sigmas_326263, num_points_326264], **kwargs_326265)
        
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___326267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), _gen_gaussians_even_call_result_326266, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_326268 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), getitem___326267, int_326261)
        
        # Assigning a type to the variable 'tuple_var_assignment_324972' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tuple_var_assignment_324972', subscript_call_result_326268)
        
        # Assigning a Name to a Name (line 256):
        # Getting the type of 'tuple_var_assignment_324971' (line 256)
        tuple_var_assignment_324971_326269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tuple_var_assignment_324971')
        # Assigning a type to the variable 'test_data' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'test_data', tuple_var_assignment_324971_326269)
        
        # Assigning a Name to a Name (line 256):
        # Getting the type of 'tuple_var_assignment_324972' (line 256)
        tuple_var_assignment_324972_326270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'tuple_var_assignment_324972')
        # Assigning a type to the variable 'act_locs' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'act_locs', tuple_var_assignment_324972_326270)
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to arange(...): (line 257)
        # Processing the call arguments (line 257)
        float_326273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 27), 'float')
        
        # Call to max(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'sigmas' (line 257)
        sigmas_326275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 36), 'sigmas', False)
        # Processing the call keyword arguments (line 257)
        kwargs_326276 = {}
        # Getting the type of 'max' (line 257)
        max_326274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 32), 'max', False)
        # Calling max(args, kwargs) (line 257)
        max_call_result_326277 = invoke(stypy.reporting.localization.Localization(__file__, 257, 32), max_326274, *[sigmas_326275], **kwargs_326276)
        
        # Processing the call keyword arguments (line 257)
        kwargs_326278 = {}
        # Getting the type of 'np' (line 257)
        np_326271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 257)
        arange_326272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 17), np_326271, 'arange')
        # Calling arange(args, kwargs) (line 257)
        arange_call_result_326279 = invoke(stypy.reporting.localization.Localization(__file__, 257, 17), arange_326272, *[float_326273, max_call_result_326277], **kwargs_326278)
        
        # Assigning a type to the variable 'widths' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'widths', arange_call_result_326279)
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to find_peaks_cwt(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'test_data' (line 258)
        test_data_326281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 36), 'test_data', False)
        # Getting the type of 'widths' (line 258)
        widths_326282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'widths', False)
        # Processing the call keyword arguments (line 258)
        int_326283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 66), 'int')
        keyword_326284 = int_326283
        int_326285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 77), 'int')
        keyword_326286 = int_326285
        # Getting the type of 'None' (line 259)
        None_326287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 52), 'None', False)
        keyword_326288 = None_326287
        kwargs_326289 = {'min_length': keyword_326288, 'gap_thresh': keyword_326284, 'min_snr': keyword_326286}
        # Getting the type of 'find_peaks_cwt' (line 258)
        find_peaks_cwt_326280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 21), 'find_peaks_cwt', False)
        # Calling find_peaks_cwt(args, kwargs) (line 258)
        find_peaks_cwt_call_result_326290 = invoke(stypy.reporting.localization.Localization(__file__, 258, 21), find_peaks_cwt_326280, *[test_data_326281, widths_326282], **kwargs_326289)
        
        # Assigning a type to the variable 'found_locs' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'found_locs', find_peaks_cwt_call_result_326290)
        
        # Call to assert_array_equal(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'found_locs' (line 260)
        found_locs_326294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 38), 'found_locs', False)
        # Getting the type of 'act_locs' (line 260)
        act_locs_326295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 50), 'act_locs', False)
        str_326296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 24), 'str', 'Found maximum locations did not equal those expected')
        # Processing the call keyword arguments (line 260)
        kwargs_326297 = {}
        # Getting the type of 'np' (line 260)
        np_326291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'np', False)
        # Obtaining the member 'testing' of a type (line 260)
        testing_326292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), np_326291, 'testing')
        # Obtaining the member 'assert_array_equal' of a type (line 260)
        assert_array_equal_326293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), testing_326292, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 260)
        assert_array_equal_call_result_326298 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), assert_array_equal_326293, *[found_locs_326294, act_locs_326295, str_326296], **kwargs_326297)
        
        
        # ################# End of 'test_find_peaks_exact(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_find_peaks_exact' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_326299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_326299)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_find_peaks_exact'
        return stypy_return_type_326299


    @norecursion
    def test_find_peaks_withnoise(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_find_peaks_withnoise'
        module_type_store = module_type_store.open_function_context('test_find_peaks_withnoise', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_localization', localization)
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_function_name', 'TestFindPeaks.test_find_peaks_withnoise')
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_param_names_list', [])
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFindPeaks.test_find_peaks_withnoise.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFindPeaks.test_find_peaks_withnoise', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_find_peaks_withnoise', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_find_peaks_withnoise(...)' code ##################

        str_326300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', '\n        Verify that peak locations are (approximately) found\n        for a series of gaussians with added noise.\n        ')
        
        # Assigning a List to a Name (line 268):
        
        # Assigning a List to a Name (line 268):
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_326301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        # Adding element type (line 268)
        float_326302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 17), list_326301, float_326302)
        # Adding element type (line 268)
        float_326303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 17), list_326301, float_326303)
        # Adding element type (line 268)
        float_326304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 17), list_326301, float_326304)
        # Adding element type (line 268)
        float_326305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 17), list_326301, float_326305)
        # Adding element type (line 268)
        float_326306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 17), list_326301, float_326306)
        # Adding element type (line 268)
        float_326307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 17), list_326301, float_326307)
        
        # Assigning a type to the variable 'sigmas' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'sigmas', list_326301)
        
        # Assigning a Num to a Name (line 269):
        
        # Assigning a Num to a Name (line 269):
        int_326308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 21), 'int')
        # Assigning a type to the variable 'num_points' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'num_points', int_326308)
        
        # Assigning a Call to a Tuple (line 270):
        
        # Assigning a Subscript to a Name (line 270):
        
        # Obtaining the type of the subscript
        int_326309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 8), 'int')
        
        # Call to _gen_gaussians_even(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'sigmas' (line 270)
        sigmas_326311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'sigmas', False)
        # Getting the type of 'num_points' (line 270)
        num_points_326312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 58), 'num_points', False)
        # Processing the call keyword arguments (line 270)
        kwargs_326313 = {}
        # Getting the type of '_gen_gaussians_even' (line 270)
        _gen_gaussians_even_326310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 30), '_gen_gaussians_even', False)
        # Calling _gen_gaussians_even(args, kwargs) (line 270)
        _gen_gaussians_even_call_result_326314 = invoke(stypy.reporting.localization.Localization(__file__, 270, 30), _gen_gaussians_even_326310, *[sigmas_326311, num_points_326312], **kwargs_326313)
        
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___326315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), _gen_gaussians_even_call_result_326314, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_326316 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), getitem___326315, int_326309)
        
        # Assigning a type to the variable 'tuple_var_assignment_324973' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_324973', subscript_call_result_326316)
        
        # Assigning a Subscript to a Name (line 270):
        
        # Obtaining the type of the subscript
        int_326317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 8), 'int')
        
        # Call to _gen_gaussians_even(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'sigmas' (line 270)
        sigmas_326319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'sigmas', False)
        # Getting the type of 'num_points' (line 270)
        num_points_326320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 58), 'num_points', False)
        # Processing the call keyword arguments (line 270)
        kwargs_326321 = {}
        # Getting the type of '_gen_gaussians_even' (line 270)
        _gen_gaussians_even_326318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 30), '_gen_gaussians_even', False)
        # Calling _gen_gaussians_even(args, kwargs) (line 270)
        _gen_gaussians_even_call_result_326322 = invoke(stypy.reporting.localization.Localization(__file__, 270, 30), _gen_gaussians_even_326318, *[sigmas_326319, num_points_326320], **kwargs_326321)
        
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___326323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), _gen_gaussians_even_call_result_326322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_326324 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), getitem___326323, int_326317)
        
        # Assigning a type to the variable 'tuple_var_assignment_324974' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_324974', subscript_call_result_326324)
        
        # Assigning a Name to a Name (line 270):
        # Getting the type of 'tuple_var_assignment_324973' (line 270)
        tuple_var_assignment_324973_326325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_324973')
        # Assigning a type to the variable 'test_data' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'test_data', tuple_var_assignment_324973_326325)
        
        # Assigning a Name to a Name (line 270):
        # Getting the type of 'tuple_var_assignment_324974' (line 270)
        tuple_var_assignment_324974_326326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'tuple_var_assignment_324974')
        # Assigning a type to the variable 'act_locs' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'act_locs', tuple_var_assignment_324974_326326)
        
        # Assigning a Call to a Name (line 271):
        
        # Assigning a Call to a Name (line 271):
        
        # Call to arange(...): (line 271)
        # Processing the call arguments (line 271)
        float_326329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 27), 'float')
        
        # Call to max(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'sigmas' (line 271)
        sigmas_326331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 36), 'sigmas', False)
        # Processing the call keyword arguments (line 271)
        kwargs_326332 = {}
        # Getting the type of 'max' (line 271)
        max_326330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 32), 'max', False)
        # Calling max(args, kwargs) (line 271)
        max_call_result_326333 = invoke(stypy.reporting.localization.Localization(__file__, 271, 32), max_326330, *[sigmas_326331], **kwargs_326332)
        
        # Processing the call keyword arguments (line 271)
        kwargs_326334 = {}
        # Getting the type of 'np' (line 271)
        np_326327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 271)
        arange_326328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 17), np_326327, 'arange')
        # Calling arange(args, kwargs) (line 271)
        arange_call_result_326335 = invoke(stypy.reporting.localization.Localization(__file__, 271, 17), arange_326328, *[float_326329, max_call_result_326333], **kwargs_326334)
        
        # Assigning a type to the variable 'widths' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'widths', arange_call_result_326335)
        
        # Assigning a Num to a Name (line 272):
        
        # Assigning a Num to a Name (line 272):
        float_326336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 20), 'float')
        # Assigning a type to the variable 'noise_amp' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'noise_amp', float_326336)
        
        # Call to seed(...): (line 273)
        # Processing the call arguments (line 273)
        int_326340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 23), 'int')
        # Processing the call keyword arguments (line 273)
        kwargs_326341 = {}
        # Getting the type of 'np' (line 273)
        np_326337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 273)
        random_326338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), np_326337, 'random')
        # Obtaining the member 'seed' of a type (line 273)
        seed_326339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), random_326338, 'seed')
        # Calling seed(args, kwargs) (line 273)
        seed_call_result_326342 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), seed_326339, *[int_326340], **kwargs_326341)
        
        
        # Getting the type of 'test_data' (line 274)
        test_data_326343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'test_data')
        
        # Call to rand(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'num_points' (line 274)
        num_points_326347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), 'num_points', False)
        # Processing the call keyword arguments (line 274)
        kwargs_326348 = {}
        # Getting the type of 'np' (line 274)
        np_326344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'np', False)
        # Obtaining the member 'random' of a type (line 274)
        random_326345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), np_326344, 'random')
        # Obtaining the member 'rand' of a type (line 274)
        rand_326346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), random_326345, 'rand')
        # Calling rand(args, kwargs) (line 274)
        rand_call_result_326349 = invoke(stypy.reporting.localization.Localization(__file__, 274, 22), rand_326346, *[num_points_326347], **kwargs_326348)
        
        float_326350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 51), 'float')
        # Applying the binary operator '-' (line 274)
        result_sub_326351 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 22), '-', rand_call_result_326349, float_326350)
        
        int_326352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 57), 'int')
        # Getting the type of 'noise_amp' (line 274)
        noise_amp_326353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 59), 'noise_amp')
        # Applying the binary operator '*' (line 274)
        result_mul_326354 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 57), '*', int_326352, noise_amp_326353)
        
        # Applying the binary operator '*' (line 274)
        result_mul_326355 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 21), '*', result_sub_326351, result_mul_326354)
        
        # Applying the binary operator '+=' (line 274)
        result_iadd_326356 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 8), '+=', test_data_326343, result_mul_326355)
        # Assigning a type to the variable 'test_data' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'test_data', result_iadd_326356)
        
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to find_peaks_cwt(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'test_data' (line 275)
        test_data_326358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 36), 'test_data', False)
        # Getting the type of 'widths' (line 275)
        widths_326359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 47), 'widths', False)
        # Processing the call keyword arguments (line 275)
        int_326360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 66), 'int')
        keyword_326361 = int_326360
        int_326362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 52), 'int')
        keyword_326363 = int_326362
        # Getting the type of 'noise_amp' (line 276)
        noise_amp_326364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 63), 'noise_amp', False)
        int_326365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 75), 'int')
        # Applying the binary operator 'div' (line 276)
        result_div_326366 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 63), 'div', noise_amp_326364, int_326365)
        
        keyword_326367 = result_div_326366
        kwargs_326368 = {'min_length': keyword_326361, 'gap_thresh': keyword_326363, 'min_snr': keyword_326367}
        # Getting the type of 'find_peaks_cwt' (line 275)
        find_peaks_cwt_326357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'find_peaks_cwt', False)
        # Calling find_peaks_cwt(args, kwargs) (line 275)
        find_peaks_cwt_call_result_326369 = invoke(stypy.reporting.localization.Localization(__file__, 275, 21), find_peaks_cwt_326357, *[test_data_326358, widths_326359], **kwargs_326368)
        
        # Assigning a type to the variable 'found_locs' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'found_locs', find_peaks_cwt_call_result_326369)
        
        # Call to assert_equal(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Call to len(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'found_locs' (line 278)
        found_locs_326374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 36), 'found_locs', False)
        # Processing the call keyword arguments (line 278)
        kwargs_326375 = {}
        # Getting the type of 'len' (line 278)
        len_326373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 32), 'len', False)
        # Calling len(args, kwargs) (line 278)
        len_call_result_326376 = invoke(stypy.reporting.localization.Localization(__file__, 278, 32), len_326373, *[found_locs_326374], **kwargs_326375)
        
        
        # Call to len(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'act_locs' (line 278)
        act_locs_326378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 53), 'act_locs', False)
        # Processing the call keyword arguments (line 278)
        kwargs_326379 = {}
        # Getting the type of 'len' (line 278)
        len_326377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 49), 'len', False)
        # Calling len(args, kwargs) (line 278)
        len_call_result_326380 = invoke(stypy.reporting.localization.Localization(__file__, 278, 49), len_326377, *[act_locs_326378], **kwargs_326379)
        
        str_326381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 64), 'str', 'Different number')
        str_326382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 32), 'str', 'of peaks found than expected')
        # Applying the binary operator '+' (line 278)
        result_add_326383 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 64), '+', str_326381, str_326382)
        
        # Processing the call keyword arguments (line 278)
        kwargs_326384 = {}
        # Getting the type of 'np' (line 278)
        np_326370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'np', False)
        # Obtaining the member 'testing' of a type (line 278)
        testing_326371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), np_326370, 'testing')
        # Obtaining the member 'assert_equal' of a type (line 278)
        assert_equal_326372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), testing_326371, 'assert_equal')
        # Calling assert_equal(args, kwargs) (line 278)
        assert_equal_call_result_326385 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), assert_equal_326372, *[len_call_result_326376, len_call_result_326380, result_add_326383], **kwargs_326384)
        
        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to abs(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'found_locs' (line 280)
        found_locs_326388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'found_locs', False)
        # Getting the type of 'act_locs' (line 280)
        act_locs_326389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 36), 'act_locs', False)
        # Applying the binary operator '-' (line 280)
        result_sub_326390 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 23), '-', found_locs_326388, act_locs_326389)
        
        # Processing the call keyword arguments (line 280)
        kwargs_326391 = {}
        # Getting the type of 'np' (line 280)
        np_326386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'np', False)
        # Obtaining the member 'abs' of a type (line 280)
        abs_326387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), np_326386, 'abs')
        # Calling abs(args, kwargs) (line 280)
        abs_call_result_326392 = invoke(stypy.reporting.localization.Localization(__file__, 280, 16), abs_326387, *[result_sub_326390], **kwargs_326391)
        
        # Assigning a type to the variable 'diffs' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'diffs', abs_call_result_326392)
        
        # Assigning a BinOp to a Name (line 281):
        
        # Assigning a BinOp to a Name (line 281):
        
        # Call to array(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'sigmas' (line 281)
        sigmas_326395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 29), 'sigmas', False)
        # Processing the call keyword arguments (line 281)
        kwargs_326396 = {}
        # Getting the type of 'np' (line 281)
        np_326393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 281)
        array_326394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 20), np_326393, 'array')
        # Calling array(args, kwargs) (line 281)
        array_call_result_326397 = invoke(stypy.reporting.localization.Localization(__file__, 281, 20), array_326394, *[sigmas_326395], **kwargs_326396)
        
        int_326398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 39), 'int')
        # Applying the binary operator 'div' (line 281)
        result_div_326399 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 20), 'div', array_call_result_326397, int_326398)
        
        # Assigning a type to the variable 'max_diffs' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'max_diffs', result_div_326399)
        
        # Call to assert_array_less(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'diffs' (line 282)
        diffs_326403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 37), 'diffs', False)
        # Getting the type of 'max_diffs' (line 282)
        max_diffs_326404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 44), 'max_diffs', False)
        str_326405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 55), 'str', 'Maximum location differed')
        str_326406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 37), 'str', 'by more than %s')
        # Getting the type of 'max_diffs' (line 283)
        max_diffs_326407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 58), 'max_diffs', False)
        # Applying the binary operator '%' (line 283)
        result_mod_326408 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 37), '%', str_326406, max_diffs_326407)
        
        # Applying the binary operator '+' (line 282)
        result_add_326409 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 55), '+', str_326405, result_mod_326408)
        
        # Processing the call keyword arguments (line 282)
        kwargs_326410 = {}
        # Getting the type of 'np' (line 282)
        np_326400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'np', False)
        # Obtaining the member 'testing' of a type (line 282)
        testing_326401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), np_326400, 'testing')
        # Obtaining the member 'assert_array_less' of a type (line 282)
        assert_array_less_326402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), testing_326401, 'assert_array_less')
        # Calling assert_array_less(args, kwargs) (line 282)
        assert_array_less_call_result_326411 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), assert_array_less_326402, *[diffs_326403, max_diffs_326404, result_add_326409], **kwargs_326410)
        
        
        # ################# End of 'test_find_peaks_withnoise(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_find_peaks_withnoise' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_326412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_326412)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_find_peaks_withnoise'
        return stypy_return_type_326412


    @norecursion
    def test_find_peaks_nopeak(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_find_peaks_nopeak'
        module_type_store = module_type_store.open_function_context('test_find_peaks_nopeak', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_localization', localization)
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_function_name', 'TestFindPeaks.test_find_peaks_nopeak')
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_param_names_list', [])
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFindPeaks.test_find_peaks_nopeak.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFindPeaks.test_find_peaks_nopeak', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_find_peaks_nopeak', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_find_peaks_nopeak(...)' code ##################

        str_326413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', "\n        Verify that no peak is found in\n        data that's just noise.\n        ")
        
        # Assigning a Num to a Name (line 290):
        
        # Assigning a Num to a Name (line 290):
        float_326414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 20), 'float')
        # Assigning a type to the variable 'noise_amp' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'noise_amp', float_326414)
        
        # Assigning a Num to a Name (line 291):
        
        # Assigning a Num to a Name (line 291):
        int_326415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 21), 'int')
        # Assigning a type to the variable 'num_points' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'num_points', int_326415)
        
        # Call to seed(...): (line 292)
        # Processing the call arguments (line 292)
        int_326419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 23), 'int')
        # Processing the call keyword arguments (line 292)
        kwargs_326420 = {}
        # Getting the type of 'np' (line 292)
        np_326416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 292)
        random_326417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), np_326416, 'random')
        # Obtaining the member 'seed' of a type (line 292)
        seed_326418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), random_326417, 'seed')
        # Calling seed(args, kwargs) (line 292)
        seed_call_result_326421 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), seed_326418, *[int_326419], **kwargs_326420)
        
        
        # Assigning a BinOp to a Name (line 293):
        
        # Assigning a BinOp to a Name (line 293):
        
        # Call to rand(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'num_points' (line 293)
        num_points_326425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 36), 'num_points', False)
        # Processing the call keyword arguments (line 293)
        kwargs_326426 = {}
        # Getting the type of 'np' (line 293)
        np_326422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 21), 'np', False)
        # Obtaining the member 'random' of a type (line 293)
        random_326423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 21), np_326422, 'random')
        # Obtaining the member 'rand' of a type (line 293)
        rand_326424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 21), random_326423, 'rand')
        # Calling rand(args, kwargs) (line 293)
        rand_call_result_326427 = invoke(stypy.reporting.localization.Localization(__file__, 293, 21), rand_326424, *[num_points_326425], **kwargs_326426)
        
        float_326428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 50), 'float')
        # Applying the binary operator '-' (line 293)
        result_sub_326429 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 21), '-', rand_call_result_326427, float_326428)
        
        int_326430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 56), 'int')
        # Getting the type of 'noise_amp' (line 293)
        noise_amp_326431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 58), 'noise_amp')
        # Applying the binary operator '*' (line 293)
        result_mul_326432 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 56), '*', int_326430, noise_amp_326431)
        
        # Applying the binary operator '*' (line 293)
        result_mul_326433 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 20), '*', result_sub_326429, result_mul_326432)
        
        # Assigning a type to the variable 'test_data' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'test_data', result_mul_326433)
        
        # Assigning a Call to a Name (line 294):
        
        # Assigning a Call to a Name (line 294):
        
        # Call to arange(...): (line 294)
        # Processing the call arguments (line 294)
        int_326436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 27), 'int')
        int_326437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 31), 'int')
        # Processing the call keyword arguments (line 294)
        kwargs_326438 = {}
        # Getting the type of 'np' (line 294)
        np_326434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'np', False)
        # Obtaining the member 'arange' of a type (line 294)
        arange_326435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 17), np_326434, 'arange')
        # Calling arange(args, kwargs) (line 294)
        arange_call_result_326439 = invoke(stypy.reporting.localization.Localization(__file__, 294, 17), arange_326435, *[int_326436, int_326437], **kwargs_326438)
        
        # Assigning a type to the variable 'widths' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'widths', arange_call_result_326439)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to find_peaks_cwt(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'test_data' (line 295)
        test_data_326441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 36), 'test_data', False)
        # Getting the type of 'widths' (line 295)
        widths_326442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 47), 'widths', False)
        # Processing the call keyword arguments (line 295)
        int_326443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 63), 'int')
        keyword_326444 = int_326443
        int_326445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 77), 'int')
        keyword_326446 = int_326445
        kwargs_326447 = {'min_snr': keyword_326444, 'noise_perc': keyword_326446}
        # Getting the type of 'find_peaks_cwt' (line 295)
        find_peaks_cwt_326440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'find_peaks_cwt', False)
        # Calling find_peaks_cwt(args, kwargs) (line 295)
        find_peaks_cwt_call_result_326448 = invoke(stypy.reporting.localization.Localization(__file__, 295, 21), find_peaks_cwt_326440, *[test_data_326441, widths_326442], **kwargs_326447)
        
        # Assigning a type to the variable 'found_locs' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'found_locs', find_peaks_cwt_call_result_326448)
        
        # Call to assert_equal(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Call to len(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'found_locs' (line 296)
        found_locs_326453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 36), 'found_locs', False)
        # Processing the call keyword arguments (line 296)
        kwargs_326454 = {}
        # Getting the type of 'len' (line 296)
        len_326452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 32), 'len', False)
        # Calling len(args, kwargs) (line 296)
        len_call_result_326455 = invoke(stypy.reporting.localization.Localization(__file__, 296, 32), len_326452, *[found_locs_326453], **kwargs_326454)
        
        int_326456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 49), 'int')
        # Processing the call keyword arguments (line 296)
        kwargs_326457 = {}
        # Getting the type of 'np' (line 296)
        np_326449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'np', False)
        # Obtaining the member 'testing' of a type (line 296)
        testing_326450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), np_326449, 'testing')
        # Obtaining the member 'assert_equal' of a type (line 296)
        assert_equal_326451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), testing_326450, 'assert_equal')
        # Calling assert_equal(args, kwargs) (line 296)
        assert_equal_call_result_326458 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), assert_equal_326451, *[len_call_result_326455, int_326456], **kwargs_326457)
        
        
        # ################# End of 'test_find_peaks_nopeak(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_find_peaks_nopeak' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_326459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_326459)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_find_peaks_nopeak'
        return stypy_return_type_326459


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 248, 0, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFindPeaks.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFindPeaks' (line 248)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'TestFindPeaks', TestFindPeaks)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

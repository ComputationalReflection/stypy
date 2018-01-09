
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os.path
4: import tempfile
5: import shutil
6: import numpy as np
7: import glob
8: 
9: import pytest
10: from pytest import raises as assert_raises
11: from numpy.testing import (assert_equal, assert_allclose,
12:                            assert_array_equal, assert_)
13: from scipy._lib._numpy_compat import suppress_warnings
14: from scipy import misc
15: from numpy.ma.testutils import assert_mask_equal
16: 
17: try:
18:     import PIL.Image
19: except ImportError:
20:     _have_PIL = False
21: else:
22:     _have_PIL = True
23: 
24: 
25: # Function / method decorator for skipping PIL tests on import failure
26: _pilskip = pytest.mark.skipif(not _have_PIL, reason='Need to import PIL for this test')
27: 
28: datapath = os.path.dirname(__file__)
29: 
30: @_pilskip
31: class TestPILUtil(object):
32:     def test_imresize(self):
33:         im = np.random.random((10, 20))
34:         for T in np.sctypes['float'] + [float]:
35:             # 1.1 rounds to below 1.1 for float16, 1.101 works
36:             with suppress_warnings() as sup:
37:                 sup.filter(DeprecationWarning)
38:                 im1 = misc.imresize(im, T(1.101))
39:             assert_equal(im1.shape, (11, 22))
40: 
41:     def test_imresize2(self):
42:         im = np.random.random((20, 30))
43:         with suppress_warnings() as sup:
44:             sup.filter(DeprecationWarning)
45:             im2 = misc.imresize(im, (30, 40), interp='bicubic')
46:         assert_equal(im2.shape, (30, 40))
47: 
48:     def test_imresize3(self):
49:         im = np.random.random((15, 30))
50:         with suppress_warnings() as sup:
51:             sup.filter(DeprecationWarning)
52:             im2 = misc.imresize(im, (30, 60), interp='nearest')
53:         assert_equal(im2.shape, (30, 60))
54: 
55:     def test_imresize4(self):
56:         im = np.array([[1, 2],
57:                        [3, 4]])
58:         # Check that resizing by target size, float and int are the same
59:         with suppress_warnings() as sup:
60:             sup.filter(DeprecationWarning)
61:             im2 = misc.imresize(im, (4, 4), mode='F')  # output size
62:             im3 = misc.imresize(im, 2., mode='F')  # fraction
63:             im4 = misc.imresize(im, 200, mode='F')  # percentage
64:         assert_equal(im2, im3)
65:         assert_equal(im2, im4)
66: 
67:     def test_imresize5(self):
68:         im = np.random.random((25, 15))
69:         with suppress_warnings() as sup:
70:             sup.filter(DeprecationWarning)
71:             im2 = misc.imresize(im, (30, 60), interp='lanczos')
72:         assert_equal(im2.shape, (30, 60))
73: 
74:     def test_bytescale(self):
75:         x = np.array([0, 1, 2], np.uint8)
76:         y = np.array([0, 1, 2])
77:         with suppress_warnings() as sup:
78:             sup.filter(DeprecationWarning)
79:             assert_equal(misc.bytescale(x), x)
80:             assert_equal(misc.bytescale(y), [0, 128, 255])
81: 
82:     def test_bytescale_keywords(self):
83:         x = np.array([40, 60, 120, 200, 300, 500])
84:         with suppress_warnings() as sup:
85:             sup.filter(DeprecationWarning)
86:             res_lowhigh = misc.bytescale(x, low=10, high=143)
87:             assert_equal(res_lowhigh, [10, 16, 33, 56, 85, 143])
88:             res_cmincmax = misc.bytescale(x, cmin=60, cmax=300)
89:             assert_equal(res_cmincmax, [0, 0, 64, 149, 255, 255])
90:             assert_equal(misc.bytescale(np.array([3, 3, 3]), low=4), [4, 4, 4])
91: 
92:     def test_bytescale_cscale_lowhigh(self):
93:         a = np.arange(10)
94:         with suppress_warnings() as sup:
95:             sup.filter(DeprecationWarning)
96:             actual = misc.bytescale(a, cmin=3, cmax=6, low=100, high=200)
97:         expected = [100, 100, 100, 100, 133, 167, 200, 200, 200, 200]
98:         assert_equal(actual, expected)
99: 
100:     def test_bytescale_mask(self):
101:         a = np.ma.MaskedArray(data=[1, 2, 3], mask=[False, False, True])
102:         with suppress_warnings() as sup:
103:             sup.filter(DeprecationWarning)
104:             actual = misc.bytescale(a)
105:         expected = [0, 255, 3]
106:         assert_equal(expected, actual)
107:         assert_mask_equal(a.mask, actual.mask)
108:         assert_(isinstance(actual, np.ma.MaskedArray))
109: 
110:     def test_bytescale_rounding(self):
111:         a = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
112:         with suppress_warnings() as sup:
113:             sup.filter(DeprecationWarning)
114:             actual = misc.bytescale(a, cmin=0, cmax=10, low=0, high=10)
115:         expected = [0, 1, 2, 3, 4]
116:         assert_equal(actual, expected)
117: 
118:     def test_bytescale_low_greaterthan_high(self):
119:         with assert_raises(ValueError):
120:             with suppress_warnings() as sup:
121:                 sup.filter(DeprecationWarning)
122:                 misc.bytescale(np.arange(3), low=10, high=5)
123: 
124:     def test_bytescale_low_lessthan_0(self):
125:         with assert_raises(ValueError):
126:             with suppress_warnings() as sup:
127:                 sup.filter(DeprecationWarning)
128:                 misc.bytescale(np.arange(3), low=-1)
129: 
130:     def test_bytescale_high_greaterthan_255(self):
131:         with assert_raises(ValueError):
132:             with suppress_warnings() as sup:
133:                 sup.filter(DeprecationWarning)
134:                 misc.bytescale(np.arange(3), high=256)
135: 
136:     def test_bytescale_low_equals_high(self):
137:         a = np.arange(3)
138:         with suppress_warnings() as sup:
139:             sup.filter(DeprecationWarning)
140:             actual = misc.bytescale(a, low=10, high=10)
141:         expected = [10, 10, 10]
142:         assert_equal(actual, expected)
143: 
144:     def test_imsave(self):
145:         picdir = os.path.join(datapath, "data")
146:         for png in glob.iglob(picdir + "/*.png"):
147:             with suppress_warnings() as sup:
148:                 # PIL causes a Py3k ResourceWarning
149:                 sup.filter(message="unclosed file")
150:                 sup.filter(DeprecationWarning)
151:                 img = misc.imread(png)
152:             tmpdir = tempfile.mkdtemp()
153:             try:
154:                 fn1 = os.path.join(tmpdir, 'test.png')
155:                 fn2 = os.path.join(tmpdir, 'testimg')
156:                 with suppress_warnings() as sup:
157:                     # PIL causes a Py3k ResourceWarning
158:                     sup.filter(message="unclosed file")
159:                     sup.filter(DeprecationWarning)
160:                     misc.imsave(fn1, img)
161:                     misc.imsave(fn2, img, 'PNG')
162: 
163:                 with suppress_warnings() as sup:
164:                     # PIL causes a Py3k ResourceWarning
165:                     sup.filter(message="unclosed file")
166:                     sup.filter(DeprecationWarning)
167:                     data1 = misc.imread(fn1)
168:                     data2 = misc.imread(fn2)
169:                 assert_allclose(data1, img)
170:                 assert_allclose(data2, img)
171:                 assert_equal(data1.shape, img.shape)
172:                 assert_equal(data2.shape, img.shape)
173:             finally:
174:                 shutil.rmtree(tmpdir)
175: 
176: 
177: def check_fromimage(filename, irange, shape):
178:     fp = open(filename, "rb")
179:     with suppress_warnings() as sup:
180:         sup.filter(DeprecationWarning)
181:         img = misc.fromimage(PIL.Image.open(fp))
182:     fp.close()
183:     imin, imax = irange
184:     assert_equal(img.min(), imin)
185:     assert_equal(img.max(), imax)
186:     assert_equal(img.shape, shape)
187: 
188: 
189: @_pilskip
190: def test_fromimage():
191:     # Test generator for parametric tests
192:     # Tuples in the list are (filename, (datamin, datamax), shape).
193:     files = [('icon.png', (0, 255), (48, 48, 4)),
194:              ('icon_mono.png', (0, 255), (48, 48, 4)),
195:              ('icon_mono_flat.png', (0, 255), (48, 48, 3))]
196:     for fn, irange, shape in files:
197:         with suppress_warnings() as sup:
198:             sup.filter(DeprecationWarning)
199:             check_fromimage(os.path.join(datapath, 'data', fn), irange, shape)
200: 
201: 
202: @_pilskip
203: def test_imread_indexed_png():
204:     # The file `foo3x5x4indexed.png` was created with this array
205:     # (3x5 is (height)x(width)):
206:     data = np.array([[[127, 0, 255, 255],
207:                       [127, 0, 255, 255],
208:                       [127, 0, 255, 255],
209:                       [127, 0, 255, 255],
210:                       [127, 0, 255, 255]],
211:                      [[192, 192, 255, 0],
212:                       [192, 192, 255, 0],
213:                       [0, 0, 255, 0],
214:                       [0, 0, 255, 0],
215:                       [0, 0, 255, 0]],
216:                      [[0, 31, 255, 255],
217:                       [0, 31, 255, 255],
218:                       [0, 31, 255, 255],
219:                       [0, 31, 255, 255],
220:                       [0, 31, 255, 255]]], dtype=np.uint8)
221: 
222:     filename = os.path.join(datapath, 'data', 'foo3x5x4indexed.png')
223:     with open(filename, 'rb') as f:
224:         with suppress_warnings() as sup:
225:             sup.filter(DeprecationWarning)
226:             im = misc.imread(f)
227:     assert_array_equal(im, data)
228: 
229: 
230: @_pilskip
231: def test_imread_1bit():
232:     # box1.png is a 48x48 grayscale image with bit depth 1.
233:     # The border pixels are 1 and the rest are 0.
234:     filename = os.path.join(datapath, 'data', 'box1.png')
235:     with open(filename, 'rb') as f:
236:         with suppress_warnings() as sup:
237:             sup.filter(DeprecationWarning)
238:             im = misc.imread(f)
239:     assert_equal(im.dtype, np.uint8)
240:     expected = np.zeros((48, 48), dtype=np.uint8)
241:     # When scaled up from 1 bit to 8 bits, 1 becomes 255.
242:     expected[:, 0] = 255
243:     expected[:, -1] = 255
244:     expected[0, :] = 255
245:     expected[-1, :] = 255
246:     assert_equal(im, expected)
247: 
248: 
249: @_pilskip
250: def test_imread_2bit():
251:     # blocks2bit.png is a 12x12 grayscale image with bit depth 2.
252:     # The pattern is 4 square subblocks of size 6x6.  Upper left
253:     # is all 0, upper right is all 1, lower left is all 2, lower
254:     # right is all 3.
255:     # When scaled up to 8 bits, the values become [0, 85, 170, 255].
256:     filename = os.path.join(datapath, 'data', 'blocks2bit.png')
257:     with open(filename, 'rb') as f:
258:         with suppress_warnings() as sup:
259:             sup.filter(DeprecationWarning)
260:             im = misc.imread(f)
261:     assert_equal(im.dtype, np.uint8)
262:     expected = np.zeros((12, 12), dtype=np.uint8)
263:     expected[:6, 6:] = 85
264:     expected[6:, :6] = 170
265:     expected[6:, 6:] = 255
266:     assert_equal(im, expected)
267: 
268: 
269: @_pilskip
270: def test_imread_4bit():
271:     # pattern4bit.png is a 12(h) x 31(w) grayscale image with bit depth 4.
272:     # The value in row j and column i is maximum(j, i) % 16.
273:     # When scaled up to 8 bits, the values become [0, 17, 34, ..., 255].
274:     filename = os.path.join(datapath, 'data', 'pattern4bit.png')
275:     with open(filename, 'rb') as f:
276:         with suppress_warnings() as sup:
277:             sup.filter(DeprecationWarning)
278:             im = misc.imread(f)
279:     assert_equal(im.dtype, np.uint8)
280:     j, i = np.meshgrid(np.arange(12), np.arange(31), indexing='ij')
281:     expected = 17*(np.maximum(j, i) % 16).astype(np.uint8)
282:     assert_equal(im, expected)
283: 
284: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os.path' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_115666) is not StypyTypeError):

    if (import_115666 != 'pyd_module'):
        __import__(import_115666)
        sys_modules_115667 = sys.modules[import_115666]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_115667.module_type_store, module_type_store)
    else:
        import os.path

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', os.path, module_type_store)

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_115666)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import tempfile' statement (line 4)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import shutil' statement (line 5)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115668 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_115668) is not StypyTypeError):

    if (import_115668 != 'pyd_module'):
        __import__(import_115668)
        sys_modules_115669 = sys.modules[import_115668]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_115669.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_115668)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import glob' statement (line 7)
import glob

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'glob', glob, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import pytest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115670 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_115670) is not StypyTypeError):

    if (import_115670 != 'pyd_module'):
        __import__(import_115670)
        sys_modules_115671 = sys.modules[import_115670]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_115671.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_115670)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from pytest import assert_raises' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115672 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest')

if (type(import_115672) is not StypyTypeError):

    if (import_115672 != 'pyd_module'):
        __import__(import_115672)
        sys_modules_115673 = sys.modules[import_115672]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', sys_modules_115673.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_115673, sys_modules_115673.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', import_115672)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.testing import assert_equal, assert_allclose, assert_array_equal, assert_' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing')

if (type(import_115674) is not StypyTypeError):

    if (import_115674 != 'pyd_module'):
        __import__(import_115674)
        sys_modules_115675 = sys.modules[import_115674]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', sys_modules_115675.module_type_store, module_type_store, ['assert_equal', 'assert_allclose', 'assert_array_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_115675, sys_modules_115675.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose, assert_array_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose', 'assert_array_equal', 'assert_'], [assert_equal, assert_allclose, assert_array_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.testing', import_115674)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115676 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._numpy_compat')

if (type(import_115676) is not StypyTypeError):

    if (import_115676 != 'pyd_module'):
        __import__(import_115676)
        sys_modules_115677 = sys.modules[import_115676]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._numpy_compat', sys_modules_115677.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_115677, sys_modules_115677.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._numpy_compat', import_115676)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy import misc' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115678 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy')

if (type(import_115678) is not StypyTypeError):

    if (import_115678 != 'pyd_module'):
        __import__(import_115678)
        sys_modules_115679 = sys.modules[import_115678]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy', sys_modules_115679.module_type_store, module_type_store, ['misc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_115679, sys_modules_115679.module_type_store, module_type_store)
    else:
        from scipy import misc

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy', None, module_type_store, ['misc'], [misc])

else:
    # Assigning a type to the variable 'scipy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy', import_115678)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.ma.testutils import assert_mask_equal' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115680 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.ma.testutils')

if (type(import_115680) is not StypyTypeError):

    if (import_115680 != 'pyd_module'):
        __import__(import_115680)
        sys_modules_115681 = sys.modules[import_115680]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.ma.testutils', sys_modules_115681.module_type_store, module_type_store, ['assert_mask_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_115681, sys_modules_115681.module_type_store, module_type_store)
    else:
        from numpy.ma.testutils import assert_mask_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.ma.testutils', None, module_type_store, ['assert_mask_equal'], [assert_mask_equal])

else:
    # Assigning a type to the variable 'numpy.ma.testutils' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.ma.testutils', import_115680)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')



# SSA begins for try-except statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))

# 'import PIL.Image' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'PIL.Image')

if (type(import_115682) is not StypyTypeError):

    if (import_115682 != 'pyd_module'):
        __import__(import_115682)
        sys_modules_115683 = sys.modules[import_115682]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'PIL.Image', sys_modules_115683.module_type_store, module_type_store)
    else:
        import PIL.Image

        import_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'PIL.Image', PIL.Image, module_type_store)

else:
    # Assigning a type to the variable 'PIL.Image' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'PIL.Image', import_115682)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

# SSA branch for the except part of a try statement (line 17)
# SSA branch for the except 'ImportError' branch of a try statement (line 17)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 20):

# Assigning a Name to a Name (line 20):
# Getting the type of 'False' (line 20)
False_115684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'False')
# Assigning a type to the variable '_have_PIL' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), '_have_PIL', False_115684)
# SSA branch for the else branch of a try statement (line 17)
module_type_store.open_ssa_branch('except else')

# Assigning a Name to a Name (line 22):

# Assigning a Name to a Name (line 22):
# Getting the type of 'True' (line 22)
True_115685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'True')
# Assigning a type to the variable '_have_PIL' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), '_have_PIL', True_115685)
# SSA join for try-except statement (line 17)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 26):

# Assigning a Call to a Name (line 26):

# Call to skipif(...): (line 26)
# Processing the call arguments (line 26)

# Getting the type of '_have_PIL' (line 26)
_have_PIL_115689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 34), '_have_PIL', False)
# Applying the 'not' unary operator (line 26)
result_not__115690 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 30), 'not', _have_PIL_115689)

# Processing the call keyword arguments (line 26)
str_115691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 52), 'str', 'Need to import PIL for this test')
keyword_115692 = str_115691
kwargs_115693 = {'reason': keyword_115692}
# Getting the type of 'pytest' (line 26)
pytest_115686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'pytest', False)
# Obtaining the member 'mark' of a type (line 26)
mark_115687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 11), pytest_115686, 'mark')
# Obtaining the member 'skipif' of a type (line 26)
skipif_115688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 11), mark_115687, 'skipif')
# Calling skipif(args, kwargs) (line 26)
skipif_call_result_115694 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), skipif_115688, *[result_not__115690], **kwargs_115693)

# Assigning a type to the variable '_pilskip' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_pilskip', skipif_call_result_115694)

# Assigning a Call to a Name (line 28):

# Assigning a Call to a Name (line 28):

# Call to dirname(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of '__file__' (line 28)
file___115698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 27), '__file__', False)
# Processing the call keyword arguments (line 28)
kwargs_115699 = {}
# Getting the type of 'os' (line 28)
os_115695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'os', False)
# Obtaining the member 'path' of a type (line 28)
path_115696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), os_115695, 'path')
# Obtaining the member 'dirname' of a type (line 28)
dirname_115697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), path_115696, 'dirname')
# Calling dirname(args, kwargs) (line 28)
dirname_call_result_115700 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), dirname_115697, *[file___115698], **kwargs_115699)

# Assigning a type to the variable 'datapath' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'datapath', dirname_call_result_115700)
# Declaration of the 'TestPILUtil' class
# Getting the type of '_pilskip' (line 30)
_pilskip_115701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_pilskip')

class TestPILUtil(object, ):

    @norecursion
    def test_imresize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_imresize'
        module_type_store = module_type_store.open_function_context('test_imresize', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_imresize')
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_imresize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_imresize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_imresize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_imresize(...)' code ##################

        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to random(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_115705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        int_115706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 31), tuple_115705, int_115706)
        # Adding element type (line 33)
        int_115707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 31), tuple_115705, int_115707)
        
        # Processing the call keyword arguments (line 33)
        kwargs_115708 = {}
        # Getting the type of 'np' (line 33)
        np_115702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'np', False)
        # Obtaining the member 'random' of a type (line 33)
        random_115703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), np_115702, 'random')
        # Obtaining the member 'random' of a type (line 33)
        random_115704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), random_115703, 'random')
        # Calling random(args, kwargs) (line 33)
        random_call_result_115709 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), random_115704, *[tuple_115705], **kwargs_115708)
        
        # Assigning a type to the variable 'im' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'im', random_call_result_115709)
        
        
        # Obtaining the type of the subscript
        str_115710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'str', 'float')
        # Getting the type of 'np' (line 34)
        np_115711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'np')
        # Obtaining the member 'sctypes' of a type (line 34)
        sctypes_115712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), np_115711, 'sctypes')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___115713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), sctypes_115712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_115714 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), getitem___115713, str_115710)
        
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_115715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        # Getting the type of 'float' (line 34)
        float_115716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 39), list_115715, float_115716)
        
        # Applying the binary operator '+' (line 34)
        result_add_115717 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 17), '+', subscript_call_result_115714, list_115715)
        
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), result_add_115717)
        # Getting the type of the for loop variable (line 34)
        for_loop_var_115718 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), result_add_115717)
        # Assigning a type to the variable 'T' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'T', for_loop_var_115718)
        # SSA begins for a for statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to suppress_warnings(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_115720 = {}
        # Getting the type of 'suppress_warnings' (line 36)
        suppress_warnings_115719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 36)
        suppress_warnings_call_result_115721 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), suppress_warnings_115719, *[], **kwargs_115720)
        
        with_115722 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 36, 17), suppress_warnings_call_result_115721, 'with parameter', '__enter__', '__exit__')

        if with_115722:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 36)
            enter___115723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 17), suppress_warnings_call_result_115721, '__enter__')
            with_enter_115724 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), enter___115723)
            # Assigning a type to the variable 'sup' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'sup', with_enter_115724)
            
            # Call to filter(...): (line 37)
            # Processing the call arguments (line 37)
            # Getting the type of 'DeprecationWarning' (line 37)
            DeprecationWarning_115727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 37)
            kwargs_115728 = {}
            # Getting the type of 'sup' (line 37)
            sup_115725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 37)
            filter_115726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), sup_115725, 'filter')
            # Calling filter(args, kwargs) (line 37)
            filter_call_result_115729 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), filter_115726, *[DeprecationWarning_115727], **kwargs_115728)
            
            
            # Assigning a Call to a Name (line 38):
            
            # Assigning a Call to a Name (line 38):
            
            # Call to imresize(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'im' (line 38)
            im_115732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 36), 'im', False)
            
            # Call to T(...): (line 38)
            # Processing the call arguments (line 38)
            float_115734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 42), 'float')
            # Processing the call keyword arguments (line 38)
            kwargs_115735 = {}
            # Getting the type of 'T' (line 38)
            T_115733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 40), 'T', False)
            # Calling T(args, kwargs) (line 38)
            T_call_result_115736 = invoke(stypy.reporting.localization.Localization(__file__, 38, 40), T_115733, *[float_115734], **kwargs_115735)
            
            # Processing the call keyword arguments (line 38)
            kwargs_115737 = {}
            # Getting the type of 'misc' (line 38)
            misc_115730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'misc', False)
            # Obtaining the member 'imresize' of a type (line 38)
            imresize_115731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 22), misc_115730, 'imresize')
            # Calling imresize(args, kwargs) (line 38)
            imresize_call_result_115738 = invoke(stypy.reporting.localization.Localization(__file__, 38, 22), imresize_115731, *[im_115732, T_call_result_115736], **kwargs_115737)
            
            # Assigning a type to the variable 'im1' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'im1', imresize_call_result_115738)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 36)
            exit___115739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 17), suppress_warnings_call_result_115721, '__exit__')
            with_exit_115740 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), exit___115739, None, None, None)

        
        # Call to assert_equal(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'im1' (line 39)
        im1_115742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'im1', False)
        # Obtaining the member 'shape' of a type (line 39)
        shape_115743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 25), im1_115742, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_115744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        int_115745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 37), tuple_115744, int_115745)
        # Adding element type (line 39)
        int_115746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 37), tuple_115744, int_115746)
        
        # Processing the call keyword arguments (line 39)
        kwargs_115747 = {}
        # Getting the type of 'assert_equal' (line 39)
        assert_equal_115741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 39)
        assert_equal_call_result_115748 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), assert_equal_115741, *[shape_115743, tuple_115744], **kwargs_115747)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_imresize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_imresize' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_115749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115749)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_imresize'
        return stypy_return_type_115749


    @norecursion
    def test_imresize2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_imresize2'
        module_type_store = module_type_store.open_function_context('test_imresize2', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_imresize2')
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_imresize2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_imresize2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_imresize2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_imresize2(...)' code ##################

        
        # Assigning a Call to a Name (line 42):
        
        # Assigning a Call to a Name (line 42):
        
        # Call to random(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_115753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        int_115754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 31), tuple_115753, int_115754)
        # Adding element type (line 42)
        int_115755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 31), tuple_115753, int_115755)
        
        # Processing the call keyword arguments (line 42)
        kwargs_115756 = {}
        # Getting the type of 'np' (line 42)
        np_115750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'np', False)
        # Obtaining the member 'random' of a type (line 42)
        random_115751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), np_115750, 'random')
        # Obtaining the member 'random' of a type (line 42)
        random_115752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), random_115751, 'random')
        # Calling random(args, kwargs) (line 42)
        random_call_result_115757 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), random_115752, *[tuple_115753], **kwargs_115756)
        
        # Assigning a type to the variable 'im' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'im', random_call_result_115757)
        
        # Call to suppress_warnings(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_115759 = {}
        # Getting the type of 'suppress_warnings' (line 43)
        suppress_warnings_115758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 43)
        suppress_warnings_call_result_115760 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), suppress_warnings_115758, *[], **kwargs_115759)
        
        with_115761 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 43, 13), suppress_warnings_call_result_115760, 'with parameter', '__enter__', '__exit__')

        if with_115761:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 43)
            enter___115762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), suppress_warnings_call_result_115760, '__enter__')
            with_enter_115763 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), enter___115762)
            # Assigning a type to the variable 'sup' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'sup', with_enter_115763)
            
            # Call to filter(...): (line 44)
            # Processing the call arguments (line 44)
            # Getting the type of 'DeprecationWarning' (line 44)
            DeprecationWarning_115766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 44)
            kwargs_115767 = {}
            # Getting the type of 'sup' (line 44)
            sup_115764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 44)
            filter_115765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), sup_115764, 'filter')
            # Calling filter(args, kwargs) (line 44)
            filter_call_result_115768 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), filter_115765, *[DeprecationWarning_115766], **kwargs_115767)
            
            
            # Assigning a Call to a Name (line 45):
            
            # Assigning a Call to a Name (line 45):
            
            # Call to imresize(...): (line 45)
            # Processing the call arguments (line 45)
            # Getting the type of 'im' (line 45)
            im_115771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 32), 'im', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 45)
            tuple_115772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 45)
            # Adding element type (line 45)
            int_115773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 37), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 37), tuple_115772, int_115773)
            # Adding element type (line 45)
            int_115774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 41), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 37), tuple_115772, int_115774)
            
            # Processing the call keyword arguments (line 45)
            str_115775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 53), 'str', 'bicubic')
            keyword_115776 = str_115775
            kwargs_115777 = {'interp': keyword_115776}
            # Getting the type of 'misc' (line 45)
            misc_115769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'misc', False)
            # Obtaining the member 'imresize' of a type (line 45)
            imresize_115770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 18), misc_115769, 'imresize')
            # Calling imresize(args, kwargs) (line 45)
            imresize_call_result_115778 = invoke(stypy.reporting.localization.Localization(__file__, 45, 18), imresize_115770, *[im_115771, tuple_115772], **kwargs_115777)
            
            # Assigning a type to the variable 'im2' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'im2', imresize_call_result_115778)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 43)
            exit___115779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), suppress_warnings_call_result_115760, '__exit__')
            with_exit_115780 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), exit___115779, None, None, None)

        
        # Call to assert_equal(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'im2' (line 46)
        im2_115782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'im2', False)
        # Obtaining the member 'shape' of a type (line 46)
        shape_115783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 21), im2_115782, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_115784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        int_115785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 33), tuple_115784, int_115785)
        # Adding element type (line 46)
        int_115786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 33), tuple_115784, int_115786)
        
        # Processing the call keyword arguments (line 46)
        kwargs_115787 = {}
        # Getting the type of 'assert_equal' (line 46)
        assert_equal_115781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 46)
        assert_equal_call_result_115788 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assert_equal_115781, *[shape_115783, tuple_115784], **kwargs_115787)
        
        
        # ################# End of 'test_imresize2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_imresize2' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_115789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_imresize2'
        return stypy_return_type_115789


    @norecursion
    def test_imresize3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_imresize3'
        module_type_store = module_type_store.open_function_context('test_imresize3', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_imresize3')
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_imresize3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_imresize3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_imresize3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_imresize3(...)' code ##################

        
        # Assigning a Call to a Name (line 49):
        
        # Assigning a Call to a Name (line 49):
        
        # Call to random(...): (line 49)
        # Processing the call arguments (line 49)
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_115793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        int_115794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 31), tuple_115793, int_115794)
        # Adding element type (line 49)
        int_115795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 31), tuple_115793, int_115795)
        
        # Processing the call keyword arguments (line 49)
        kwargs_115796 = {}
        # Getting the type of 'np' (line 49)
        np_115790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 13), 'np', False)
        # Obtaining the member 'random' of a type (line 49)
        random_115791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 13), np_115790, 'random')
        # Obtaining the member 'random' of a type (line 49)
        random_115792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 13), random_115791, 'random')
        # Calling random(args, kwargs) (line 49)
        random_call_result_115797 = invoke(stypy.reporting.localization.Localization(__file__, 49, 13), random_115792, *[tuple_115793], **kwargs_115796)
        
        # Assigning a type to the variable 'im' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'im', random_call_result_115797)
        
        # Call to suppress_warnings(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_115799 = {}
        # Getting the type of 'suppress_warnings' (line 50)
        suppress_warnings_115798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 50)
        suppress_warnings_call_result_115800 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), suppress_warnings_115798, *[], **kwargs_115799)
        
        with_115801 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 50, 13), suppress_warnings_call_result_115800, 'with parameter', '__enter__', '__exit__')

        if with_115801:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 50)
            enter___115802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 13), suppress_warnings_call_result_115800, '__enter__')
            with_enter_115803 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), enter___115802)
            # Assigning a type to the variable 'sup' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'sup', with_enter_115803)
            
            # Call to filter(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'DeprecationWarning' (line 51)
            DeprecationWarning_115806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 51)
            kwargs_115807 = {}
            # Getting the type of 'sup' (line 51)
            sup_115804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 51)
            filter_115805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), sup_115804, 'filter')
            # Calling filter(args, kwargs) (line 51)
            filter_call_result_115808 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), filter_115805, *[DeprecationWarning_115806], **kwargs_115807)
            
            
            # Assigning a Call to a Name (line 52):
            
            # Assigning a Call to a Name (line 52):
            
            # Call to imresize(...): (line 52)
            # Processing the call arguments (line 52)
            # Getting the type of 'im' (line 52)
            im_115811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 'im', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 52)
            tuple_115812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 52)
            # Adding element type (line 52)
            int_115813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 37), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), tuple_115812, int_115813)
            # Adding element type (line 52)
            int_115814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 41), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), tuple_115812, int_115814)
            
            # Processing the call keyword arguments (line 52)
            str_115815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 53), 'str', 'nearest')
            keyword_115816 = str_115815
            kwargs_115817 = {'interp': keyword_115816}
            # Getting the type of 'misc' (line 52)
            misc_115809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'misc', False)
            # Obtaining the member 'imresize' of a type (line 52)
            imresize_115810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 18), misc_115809, 'imresize')
            # Calling imresize(args, kwargs) (line 52)
            imresize_call_result_115818 = invoke(stypy.reporting.localization.Localization(__file__, 52, 18), imresize_115810, *[im_115811, tuple_115812], **kwargs_115817)
            
            # Assigning a type to the variable 'im2' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'im2', imresize_call_result_115818)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 50)
            exit___115819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 13), suppress_warnings_call_result_115800, '__exit__')
            with_exit_115820 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), exit___115819, None, None, None)

        
        # Call to assert_equal(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'im2' (line 53)
        im2_115822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'im2', False)
        # Obtaining the member 'shape' of a type (line 53)
        shape_115823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 21), im2_115822, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_115824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        int_115825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 33), tuple_115824, int_115825)
        # Adding element type (line 53)
        int_115826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 33), tuple_115824, int_115826)
        
        # Processing the call keyword arguments (line 53)
        kwargs_115827 = {}
        # Getting the type of 'assert_equal' (line 53)
        assert_equal_115821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 53)
        assert_equal_call_result_115828 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert_equal_115821, *[shape_115823, tuple_115824], **kwargs_115827)
        
        
        # ################# End of 'test_imresize3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_imresize3' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_115829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115829)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_imresize3'
        return stypy_return_type_115829


    @norecursion
    def test_imresize4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_imresize4'
        module_type_store = module_type_store.open_function_context('test_imresize4', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_imresize4')
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_imresize4.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_imresize4', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_imresize4', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_imresize4(...)' code ##################

        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to array(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_115832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_115833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        int_115834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 23), list_115833, int_115834)
        # Adding element type (line 56)
        int_115835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 23), list_115833, int_115835)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 22), list_115832, list_115833)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_115836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_115837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 23), list_115836, int_115837)
        # Adding element type (line 57)
        int_115838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 23), list_115836, int_115838)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 22), list_115832, list_115836)
        
        # Processing the call keyword arguments (line 56)
        kwargs_115839 = {}
        # Getting the type of 'np' (line 56)
        np_115830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 56)
        array_115831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 13), np_115830, 'array')
        # Calling array(args, kwargs) (line 56)
        array_call_result_115840 = invoke(stypy.reporting.localization.Localization(__file__, 56, 13), array_115831, *[list_115832], **kwargs_115839)
        
        # Assigning a type to the variable 'im' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'im', array_call_result_115840)
        
        # Call to suppress_warnings(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_115842 = {}
        # Getting the type of 'suppress_warnings' (line 59)
        suppress_warnings_115841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 59)
        suppress_warnings_call_result_115843 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), suppress_warnings_115841, *[], **kwargs_115842)
        
        with_115844 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 59, 13), suppress_warnings_call_result_115843, 'with parameter', '__enter__', '__exit__')

        if with_115844:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 59)
            enter___115845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 13), suppress_warnings_call_result_115843, '__enter__')
            with_enter_115846 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), enter___115845)
            # Assigning a type to the variable 'sup' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'sup', with_enter_115846)
            
            # Call to filter(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'DeprecationWarning' (line 60)
            DeprecationWarning_115849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 60)
            kwargs_115850 = {}
            # Getting the type of 'sup' (line 60)
            sup_115847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 60)
            filter_115848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), sup_115847, 'filter')
            # Calling filter(args, kwargs) (line 60)
            filter_call_result_115851 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), filter_115848, *[DeprecationWarning_115849], **kwargs_115850)
            
            
            # Assigning a Call to a Name (line 61):
            
            # Assigning a Call to a Name (line 61):
            
            # Call to imresize(...): (line 61)
            # Processing the call arguments (line 61)
            # Getting the type of 'im' (line 61)
            im_115854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'im', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 61)
            tuple_115855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 61)
            # Adding element type (line 61)
            int_115856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 37), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 37), tuple_115855, int_115856)
            # Adding element type (line 61)
            int_115857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 37), tuple_115855, int_115857)
            
            # Processing the call keyword arguments (line 61)
            str_115858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 49), 'str', 'F')
            keyword_115859 = str_115858
            kwargs_115860 = {'mode': keyword_115859}
            # Getting the type of 'misc' (line 61)
            misc_115852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'misc', False)
            # Obtaining the member 'imresize' of a type (line 61)
            imresize_115853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), misc_115852, 'imresize')
            # Calling imresize(args, kwargs) (line 61)
            imresize_call_result_115861 = invoke(stypy.reporting.localization.Localization(__file__, 61, 18), imresize_115853, *[im_115854, tuple_115855], **kwargs_115860)
            
            # Assigning a type to the variable 'im2' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'im2', imresize_call_result_115861)
            
            # Assigning a Call to a Name (line 62):
            
            # Assigning a Call to a Name (line 62):
            
            # Call to imresize(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'im' (line 62)
            im_115864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'im', False)
            float_115865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 36), 'float')
            # Processing the call keyword arguments (line 62)
            str_115866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 45), 'str', 'F')
            keyword_115867 = str_115866
            kwargs_115868 = {'mode': keyword_115867}
            # Getting the type of 'misc' (line 62)
            misc_115862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'misc', False)
            # Obtaining the member 'imresize' of a type (line 62)
            imresize_115863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 18), misc_115862, 'imresize')
            # Calling imresize(args, kwargs) (line 62)
            imresize_call_result_115869 = invoke(stypy.reporting.localization.Localization(__file__, 62, 18), imresize_115863, *[im_115864, float_115865], **kwargs_115868)
            
            # Assigning a type to the variable 'im3' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'im3', imresize_call_result_115869)
            
            # Assigning a Call to a Name (line 63):
            
            # Assigning a Call to a Name (line 63):
            
            # Call to imresize(...): (line 63)
            # Processing the call arguments (line 63)
            # Getting the type of 'im' (line 63)
            im_115872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'im', False)
            int_115873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 36), 'int')
            # Processing the call keyword arguments (line 63)
            str_115874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 46), 'str', 'F')
            keyword_115875 = str_115874
            kwargs_115876 = {'mode': keyword_115875}
            # Getting the type of 'misc' (line 63)
            misc_115870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'misc', False)
            # Obtaining the member 'imresize' of a type (line 63)
            imresize_115871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 18), misc_115870, 'imresize')
            # Calling imresize(args, kwargs) (line 63)
            imresize_call_result_115877 = invoke(stypy.reporting.localization.Localization(__file__, 63, 18), imresize_115871, *[im_115872, int_115873], **kwargs_115876)
            
            # Assigning a type to the variable 'im4' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'im4', imresize_call_result_115877)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 59)
            exit___115878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 13), suppress_warnings_call_result_115843, '__exit__')
            with_exit_115879 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), exit___115878, None, None, None)

        
        # Call to assert_equal(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'im2' (line 64)
        im2_115881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'im2', False)
        # Getting the type of 'im3' (line 64)
        im3_115882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'im3', False)
        # Processing the call keyword arguments (line 64)
        kwargs_115883 = {}
        # Getting the type of 'assert_equal' (line 64)
        assert_equal_115880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 64)
        assert_equal_call_result_115884 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_equal_115880, *[im2_115881, im3_115882], **kwargs_115883)
        
        
        # Call to assert_equal(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'im2' (line 65)
        im2_115886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'im2', False)
        # Getting the type of 'im4' (line 65)
        im4_115887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'im4', False)
        # Processing the call keyword arguments (line 65)
        kwargs_115888 = {}
        # Getting the type of 'assert_equal' (line 65)
        assert_equal_115885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 65)
        assert_equal_call_result_115889 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_equal_115885, *[im2_115886, im4_115887], **kwargs_115888)
        
        
        # ################# End of 'test_imresize4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_imresize4' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_115890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115890)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_imresize4'
        return stypy_return_type_115890


    @norecursion
    def test_imresize5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_imresize5'
        module_type_store = module_type_store.open_function_context('test_imresize5', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_imresize5')
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_imresize5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_imresize5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_imresize5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_imresize5(...)' code ##################

        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to random(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_115894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        int_115895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 31), tuple_115894, int_115895)
        # Adding element type (line 68)
        int_115896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 31), tuple_115894, int_115896)
        
        # Processing the call keyword arguments (line 68)
        kwargs_115897 = {}
        # Getting the type of 'np' (line 68)
        np_115891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'np', False)
        # Obtaining the member 'random' of a type (line 68)
        random_115892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 13), np_115891, 'random')
        # Obtaining the member 'random' of a type (line 68)
        random_115893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 13), random_115892, 'random')
        # Calling random(args, kwargs) (line 68)
        random_call_result_115898 = invoke(stypy.reporting.localization.Localization(__file__, 68, 13), random_115893, *[tuple_115894], **kwargs_115897)
        
        # Assigning a type to the variable 'im' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'im', random_call_result_115898)
        
        # Call to suppress_warnings(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_115900 = {}
        # Getting the type of 'suppress_warnings' (line 69)
        suppress_warnings_115899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 69)
        suppress_warnings_call_result_115901 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), suppress_warnings_115899, *[], **kwargs_115900)
        
        with_115902 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 69, 13), suppress_warnings_call_result_115901, 'with parameter', '__enter__', '__exit__')

        if with_115902:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 69)
            enter___115903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), suppress_warnings_call_result_115901, '__enter__')
            with_enter_115904 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), enter___115903)
            # Assigning a type to the variable 'sup' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'sup', with_enter_115904)
            
            # Call to filter(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'DeprecationWarning' (line 70)
            DeprecationWarning_115907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 70)
            kwargs_115908 = {}
            # Getting the type of 'sup' (line 70)
            sup_115905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 70)
            filter_115906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), sup_115905, 'filter')
            # Calling filter(args, kwargs) (line 70)
            filter_call_result_115909 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), filter_115906, *[DeprecationWarning_115907], **kwargs_115908)
            
            
            # Assigning a Call to a Name (line 71):
            
            # Assigning a Call to a Name (line 71):
            
            # Call to imresize(...): (line 71)
            # Processing the call arguments (line 71)
            # Getting the type of 'im' (line 71)
            im_115912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'im', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 71)
            tuple_115913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 71)
            # Adding element type (line 71)
            int_115914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 37), tuple_115913, int_115914)
            # Adding element type (line 71)
            int_115915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 41), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 37), tuple_115913, int_115915)
            
            # Processing the call keyword arguments (line 71)
            str_115916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 53), 'str', 'lanczos')
            keyword_115917 = str_115916
            kwargs_115918 = {'interp': keyword_115917}
            # Getting the type of 'misc' (line 71)
            misc_115910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'misc', False)
            # Obtaining the member 'imresize' of a type (line 71)
            imresize_115911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 18), misc_115910, 'imresize')
            # Calling imresize(args, kwargs) (line 71)
            imresize_call_result_115919 = invoke(stypy.reporting.localization.Localization(__file__, 71, 18), imresize_115911, *[im_115912, tuple_115913], **kwargs_115918)
            
            # Assigning a type to the variable 'im2' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'im2', imresize_call_result_115919)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 69)
            exit___115920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), suppress_warnings_call_result_115901, '__exit__')
            with_exit_115921 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), exit___115920, None, None, None)

        
        # Call to assert_equal(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'im2' (line 72)
        im2_115923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'im2', False)
        # Obtaining the member 'shape' of a type (line 72)
        shape_115924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 21), im2_115923, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_115925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        int_115926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 33), tuple_115925, int_115926)
        # Adding element type (line 72)
        int_115927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 33), tuple_115925, int_115927)
        
        # Processing the call keyword arguments (line 72)
        kwargs_115928 = {}
        # Getting the type of 'assert_equal' (line 72)
        assert_equal_115922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 72)
        assert_equal_call_result_115929 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert_equal_115922, *[shape_115924, tuple_115925], **kwargs_115928)
        
        
        # ################# End of 'test_imresize5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_imresize5' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_115930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_imresize5'
        return stypy_return_type_115930


    @norecursion
    def test_bytescale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale'
        module_type_store = module_type_store.open_function_context('test_bytescale', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale')
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale(...)' code ##################

        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to array(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_115933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        int_115934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), list_115933, int_115934)
        # Adding element type (line 75)
        int_115935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), list_115933, int_115935)
        # Adding element type (line 75)
        int_115936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), list_115933, int_115936)
        
        # Getting the type of 'np' (line 75)
        np_115937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'np', False)
        # Obtaining the member 'uint8' of a type (line 75)
        uint8_115938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), np_115937, 'uint8')
        # Processing the call keyword arguments (line 75)
        kwargs_115939 = {}
        # Getting the type of 'np' (line 75)
        np_115931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 75)
        array_115932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), np_115931, 'array')
        # Calling array(args, kwargs) (line 75)
        array_call_result_115940 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), array_115932, *[list_115933, uint8_115938], **kwargs_115939)
        
        # Assigning a type to the variable 'x' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'x', array_call_result_115940)
        
        # Assigning a Call to a Name (line 76):
        
        # Assigning a Call to a Name (line 76):
        
        # Call to array(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_115943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        int_115944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 21), list_115943, int_115944)
        # Adding element type (line 76)
        int_115945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 21), list_115943, int_115945)
        # Adding element type (line 76)
        int_115946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 21), list_115943, int_115946)
        
        # Processing the call keyword arguments (line 76)
        kwargs_115947 = {}
        # Getting the type of 'np' (line 76)
        np_115941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 76)
        array_115942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), np_115941, 'array')
        # Calling array(args, kwargs) (line 76)
        array_call_result_115948 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), array_115942, *[list_115943], **kwargs_115947)
        
        # Assigning a type to the variable 'y' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'y', array_call_result_115948)
        
        # Call to suppress_warnings(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_115950 = {}
        # Getting the type of 'suppress_warnings' (line 77)
        suppress_warnings_115949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 77)
        suppress_warnings_call_result_115951 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), suppress_warnings_115949, *[], **kwargs_115950)
        
        with_115952 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 77, 13), suppress_warnings_call_result_115951, 'with parameter', '__enter__', '__exit__')

        if with_115952:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 77)
            enter___115953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), suppress_warnings_call_result_115951, '__enter__')
            with_enter_115954 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), enter___115953)
            # Assigning a type to the variable 'sup' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'sup', with_enter_115954)
            
            # Call to filter(...): (line 78)
            # Processing the call arguments (line 78)
            # Getting the type of 'DeprecationWarning' (line 78)
            DeprecationWarning_115957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 78)
            kwargs_115958 = {}
            # Getting the type of 'sup' (line 78)
            sup_115955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 78)
            filter_115956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), sup_115955, 'filter')
            # Calling filter(args, kwargs) (line 78)
            filter_call_result_115959 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), filter_115956, *[DeprecationWarning_115957], **kwargs_115958)
            
            
            # Call to assert_equal(...): (line 79)
            # Processing the call arguments (line 79)
            
            # Call to bytescale(...): (line 79)
            # Processing the call arguments (line 79)
            # Getting the type of 'x' (line 79)
            x_115963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'x', False)
            # Processing the call keyword arguments (line 79)
            kwargs_115964 = {}
            # Getting the type of 'misc' (line 79)
            misc_115961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 79)
            bytescale_115962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), misc_115961, 'bytescale')
            # Calling bytescale(args, kwargs) (line 79)
            bytescale_call_result_115965 = invoke(stypy.reporting.localization.Localization(__file__, 79, 25), bytescale_115962, *[x_115963], **kwargs_115964)
            
            # Getting the type of 'x' (line 79)
            x_115966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 44), 'x', False)
            # Processing the call keyword arguments (line 79)
            kwargs_115967 = {}
            # Getting the type of 'assert_equal' (line 79)
            assert_equal_115960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 79)
            assert_equal_call_result_115968 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), assert_equal_115960, *[bytescale_call_result_115965, x_115966], **kwargs_115967)
            
            
            # Call to assert_equal(...): (line 80)
            # Processing the call arguments (line 80)
            
            # Call to bytescale(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'y' (line 80)
            y_115972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 40), 'y', False)
            # Processing the call keyword arguments (line 80)
            kwargs_115973 = {}
            # Getting the type of 'misc' (line 80)
            misc_115970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 80)
            bytescale_115971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), misc_115970, 'bytescale')
            # Calling bytescale(args, kwargs) (line 80)
            bytescale_call_result_115974 = invoke(stypy.reporting.localization.Localization(__file__, 80, 25), bytescale_115971, *[y_115972], **kwargs_115973)
            
            
            # Obtaining an instance of the builtin type 'list' (line 80)
            list_115975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 44), 'list')
            # Adding type elements to the builtin type 'list' instance (line 80)
            # Adding element type (line 80)
            int_115976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 45), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 44), list_115975, int_115976)
            # Adding element type (line 80)
            int_115977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 48), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 44), list_115975, int_115977)
            # Adding element type (line 80)
            int_115978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 53), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 44), list_115975, int_115978)
            
            # Processing the call keyword arguments (line 80)
            kwargs_115979 = {}
            # Getting the type of 'assert_equal' (line 80)
            assert_equal_115969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 80)
            assert_equal_call_result_115980 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), assert_equal_115969, *[bytescale_call_result_115974, list_115975], **kwargs_115979)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 77)
            exit___115981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), suppress_warnings_call_result_115951, '__exit__')
            with_exit_115982 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), exit___115981, None, None, None)

        
        # ################# End of 'test_bytescale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_115983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115983)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale'
        return stypy_return_type_115983


    @norecursion
    def test_bytescale_keywords(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale_keywords'
        module_type_store = module_type_store.open_function_context('test_bytescale_keywords', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale_keywords')
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale_keywords.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale_keywords', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale_keywords', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale_keywords(...)' code ##################

        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to array(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_115986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        int_115987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_115986, int_115987)
        # Adding element type (line 83)
        int_115988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_115986, int_115988)
        # Adding element type (line 83)
        int_115989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_115986, int_115989)
        # Adding element type (line 83)
        int_115990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_115986, int_115990)
        # Adding element type (line 83)
        int_115991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_115986, int_115991)
        # Adding element type (line 83)
        int_115992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_115986, int_115992)
        
        # Processing the call keyword arguments (line 83)
        kwargs_115993 = {}
        # Getting the type of 'np' (line 83)
        np_115984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 83)
        array_115985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), np_115984, 'array')
        # Calling array(args, kwargs) (line 83)
        array_call_result_115994 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), array_115985, *[list_115986], **kwargs_115993)
        
        # Assigning a type to the variable 'x' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'x', array_call_result_115994)
        
        # Call to suppress_warnings(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_115996 = {}
        # Getting the type of 'suppress_warnings' (line 84)
        suppress_warnings_115995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 84)
        suppress_warnings_call_result_115997 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), suppress_warnings_115995, *[], **kwargs_115996)
        
        with_115998 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 84, 13), suppress_warnings_call_result_115997, 'with parameter', '__enter__', '__exit__')

        if with_115998:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 84)
            enter___115999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), suppress_warnings_call_result_115997, '__enter__')
            with_enter_116000 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), enter___115999)
            # Assigning a type to the variable 'sup' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'sup', with_enter_116000)
            
            # Call to filter(...): (line 85)
            # Processing the call arguments (line 85)
            # Getting the type of 'DeprecationWarning' (line 85)
            DeprecationWarning_116003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 85)
            kwargs_116004 = {}
            # Getting the type of 'sup' (line 85)
            sup_116001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 85)
            filter_116002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), sup_116001, 'filter')
            # Calling filter(args, kwargs) (line 85)
            filter_call_result_116005 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), filter_116002, *[DeprecationWarning_116003], **kwargs_116004)
            
            
            # Assigning a Call to a Name (line 86):
            
            # Assigning a Call to a Name (line 86):
            
            # Call to bytescale(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'x' (line 86)
            x_116008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 41), 'x', False)
            # Processing the call keyword arguments (line 86)
            int_116009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 48), 'int')
            keyword_116010 = int_116009
            int_116011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 57), 'int')
            keyword_116012 = int_116011
            kwargs_116013 = {'high': keyword_116012, 'low': keyword_116010}
            # Getting the type of 'misc' (line 86)
            misc_116006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 86)
            bytescale_116007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 26), misc_116006, 'bytescale')
            # Calling bytescale(args, kwargs) (line 86)
            bytescale_call_result_116014 = invoke(stypy.reporting.localization.Localization(__file__, 86, 26), bytescale_116007, *[x_116008], **kwargs_116013)
            
            # Assigning a type to the variable 'res_lowhigh' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'res_lowhigh', bytescale_call_result_116014)
            
            # Call to assert_equal(...): (line 87)
            # Processing the call arguments (line 87)
            # Getting the type of 'res_lowhigh' (line 87)
            res_lowhigh_116016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'res_lowhigh', False)
            
            # Obtaining an instance of the builtin type 'list' (line 87)
            list_116017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 38), 'list')
            # Adding type elements to the builtin type 'list' instance (line 87)
            # Adding element type (line 87)
            int_116018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 39), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 38), list_116017, int_116018)
            # Adding element type (line 87)
            int_116019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 43), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 38), list_116017, int_116019)
            # Adding element type (line 87)
            int_116020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 47), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 38), list_116017, int_116020)
            # Adding element type (line 87)
            int_116021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 51), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 38), list_116017, int_116021)
            # Adding element type (line 87)
            int_116022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 55), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 38), list_116017, int_116022)
            # Adding element type (line 87)
            int_116023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 59), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 38), list_116017, int_116023)
            
            # Processing the call keyword arguments (line 87)
            kwargs_116024 = {}
            # Getting the type of 'assert_equal' (line 87)
            assert_equal_116015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 87)
            assert_equal_call_result_116025 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), assert_equal_116015, *[res_lowhigh_116016, list_116017], **kwargs_116024)
            
            
            # Assigning a Call to a Name (line 88):
            
            # Assigning a Call to a Name (line 88):
            
            # Call to bytescale(...): (line 88)
            # Processing the call arguments (line 88)
            # Getting the type of 'x' (line 88)
            x_116028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 42), 'x', False)
            # Processing the call keyword arguments (line 88)
            int_116029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 50), 'int')
            keyword_116030 = int_116029
            int_116031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 59), 'int')
            keyword_116032 = int_116031
            kwargs_116033 = {'cmax': keyword_116032, 'cmin': keyword_116030}
            # Getting the type of 'misc' (line 88)
            misc_116026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 88)
            bytescale_116027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 27), misc_116026, 'bytescale')
            # Calling bytescale(args, kwargs) (line 88)
            bytescale_call_result_116034 = invoke(stypy.reporting.localization.Localization(__file__, 88, 27), bytescale_116027, *[x_116028], **kwargs_116033)
            
            # Assigning a type to the variable 'res_cmincmax' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'res_cmincmax', bytescale_call_result_116034)
            
            # Call to assert_equal(...): (line 89)
            # Processing the call arguments (line 89)
            # Getting the type of 'res_cmincmax' (line 89)
            res_cmincmax_116036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'res_cmincmax', False)
            
            # Obtaining an instance of the builtin type 'list' (line 89)
            list_116037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'list')
            # Adding type elements to the builtin type 'list' instance (line 89)
            # Adding element type (line 89)
            int_116038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 40), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 39), list_116037, int_116038)
            # Adding element type (line 89)
            int_116039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 43), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 39), list_116037, int_116039)
            # Adding element type (line 89)
            int_116040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 46), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 39), list_116037, int_116040)
            # Adding element type (line 89)
            int_116041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 50), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 39), list_116037, int_116041)
            # Adding element type (line 89)
            int_116042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 55), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 39), list_116037, int_116042)
            # Adding element type (line 89)
            int_116043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 60), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 39), list_116037, int_116043)
            
            # Processing the call keyword arguments (line 89)
            kwargs_116044 = {}
            # Getting the type of 'assert_equal' (line 89)
            assert_equal_116035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 89)
            assert_equal_call_result_116045 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), assert_equal_116035, *[res_cmincmax_116036, list_116037], **kwargs_116044)
            
            
            # Call to assert_equal(...): (line 90)
            # Processing the call arguments (line 90)
            
            # Call to bytescale(...): (line 90)
            # Processing the call arguments (line 90)
            
            # Call to array(...): (line 90)
            # Processing the call arguments (line 90)
            
            # Obtaining an instance of the builtin type 'list' (line 90)
            list_116051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 49), 'list')
            # Adding type elements to the builtin type 'list' instance (line 90)
            # Adding element type (line 90)
            int_116052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 50), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 49), list_116051, int_116052)
            # Adding element type (line 90)
            int_116053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 53), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 49), list_116051, int_116053)
            # Adding element type (line 90)
            int_116054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 56), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 49), list_116051, int_116054)
            
            # Processing the call keyword arguments (line 90)
            kwargs_116055 = {}
            # Getting the type of 'np' (line 90)
            np_116049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'np', False)
            # Obtaining the member 'array' of a type (line 90)
            array_116050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 40), np_116049, 'array')
            # Calling array(args, kwargs) (line 90)
            array_call_result_116056 = invoke(stypy.reporting.localization.Localization(__file__, 90, 40), array_116050, *[list_116051], **kwargs_116055)
            
            # Processing the call keyword arguments (line 90)
            int_116057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 65), 'int')
            keyword_116058 = int_116057
            kwargs_116059 = {'low': keyword_116058}
            # Getting the type of 'misc' (line 90)
            misc_116047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 90)
            bytescale_116048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 25), misc_116047, 'bytescale')
            # Calling bytescale(args, kwargs) (line 90)
            bytescale_call_result_116060 = invoke(stypy.reporting.localization.Localization(__file__, 90, 25), bytescale_116048, *[array_call_result_116056], **kwargs_116059)
            
            
            # Obtaining an instance of the builtin type 'list' (line 90)
            list_116061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 69), 'list')
            # Adding type elements to the builtin type 'list' instance (line 90)
            # Adding element type (line 90)
            int_116062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 70), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 69), list_116061, int_116062)
            # Adding element type (line 90)
            int_116063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 73), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 69), list_116061, int_116063)
            # Adding element type (line 90)
            int_116064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 76), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 69), list_116061, int_116064)
            
            # Processing the call keyword arguments (line 90)
            kwargs_116065 = {}
            # Getting the type of 'assert_equal' (line 90)
            assert_equal_116046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 90)
            assert_equal_call_result_116066 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), assert_equal_116046, *[bytescale_call_result_116060, list_116061], **kwargs_116065)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 84)
            exit___116067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), suppress_warnings_call_result_115997, '__exit__')
            with_exit_116068 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), exit___116067, None, None, None)

        
        # ################# End of 'test_bytescale_keywords(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale_keywords' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_116069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale_keywords'
        return stypy_return_type_116069


    @norecursion
    def test_bytescale_cscale_lowhigh(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale_cscale_lowhigh'
        module_type_store = module_type_store.open_function_context('test_bytescale_cscale_lowhigh', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale_cscale_lowhigh')
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale_cscale_lowhigh.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale_cscale_lowhigh', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale_cscale_lowhigh', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale_cscale_lowhigh(...)' code ##################

        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to arange(...): (line 93)
        # Processing the call arguments (line 93)
        int_116072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_116073 = {}
        # Getting the type of 'np' (line 93)
        np_116070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 93)
        arange_116071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), np_116070, 'arange')
        # Calling arange(args, kwargs) (line 93)
        arange_call_result_116074 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), arange_116071, *[int_116072], **kwargs_116073)
        
        # Assigning a type to the variable 'a' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'a', arange_call_result_116074)
        
        # Call to suppress_warnings(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_116076 = {}
        # Getting the type of 'suppress_warnings' (line 94)
        suppress_warnings_116075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 94)
        suppress_warnings_call_result_116077 = invoke(stypy.reporting.localization.Localization(__file__, 94, 13), suppress_warnings_116075, *[], **kwargs_116076)
        
        with_116078 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 94, 13), suppress_warnings_call_result_116077, 'with parameter', '__enter__', '__exit__')

        if with_116078:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 94)
            enter___116079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 13), suppress_warnings_call_result_116077, '__enter__')
            with_enter_116080 = invoke(stypy.reporting.localization.Localization(__file__, 94, 13), enter___116079)
            # Assigning a type to the variable 'sup' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'sup', with_enter_116080)
            
            # Call to filter(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'DeprecationWarning' (line 95)
            DeprecationWarning_116083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 95)
            kwargs_116084 = {}
            # Getting the type of 'sup' (line 95)
            sup_116081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 95)
            filter_116082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), sup_116081, 'filter')
            # Calling filter(args, kwargs) (line 95)
            filter_call_result_116085 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), filter_116082, *[DeprecationWarning_116083], **kwargs_116084)
            
            
            # Assigning a Call to a Name (line 96):
            
            # Assigning a Call to a Name (line 96):
            
            # Call to bytescale(...): (line 96)
            # Processing the call arguments (line 96)
            # Getting the type of 'a' (line 96)
            a_116088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 36), 'a', False)
            # Processing the call keyword arguments (line 96)
            int_116089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 44), 'int')
            keyword_116090 = int_116089
            int_116091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 52), 'int')
            keyword_116092 = int_116091
            int_116093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 59), 'int')
            keyword_116094 = int_116093
            int_116095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 69), 'int')
            keyword_116096 = int_116095
            kwargs_116097 = {'high': keyword_116096, 'cmax': keyword_116092, 'cmin': keyword_116090, 'low': keyword_116094}
            # Getting the type of 'misc' (line 96)
            misc_116086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 96)
            bytescale_116087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 21), misc_116086, 'bytescale')
            # Calling bytescale(args, kwargs) (line 96)
            bytescale_call_result_116098 = invoke(stypy.reporting.localization.Localization(__file__, 96, 21), bytescale_116087, *[a_116088], **kwargs_116097)
            
            # Assigning a type to the variable 'actual' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'actual', bytescale_call_result_116098)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 94)
            exit___116099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 13), suppress_warnings_call_result_116077, '__exit__')
            with_exit_116100 = invoke(stypy.reporting.localization.Localization(__file__, 94, 13), exit___116099, None, None, None)

        
        # Assigning a List to a Name (line 97):
        
        # Assigning a List to a Name (line 97):
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_116101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        int_116102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116102)
        # Adding element type (line 97)
        int_116103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116103)
        # Adding element type (line 97)
        int_116104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116104)
        # Adding element type (line 97)
        int_116105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116105)
        # Adding element type (line 97)
        int_116106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116106)
        # Adding element type (line 97)
        int_116107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116107)
        # Adding element type (line 97)
        int_116108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116108)
        # Adding element type (line 97)
        int_116109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116109)
        # Adding element type (line 97)
        int_116110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116110)
        # Adding element type (line 97)
        int_116111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 19), list_116101, int_116111)
        
        # Assigning a type to the variable 'expected' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'expected', list_116101)
        
        # Call to assert_equal(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'actual' (line 98)
        actual_116113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'actual', False)
        # Getting the type of 'expected' (line 98)
        expected_116114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 29), 'expected', False)
        # Processing the call keyword arguments (line 98)
        kwargs_116115 = {}
        # Getting the type of 'assert_equal' (line 98)
        assert_equal_116112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 98)
        assert_equal_call_result_116116 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), assert_equal_116112, *[actual_116113, expected_116114], **kwargs_116115)
        
        
        # ################# End of 'test_bytescale_cscale_lowhigh(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale_cscale_lowhigh' in the type store
        # Getting the type of 'stypy_return_type' (line 92)
        stypy_return_type_116117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale_cscale_lowhigh'
        return stypy_return_type_116117


    @norecursion
    def test_bytescale_mask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale_mask'
        module_type_store = module_type_store.open_function_context('test_bytescale_mask', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale_mask')
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale_mask.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale_mask', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale_mask', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale_mask(...)' code ##################

        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to MaskedArray(...): (line 101)
        # Processing the call keyword arguments (line 101)
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_116121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        int_116122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 35), list_116121, int_116122)
        # Adding element type (line 101)
        int_116123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 35), list_116121, int_116123)
        # Adding element type (line 101)
        int_116124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 35), list_116121, int_116124)
        
        keyword_116125 = list_116121
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_116126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        # Getting the type of 'False' (line 101)
        False_116127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 52), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 51), list_116126, False_116127)
        # Adding element type (line 101)
        # Getting the type of 'False' (line 101)
        False_116128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 59), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 51), list_116126, False_116128)
        # Adding element type (line 101)
        # Getting the type of 'True' (line 101)
        True_116129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 66), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 51), list_116126, True_116129)
        
        keyword_116130 = list_116126
        kwargs_116131 = {'mask': keyword_116130, 'data': keyword_116125}
        # Getting the type of 'np' (line 101)
        np_116118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'np', False)
        # Obtaining the member 'ma' of a type (line 101)
        ma_116119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), np_116118, 'ma')
        # Obtaining the member 'MaskedArray' of a type (line 101)
        MaskedArray_116120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), ma_116119, 'MaskedArray')
        # Calling MaskedArray(args, kwargs) (line 101)
        MaskedArray_call_result_116132 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), MaskedArray_116120, *[], **kwargs_116131)
        
        # Assigning a type to the variable 'a' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'a', MaskedArray_call_result_116132)
        
        # Call to suppress_warnings(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_116134 = {}
        # Getting the type of 'suppress_warnings' (line 102)
        suppress_warnings_116133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 102)
        suppress_warnings_call_result_116135 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), suppress_warnings_116133, *[], **kwargs_116134)
        
        with_116136 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 102, 13), suppress_warnings_call_result_116135, 'with parameter', '__enter__', '__exit__')

        if with_116136:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 102)
            enter___116137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), suppress_warnings_call_result_116135, '__enter__')
            with_enter_116138 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), enter___116137)
            # Assigning a type to the variable 'sup' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'sup', with_enter_116138)
            
            # Call to filter(...): (line 103)
            # Processing the call arguments (line 103)
            # Getting the type of 'DeprecationWarning' (line 103)
            DeprecationWarning_116141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 103)
            kwargs_116142 = {}
            # Getting the type of 'sup' (line 103)
            sup_116139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 103)
            filter_116140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), sup_116139, 'filter')
            # Calling filter(args, kwargs) (line 103)
            filter_call_result_116143 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), filter_116140, *[DeprecationWarning_116141], **kwargs_116142)
            
            
            # Assigning a Call to a Name (line 104):
            
            # Assigning a Call to a Name (line 104):
            
            # Call to bytescale(...): (line 104)
            # Processing the call arguments (line 104)
            # Getting the type of 'a' (line 104)
            a_116146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 36), 'a', False)
            # Processing the call keyword arguments (line 104)
            kwargs_116147 = {}
            # Getting the type of 'misc' (line 104)
            misc_116144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 104)
            bytescale_116145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 21), misc_116144, 'bytescale')
            # Calling bytescale(args, kwargs) (line 104)
            bytescale_call_result_116148 = invoke(stypy.reporting.localization.Localization(__file__, 104, 21), bytescale_116145, *[a_116146], **kwargs_116147)
            
            # Assigning a type to the variable 'actual' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'actual', bytescale_call_result_116148)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 102)
            exit___116149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), suppress_warnings_call_result_116135, '__exit__')
            with_exit_116150 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), exit___116149, None, None, None)

        
        # Assigning a List to a Name (line 105):
        
        # Assigning a List to a Name (line 105):
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_116151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        int_116152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_116151, int_116152)
        # Adding element type (line 105)
        int_116153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_116151, int_116153)
        # Adding element type (line 105)
        int_116154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_116151, int_116154)
        
        # Assigning a type to the variable 'expected' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'expected', list_116151)
        
        # Call to assert_equal(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'expected' (line 106)
        expected_116156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'expected', False)
        # Getting the type of 'actual' (line 106)
        actual_116157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 31), 'actual', False)
        # Processing the call keyword arguments (line 106)
        kwargs_116158 = {}
        # Getting the type of 'assert_equal' (line 106)
        assert_equal_116155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 106)
        assert_equal_call_result_116159 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_equal_116155, *[expected_116156, actual_116157], **kwargs_116158)
        
        
        # Call to assert_mask_equal(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'a' (line 107)
        a_116161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 26), 'a', False)
        # Obtaining the member 'mask' of a type (line 107)
        mask_116162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 26), a_116161, 'mask')
        # Getting the type of 'actual' (line 107)
        actual_116163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'actual', False)
        # Obtaining the member 'mask' of a type (line 107)
        mask_116164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 34), actual_116163, 'mask')
        # Processing the call keyword arguments (line 107)
        kwargs_116165 = {}
        # Getting the type of 'assert_mask_equal' (line 107)
        assert_mask_equal_116160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_mask_equal', False)
        # Calling assert_mask_equal(args, kwargs) (line 107)
        assert_mask_equal_call_result_116166 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert_mask_equal_116160, *[mask_116162, mask_116164], **kwargs_116165)
        
        
        # Call to assert_(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to isinstance(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'actual' (line 108)
        actual_116169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'actual', False)
        # Getting the type of 'np' (line 108)
        np_116170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'np', False)
        # Obtaining the member 'ma' of a type (line 108)
        ma_116171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 35), np_116170, 'ma')
        # Obtaining the member 'MaskedArray' of a type (line 108)
        MaskedArray_116172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 35), ma_116171, 'MaskedArray')
        # Processing the call keyword arguments (line 108)
        kwargs_116173 = {}
        # Getting the type of 'isinstance' (line 108)
        isinstance_116168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 108)
        isinstance_call_result_116174 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), isinstance_116168, *[actual_116169, MaskedArray_116172], **kwargs_116173)
        
        # Processing the call keyword arguments (line 108)
        kwargs_116175 = {}
        # Getting the type of 'assert_' (line 108)
        assert__116167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 108)
        assert__call_result_116176 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assert__116167, *[isinstance_call_result_116174], **kwargs_116175)
        
        
        # ################# End of 'test_bytescale_mask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale_mask' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_116177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale_mask'
        return stypy_return_type_116177


    @norecursion
    def test_bytescale_rounding(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale_rounding'
        module_type_store = module_type_store.open_function_context('test_bytescale_rounding', 110, 4, False)
        # Assigning a type to the variable 'self' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale_rounding')
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale_rounding.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale_rounding', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale_rounding', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale_rounding(...)' code ##################

        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to array(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_116180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        float_116181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_116180, float_116181)
        # Adding element type (line 111)
        float_116182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_116180, float_116182)
        # Adding element type (line 111)
        float_116183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_116180, float_116183)
        # Adding element type (line 111)
        float_116184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_116180, float_116184)
        # Adding element type (line 111)
        float_116185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 21), list_116180, float_116185)
        
        # Processing the call keyword arguments (line 111)
        kwargs_116186 = {}
        # Getting the type of 'np' (line 111)
        np_116178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 111)
        array_116179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), np_116178, 'array')
        # Calling array(args, kwargs) (line 111)
        array_call_result_116187 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), array_116179, *[list_116180], **kwargs_116186)
        
        # Assigning a type to the variable 'a' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'a', array_call_result_116187)
        
        # Call to suppress_warnings(...): (line 112)
        # Processing the call keyword arguments (line 112)
        kwargs_116189 = {}
        # Getting the type of 'suppress_warnings' (line 112)
        suppress_warnings_116188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 112)
        suppress_warnings_call_result_116190 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), suppress_warnings_116188, *[], **kwargs_116189)
        
        with_116191 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 112, 13), suppress_warnings_call_result_116190, 'with parameter', '__enter__', '__exit__')

        if with_116191:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 112)
            enter___116192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), suppress_warnings_call_result_116190, '__enter__')
            with_enter_116193 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), enter___116192)
            # Assigning a type to the variable 'sup' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'sup', with_enter_116193)
            
            # Call to filter(...): (line 113)
            # Processing the call arguments (line 113)
            # Getting the type of 'DeprecationWarning' (line 113)
            DeprecationWarning_116196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 113)
            kwargs_116197 = {}
            # Getting the type of 'sup' (line 113)
            sup_116194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 113)
            filter_116195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), sup_116194, 'filter')
            # Calling filter(args, kwargs) (line 113)
            filter_call_result_116198 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), filter_116195, *[DeprecationWarning_116196], **kwargs_116197)
            
            
            # Assigning a Call to a Name (line 114):
            
            # Assigning a Call to a Name (line 114):
            
            # Call to bytescale(...): (line 114)
            # Processing the call arguments (line 114)
            # Getting the type of 'a' (line 114)
            a_116201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 36), 'a', False)
            # Processing the call keyword arguments (line 114)
            int_116202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 44), 'int')
            keyword_116203 = int_116202
            int_116204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 52), 'int')
            keyword_116205 = int_116204
            int_116206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 60), 'int')
            keyword_116207 = int_116206
            int_116208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 68), 'int')
            keyword_116209 = int_116208
            kwargs_116210 = {'high': keyword_116209, 'cmax': keyword_116205, 'cmin': keyword_116203, 'low': keyword_116207}
            # Getting the type of 'misc' (line 114)
            misc_116199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 114)
            bytescale_116200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 21), misc_116199, 'bytescale')
            # Calling bytescale(args, kwargs) (line 114)
            bytescale_call_result_116211 = invoke(stypy.reporting.localization.Localization(__file__, 114, 21), bytescale_116200, *[a_116201], **kwargs_116210)
            
            # Assigning a type to the variable 'actual' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'actual', bytescale_call_result_116211)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 112)
            exit___116212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), suppress_warnings_call_result_116190, '__exit__')
            with_exit_116213 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), exit___116212, None, None, None)

        
        # Assigning a List to a Name (line 115):
        
        # Assigning a List to a Name (line 115):
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_116214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        int_116215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_116214, int_116215)
        # Adding element type (line 115)
        int_116216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_116214, int_116216)
        # Adding element type (line 115)
        int_116217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_116214, int_116217)
        # Adding element type (line 115)
        int_116218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_116214, int_116218)
        # Adding element type (line 115)
        int_116219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 19), list_116214, int_116219)
        
        # Assigning a type to the variable 'expected' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'expected', list_116214)
        
        # Call to assert_equal(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'actual' (line 116)
        actual_116221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'actual', False)
        # Getting the type of 'expected' (line 116)
        expected_116222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'expected', False)
        # Processing the call keyword arguments (line 116)
        kwargs_116223 = {}
        # Getting the type of 'assert_equal' (line 116)
        assert_equal_116220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 116)
        assert_equal_call_result_116224 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert_equal_116220, *[actual_116221, expected_116222], **kwargs_116223)
        
        
        # ################# End of 'test_bytescale_rounding(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale_rounding' in the type store
        # Getting the type of 'stypy_return_type' (line 110)
        stypy_return_type_116225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale_rounding'
        return stypy_return_type_116225


    @norecursion
    def test_bytescale_low_greaterthan_high(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale_low_greaterthan_high'
        module_type_store = module_type_store.open_function_context('test_bytescale_low_greaterthan_high', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale_low_greaterthan_high')
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale_low_greaterthan_high.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale_low_greaterthan_high', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale_low_greaterthan_high', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale_low_greaterthan_high(...)' code ##################

        
        # Call to assert_raises(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'ValueError' (line 119)
        ValueError_116227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 119)
        kwargs_116228 = {}
        # Getting the type of 'assert_raises' (line 119)
        assert_raises_116226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 119)
        assert_raises_call_result_116229 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), assert_raises_116226, *[ValueError_116227], **kwargs_116228)
        
        with_116230 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 119, 13), assert_raises_call_result_116229, 'with parameter', '__enter__', '__exit__')

        if with_116230:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 119)
            enter___116231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 13), assert_raises_call_result_116229, '__enter__')
            with_enter_116232 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), enter___116231)
            
            # Call to suppress_warnings(...): (line 120)
            # Processing the call keyword arguments (line 120)
            kwargs_116234 = {}
            # Getting the type of 'suppress_warnings' (line 120)
            suppress_warnings_116233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'suppress_warnings', False)
            # Calling suppress_warnings(args, kwargs) (line 120)
            suppress_warnings_call_result_116235 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), suppress_warnings_116233, *[], **kwargs_116234)
            
            with_116236 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 120, 17), suppress_warnings_call_result_116235, 'with parameter', '__enter__', '__exit__')

            if with_116236:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 120)
                enter___116237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), suppress_warnings_call_result_116235, '__enter__')
                with_enter_116238 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), enter___116237)
                # Assigning a type to the variable 'sup' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'sup', with_enter_116238)
                
                # Call to filter(...): (line 121)
                # Processing the call arguments (line 121)
                # Getting the type of 'DeprecationWarning' (line 121)
                DeprecationWarning_116241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'DeprecationWarning', False)
                # Processing the call keyword arguments (line 121)
                kwargs_116242 = {}
                # Getting the type of 'sup' (line 121)
                sup_116239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'sup', False)
                # Obtaining the member 'filter' of a type (line 121)
                filter_116240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), sup_116239, 'filter')
                # Calling filter(args, kwargs) (line 121)
                filter_call_result_116243 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), filter_116240, *[DeprecationWarning_116241], **kwargs_116242)
                
                
                # Call to bytescale(...): (line 122)
                # Processing the call arguments (line 122)
                
                # Call to arange(...): (line 122)
                # Processing the call arguments (line 122)
                int_116248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 41), 'int')
                # Processing the call keyword arguments (line 122)
                kwargs_116249 = {}
                # Getting the type of 'np' (line 122)
                np_116246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'np', False)
                # Obtaining the member 'arange' of a type (line 122)
                arange_116247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 31), np_116246, 'arange')
                # Calling arange(args, kwargs) (line 122)
                arange_call_result_116250 = invoke(stypy.reporting.localization.Localization(__file__, 122, 31), arange_116247, *[int_116248], **kwargs_116249)
                
                # Processing the call keyword arguments (line 122)
                int_116251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 49), 'int')
                keyword_116252 = int_116251
                int_116253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 58), 'int')
                keyword_116254 = int_116253
                kwargs_116255 = {'high': keyword_116254, 'low': keyword_116252}
                # Getting the type of 'misc' (line 122)
                misc_116244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'misc', False)
                # Obtaining the member 'bytescale' of a type (line 122)
                bytescale_116245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), misc_116244, 'bytescale')
                # Calling bytescale(args, kwargs) (line 122)
                bytescale_call_result_116256 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), bytescale_116245, *[arange_call_result_116250], **kwargs_116255)
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 120)
                exit___116257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), suppress_warnings_call_result_116235, '__exit__')
                with_exit_116258 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), exit___116257, None, None, None)

            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 119)
            exit___116259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 13), assert_raises_call_result_116229, '__exit__')
            with_exit_116260 = invoke(stypy.reporting.localization.Localization(__file__, 119, 13), exit___116259, None, None, None)

        
        # ################# End of 'test_bytescale_low_greaterthan_high(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale_low_greaterthan_high' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_116261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale_low_greaterthan_high'
        return stypy_return_type_116261


    @norecursion
    def test_bytescale_low_lessthan_0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale_low_lessthan_0'
        module_type_store = module_type_store.open_function_context('test_bytescale_low_lessthan_0', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale_low_lessthan_0')
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale_low_lessthan_0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale_low_lessthan_0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale_low_lessthan_0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale_low_lessthan_0(...)' code ##################

        
        # Call to assert_raises(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'ValueError' (line 125)
        ValueError_116263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 125)
        kwargs_116264 = {}
        # Getting the type of 'assert_raises' (line 125)
        assert_raises_116262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 125)
        assert_raises_call_result_116265 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), assert_raises_116262, *[ValueError_116263], **kwargs_116264)
        
        with_116266 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 125, 13), assert_raises_call_result_116265, 'with parameter', '__enter__', '__exit__')

        if with_116266:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 125)
            enter___116267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), assert_raises_call_result_116265, '__enter__')
            with_enter_116268 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), enter___116267)
            
            # Call to suppress_warnings(...): (line 126)
            # Processing the call keyword arguments (line 126)
            kwargs_116270 = {}
            # Getting the type of 'suppress_warnings' (line 126)
            suppress_warnings_116269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'suppress_warnings', False)
            # Calling suppress_warnings(args, kwargs) (line 126)
            suppress_warnings_call_result_116271 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), suppress_warnings_116269, *[], **kwargs_116270)
            
            with_116272 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 126, 17), suppress_warnings_call_result_116271, 'with parameter', '__enter__', '__exit__')

            if with_116272:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 126)
                enter___116273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), suppress_warnings_call_result_116271, '__enter__')
                with_enter_116274 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), enter___116273)
                # Assigning a type to the variable 'sup' (line 126)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'sup', with_enter_116274)
                
                # Call to filter(...): (line 127)
                # Processing the call arguments (line 127)
                # Getting the type of 'DeprecationWarning' (line 127)
                DeprecationWarning_116277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'DeprecationWarning', False)
                # Processing the call keyword arguments (line 127)
                kwargs_116278 = {}
                # Getting the type of 'sup' (line 127)
                sup_116275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'sup', False)
                # Obtaining the member 'filter' of a type (line 127)
                filter_116276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), sup_116275, 'filter')
                # Calling filter(args, kwargs) (line 127)
                filter_call_result_116279 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), filter_116276, *[DeprecationWarning_116277], **kwargs_116278)
                
                
                # Call to bytescale(...): (line 128)
                # Processing the call arguments (line 128)
                
                # Call to arange(...): (line 128)
                # Processing the call arguments (line 128)
                int_116284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 41), 'int')
                # Processing the call keyword arguments (line 128)
                kwargs_116285 = {}
                # Getting the type of 'np' (line 128)
                np_116282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 31), 'np', False)
                # Obtaining the member 'arange' of a type (line 128)
                arange_116283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 31), np_116282, 'arange')
                # Calling arange(args, kwargs) (line 128)
                arange_call_result_116286 = invoke(stypy.reporting.localization.Localization(__file__, 128, 31), arange_116283, *[int_116284], **kwargs_116285)
                
                # Processing the call keyword arguments (line 128)
                int_116287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 49), 'int')
                keyword_116288 = int_116287
                kwargs_116289 = {'low': keyword_116288}
                # Getting the type of 'misc' (line 128)
                misc_116280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'misc', False)
                # Obtaining the member 'bytescale' of a type (line 128)
                bytescale_116281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), misc_116280, 'bytescale')
                # Calling bytescale(args, kwargs) (line 128)
                bytescale_call_result_116290 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), bytescale_116281, *[arange_call_result_116286], **kwargs_116289)
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 126)
                exit___116291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), suppress_warnings_call_result_116271, '__exit__')
                with_exit_116292 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), exit___116291, None, None, None)

            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 125)
            exit___116293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), assert_raises_call_result_116265, '__exit__')
            with_exit_116294 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), exit___116293, None, None, None)

        
        # ################# End of 'test_bytescale_low_lessthan_0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale_low_lessthan_0' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_116295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116295)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale_low_lessthan_0'
        return stypy_return_type_116295


    @norecursion
    def test_bytescale_high_greaterthan_255(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale_high_greaterthan_255'
        module_type_store = module_type_store.open_function_context('test_bytescale_high_greaterthan_255', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale_high_greaterthan_255')
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale_high_greaterthan_255.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale_high_greaterthan_255', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale_high_greaterthan_255', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale_high_greaterthan_255(...)' code ##################

        
        # Call to assert_raises(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'ValueError' (line 131)
        ValueError_116297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'ValueError', False)
        # Processing the call keyword arguments (line 131)
        kwargs_116298 = {}
        # Getting the type of 'assert_raises' (line 131)
        assert_raises_116296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 131)
        assert_raises_call_result_116299 = invoke(stypy.reporting.localization.Localization(__file__, 131, 13), assert_raises_116296, *[ValueError_116297], **kwargs_116298)
        
        with_116300 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 131, 13), assert_raises_call_result_116299, 'with parameter', '__enter__', '__exit__')

        if with_116300:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 131)
            enter___116301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 13), assert_raises_call_result_116299, '__enter__')
            with_enter_116302 = invoke(stypy.reporting.localization.Localization(__file__, 131, 13), enter___116301)
            
            # Call to suppress_warnings(...): (line 132)
            # Processing the call keyword arguments (line 132)
            kwargs_116304 = {}
            # Getting the type of 'suppress_warnings' (line 132)
            suppress_warnings_116303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'suppress_warnings', False)
            # Calling suppress_warnings(args, kwargs) (line 132)
            suppress_warnings_call_result_116305 = invoke(stypy.reporting.localization.Localization(__file__, 132, 17), suppress_warnings_116303, *[], **kwargs_116304)
            
            with_116306 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 132, 17), suppress_warnings_call_result_116305, 'with parameter', '__enter__', '__exit__')

            if with_116306:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 132)
                enter___116307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 17), suppress_warnings_call_result_116305, '__enter__')
                with_enter_116308 = invoke(stypy.reporting.localization.Localization(__file__, 132, 17), enter___116307)
                # Assigning a type to the variable 'sup' (line 132)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'sup', with_enter_116308)
                
                # Call to filter(...): (line 133)
                # Processing the call arguments (line 133)
                # Getting the type of 'DeprecationWarning' (line 133)
                DeprecationWarning_116311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), 'DeprecationWarning', False)
                # Processing the call keyword arguments (line 133)
                kwargs_116312 = {}
                # Getting the type of 'sup' (line 133)
                sup_116309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'sup', False)
                # Obtaining the member 'filter' of a type (line 133)
                filter_116310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), sup_116309, 'filter')
                # Calling filter(args, kwargs) (line 133)
                filter_call_result_116313 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), filter_116310, *[DeprecationWarning_116311], **kwargs_116312)
                
                
                # Call to bytescale(...): (line 134)
                # Processing the call arguments (line 134)
                
                # Call to arange(...): (line 134)
                # Processing the call arguments (line 134)
                int_116318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 41), 'int')
                # Processing the call keyword arguments (line 134)
                kwargs_116319 = {}
                # Getting the type of 'np' (line 134)
                np_116316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'np', False)
                # Obtaining the member 'arange' of a type (line 134)
                arange_116317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 31), np_116316, 'arange')
                # Calling arange(args, kwargs) (line 134)
                arange_call_result_116320 = invoke(stypy.reporting.localization.Localization(__file__, 134, 31), arange_116317, *[int_116318], **kwargs_116319)
                
                # Processing the call keyword arguments (line 134)
                int_116321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 50), 'int')
                keyword_116322 = int_116321
                kwargs_116323 = {'high': keyword_116322}
                # Getting the type of 'misc' (line 134)
                misc_116314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'misc', False)
                # Obtaining the member 'bytescale' of a type (line 134)
                bytescale_116315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), misc_116314, 'bytescale')
                # Calling bytescale(args, kwargs) (line 134)
                bytescale_call_result_116324 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), bytescale_116315, *[arange_call_result_116320], **kwargs_116323)
                
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 132)
                exit___116325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 17), suppress_warnings_call_result_116305, '__exit__')
                with_exit_116326 = invoke(stypy.reporting.localization.Localization(__file__, 132, 17), exit___116325, None, None, None)

            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 131)
            exit___116327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 13), assert_raises_call_result_116299, '__exit__')
            with_exit_116328 = invoke(stypy.reporting.localization.Localization(__file__, 131, 13), exit___116327, None, None, None)

        
        # ################# End of 'test_bytescale_high_greaterthan_255(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale_high_greaterthan_255' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_116329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116329)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale_high_greaterthan_255'
        return stypy_return_type_116329


    @norecursion
    def test_bytescale_low_equals_high(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytescale_low_equals_high'
        module_type_store = module_type_store.open_function_context('test_bytescale_low_equals_high', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_bytescale_low_equals_high')
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_bytescale_low_equals_high.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_bytescale_low_equals_high', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytescale_low_equals_high', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytescale_low_equals_high(...)' code ##################

        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to arange(...): (line 137)
        # Processing the call arguments (line 137)
        int_116332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'int')
        # Processing the call keyword arguments (line 137)
        kwargs_116333 = {}
        # Getting the type of 'np' (line 137)
        np_116330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 137)
        arange_116331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), np_116330, 'arange')
        # Calling arange(args, kwargs) (line 137)
        arange_call_result_116334 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), arange_116331, *[int_116332], **kwargs_116333)
        
        # Assigning a type to the variable 'a' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'a', arange_call_result_116334)
        
        # Call to suppress_warnings(...): (line 138)
        # Processing the call keyword arguments (line 138)
        kwargs_116336 = {}
        # Getting the type of 'suppress_warnings' (line 138)
        suppress_warnings_116335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 138)
        suppress_warnings_call_result_116337 = invoke(stypy.reporting.localization.Localization(__file__, 138, 13), suppress_warnings_116335, *[], **kwargs_116336)
        
        with_116338 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 138, 13), suppress_warnings_call_result_116337, 'with parameter', '__enter__', '__exit__')

        if with_116338:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 138)
            enter___116339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 13), suppress_warnings_call_result_116337, '__enter__')
            with_enter_116340 = invoke(stypy.reporting.localization.Localization(__file__, 138, 13), enter___116339)
            # Assigning a type to the variable 'sup' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'sup', with_enter_116340)
            
            # Call to filter(...): (line 139)
            # Processing the call arguments (line 139)
            # Getting the type of 'DeprecationWarning' (line 139)
            DeprecationWarning_116343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 139)
            kwargs_116344 = {}
            # Getting the type of 'sup' (line 139)
            sup_116341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 139)
            filter_116342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), sup_116341, 'filter')
            # Calling filter(args, kwargs) (line 139)
            filter_call_result_116345 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), filter_116342, *[DeprecationWarning_116343], **kwargs_116344)
            
            
            # Assigning a Call to a Name (line 140):
            
            # Assigning a Call to a Name (line 140):
            
            # Call to bytescale(...): (line 140)
            # Processing the call arguments (line 140)
            # Getting the type of 'a' (line 140)
            a_116348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 36), 'a', False)
            # Processing the call keyword arguments (line 140)
            int_116349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 43), 'int')
            keyword_116350 = int_116349
            int_116351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 52), 'int')
            keyword_116352 = int_116351
            kwargs_116353 = {'high': keyword_116352, 'low': keyword_116350}
            # Getting the type of 'misc' (line 140)
            misc_116346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'misc', False)
            # Obtaining the member 'bytescale' of a type (line 140)
            bytescale_116347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), misc_116346, 'bytescale')
            # Calling bytescale(args, kwargs) (line 140)
            bytescale_call_result_116354 = invoke(stypy.reporting.localization.Localization(__file__, 140, 21), bytescale_116347, *[a_116348], **kwargs_116353)
            
            # Assigning a type to the variable 'actual' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'actual', bytescale_call_result_116354)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 138)
            exit___116355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 13), suppress_warnings_call_result_116337, '__exit__')
            with_exit_116356 = invoke(stypy.reporting.localization.Localization(__file__, 138, 13), exit___116355, None, None, None)

        
        # Assigning a List to a Name (line 141):
        
        # Assigning a List to a Name (line 141):
        
        # Obtaining an instance of the builtin type 'list' (line 141)
        list_116357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 141)
        # Adding element type (line 141)
        int_116358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 19), list_116357, int_116358)
        # Adding element type (line 141)
        int_116359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 19), list_116357, int_116359)
        # Adding element type (line 141)
        int_116360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 19), list_116357, int_116360)
        
        # Assigning a type to the variable 'expected' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'expected', list_116357)
        
        # Call to assert_equal(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'actual' (line 142)
        actual_116362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'actual', False)
        # Getting the type of 'expected' (line 142)
        expected_116363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'expected', False)
        # Processing the call keyword arguments (line 142)
        kwargs_116364 = {}
        # Getting the type of 'assert_equal' (line 142)
        assert_equal_116361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 142)
        assert_equal_call_result_116365 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assert_equal_116361, *[actual_116362, expected_116363], **kwargs_116364)
        
        
        # ################# End of 'test_bytescale_low_equals_high(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytescale_low_equals_high' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_116366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytescale_low_equals_high'
        return stypy_return_type_116366


    @norecursion
    def test_imsave(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_imsave'
        module_type_store = module_type_store.open_function_context('test_imsave', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_localization', localization)
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_function_name', 'TestPILUtil.test_imsave')
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_param_names_list', [])
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPILUtil.test_imsave.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.test_imsave', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_imsave', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_imsave(...)' code ##################

        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to join(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'datapath' (line 145)
        datapath_116370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), 'datapath', False)
        str_116371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 40), 'str', 'data')
        # Processing the call keyword arguments (line 145)
        kwargs_116372 = {}
        # Getting the type of 'os' (line 145)
        os_116367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 145)
        path_116368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 17), os_116367, 'path')
        # Obtaining the member 'join' of a type (line 145)
        join_116369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 17), path_116368, 'join')
        # Calling join(args, kwargs) (line 145)
        join_call_result_116373 = invoke(stypy.reporting.localization.Localization(__file__, 145, 17), join_116369, *[datapath_116370, str_116371], **kwargs_116372)
        
        # Assigning a type to the variable 'picdir' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'picdir', join_call_result_116373)
        
        
        # Call to iglob(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'picdir' (line 146)
        picdir_116376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'picdir', False)
        str_116377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 39), 'str', '/*.png')
        # Applying the binary operator '+' (line 146)
        result_add_116378 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 30), '+', picdir_116376, str_116377)
        
        # Processing the call keyword arguments (line 146)
        kwargs_116379 = {}
        # Getting the type of 'glob' (line 146)
        glob_116374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'glob', False)
        # Obtaining the member 'iglob' of a type (line 146)
        iglob_116375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 19), glob_116374, 'iglob')
        # Calling iglob(args, kwargs) (line 146)
        iglob_call_result_116380 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), iglob_116375, *[result_add_116378], **kwargs_116379)
        
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), iglob_call_result_116380)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_116381 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), iglob_call_result_116380)
        # Assigning a type to the variable 'png' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'png', for_loop_var_116381)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to suppress_warnings(...): (line 147)
        # Processing the call keyword arguments (line 147)
        kwargs_116383 = {}
        # Getting the type of 'suppress_warnings' (line 147)
        suppress_warnings_116382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 147)
        suppress_warnings_call_result_116384 = invoke(stypy.reporting.localization.Localization(__file__, 147, 17), suppress_warnings_116382, *[], **kwargs_116383)
        
        with_116385 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 147, 17), suppress_warnings_call_result_116384, 'with parameter', '__enter__', '__exit__')

        if with_116385:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 147)
            enter___116386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 17), suppress_warnings_call_result_116384, '__enter__')
            with_enter_116387 = invoke(stypy.reporting.localization.Localization(__file__, 147, 17), enter___116386)
            # Assigning a type to the variable 'sup' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'sup', with_enter_116387)
            
            # Call to filter(...): (line 149)
            # Processing the call keyword arguments (line 149)
            str_116390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'str', 'unclosed file')
            keyword_116391 = str_116390
            kwargs_116392 = {'message': keyword_116391}
            # Getting the type of 'sup' (line 149)
            sup_116388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 149)
            filter_116389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), sup_116388, 'filter')
            # Calling filter(args, kwargs) (line 149)
            filter_call_result_116393 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), filter_116389, *[], **kwargs_116392)
            
            
            # Call to filter(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'DeprecationWarning' (line 150)
            DeprecationWarning_116396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 150)
            kwargs_116397 = {}
            # Getting the type of 'sup' (line 150)
            sup_116394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'sup', False)
            # Obtaining the member 'filter' of a type (line 150)
            filter_116395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), sup_116394, 'filter')
            # Calling filter(args, kwargs) (line 150)
            filter_call_result_116398 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), filter_116395, *[DeprecationWarning_116396], **kwargs_116397)
            
            
            # Assigning a Call to a Name (line 151):
            
            # Assigning a Call to a Name (line 151):
            
            # Call to imread(...): (line 151)
            # Processing the call arguments (line 151)
            # Getting the type of 'png' (line 151)
            png_116401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'png', False)
            # Processing the call keyword arguments (line 151)
            kwargs_116402 = {}
            # Getting the type of 'misc' (line 151)
            misc_116399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'misc', False)
            # Obtaining the member 'imread' of a type (line 151)
            imread_116400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 22), misc_116399, 'imread')
            # Calling imread(args, kwargs) (line 151)
            imread_call_result_116403 = invoke(stypy.reporting.localization.Localization(__file__, 151, 22), imread_116400, *[png_116401], **kwargs_116402)
            
            # Assigning a type to the variable 'img' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'img', imread_call_result_116403)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 147)
            exit___116404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 17), suppress_warnings_call_result_116384, '__exit__')
            with_exit_116405 = invoke(stypy.reporting.localization.Localization(__file__, 147, 17), exit___116404, None, None, None)

        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to mkdtemp(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_116408 = {}
        # Getting the type of 'tempfile' (line 152)
        tempfile_116406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'tempfile', False)
        # Obtaining the member 'mkdtemp' of a type (line 152)
        mkdtemp_116407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 21), tempfile_116406, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 152)
        mkdtemp_call_result_116409 = invoke(stypy.reporting.localization.Localization(__file__, 152, 21), mkdtemp_116407, *[], **kwargs_116408)
        
        # Assigning a type to the variable 'tmpdir' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'tmpdir', mkdtemp_call_result_116409)
        
        # Try-finally block (line 153)
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to join(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'tmpdir' (line 154)
        tmpdir_116413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 35), 'tmpdir', False)
        str_116414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 43), 'str', 'test.png')
        # Processing the call keyword arguments (line 154)
        kwargs_116415 = {}
        # Getting the type of 'os' (line 154)
        os_116410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 154)
        path_116411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), os_116410, 'path')
        # Obtaining the member 'join' of a type (line 154)
        join_116412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), path_116411, 'join')
        # Calling join(args, kwargs) (line 154)
        join_call_result_116416 = invoke(stypy.reporting.localization.Localization(__file__, 154, 22), join_116412, *[tmpdir_116413, str_116414], **kwargs_116415)
        
        # Assigning a type to the variable 'fn1' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'fn1', join_call_result_116416)
        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to join(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'tmpdir' (line 155)
        tmpdir_116420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'tmpdir', False)
        str_116421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 43), 'str', 'testimg')
        # Processing the call keyword arguments (line 155)
        kwargs_116422 = {}
        # Getting the type of 'os' (line 155)
        os_116417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 155)
        path_116418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 22), os_116417, 'path')
        # Obtaining the member 'join' of a type (line 155)
        join_116419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 22), path_116418, 'join')
        # Calling join(args, kwargs) (line 155)
        join_call_result_116423 = invoke(stypy.reporting.localization.Localization(__file__, 155, 22), join_116419, *[tmpdir_116420, str_116421], **kwargs_116422)
        
        # Assigning a type to the variable 'fn2' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'fn2', join_call_result_116423)
        
        # Call to suppress_warnings(...): (line 156)
        # Processing the call keyword arguments (line 156)
        kwargs_116425 = {}
        # Getting the type of 'suppress_warnings' (line 156)
        suppress_warnings_116424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 156)
        suppress_warnings_call_result_116426 = invoke(stypy.reporting.localization.Localization(__file__, 156, 21), suppress_warnings_116424, *[], **kwargs_116425)
        
        with_116427 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 156, 21), suppress_warnings_call_result_116426, 'with parameter', '__enter__', '__exit__')

        if with_116427:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 156)
            enter___116428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 21), suppress_warnings_call_result_116426, '__enter__')
            with_enter_116429 = invoke(stypy.reporting.localization.Localization(__file__, 156, 21), enter___116428)
            # Assigning a type to the variable 'sup' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'sup', with_enter_116429)
            
            # Call to filter(...): (line 158)
            # Processing the call keyword arguments (line 158)
            str_116432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 39), 'str', 'unclosed file')
            keyword_116433 = str_116432
            kwargs_116434 = {'message': keyword_116433}
            # Getting the type of 'sup' (line 158)
            sup_116430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'sup', False)
            # Obtaining the member 'filter' of a type (line 158)
            filter_116431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 20), sup_116430, 'filter')
            # Calling filter(args, kwargs) (line 158)
            filter_call_result_116435 = invoke(stypy.reporting.localization.Localization(__file__, 158, 20), filter_116431, *[], **kwargs_116434)
            
            
            # Call to filter(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 'DeprecationWarning' (line 159)
            DeprecationWarning_116438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 159)
            kwargs_116439 = {}
            # Getting the type of 'sup' (line 159)
            sup_116436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'sup', False)
            # Obtaining the member 'filter' of a type (line 159)
            filter_116437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 20), sup_116436, 'filter')
            # Calling filter(args, kwargs) (line 159)
            filter_call_result_116440 = invoke(stypy.reporting.localization.Localization(__file__, 159, 20), filter_116437, *[DeprecationWarning_116438], **kwargs_116439)
            
            
            # Call to imsave(...): (line 160)
            # Processing the call arguments (line 160)
            # Getting the type of 'fn1' (line 160)
            fn1_116443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'fn1', False)
            # Getting the type of 'img' (line 160)
            img_116444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'img', False)
            # Processing the call keyword arguments (line 160)
            kwargs_116445 = {}
            # Getting the type of 'misc' (line 160)
            misc_116441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'misc', False)
            # Obtaining the member 'imsave' of a type (line 160)
            imsave_116442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), misc_116441, 'imsave')
            # Calling imsave(args, kwargs) (line 160)
            imsave_call_result_116446 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), imsave_116442, *[fn1_116443, img_116444], **kwargs_116445)
            
            
            # Call to imsave(...): (line 161)
            # Processing the call arguments (line 161)
            # Getting the type of 'fn2' (line 161)
            fn2_116449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'fn2', False)
            # Getting the type of 'img' (line 161)
            img_116450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 37), 'img', False)
            str_116451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 42), 'str', 'PNG')
            # Processing the call keyword arguments (line 161)
            kwargs_116452 = {}
            # Getting the type of 'misc' (line 161)
            misc_116447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'misc', False)
            # Obtaining the member 'imsave' of a type (line 161)
            imsave_116448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 20), misc_116447, 'imsave')
            # Calling imsave(args, kwargs) (line 161)
            imsave_call_result_116453 = invoke(stypy.reporting.localization.Localization(__file__, 161, 20), imsave_116448, *[fn2_116449, img_116450, str_116451], **kwargs_116452)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 156)
            exit___116454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 21), suppress_warnings_call_result_116426, '__exit__')
            with_exit_116455 = invoke(stypy.reporting.localization.Localization(__file__, 156, 21), exit___116454, None, None, None)

        
        # Call to suppress_warnings(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_116457 = {}
        # Getting the type of 'suppress_warnings' (line 163)
        suppress_warnings_116456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 163)
        suppress_warnings_call_result_116458 = invoke(stypy.reporting.localization.Localization(__file__, 163, 21), suppress_warnings_116456, *[], **kwargs_116457)
        
        with_116459 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 163, 21), suppress_warnings_call_result_116458, 'with parameter', '__enter__', '__exit__')

        if with_116459:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 163)
            enter___116460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 21), suppress_warnings_call_result_116458, '__enter__')
            with_enter_116461 = invoke(stypy.reporting.localization.Localization(__file__, 163, 21), enter___116460)
            # Assigning a type to the variable 'sup' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'sup', with_enter_116461)
            
            # Call to filter(...): (line 165)
            # Processing the call keyword arguments (line 165)
            str_116464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 39), 'str', 'unclosed file')
            keyword_116465 = str_116464
            kwargs_116466 = {'message': keyword_116465}
            # Getting the type of 'sup' (line 165)
            sup_116462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'sup', False)
            # Obtaining the member 'filter' of a type (line 165)
            filter_116463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 20), sup_116462, 'filter')
            # Calling filter(args, kwargs) (line 165)
            filter_call_result_116467 = invoke(stypy.reporting.localization.Localization(__file__, 165, 20), filter_116463, *[], **kwargs_116466)
            
            
            # Call to filter(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'DeprecationWarning' (line 166)
            DeprecationWarning_116470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 166)
            kwargs_116471 = {}
            # Getting the type of 'sup' (line 166)
            sup_116468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'sup', False)
            # Obtaining the member 'filter' of a type (line 166)
            filter_116469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 20), sup_116468, 'filter')
            # Calling filter(args, kwargs) (line 166)
            filter_call_result_116472 = invoke(stypy.reporting.localization.Localization(__file__, 166, 20), filter_116469, *[DeprecationWarning_116470], **kwargs_116471)
            
            
            # Assigning a Call to a Name (line 167):
            
            # Assigning a Call to a Name (line 167):
            
            # Call to imread(...): (line 167)
            # Processing the call arguments (line 167)
            # Getting the type of 'fn1' (line 167)
            fn1_116475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 40), 'fn1', False)
            # Processing the call keyword arguments (line 167)
            kwargs_116476 = {}
            # Getting the type of 'misc' (line 167)
            misc_116473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'misc', False)
            # Obtaining the member 'imread' of a type (line 167)
            imread_116474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 28), misc_116473, 'imread')
            # Calling imread(args, kwargs) (line 167)
            imread_call_result_116477 = invoke(stypy.reporting.localization.Localization(__file__, 167, 28), imread_116474, *[fn1_116475], **kwargs_116476)
            
            # Assigning a type to the variable 'data1' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'data1', imread_call_result_116477)
            
            # Assigning a Call to a Name (line 168):
            
            # Assigning a Call to a Name (line 168):
            
            # Call to imread(...): (line 168)
            # Processing the call arguments (line 168)
            # Getting the type of 'fn2' (line 168)
            fn2_116480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 40), 'fn2', False)
            # Processing the call keyword arguments (line 168)
            kwargs_116481 = {}
            # Getting the type of 'misc' (line 168)
            misc_116478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'misc', False)
            # Obtaining the member 'imread' of a type (line 168)
            imread_116479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 28), misc_116478, 'imread')
            # Calling imread(args, kwargs) (line 168)
            imread_call_result_116482 = invoke(stypy.reporting.localization.Localization(__file__, 168, 28), imread_116479, *[fn2_116480], **kwargs_116481)
            
            # Assigning a type to the variable 'data2' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'data2', imread_call_result_116482)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 163)
            exit___116483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 21), suppress_warnings_call_result_116458, '__exit__')
            with_exit_116484 = invoke(stypy.reporting.localization.Localization(__file__, 163, 21), exit___116483, None, None, None)

        
        # Call to assert_allclose(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'data1' (line 169)
        data1_116486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 32), 'data1', False)
        # Getting the type of 'img' (line 169)
        img_116487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 39), 'img', False)
        # Processing the call keyword arguments (line 169)
        kwargs_116488 = {}
        # Getting the type of 'assert_allclose' (line 169)
        assert_allclose_116485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 169)
        assert_allclose_call_result_116489 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), assert_allclose_116485, *[data1_116486, img_116487], **kwargs_116488)
        
        
        # Call to assert_allclose(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'data2' (line 170)
        data2_116491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 32), 'data2', False)
        # Getting the type of 'img' (line 170)
        img_116492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 39), 'img', False)
        # Processing the call keyword arguments (line 170)
        kwargs_116493 = {}
        # Getting the type of 'assert_allclose' (line 170)
        assert_allclose_116490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 170)
        assert_allclose_call_result_116494 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), assert_allclose_116490, *[data2_116491, img_116492], **kwargs_116493)
        
        
        # Call to assert_equal(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'data1' (line 171)
        data1_116496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'data1', False)
        # Obtaining the member 'shape' of a type (line 171)
        shape_116497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 29), data1_116496, 'shape')
        # Getting the type of 'img' (line 171)
        img_116498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'img', False)
        # Obtaining the member 'shape' of a type (line 171)
        shape_116499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 42), img_116498, 'shape')
        # Processing the call keyword arguments (line 171)
        kwargs_116500 = {}
        # Getting the type of 'assert_equal' (line 171)
        assert_equal_116495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 171)
        assert_equal_call_result_116501 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), assert_equal_116495, *[shape_116497, shape_116499], **kwargs_116500)
        
        
        # Call to assert_equal(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'data2' (line 172)
        data2_116503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 29), 'data2', False)
        # Obtaining the member 'shape' of a type (line 172)
        shape_116504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 29), data2_116503, 'shape')
        # Getting the type of 'img' (line 172)
        img_116505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 42), 'img', False)
        # Obtaining the member 'shape' of a type (line 172)
        shape_116506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 42), img_116505, 'shape')
        # Processing the call keyword arguments (line 172)
        kwargs_116507 = {}
        # Getting the type of 'assert_equal' (line 172)
        assert_equal_116502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 172)
        assert_equal_call_result_116508 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), assert_equal_116502, *[shape_116504, shape_116506], **kwargs_116507)
        
        
        # finally branch of the try-finally block (line 153)
        
        # Call to rmtree(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'tmpdir' (line 174)
        tmpdir_116511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'tmpdir', False)
        # Processing the call keyword arguments (line 174)
        kwargs_116512 = {}
        # Getting the type of 'shutil' (line 174)
        shutil_116509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 174)
        rmtree_116510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), shutil_116509, 'rmtree')
        # Calling rmtree(args, kwargs) (line 174)
        rmtree_call_result_116513 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), rmtree_116510, *[tmpdir_116511], **kwargs_116512)
        
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_imsave(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_imsave' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_116514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_116514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_imsave'
        return stypy_return_type_116514


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 0, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPILUtil.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestPILUtil' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'TestPILUtil', TestPILUtil)
# Getting the type of 'TestPILUtil' (line 30)
TestPILUtil_116515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'TestPILUtil')
class_116516 = invoke(stypy.reporting.localization.Localization(__file__, 30, 0), _pilskip_115701, TestPILUtil_116515)
# Assigning a type to the variable 'TestPILUtil' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'TestPILUtil', class_116516)

@norecursion
def check_fromimage(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_fromimage'
    module_type_store = module_type_store.open_function_context('check_fromimage', 177, 0, False)
    
    # Passed parameters checking function
    check_fromimage.stypy_localization = localization
    check_fromimage.stypy_type_of_self = None
    check_fromimage.stypy_type_store = module_type_store
    check_fromimage.stypy_function_name = 'check_fromimage'
    check_fromimage.stypy_param_names_list = ['filename', 'irange', 'shape']
    check_fromimage.stypy_varargs_param_name = None
    check_fromimage.stypy_kwargs_param_name = None
    check_fromimage.stypy_call_defaults = defaults
    check_fromimage.stypy_call_varargs = varargs
    check_fromimage.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_fromimage', ['filename', 'irange', 'shape'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_fromimage', localization, ['filename', 'irange', 'shape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_fromimage(...)' code ##################

    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to open(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'filename' (line 178)
    filename_116518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 14), 'filename', False)
    str_116519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 24), 'str', 'rb')
    # Processing the call keyword arguments (line 178)
    kwargs_116520 = {}
    # Getting the type of 'open' (line 178)
    open_116517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 9), 'open', False)
    # Calling open(args, kwargs) (line 178)
    open_call_result_116521 = invoke(stypy.reporting.localization.Localization(__file__, 178, 9), open_116517, *[filename_116518, str_116519], **kwargs_116520)
    
    # Assigning a type to the variable 'fp' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'fp', open_call_result_116521)
    
    # Call to suppress_warnings(...): (line 179)
    # Processing the call keyword arguments (line 179)
    kwargs_116523 = {}
    # Getting the type of 'suppress_warnings' (line 179)
    suppress_warnings_116522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 179)
    suppress_warnings_call_result_116524 = invoke(stypy.reporting.localization.Localization(__file__, 179, 9), suppress_warnings_116522, *[], **kwargs_116523)
    
    with_116525 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 179, 9), suppress_warnings_call_result_116524, 'with parameter', '__enter__', '__exit__')

    if with_116525:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 179)
        enter___116526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 9), suppress_warnings_call_result_116524, '__enter__')
        with_enter_116527 = invoke(stypy.reporting.localization.Localization(__file__, 179, 9), enter___116526)
        # Assigning a type to the variable 'sup' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 9), 'sup', with_enter_116527)
        
        # Call to filter(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'DeprecationWarning' (line 180)
        DeprecationWarning_116530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 180)
        kwargs_116531 = {}
        # Getting the type of 'sup' (line 180)
        sup_116528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 180)
        filter_116529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), sup_116528, 'filter')
        # Calling filter(args, kwargs) (line 180)
        filter_call_result_116532 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), filter_116529, *[DeprecationWarning_116530], **kwargs_116531)
        
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to fromimage(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to open(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'fp' (line 181)
        fp_116538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 44), 'fp', False)
        # Processing the call keyword arguments (line 181)
        kwargs_116539 = {}
        # Getting the type of 'PIL' (line 181)
        PIL_116535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'PIL', False)
        # Obtaining the member 'Image' of a type (line 181)
        Image_116536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 29), PIL_116535, 'Image')
        # Obtaining the member 'open' of a type (line 181)
        open_116537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 29), Image_116536, 'open')
        # Calling open(args, kwargs) (line 181)
        open_call_result_116540 = invoke(stypy.reporting.localization.Localization(__file__, 181, 29), open_116537, *[fp_116538], **kwargs_116539)
        
        # Processing the call keyword arguments (line 181)
        kwargs_116541 = {}
        # Getting the type of 'misc' (line 181)
        misc_116533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'misc', False)
        # Obtaining the member 'fromimage' of a type (line 181)
        fromimage_116534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 14), misc_116533, 'fromimage')
        # Calling fromimage(args, kwargs) (line 181)
        fromimage_call_result_116542 = invoke(stypy.reporting.localization.Localization(__file__, 181, 14), fromimage_116534, *[open_call_result_116540], **kwargs_116541)
        
        # Assigning a type to the variable 'img' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'img', fromimage_call_result_116542)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 179)
        exit___116543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 9), suppress_warnings_call_result_116524, '__exit__')
        with_exit_116544 = invoke(stypy.reporting.localization.Localization(__file__, 179, 9), exit___116543, None, None, None)

    
    # Call to close(...): (line 182)
    # Processing the call keyword arguments (line 182)
    kwargs_116547 = {}
    # Getting the type of 'fp' (line 182)
    fp_116545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'fp', False)
    # Obtaining the member 'close' of a type (line 182)
    close_116546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 4), fp_116545, 'close')
    # Calling close(args, kwargs) (line 182)
    close_call_result_116548 = invoke(stypy.reporting.localization.Localization(__file__, 182, 4), close_116546, *[], **kwargs_116547)
    
    
    # Assigning a Name to a Tuple (line 183):
    
    # Assigning a Subscript to a Name (line 183):
    
    # Obtaining the type of the subscript
    int_116549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    # Getting the type of 'irange' (line 183)
    irange_116550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'irange')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___116551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), irange_116550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_116552 = invoke(stypy.reporting.localization.Localization(__file__, 183, 4), getitem___116551, int_116549)
    
    # Assigning a type to the variable 'tuple_var_assignment_115662' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'tuple_var_assignment_115662', subscript_call_result_116552)
    
    # Assigning a Subscript to a Name (line 183):
    
    # Obtaining the type of the subscript
    int_116553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'int')
    # Getting the type of 'irange' (line 183)
    irange_116554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'irange')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___116555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 4), irange_116554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_116556 = invoke(stypy.reporting.localization.Localization(__file__, 183, 4), getitem___116555, int_116553)
    
    # Assigning a type to the variable 'tuple_var_assignment_115663' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'tuple_var_assignment_115663', subscript_call_result_116556)
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'tuple_var_assignment_115662' (line 183)
    tuple_var_assignment_115662_116557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'tuple_var_assignment_115662')
    # Assigning a type to the variable 'imin' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'imin', tuple_var_assignment_115662_116557)
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'tuple_var_assignment_115663' (line 183)
    tuple_var_assignment_115663_116558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'tuple_var_assignment_115663')
    # Assigning a type to the variable 'imax' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 10), 'imax', tuple_var_assignment_115663_116558)
    
    # Call to assert_equal(...): (line 184)
    # Processing the call arguments (line 184)
    
    # Call to min(...): (line 184)
    # Processing the call keyword arguments (line 184)
    kwargs_116562 = {}
    # Getting the type of 'img' (line 184)
    img_116560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 17), 'img', False)
    # Obtaining the member 'min' of a type (line 184)
    min_116561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 17), img_116560, 'min')
    # Calling min(args, kwargs) (line 184)
    min_call_result_116563 = invoke(stypy.reporting.localization.Localization(__file__, 184, 17), min_116561, *[], **kwargs_116562)
    
    # Getting the type of 'imin' (line 184)
    imin_116564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'imin', False)
    # Processing the call keyword arguments (line 184)
    kwargs_116565 = {}
    # Getting the type of 'assert_equal' (line 184)
    assert_equal_116559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 184)
    assert_equal_call_result_116566 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), assert_equal_116559, *[min_call_result_116563, imin_116564], **kwargs_116565)
    
    
    # Call to assert_equal(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Call to max(...): (line 185)
    # Processing the call keyword arguments (line 185)
    kwargs_116570 = {}
    # Getting the type of 'img' (line 185)
    img_116568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 17), 'img', False)
    # Obtaining the member 'max' of a type (line 185)
    max_116569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 17), img_116568, 'max')
    # Calling max(args, kwargs) (line 185)
    max_call_result_116571 = invoke(stypy.reporting.localization.Localization(__file__, 185, 17), max_116569, *[], **kwargs_116570)
    
    # Getting the type of 'imax' (line 185)
    imax_116572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 28), 'imax', False)
    # Processing the call keyword arguments (line 185)
    kwargs_116573 = {}
    # Getting the type of 'assert_equal' (line 185)
    assert_equal_116567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 185)
    assert_equal_call_result_116574 = invoke(stypy.reporting.localization.Localization(__file__, 185, 4), assert_equal_116567, *[max_call_result_116571, imax_116572], **kwargs_116573)
    
    
    # Call to assert_equal(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'img' (line 186)
    img_116576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 17), 'img', False)
    # Obtaining the member 'shape' of a type (line 186)
    shape_116577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 17), img_116576, 'shape')
    # Getting the type of 'shape' (line 186)
    shape_116578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'shape', False)
    # Processing the call keyword arguments (line 186)
    kwargs_116579 = {}
    # Getting the type of 'assert_equal' (line 186)
    assert_equal_116575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 186)
    assert_equal_call_result_116580 = invoke(stypy.reporting.localization.Localization(__file__, 186, 4), assert_equal_116575, *[shape_116577, shape_116578], **kwargs_116579)
    
    
    # ################# End of 'check_fromimage(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_fromimage' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_116581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116581)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_fromimage'
    return stypy_return_type_116581

# Assigning a type to the variable 'check_fromimage' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'check_fromimage', check_fromimage)

@norecursion
def test_fromimage(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_fromimage'
    module_type_store = module_type_store.open_function_context('test_fromimage', 189, 0, False)
    
    # Passed parameters checking function
    test_fromimage.stypy_localization = localization
    test_fromimage.stypy_type_of_self = None
    test_fromimage.stypy_type_store = module_type_store
    test_fromimage.stypy_function_name = 'test_fromimage'
    test_fromimage.stypy_param_names_list = []
    test_fromimage.stypy_varargs_param_name = None
    test_fromimage.stypy_kwargs_param_name = None
    test_fromimage.stypy_call_defaults = defaults
    test_fromimage.stypy_call_varargs = varargs
    test_fromimage.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_fromimage', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_fromimage', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_fromimage(...)' code ##################

    
    # Assigning a List to a Name (line 193):
    
    # Assigning a List to a Name (line 193):
    
    # Obtaining an instance of the builtin type 'list' (line 193)
    list_116582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 193)
    # Adding element type (line 193)
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_116583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    str_116584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 14), 'str', 'icon.png')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 14), tuple_116583, str_116584)
    # Adding element type (line 193)
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_116585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    int_116586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 27), tuple_116585, int_116586)
    # Adding element type (line 193)
    int_116587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 27), tuple_116585, int_116587)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 14), tuple_116583, tuple_116585)
    # Adding element type (line 193)
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_116588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    int_116589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 37), tuple_116588, int_116589)
    # Adding element type (line 193)
    int_116590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 37), tuple_116588, int_116590)
    # Adding element type (line 193)
    int_116591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 37), tuple_116588, int_116591)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 14), tuple_116583, tuple_116588)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 12), list_116582, tuple_116583)
    # Adding element type (line 193)
    
    # Obtaining an instance of the builtin type 'tuple' (line 194)
    tuple_116592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 194)
    # Adding element type (line 194)
    str_116593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 14), 'str', 'icon_mono.png')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 14), tuple_116592, str_116593)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'tuple' (line 194)
    tuple_116594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 194)
    # Adding element type (line 194)
    int_116595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 32), tuple_116594, int_116595)
    # Adding element type (line 194)
    int_116596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 32), tuple_116594, int_116596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 14), tuple_116592, tuple_116594)
    # Adding element type (line 194)
    
    # Obtaining an instance of the builtin type 'tuple' (line 194)
    tuple_116597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 194)
    # Adding element type (line 194)
    int_116598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 42), tuple_116597, int_116598)
    # Adding element type (line 194)
    int_116599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 42), tuple_116597, int_116599)
    # Adding element type (line 194)
    int_116600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 42), tuple_116597, int_116600)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 14), tuple_116592, tuple_116597)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 12), list_116582, tuple_116592)
    # Adding element type (line 193)
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_116601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    # Adding element type (line 195)
    str_116602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 14), 'str', 'icon_mono_flat.png')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 14), tuple_116601, str_116602)
    # Adding element type (line 195)
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_116603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    # Adding element type (line 195)
    int_116604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 37), tuple_116603, int_116604)
    # Adding element type (line 195)
    int_116605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 37), tuple_116603, int_116605)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 14), tuple_116601, tuple_116603)
    # Adding element type (line 195)
    
    # Obtaining an instance of the builtin type 'tuple' (line 195)
    tuple_116606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 195)
    # Adding element type (line 195)
    int_116607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 47), tuple_116606, int_116607)
    # Adding element type (line 195)
    int_116608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 51), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 47), tuple_116606, int_116608)
    # Adding element type (line 195)
    int_116609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 47), tuple_116606, int_116609)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 14), tuple_116601, tuple_116606)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 12), list_116582, tuple_116601)
    
    # Assigning a type to the variable 'files' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'files', list_116582)
    
    # Getting the type of 'files' (line 196)
    files_116610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 29), 'files')
    # Testing the type of a for loop iterable (line 196)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 4), files_116610)
    # Getting the type of the for loop variable (line 196)
    for_loop_var_116611 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 4), files_116610)
    # Assigning a type to the variable 'fn' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'fn', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 4), for_loop_var_116611))
    # Assigning a type to the variable 'irange' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'irange', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 4), for_loop_var_116611))
    # Assigning a type to the variable 'shape' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'shape', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 4), for_loop_var_116611))
    # SSA begins for a for statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to suppress_warnings(...): (line 197)
    # Processing the call keyword arguments (line 197)
    kwargs_116613 = {}
    # Getting the type of 'suppress_warnings' (line 197)
    suppress_warnings_116612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 197)
    suppress_warnings_call_result_116614 = invoke(stypy.reporting.localization.Localization(__file__, 197, 13), suppress_warnings_116612, *[], **kwargs_116613)
    
    with_116615 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 197, 13), suppress_warnings_call_result_116614, 'with parameter', '__enter__', '__exit__')

    if with_116615:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 197)
        enter___116616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 13), suppress_warnings_call_result_116614, '__enter__')
        with_enter_116617 = invoke(stypy.reporting.localization.Localization(__file__, 197, 13), enter___116616)
        # Assigning a type to the variable 'sup' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 13), 'sup', with_enter_116617)
        
        # Call to filter(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'DeprecationWarning' (line 198)
        DeprecationWarning_116620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 198)
        kwargs_116621 = {}
        # Getting the type of 'sup' (line 198)
        sup_116618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'sup', False)
        # Obtaining the member 'filter' of a type (line 198)
        filter_116619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), sup_116618, 'filter')
        # Calling filter(args, kwargs) (line 198)
        filter_call_result_116622 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), filter_116619, *[DeprecationWarning_116620], **kwargs_116621)
        
        
        # Call to check_fromimage(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Call to join(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'datapath' (line 199)
        datapath_116627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'datapath', False)
        str_116628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 51), 'str', 'data')
        # Getting the type of 'fn' (line 199)
        fn_116629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 59), 'fn', False)
        # Processing the call keyword arguments (line 199)
        kwargs_116630 = {}
        # Getting the type of 'os' (line 199)
        os_116624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 199)
        path_116625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 28), os_116624, 'path')
        # Obtaining the member 'join' of a type (line 199)
        join_116626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 28), path_116625, 'join')
        # Calling join(args, kwargs) (line 199)
        join_call_result_116631 = invoke(stypy.reporting.localization.Localization(__file__, 199, 28), join_116626, *[datapath_116627, str_116628, fn_116629], **kwargs_116630)
        
        # Getting the type of 'irange' (line 199)
        irange_116632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 64), 'irange', False)
        # Getting the type of 'shape' (line 199)
        shape_116633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 72), 'shape', False)
        # Processing the call keyword arguments (line 199)
        kwargs_116634 = {}
        # Getting the type of 'check_fromimage' (line 199)
        check_fromimage_116623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'check_fromimage', False)
        # Calling check_fromimage(args, kwargs) (line 199)
        check_fromimage_call_result_116635 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), check_fromimage_116623, *[join_call_result_116631, irange_116632, shape_116633], **kwargs_116634)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 197)
        exit___116636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 13), suppress_warnings_call_result_116614, '__exit__')
        with_exit_116637 = invoke(stypy.reporting.localization.Localization(__file__, 197, 13), exit___116636, None, None, None)

    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_fromimage(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_fromimage' in the type store
    # Getting the type of 'stypy_return_type' (line 189)
    stypy_return_type_116638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116638)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_fromimage'
    return stypy_return_type_116638

# Assigning a type to the variable 'test_fromimage' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'test_fromimage', test_fromimage)

@norecursion
def test_imread_indexed_png(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_imread_indexed_png'
    module_type_store = module_type_store.open_function_context('test_imread_indexed_png', 202, 0, False)
    
    # Passed parameters checking function
    test_imread_indexed_png.stypy_localization = localization
    test_imread_indexed_png.stypy_type_of_self = None
    test_imread_indexed_png.stypy_type_store = module_type_store
    test_imread_indexed_png.stypy_function_name = 'test_imread_indexed_png'
    test_imread_indexed_png.stypy_param_names_list = []
    test_imread_indexed_png.stypy_varargs_param_name = None
    test_imread_indexed_png.stypy_kwargs_param_name = None
    test_imread_indexed_png.stypy_call_defaults = defaults
    test_imread_indexed_png.stypy_call_varargs = varargs
    test_imread_indexed_png.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_imread_indexed_png', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_imread_indexed_png', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_imread_indexed_png(...)' code ##################

    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to array(...): (line 206)
    # Processing the call arguments (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 206)
    list_116641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 206)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 206)
    list_116642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 206)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 206)
    list_116643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 206)
    # Adding element type (line 206)
    int_116644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 22), list_116643, int_116644)
    # Adding element type (line 206)
    int_116645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 22), list_116643, int_116645)
    # Adding element type (line 206)
    int_116646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 22), list_116643, int_116646)
    # Adding element type (line 206)
    int_116647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 22), list_116643, int_116647)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 21), list_116642, list_116643)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 207)
    list_116648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 207)
    # Adding element type (line 207)
    int_116649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 22), list_116648, int_116649)
    # Adding element type (line 207)
    int_116650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 22), list_116648, int_116650)
    # Adding element type (line 207)
    int_116651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 22), list_116648, int_116651)
    # Adding element type (line 207)
    int_116652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 22), list_116648, int_116652)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 21), list_116642, list_116648)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 208)
    list_116653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 208)
    # Adding element type (line 208)
    int_116654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 22), list_116653, int_116654)
    # Adding element type (line 208)
    int_116655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 22), list_116653, int_116655)
    # Adding element type (line 208)
    int_116656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 22), list_116653, int_116656)
    # Adding element type (line 208)
    int_116657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 22), list_116653, int_116657)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 21), list_116642, list_116653)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 209)
    list_116658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 209)
    # Adding element type (line 209)
    int_116659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 22), list_116658, int_116659)
    # Adding element type (line 209)
    int_116660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 22), list_116658, int_116660)
    # Adding element type (line 209)
    int_116661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 22), list_116658, int_116661)
    # Adding element type (line 209)
    int_116662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 22), list_116658, int_116662)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 21), list_116642, list_116658)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_116663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    # Adding element type (line 210)
    int_116664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 22), list_116663, int_116664)
    # Adding element type (line 210)
    int_116665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 22), list_116663, int_116665)
    # Adding element type (line 210)
    int_116666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 22), list_116663, int_116666)
    # Adding element type (line 210)
    int_116667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 22), list_116663, int_116667)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 21), list_116642, list_116663)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 20), list_116641, list_116642)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 211)
    list_116668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 211)
    # Adding element type (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 211)
    list_116669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 211)
    # Adding element type (line 211)
    int_116670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 22), list_116669, int_116670)
    # Adding element type (line 211)
    int_116671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 22), list_116669, int_116671)
    # Adding element type (line 211)
    int_116672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 22), list_116669, int_116672)
    # Adding element type (line 211)
    int_116673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 22), list_116669, int_116673)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 21), list_116668, list_116669)
    # Adding element type (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 212)
    list_116674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 212)
    # Adding element type (line 212)
    int_116675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 22), list_116674, int_116675)
    # Adding element type (line 212)
    int_116676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 22), list_116674, int_116676)
    # Adding element type (line 212)
    int_116677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 22), list_116674, int_116677)
    # Adding element type (line 212)
    int_116678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 22), list_116674, int_116678)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 21), list_116668, list_116674)
    # Adding element type (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 213)
    list_116679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 213)
    # Adding element type (line 213)
    int_116680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 22), list_116679, int_116680)
    # Adding element type (line 213)
    int_116681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 22), list_116679, int_116681)
    # Adding element type (line 213)
    int_116682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 22), list_116679, int_116682)
    # Adding element type (line 213)
    int_116683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 22), list_116679, int_116683)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 21), list_116668, list_116679)
    # Adding element type (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 214)
    list_116684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 214)
    # Adding element type (line 214)
    int_116685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 22), list_116684, int_116685)
    # Adding element type (line 214)
    int_116686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 22), list_116684, int_116686)
    # Adding element type (line 214)
    int_116687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 22), list_116684, int_116687)
    # Adding element type (line 214)
    int_116688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 22), list_116684, int_116688)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 21), list_116668, list_116684)
    # Adding element type (line 211)
    
    # Obtaining an instance of the builtin type 'list' (line 215)
    list_116689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 215)
    # Adding element type (line 215)
    int_116690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 22), list_116689, int_116690)
    # Adding element type (line 215)
    int_116691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 22), list_116689, int_116691)
    # Adding element type (line 215)
    int_116692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 22), list_116689, int_116692)
    # Adding element type (line 215)
    int_116693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 22), list_116689, int_116693)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 21), list_116668, list_116689)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 20), list_116641, list_116668)
    # Adding element type (line 206)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_116694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    # Adding element type (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_116695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    # Adding element type (line 216)
    int_116696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 22), list_116695, int_116696)
    # Adding element type (line 216)
    int_116697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 22), list_116695, int_116697)
    # Adding element type (line 216)
    int_116698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 22), list_116695, int_116698)
    # Adding element type (line 216)
    int_116699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 22), list_116695, int_116699)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 21), list_116694, list_116695)
    # Adding element type (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 217)
    list_116700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 217)
    # Adding element type (line 217)
    int_116701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 22), list_116700, int_116701)
    # Adding element type (line 217)
    int_116702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 22), list_116700, int_116702)
    # Adding element type (line 217)
    int_116703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 22), list_116700, int_116703)
    # Adding element type (line 217)
    int_116704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 22), list_116700, int_116704)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 21), list_116694, list_116700)
    # Adding element type (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 218)
    list_116705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 218)
    # Adding element type (line 218)
    int_116706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 22), list_116705, int_116706)
    # Adding element type (line 218)
    int_116707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 22), list_116705, int_116707)
    # Adding element type (line 218)
    int_116708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 22), list_116705, int_116708)
    # Adding element type (line 218)
    int_116709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 22), list_116705, int_116709)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 21), list_116694, list_116705)
    # Adding element type (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 219)
    list_116710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 219)
    # Adding element type (line 219)
    int_116711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 22), list_116710, int_116711)
    # Adding element type (line 219)
    int_116712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 22), list_116710, int_116712)
    # Adding element type (line 219)
    int_116713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 22), list_116710, int_116713)
    # Adding element type (line 219)
    int_116714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 22), list_116710, int_116714)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 21), list_116694, list_116710)
    # Adding element type (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_116715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    # Adding element type (line 220)
    int_116716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 22), list_116715, int_116716)
    # Adding element type (line 220)
    int_116717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 22), list_116715, int_116717)
    # Adding element type (line 220)
    int_116718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 22), list_116715, int_116718)
    # Adding element type (line 220)
    int_116719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 22), list_116715, int_116719)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 21), list_116694, list_116715)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 20), list_116641, list_116694)
    
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'np' (line 220)
    np_116720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 49), 'np', False)
    # Obtaining the member 'uint8' of a type (line 220)
    uint8_116721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 49), np_116720, 'uint8')
    keyword_116722 = uint8_116721
    kwargs_116723 = {'dtype': keyword_116722}
    # Getting the type of 'np' (line 206)
    np_116639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 206)
    array_116640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 11), np_116639, 'array')
    # Calling array(args, kwargs) (line 206)
    array_call_result_116724 = invoke(stypy.reporting.localization.Localization(__file__, 206, 11), array_116640, *[list_116641], **kwargs_116723)
    
    # Assigning a type to the variable 'data' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'data', array_call_result_116724)
    
    # Assigning a Call to a Name (line 222):
    
    # Assigning a Call to a Name (line 222):
    
    # Call to join(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'datapath' (line 222)
    datapath_116728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'datapath', False)
    str_116729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 38), 'str', 'data')
    str_116730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 46), 'str', 'foo3x5x4indexed.png')
    # Processing the call keyword arguments (line 222)
    kwargs_116731 = {}
    # Getting the type of 'os' (line 222)
    os_116725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 222)
    path_116726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), os_116725, 'path')
    # Obtaining the member 'join' of a type (line 222)
    join_116727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), path_116726, 'join')
    # Calling join(args, kwargs) (line 222)
    join_call_result_116732 = invoke(stypy.reporting.localization.Localization(__file__, 222, 15), join_116727, *[datapath_116728, str_116729, str_116730], **kwargs_116731)
    
    # Assigning a type to the variable 'filename' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'filename', join_call_result_116732)
    
    # Call to open(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'filename' (line 223)
    filename_116734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'filename', False)
    str_116735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 24), 'str', 'rb')
    # Processing the call keyword arguments (line 223)
    kwargs_116736 = {}
    # Getting the type of 'open' (line 223)
    open_116733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 9), 'open', False)
    # Calling open(args, kwargs) (line 223)
    open_call_result_116737 = invoke(stypy.reporting.localization.Localization(__file__, 223, 9), open_116733, *[filename_116734, str_116735], **kwargs_116736)
    
    with_116738 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 223, 9), open_call_result_116737, 'with parameter', '__enter__', '__exit__')

    if with_116738:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 223)
        enter___116739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 9), open_call_result_116737, '__enter__')
        with_enter_116740 = invoke(stypy.reporting.localization.Localization(__file__, 223, 9), enter___116739)
        # Assigning a type to the variable 'f' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 9), 'f', with_enter_116740)
        
        # Call to suppress_warnings(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_116742 = {}
        # Getting the type of 'suppress_warnings' (line 224)
        suppress_warnings_116741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 224)
        suppress_warnings_call_result_116743 = invoke(stypy.reporting.localization.Localization(__file__, 224, 13), suppress_warnings_116741, *[], **kwargs_116742)
        
        with_116744 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 224, 13), suppress_warnings_call_result_116743, 'with parameter', '__enter__', '__exit__')

        if with_116744:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 224)
            enter___116745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 13), suppress_warnings_call_result_116743, '__enter__')
            with_enter_116746 = invoke(stypy.reporting.localization.Localization(__file__, 224, 13), enter___116745)
            # Assigning a type to the variable 'sup' (line 224)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 13), 'sup', with_enter_116746)
            
            # Call to filter(...): (line 225)
            # Processing the call arguments (line 225)
            # Getting the type of 'DeprecationWarning' (line 225)
            DeprecationWarning_116749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 225)
            kwargs_116750 = {}
            # Getting the type of 'sup' (line 225)
            sup_116747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 225)
            filter_116748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), sup_116747, 'filter')
            # Calling filter(args, kwargs) (line 225)
            filter_call_result_116751 = invoke(stypy.reporting.localization.Localization(__file__, 225, 12), filter_116748, *[DeprecationWarning_116749], **kwargs_116750)
            
            
            # Assigning a Call to a Name (line 226):
            
            # Assigning a Call to a Name (line 226):
            
            # Call to imread(...): (line 226)
            # Processing the call arguments (line 226)
            # Getting the type of 'f' (line 226)
            f_116754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'f', False)
            # Processing the call keyword arguments (line 226)
            kwargs_116755 = {}
            # Getting the type of 'misc' (line 226)
            misc_116752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'misc', False)
            # Obtaining the member 'imread' of a type (line 226)
            imread_116753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 17), misc_116752, 'imread')
            # Calling imread(args, kwargs) (line 226)
            imread_call_result_116756 = invoke(stypy.reporting.localization.Localization(__file__, 226, 17), imread_116753, *[f_116754], **kwargs_116755)
            
            # Assigning a type to the variable 'im' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'im', imread_call_result_116756)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 224)
            exit___116757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 13), suppress_warnings_call_result_116743, '__exit__')
            with_exit_116758 = invoke(stypy.reporting.localization.Localization(__file__, 224, 13), exit___116757, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 223)
        exit___116759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 9), open_call_result_116737, '__exit__')
        with_exit_116760 = invoke(stypy.reporting.localization.Localization(__file__, 223, 9), exit___116759, None, None, None)

    
    # Call to assert_array_equal(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'im' (line 227)
    im_116762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'im', False)
    # Getting the type of 'data' (line 227)
    data_116763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'data', False)
    # Processing the call keyword arguments (line 227)
    kwargs_116764 = {}
    # Getting the type of 'assert_array_equal' (line 227)
    assert_array_equal_116761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 227)
    assert_array_equal_call_result_116765 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), assert_array_equal_116761, *[im_116762, data_116763], **kwargs_116764)
    
    
    # ################# End of 'test_imread_indexed_png(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_imread_indexed_png' in the type store
    # Getting the type of 'stypy_return_type' (line 202)
    stypy_return_type_116766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116766)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_imread_indexed_png'
    return stypy_return_type_116766

# Assigning a type to the variable 'test_imread_indexed_png' (line 202)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 0), 'test_imread_indexed_png', test_imread_indexed_png)

@norecursion
def test_imread_1bit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_imread_1bit'
    module_type_store = module_type_store.open_function_context('test_imread_1bit', 230, 0, False)
    
    # Passed parameters checking function
    test_imread_1bit.stypy_localization = localization
    test_imread_1bit.stypy_type_of_self = None
    test_imread_1bit.stypy_type_store = module_type_store
    test_imread_1bit.stypy_function_name = 'test_imread_1bit'
    test_imread_1bit.stypy_param_names_list = []
    test_imread_1bit.stypy_varargs_param_name = None
    test_imread_1bit.stypy_kwargs_param_name = None
    test_imread_1bit.stypy_call_defaults = defaults
    test_imread_1bit.stypy_call_varargs = varargs
    test_imread_1bit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_imread_1bit', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_imread_1bit', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_imread_1bit(...)' code ##################

    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to join(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'datapath' (line 234)
    datapath_116770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 28), 'datapath', False)
    str_116771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 38), 'str', 'data')
    str_116772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 46), 'str', 'box1.png')
    # Processing the call keyword arguments (line 234)
    kwargs_116773 = {}
    # Getting the type of 'os' (line 234)
    os_116767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 234)
    path_116768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 15), os_116767, 'path')
    # Obtaining the member 'join' of a type (line 234)
    join_116769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 15), path_116768, 'join')
    # Calling join(args, kwargs) (line 234)
    join_call_result_116774 = invoke(stypy.reporting.localization.Localization(__file__, 234, 15), join_116769, *[datapath_116770, str_116771, str_116772], **kwargs_116773)
    
    # Assigning a type to the variable 'filename' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'filename', join_call_result_116774)
    
    # Call to open(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'filename' (line 235)
    filename_116776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 14), 'filename', False)
    str_116777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 24), 'str', 'rb')
    # Processing the call keyword arguments (line 235)
    kwargs_116778 = {}
    # Getting the type of 'open' (line 235)
    open_116775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 9), 'open', False)
    # Calling open(args, kwargs) (line 235)
    open_call_result_116779 = invoke(stypy.reporting.localization.Localization(__file__, 235, 9), open_116775, *[filename_116776, str_116777], **kwargs_116778)
    
    with_116780 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 235, 9), open_call_result_116779, 'with parameter', '__enter__', '__exit__')

    if with_116780:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 235)
        enter___116781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 9), open_call_result_116779, '__enter__')
        with_enter_116782 = invoke(stypy.reporting.localization.Localization(__file__, 235, 9), enter___116781)
        # Assigning a type to the variable 'f' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 9), 'f', with_enter_116782)
        
        # Call to suppress_warnings(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_116784 = {}
        # Getting the type of 'suppress_warnings' (line 236)
        suppress_warnings_116783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 236)
        suppress_warnings_call_result_116785 = invoke(stypy.reporting.localization.Localization(__file__, 236, 13), suppress_warnings_116783, *[], **kwargs_116784)
        
        with_116786 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 236, 13), suppress_warnings_call_result_116785, 'with parameter', '__enter__', '__exit__')

        if with_116786:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 236)
            enter___116787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 13), suppress_warnings_call_result_116785, '__enter__')
            with_enter_116788 = invoke(stypy.reporting.localization.Localization(__file__, 236, 13), enter___116787)
            # Assigning a type to the variable 'sup' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'sup', with_enter_116788)
            
            # Call to filter(...): (line 237)
            # Processing the call arguments (line 237)
            # Getting the type of 'DeprecationWarning' (line 237)
            DeprecationWarning_116791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 237)
            kwargs_116792 = {}
            # Getting the type of 'sup' (line 237)
            sup_116789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 237)
            filter_116790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), sup_116789, 'filter')
            # Calling filter(args, kwargs) (line 237)
            filter_call_result_116793 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), filter_116790, *[DeprecationWarning_116791], **kwargs_116792)
            
            
            # Assigning a Call to a Name (line 238):
            
            # Assigning a Call to a Name (line 238):
            
            # Call to imread(...): (line 238)
            # Processing the call arguments (line 238)
            # Getting the type of 'f' (line 238)
            f_116796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 29), 'f', False)
            # Processing the call keyword arguments (line 238)
            kwargs_116797 = {}
            # Getting the type of 'misc' (line 238)
            misc_116794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 17), 'misc', False)
            # Obtaining the member 'imread' of a type (line 238)
            imread_116795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 17), misc_116794, 'imread')
            # Calling imread(args, kwargs) (line 238)
            imread_call_result_116798 = invoke(stypy.reporting.localization.Localization(__file__, 238, 17), imread_116795, *[f_116796], **kwargs_116797)
            
            # Assigning a type to the variable 'im' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'im', imread_call_result_116798)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 236)
            exit___116799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 13), suppress_warnings_call_result_116785, '__exit__')
            with_exit_116800 = invoke(stypy.reporting.localization.Localization(__file__, 236, 13), exit___116799, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 235)
        exit___116801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 9), open_call_result_116779, '__exit__')
        with_exit_116802 = invoke(stypy.reporting.localization.Localization(__file__, 235, 9), exit___116801, None, None, None)

    
    # Call to assert_equal(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'im' (line 239)
    im_116804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 17), 'im', False)
    # Obtaining the member 'dtype' of a type (line 239)
    dtype_116805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 17), im_116804, 'dtype')
    # Getting the type of 'np' (line 239)
    np_116806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'np', False)
    # Obtaining the member 'uint8' of a type (line 239)
    uint8_116807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 27), np_116806, 'uint8')
    # Processing the call keyword arguments (line 239)
    kwargs_116808 = {}
    # Getting the type of 'assert_equal' (line 239)
    assert_equal_116803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 239)
    assert_equal_call_result_116809 = invoke(stypy.reporting.localization.Localization(__file__, 239, 4), assert_equal_116803, *[dtype_116805, uint8_116807], **kwargs_116808)
    
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to zeros(...): (line 240)
    # Processing the call arguments (line 240)
    
    # Obtaining an instance of the builtin type 'tuple' (line 240)
    tuple_116812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 240)
    # Adding element type (line 240)
    int_116813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 25), tuple_116812, int_116813)
    # Adding element type (line 240)
    int_116814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 25), tuple_116812, int_116814)
    
    # Processing the call keyword arguments (line 240)
    # Getting the type of 'np' (line 240)
    np_116815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 40), 'np', False)
    # Obtaining the member 'uint8' of a type (line 240)
    uint8_116816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 40), np_116815, 'uint8')
    keyword_116817 = uint8_116816
    kwargs_116818 = {'dtype': keyword_116817}
    # Getting the type of 'np' (line 240)
    np_116810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 240)
    zeros_116811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), np_116810, 'zeros')
    # Calling zeros(args, kwargs) (line 240)
    zeros_call_result_116819 = invoke(stypy.reporting.localization.Localization(__file__, 240, 15), zeros_116811, *[tuple_116812], **kwargs_116818)
    
    # Assigning a type to the variable 'expected' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'expected', zeros_call_result_116819)
    
    # Assigning a Num to a Subscript (line 242):
    
    # Assigning a Num to a Subscript (line 242):
    int_116820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 21), 'int')
    # Getting the type of 'expected' (line 242)
    expected_116821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'expected')
    slice_116822 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 242, 4), None, None, None)
    int_116823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 16), 'int')
    # Storing an element on a container (line 242)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 4), expected_116821, ((slice_116822, int_116823), int_116820))
    
    # Assigning a Num to a Subscript (line 243):
    
    # Assigning a Num to a Subscript (line 243):
    int_116824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 22), 'int')
    # Getting the type of 'expected' (line 243)
    expected_116825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'expected')
    slice_116826 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 243, 4), None, None, None)
    int_116827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 16), 'int')
    # Storing an element on a container (line 243)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 4), expected_116825, ((slice_116826, int_116827), int_116824))
    
    # Assigning a Num to a Subscript (line 244):
    
    # Assigning a Num to a Subscript (line 244):
    int_116828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 21), 'int')
    # Getting the type of 'expected' (line 244)
    expected_116829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'expected')
    int_116830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 13), 'int')
    slice_116831 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 244, 4), None, None, None)
    # Storing an element on a container (line 244)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 4), expected_116829, ((int_116830, slice_116831), int_116828))
    
    # Assigning a Num to a Subscript (line 245):
    
    # Assigning a Num to a Subscript (line 245):
    int_116832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'int')
    # Getting the type of 'expected' (line 245)
    expected_116833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'expected')
    int_116834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 13), 'int')
    slice_116835 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 4), None, None, None)
    # Storing an element on a container (line 245)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 4), expected_116833, ((int_116834, slice_116835), int_116832))
    
    # Call to assert_equal(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'im' (line 246)
    im_116837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'im', False)
    # Getting the type of 'expected' (line 246)
    expected_116838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'expected', False)
    # Processing the call keyword arguments (line 246)
    kwargs_116839 = {}
    # Getting the type of 'assert_equal' (line 246)
    assert_equal_116836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 246)
    assert_equal_call_result_116840 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), assert_equal_116836, *[im_116837, expected_116838], **kwargs_116839)
    
    
    # ################# End of 'test_imread_1bit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_imread_1bit' in the type store
    # Getting the type of 'stypy_return_type' (line 230)
    stypy_return_type_116841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116841)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_imread_1bit'
    return stypy_return_type_116841

# Assigning a type to the variable 'test_imread_1bit' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'test_imread_1bit', test_imread_1bit)

@norecursion
def test_imread_2bit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_imread_2bit'
    module_type_store = module_type_store.open_function_context('test_imread_2bit', 249, 0, False)
    
    # Passed parameters checking function
    test_imread_2bit.stypy_localization = localization
    test_imread_2bit.stypy_type_of_self = None
    test_imread_2bit.stypy_type_store = module_type_store
    test_imread_2bit.stypy_function_name = 'test_imread_2bit'
    test_imread_2bit.stypy_param_names_list = []
    test_imread_2bit.stypy_varargs_param_name = None
    test_imread_2bit.stypy_kwargs_param_name = None
    test_imread_2bit.stypy_call_defaults = defaults
    test_imread_2bit.stypy_call_varargs = varargs
    test_imread_2bit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_imread_2bit', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_imread_2bit', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_imread_2bit(...)' code ##################

    
    # Assigning a Call to a Name (line 256):
    
    # Assigning a Call to a Name (line 256):
    
    # Call to join(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'datapath' (line 256)
    datapath_116845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 28), 'datapath', False)
    str_116846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 38), 'str', 'data')
    str_116847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 46), 'str', 'blocks2bit.png')
    # Processing the call keyword arguments (line 256)
    kwargs_116848 = {}
    # Getting the type of 'os' (line 256)
    os_116842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 256)
    path_116843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 15), os_116842, 'path')
    # Obtaining the member 'join' of a type (line 256)
    join_116844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 15), path_116843, 'join')
    # Calling join(args, kwargs) (line 256)
    join_call_result_116849 = invoke(stypy.reporting.localization.Localization(__file__, 256, 15), join_116844, *[datapath_116845, str_116846, str_116847], **kwargs_116848)
    
    # Assigning a type to the variable 'filename' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'filename', join_call_result_116849)
    
    # Call to open(...): (line 257)
    # Processing the call arguments (line 257)
    # Getting the type of 'filename' (line 257)
    filename_116851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 14), 'filename', False)
    str_116852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 24), 'str', 'rb')
    # Processing the call keyword arguments (line 257)
    kwargs_116853 = {}
    # Getting the type of 'open' (line 257)
    open_116850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 9), 'open', False)
    # Calling open(args, kwargs) (line 257)
    open_call_result_116854 = invoke(stypy.reporting.localization.Localization(__file__, 257, 9), open_116850, *[filename_116851, str_116852], **kwargs_116853)
    
    with_116855 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 257, 9), open_call_result_116854, 'with parameter', '__enter__', '__exit__')

    if with_116855:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 257)
        enter___116856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 9), open_call_result_116854, '__enter__')
        with_enter_116857 = invoke(stypy.reporting.localization.Localization(__file__, 257, 9), enter___116856)
        # Assigning a type to the variable 'f' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 9), 'f', with_enter_116857)
        
        # Call to suppress_warnings(...): (line 258)
        # Processing the call keyword arguments (line 258)
        kwargs_116859 = {}
        # Getting the type of 'suppress_warnings' (line 258)
        suppress_warnings_116858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 258)
        suppress_warnings_call_result_116860 = invoke(stypy.reporting.localization.Localization(__file__, 258, 13), suppress_warnings_116858, *[], **kwargs_116859)
        
        with_116861 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 258, 13), suppress_warnings_call_result_116860, 'with parameter', '__enter__', '__exit__')

        if with_116861:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 258)
            enter___116862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 13), suppress_warnings_call_result_116860, '__enter__')
            with_enter_116863 = invoke(stypy.reporting.localization.Localization(__file__, 258, 13), enter___116862)
            # Assigning a type to the variable 'sup' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'sup', with_enter_116863)
            
            # Call to filter(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'DeprecationWarning' (line 259)
            DeprecationWarning_116866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 259)
            kwargs_116867 = {}
            # Getting the type of 'sup' (line 259)
            sup_116864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 259)
            filter_116865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), sup_116864, 'filter')
            # Calling filter(args, kwargs) (line 259)
            filter_call_result_116868 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), filter_116865, *[DeprecationWarning_116866], **kwargs_116867)
            
            
            # Assigning a Call to a Name (line 260):
            
            # Assigning a Call to a Name (line 260):
            
            # Call to imread(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 'f' (line 260)
            f_116871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 'f', False)
            # Processing the call keyword arguments (line 260)
            kwargs_116872 = {}
            # Getting the type of 'misc' (line 260)
            misc_116869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'misc', False)
            # Obtaining the member 'imread' of a type (line 260)
            imread_116870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 17), misc_116869, 'imread')
            # Calling imread(args, kwargs) (line 260)
            imread_call_result_116873 = invoke(stypy.reporting.localization.Localization(__file__, 260, 17), imread_116870, *[f_116871], **kwargs_116872)
            
            # Assigning a type to the variable 'im' (line 260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'im', imread_call_result_116873)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 258)
            exit___116874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 13), suppress_warnings_call_result_116860, '__exit__')
            with_exit_116875 = invoke(stypy.reporting.localization.Localization(__file__, 258, 13), exit___116874, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 257)
        exit___116876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 9), open_call_result_116854, '__exit__')
        with_exit_116877 = invoke(stypy.reporting.localization.Localization(__file__, 257, 9), exit___116876, None, None, None)

    
    # Call to assert_equal(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'im' (line 261)
    im_116879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 17), 'im', False)
    # Obtaining the member 'dtype' of a type (line 261)
    dtype_116880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 17), im_116879, 'dtype')
    # Getting the type of 'np' (line 261)
    np_116881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 27), 'np', False)
    # Obtaining the member 'uint8' of a type (line 261)
    uint8_116882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 27), np_116881, 'uint8')
    # Processing the call keyword arguments (line 261)
    kwargs_116883 = {}
    # Getting the type of 'assert_equal' (line 261)
    assert_equal_116878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 261)
    assert_equal_call_result_116884 = invoke(stypy.reporting.localization.Localization(__file__, 261, 4), assert_equal_116878, *[dtype_116880, uint8_116882], **kwargs_116883)
    
    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 262):
    
    # Call to zeros(...): (line 262)
    # Processing the call arguments (line 262)
    
    # Obtaining an instance of the builtin type 'tuple' (line 262)
    tuple_116887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 262)
    # Adding element type (line 262)
    int_116888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), tuple_116887, int_116888)
    # Adding element type (line 262)
    int_116889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 25), tuple_116887, int_116889)
    
    # Processing the call keyword arguments (line 262)
    # Getting the type of 'np' (line 262)
    np_116890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 40), 'np', False)
    # Obtaining the member 'uint8' of a type (line 262)
    uint8_116891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 40), np_116890, 'uint8')
    keyword_116892 = uint8_116891
    kwargs_116893 = {'dtype': keyword_116892}
    # Getting the type of 'np' (line 262)
    np_116885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 262)
    zeros_116886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 15), np_116885, 'zeros')
    # Calling zeros(args, kwargs) (line 262)
    zeros_call_result_116894 = invoke(stypy.reporting.localization.Localization(__file__, 262, 15), zeros_116886, *[tuple_116887], **kwargs_116893)
    
    # Assigning a type to the variable 'expected' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'expected', zeros_call_result_116894)
    
    # Assigning a Num to a Subscript (line 263):
    
    # Assigning a Num to a Subscript (line 263):
    int_116895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 23), 'int')
    # Getting the type of 'expected' (line 263)
    expected_116896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'expected')
    int_116897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 14), 'int')
    slice_116898 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 4), None, int_116897, None)
    int_116899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 17), 'int')
    slice_116900 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 4), int_116899, None, None)
    # Storing an element on a container (line 263)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 4), expected_116896, ((slice_116898, slice_116900), int_116895))
    
    # Assigning a Num to a Subscript (line 264):
    
    # Assigning a Num to a Subscript (line 264):
    int_116901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 23), 'int')
    # Getting the type of 'expected' (line 264)
    expected_116902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'expected')
    int_116903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 13), 'int')
    slice_116904 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 4), int_116903, None, None)
    int_116905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 18), 'int')
    slice_116906 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 264, 4), None, int_116905, None)
    # Storing an element on a container (line 264)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 4), expected_116902, ((slice_116904, slice_116906), int_116901))
    
    # Assigning a Num to a Subscript (line 265):
    
    # Assigning a Num to a Subscript (line 265):
    int_116907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 23), 'int')
    # Getting the type of 'expected' (line 265)
    expected_116908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'expected')
    int_116909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 13), 'int')
    slice_116910 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 4), int_116909, None, None)
    int_116911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 17), 'int')
    slice_116912 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 265, 4), int_116911, None, None)
    # Storing an element on a container (line 265)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 4), expected_116908, ((slice_116910, slice_116912), int_116907))
    
    # Call to assert_equal(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'im' (line 266)
    im_116914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'im', False)
    # Getting the type of 'expected' (line 266)
    expected_116915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 21), 'expected', False)
    # Processing the call keyword arguments (line 266)
    kwargs_116916 = {}
    # Getting the type of 'assert_equal' (line 266)
    assert_equal_116913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 266)
    assert_equal_call_result_116917 = invoke(stypy.reporting.localization.Localization(__file__, 266, 4), assert_equal_116913, *[im_116914, expected_116915], **kwargs_116916)
    
    
    # ################# End of 'test_imread_2bit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_imread_2bit' in the type store
    # Getting the type of 'stypy_return_type' (line 249)
    stypy_return_type_116918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116918)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_imread_2bit'
    return stypy_return_type_116918

# Assigning a type to the variable 'test_imread_2bit' (line 249)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'test_imread_2bit', test_imread_2bit)

@norecursion
def test_imread_4bit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_imread_4bit'
    module_type_store = module_type_store.open_function_context('test_imread_4bit', 269, 0, False)
    
    # Passed parameters checking function
    test_imread_4bit.stypy_localization = localization
    test_imread_4bit.stypy_type_of_self = None
    test_imread_4bit.stypy_type_store = module_type_store
    test_imread_4bit.stypy_function_name = 'test_imread_4bit'
    test_imread_4bit.stypy_param_names_list = []
    test_imread_4bit.stypy_varargs_param_name = None
    test_imread_4bit.stypy_kwargs_param_name = None
    test_imread_4bit.stypy_call_defaults = defaults
    test_imread_4bit.stypy_call_varargs = varargs
    test_imread_4bit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_imread_4bit', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_imread_4bit', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_imread_4bit(...)' code ##################

    
    # Assigning a Call to a Name (line 274):
    
    # Assigning a Call to a Name (line 274):
    
    # Call to join(...): (line 274)
    # Processing the call arguments (line 274)
    # Getting the type of 'datapath' (line 274)
    datapath_116922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 28), 'datapath', False)
    str_116923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 38), 'str', 'data')
    str_116924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 46), 'str', 'pattern4bit.png')
    # Processing the call keyword arguments (line 274)
    kwargs_116925 = {}
    # Getting the type of 'os' (line 274)
    os_116919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 274)
    path_116920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 15), os_116919, 'path')
    # Obtaining the member 'join' of a type (line 274)
    join_116921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 15), path_116920, 'join')
    # Calling join(args, kwargs) (line 274)
    join_call_result_116926 = invoke(stypy.reporting.localization.Localization(__file__, 274, 15), join_116921, *[datapath_116922, str_116923, str_116924], **kwargs_116925)
    
    # Assigning a type to the variable 'filename' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'filename', join_call_result_116926)
    
    # Call to open(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'filename' (line 275)
    filename_116928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 14), 'filename', False)
    str_116929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 24), 'str', 'rb')
    # Processing the call keyword arguments (line 275)
    kwargs_116930 = {}
    # Getting the type of 'open' (line 275)
    open_116927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), 'open', False)
    # Calling open(args, kwargs) (line 275)
    open_call_result_116931 = invoke(stypy.reporting.localization.Localization(__file__, 275, 9), open_116927, *[filename_116928, str_116929], **kwargs_116930)
    
    with_116932 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 275, 9), open_call_result_116931, 'with parameter', '__enter__', '__exit__')

    if with_116932:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 275)
        enter___116933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 9), open_call_result_116931, '__enter__')
        with_enter_116934 = invoke(stypy.reporting.localization.Localization(__file__, 275, 9), enter___116933)
        # Assigning a type to the variable 'f' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), 'f', with_enter_116934)
        
        # Call to suppress_warnings(...): (line 276)
        # Processing the call keyword arguments (line 276)
        kwargs_116936 = {}
        # Getting the type of 'suppress_warnings' (line 276)
        suppress_warnings_116935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 276)
        suppress_warnings_call_result_116937 = invoke(stypy.reporting.localization.Localization(__file__, 276, 13), suppress_warnings_116935, *[], **kwargs_116936)
        
        with_116938 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 276, 13), suppress_warnings_call_result_116937, 'with parameter', '__enter__', '__exit__')

        if with_116938:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 276)
            enter___116939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 13), suppress_warnings_call_result_116937, '__enter__')
            with_enter_116940 = invoke(stypy.reporting.localization.Localization(__file__, 276, 13), enter___116939)
            # Assigning a type to the variable 'sup' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 13), 'sup', with_enter_116940)
            
            # Call to filter(...): (line 277)
            # Processing the call arguments (line 277)
            # Getting the type of 'DeprecationWarning' (line 277)
            DeprecationWarning_116943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 23), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 277)
            kwargs_116944 = {}
            # Getting the type of 'sup' (line 277)
            sup_116941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 277)
            filter_116942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), sup_116941, 'filter')
            # Calling filter(args, kwargs) (line 277)
            filter_call_result_116945 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), filter_116942, *[DeprecationWarning_116943], **kwargs_116944)
            
            
            # Assigning a Call to a Name (line 278):
            
            # Assigning a Call to a Name (line 278):
            
            # Call to imread(...): (line 278)
            # Processing the call arguments (line 278)
            # Getting the type of 'f' (line 278)
            f_116948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 29), 'f', False)
            # Processing the call keyword arguments (line 278)
            kwargs_116949 = {}
            # Getting the type of 'misc' (line 278)
            misc_116946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'misc', False)
            # Obtaining the member 'imread' of a type (line 278)
            imread_116947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 17), misc_116946, 'imread')
            # Calling imread(args, kwargs) (line 278)
            imread_call_result_116950 = invoke(stypy.reporting.localization.Localization(__file__, 278, 17), imread_116947, *[f_116948], **kwargs_116949)
            
            # Assigning a type to the variable 'im' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'im', imread_call_result_116950)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 276)
            exit___116951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 13), suppress_warnings_call_result_116937, '__exit__')
            with_exit_116952 = invoke(stypy.reporting.localization.Localization(__file__, 276, 13), exit___116951, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 275)
        exit___116953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 9), open_call_result_116931, '__exit__')
        with_exit_116954 = invoke(stypy.reporting.localization.Localization(__file__, 275, 9), exit___116953, None, None, None)

    
    # Call to assert_equal(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'im' (line 279)
    im_116956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 17), 'im', False)
    # Obtaining the member 'dtype' of a type (line 279)
    dtype_116957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 17), im_116956, 'dtype')
    # Getting the type of 'np' (line 279)
    np_116958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 27), 'np', False)
    # Obtaining the member 'uint8' of a type (line 279)
    uint8_116959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 27), np_116958, 'uint8')
    # Processing the call keyword arguments (line 279)
    kwargs_116960 = {}
    # Getting the type of 'assert_equal' (line 279)
    assert_equal_116955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 279)
    assert_equal_call_result_116961 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), assert_equal_116955, *[dtype_116957, uint8_116959], **kwargs_116960)
    
    
    # Assigning a Call to a Tuple (line 280):
    
    # Assigning a Subscript to a Name (line 280):
    
    # Obtaining the type of the subscript
    int_116962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 4), 'int')
    
    # Call to meshgrid(...): (line 280)
    # Processing the call arguments (line 280)
    
    # Call to arange(...): (line 280)
    # Processing the call arguments (line 280)
    int_116967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 33), 'int')
    # Processing the call keyword arguments (line 280)
    kwargs_116968 = {}
    # Getting the type of 'np' (line 280)
    np_116965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'np', False)
    # Obtaining the member 'arange' of a type (line 280)
    arange_116966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 23), np_116965, 'arange')
    # Calling arange(args, kwargs) (line 280)
    arange_call_result_116969 = invoke(stypy.reporting.localization.Localization(__file__, 280, 23), arange_116966, *[int_116967], **kwargs_116968)
    
    
    # Call to arange(...): (line 280)
    # Processing the call arguments (line 280)
    int_116972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 48), 'int')
    # Processing the call keyword arguments (line 280)
    kwargs_116973 = {}
    # Getting the type of 'np' (line 280)
    np_116970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 38), 'np', False)
    # Obtaining the member 'arange' of a type (line 280)
    arange_116971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 38), np_116970, 'arange')
    # Calling arange(args, kwargs) (line 280)
    arange_call_result_116974 = invoke(stypy.reporting.localization.Localization(__file__, 280, 38), arange_116971, *[int_116972], **kwargs_116973)
    
    # Processing the call keyword arguments (line 280)
    str_116975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 62), 'str', 'ij')
    keyword_116976 = str_116975
    kwargs_116977 = {'indexing': keyword_116976}
    # Getting the type of 'np' (line 280)
    np_116963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 280)
    meshgrid_116964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), np_116963, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 280)
    meshgrid_call_result_116978 = invoke(stypy.reporting.localization.Localization(__file__, 280, 11), meshgrid_116964, *[arange_call_result_116969, arange_call_result_116974], **kwargs_116977)
    
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___116979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 4), meshgrid_call_result_116978, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_116980 = invoke(stypy.reporting.localization.Localization(__file__, 280, 4), getitem___116979, int_116962)
    
    # Assigning a type to the variable 'tuple_var_assignment_115664' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'tuple_var_assignment_115664', subscript_call_result_116980)
    
    # Assigning a Subscript to a Name (line 280):
    
    # Obtaining the type of the subscript
    int_116981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 4), 'int')
    
    # Call to meshgrid(...): (line 280)
    # Processing the call arguments (line 280)
    
    # Call to arange(...): (line 280)
    # Processing the call arguments (line 280)
    int_116986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 33), 'int')
    # Processing the call keyword arguments (line 280)
    kwargs_116987 = {}
    # Getting the type of 'np' (line 280)
    np_116984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'np', False)
    # Obtaining the member 'arange' of a type (line 280)
    arange_116985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 23), np_116984, 'arange')
    # Calling arange(args, kwargs) (line 280)
    arange_call_result_116988 = invoke(stypy.reporting.localization.Localization(__file__, 280, 23), arange_116985, *[int_116986], **kwargs_116987)
    
    
    # Call to arange(...): (line 280)
    # Processing the call arguments (line 280)
    int_116991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 48), 'int')
    # Processing the call keyword arguments (line 280)
    kwargs_116992 = {}
    # Getting the type of 'np' (line 280)
    np_116989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 38), 'np', False)
    # Obtaining the member 'arange' of a type (line 280)
    arange_116990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 38), np_116989, 'arange')
    # Calling arange(args, kwargs) (line 280)
    arange_call_result_116993 = invoke(stypy.reporting.localization.Localization(__file__, 280, 38), arange_116990, *[int_116991], **kwargs_116992)
    
    # Processing the call keyword arguments (line 280)
    str_116994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 62), 'str', 'ij')
    keyword_116995 = str_116994
    kwargs_116996 = {'indexing': keyword_116995}
    # Getting the type of 'np' (line 280)
    np_116982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 280)
    meshgrid_116983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), np_116982, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 280)
    meshgrid_call_result_116997 = invoke(stypy.reporting.localization.Localization(__file__, 280, 11), meshgrid_116983, *[arange_call_result_116988, arange_call_result_116993], **kwargs_116996)
    
    # Obtaining the member '__getitem__' of a type (line 280)
    getitem___116998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 4), meshgrid_call_result_116997, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 280)
    subscript_call_result_116999 = invoke(stypy.reporting.localization.Localization(__file__, 280, 4), getitem___116998, int_116981)
    
    # Assigning a type to the variable 'tuple_var_assignment_115665' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'tuple_var_assignment_115665', subscript_call_result_116999)
    
    # Assigning a Name to a Name (line 280):
    # Getting the type of 'tuple_var_assignment_115664' (line 280)
    tuple_var_assignment_115664_117000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'tuple_var_assignment_115664')
    # Assigning a type to the variable 'j' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'j', tuple_var_assignment_115664_117000)
    
    # Assigning a Name to a Name (line 280):
    # Getting the type of 'tuple_var_assignment_115665' (line 280)
    tuple_var_assignment_115665_117001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'tuple_var_assignment_115665')
    # Assigning a type to the variable 'i' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 7), 'i', tuple_var_assignment_115665_117001)
    
    # Assigning a BinOp to a Name (line 281):
    
    # Assigning a BinOp to a Name (line 281):
    int_117002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 15), 'int')
    
    # Call to astype(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'np' (line 281)
    np_117012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 49), 'np', False)
    # Obtaining the member 'uint8' of a type (line 281)
    uint8_117013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 49), np_117012, 'uint8')
    # Processing the call keyword arguments (line 281)
    kwargs_117014 = {}
    
    # Call to maximum(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'j' (line 281)
    j_117005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 30), 'j', False)
    # Getting the type of 'i' (line 281)
    i_117006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 33), 'i', False)
    # Processing the call keyword arguments (line 281)
    kwargs_117007 = {}
    # Getting the type of 'np' (line 281)
    np_117003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 19), 'np', False)
    # Obtaining the member 'maximum' of a type (line 281)
    maximum_117004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), np_117003, 'maximum')
    # Calling maximum(args, kwargs) (line 281)
    maximum_call_result_117008 = invoke(stypy.reporting.localization.Localization(__file__, 281, 19), maximum_117004, *[j_117005, i_117006], **kwargs_117007)
    
    int_117009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 38), 'int')
    # Applying the binary operator '%' (line 281)
    result_mod_117010 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 19), '%', maximum_call_result_117008, int_117009)
    
    # Obtaining the member 'astype' of a type (line 281)
    astype_117011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 19), result_mod_117010, 'astype')
    # Calling astype(args, kwargs) (line 281)
    astype_call_result_117015 = invoke(stypy.reporting.localization.Localization(__file__, 281, 19), astype_117011, *[uint8_117013], **kwargs_117014)
    
    # Applying the binary operator '*' (line 281)
    result_mul_117016 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 15), '*', int_117002, astype_call_result_117015)
    
    # Assigning a type to the variable 'expected' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'expected', result_mul_117016)
    
    # Call to assert_equal(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'im' (line 282)
    im_117018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'im', False)
    # Getting the type of 'expected' (line 282)
    expected_117019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 21), 'expected', False)
    # Processing the call keyword arguments (line 282)
    kwargs_117020 = {}
    # Getting the type of 'assert_equal' (line 282)
    assert_equal_117017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 282)
    assert_equal_call_result_117021 = invoke(stypy.reporting.localization.Localization(__file__, 282, 4), assert_equal_117017, *[im_117018, expected_117019], **kwargs_117020)
    
    
    # ################# End of 'test_imread_4bit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_imread_4bit' in the type store
    # Getting the type of 'stypy_return_type' (line 269)
    stypy_return_type_117022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_imread_4bit'
    return stypy_return_type_117022

# Assigning a type to the variable 'test_imread_4bit' (line 269)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 0), 'test_imread_4bit', test_imread_4bit)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import datetime
4: import os
5: import sys
6: from os.path import join as pjoin
7: 
8: if sys.version_info[0] >= 3:
9:     from io import StringIO
10: else:
11:     from cStringIO import StringIO
12: 
13: import numpy as np
14: 
15: from numpy.testing import (assert_array_almost_equal,
16:                            assert_array_equal, assert_equal, assert_)
17: import pytest
18: from pytest import raises as assert_raises
19: 
20: from scipy.io.arff.arffread import loadarff
21: from scipy.io.arff.arffread import read_header, parse_type, ParseArffError
22: 
23: 
24: data_path = pjoin(os.path.dirname(__file__), 'data')
25: 
26: test1 = pjoin(data_path, 'test1.arff')
27: test2 = pjoin(data_path, 'test2.arff')
28: test3 = pjoin(data_path, 'test3.arff')
29: 
30: test4 = pjoin(data_path, 'test4.arff')
31: test5 = pjoin(data_path, 'test5.arff')
32: test6 = pjoin(data_path, 'test6.arff')
33: test7 = pjoin(data_path, 'test7.arff')
34: test8 = pjoin(data_path, 'test8.arff')
35: expect4_data = [(0.1, 0.2, 0.3, 0.4, 'class1'),
36:                 (-0.1, -0.2, -0.3, -0.4, 'class2'),
37:                 (1, 2, 3, 4, 'class3')]
38: expected_types = ['numeric', 'numeric', 'numeric', 'numeric', 'nominal']
39: 
40: missing = pjoin(data_path, 'missing.arff')
41: expect_missing_raw = np.array([[1, 5], [2, 4], [np.nan, np.nan]])
42: expect_missing = np.empty(3, [('yop', float), ('yap', float)])
43: expect_missing['yop'] = expect_missing_raw[:, 0]
44: expect_missing['yap'] = expect_missing_raw[:, 1]
45: 
46: 
47: class TestData(object):
48:     def test1(self):
49:         # Parsing trivial file with nothing.
50:         self._test(test4)
51: 
52:     def test2(self):
53:         # Parsing trivial file with some comments in the data section.
54:         self._test(test5)
55: 
56:     def test3(self):
57:         # Parsing trivial file with nominal attribute of 1 character.
58:         self._test(test6)
59: 
60:     def _test(self, test_file):
61:         data, meta = loadarff(test_file)
62:         for i in range(len(data)):
63:             for j in range(4):
64:                 assert_array_almost_equal(expect4_data[i][j], data[i][j])
65:         assert_equal(meta.types(), expected_types)
66: 
67:     def test_filelike(self):
68:         # Test reading from file-like object (StringIO)
69:         f1 = open(test1)
70:         data1, meta1 = loadarff(f1)
71:         f1.close()
72:         f2 = open(test1)
73:         data2, meta2 = loadarff(StringIO(f2.read()))
74:         f2.close()
75:         assert_(data1 == data2)
76:         assert_(repr(meta1) == repr(meta2))
77: 
78:     @pytest.mark.skipif(sys.version_info < (3, 6),
79:                         reason='Passing path-like objects to IO functions requires Python >= 3.6')
80:     def test_path(self):
81:         # Test reading from `pathlib.Path` object
82:         from pathlib import Path
83: 
84:         with open(test1) as f1:
85:             data1, meta1 = loadarff(f1)
86: 
87:         data2, meta2 = loadarff(Path(test1))
88: 
89:         assert_(data1 == data2)
90:         assert_(repr(meta1) == repr(meta2))
91: 
92: class TestMissingData(object):
93:     def test_missing(self):
94:         data, meta = loadarff(missing)
95:         for i in ['yop', 'yap']:
96:             assert_array_almost_equal(data[i], expect_missing[i])
97: 
98: 
99: class TestNoData(object):
100:     def test_nodata(self):
101:         # The file nodata.arff has no data in the @DATA section.
102:         # Reading it should result in an array with length 0.
103:         nodata_filename = os.path.join(data_path, 'nodata.arff')
104:         data, meta = loadarff(nodata_filename)
105:         expected_dtype = np.dtype([('sepallength', '<f8'),
106:                                    ('sepalwidth', '<f8'),
107:                                    ('petallength', '<f8'),
108:                                    ('petalwidth', '<f8'),
109:                                    ('class', 'S15')])
110:         assert_equal(data.dtype, expected_dtype)
111:         assert_equal(data.size, 0)
112: 
113: 
114: class TestHeader(object):
115:     def test_type_parsing(self):
116:         # Test parsing type of attribute from their value.
117:         ofile = open(test2)
118:         rel, attrs = read_header(ofile)
119:         ofile.close()
120: 
121:         expected = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric',
122:                     'numeric', 'string', 'string', 'nominal', 'nominal']
123: 
124:         for i in range(len(attrs)):
125:             assert_(parse_type(attrs[i][1]) == expected[i])
126: 
127:     def test_badtype_parsing(self):
128:         # Test parsing wrong type of attribute from their value.
129:         ofile = open(test3)
130:         rel, attrs = read_header(ofile)
131:         ofile.close()
132: 
133:         for name, value in attrs:
134:             assert_raises(ParseArffError, parse_type, value)
135: 
136:     def test_fullheader1(self):
137:         # Parsing trivial header with nothing.
138:         ofile = open(test1)
139:         rel, attrs = read_header(ofile)
140:         ofile.close()
141: 
142:         # Test relation
143:         assert_(rel == 'test1')
144: 
145:         # Test numerical attributes
146:         assert_(len(attrs) == 5)
147:         for i in range(4):
148:             assert_(attrs[i][0] == 'attr%d' % i)
149:             assert_(attrs[i][1] == 'REAL')
150: 
151:         # Test nominal attribute
152:         assert_(attrs[4][0] == 'class')
153:         assert_(attrs[4][1] == '{class0, class1, class2, class3}')
154: 
155:     def test_dateheader(self):
156:         ofile = open(test7)
157:         rel, attrs = read_header(ofile)
158:         ofile.close()
159: 
160:         assert_(rel == 'test7')
161: 
162:         assert_(len(attrs) == 5)
163: 
164:         assert_(attrs[0][0] == 'attr_year')
165:         assert_(attrs[0][1] == 'DATE yyyy')
166: 
167:         assert_(attrs[1][0] == 'attr_month')
168:         assert_(attrs[1][1] == 'DATE yyyy-MM')
169: 
170:         assert_(attrs[2][0] == 'attr_date')
171:         assert_(attrs[2][1] == 'DATE yyyy-MM-dd')
172: 
173:         assert_(attrs[3][0] == 'attr_datetime_local')
174:         assert_(attrs[3][1] == 'DATE "yyyy-MM-dd HH:mm"')
175: 
176:         assert_(attrs[4][0] == 'attr_datetime_missing')
177:         assert_(attrs[4][1] == 'DATE "yyyy-MM-dd HH:mm"')
178: 
179:     def test_dateheader_unsupported(self):
180:         ofile = open(test8)
181:         rel, attrs = read_header(ofile)
182:         ofile.close()
183: 
184:         assert_(rel == 'test8')
185: 
186:         assert_(len(attrs) == 2)
187:         assert_(attrs[0][0] == 'attr_datetime_utc')
188:         assert_(attrs[0][1] == 'DATE "yyyy-MM-dd HH:mm Z"')
189: 
190:         assert_(attrs[1][0] == 'attr_datetime_full')
191:         assert_(attrs[1][1] == 'DATE "yy-MM-dd HH:mm:ss z"')
192: 
193: 
194: class TestDateAttribute(object):
195:     def setup_method(self):
196:         self.data, self.meta = loadarff(test7)
197: 
198:     def test_year_attribute(self):
199:         expected = np.array([
200:             '1999',
201:             '2004',
202:             '1817',
203:             '2100',
204:             '2013',
205:             '1631'
206:         ], dtype='datetime64[Y]')
207: 
208:         assert_array_equal(self.data["attr_year"], expected)
209: 
210:     def test_month_attribute(self):
211:         expected = np.array([
212:             '1999-01',
213:             '2004-12',
214:             '1817-04',
215:             '2100-09',
216:             '2013-11',
217:             '1631-10'
218:         ], dtype='datetime64[M]')
219: 
220:         assert_array_equal(self.data["attr_month"], expected)
221: 
222:     def test_date_attribute(self):
223:         expected = np.array([
224:             '1999-01-31',
225:             '2004-12-01',
226:             '1817-04-28',
227:             '2100-09-10',
228:             '2013-11-30',
229:             '1631-10-15'
230:         ], dtype='datetime64[D]')
231: 
232:         assert_array_equal(self.data["attr_date"], expected)
233: 
234:     def test_datetime_local_attribute(self):
235:         expected = np.array([
236:             datetime.datetime(year=1999, month=1, day=31, hour=0, minute=1),
237:             datetime.datetime(year=2004, month=12, day=1, hour=23, minute=59),
238:             datetime.datetime(year=1817, month=4, day=28, hour=13, minute=0),
239:             datetime.datetime(year=2100, month=9, day=10, hour=12, minute=0),
240:             datetime.datetime(year=2013, month=11, day=30, hour=4, minute=55),
241:             datetime.datetime(year=1631, month=10, day=15, hour=20, minute=4)
242:         ], dtype='datetime64[m]')
243: 
244:         assert_array_equal(self.data["attr_datetime_local"], expected)
245: 
246:     def test_datetime_missing(self):
247:         expected = np.array([
248:             'nat',
249:             '2004-12-01T23:59',
250:             'nat',
251:             'nat',
252:             '2013-11-30T04:55',
253:             '1631-10-15T20:04'
254:         ], dtype='datetime64[m]')
255: 
256:         assert_array_equal(self.data["attr_datetime_missing"], expected)
257: 
258:     def test_datetime_timezone(self):
259:         assert_raises(ValueError, loadarff, test8)
260: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import datetime' statement (line 3)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from os.path import pjoin' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
import_129668 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os.path')

if (type(import_129668) is not StypyTypeError):

    if (import_129668 != 'pyd_module'):
        __import__(import_129668)
        sys_modules_129669 = sys.modules[import_129668]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os.path', sys_modules_129669.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_129669, sys_modules_129669.module_type_store, module_type_store)
    else:
        from os.path import join as pjoin

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os.path', None, module_type_store, ['join'], [pjoin])

else:
    # Assigning a type to the variable 'os.path' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'os.path', import_129668)

# Adding an alias
module_type_store.add_alias('pjoin', 'join')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/tests/')




# Obtaining the type of the subscript
int_129670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
# Getting the type of 'sys' (line 8)
sys_129671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 8)
version_info_129672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 3), sys_129671, 'version_info')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___129673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 3), version_info_129672, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_129674 = invoke(stypy.reporting.localization.Localization(__file__, 8, 3), getitem___129673, int_129670)

int_129675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 26), 'int')
# Applying the binary operator '>=' (line 8)
result_ge_129676 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 3), '>=', subscript_call_result_129674, int_129675)

# Testing the type of an if condition (line 8)
if_condition_129677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 0), result_ge_129676)
# Assigning a type to the variable 'if_condition_129677' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'if_condition_129677', if_condition_129677)
# SSA begins for if statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))

# 'from io import StringIO' statement (line 9)
try:
    from io import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'io', None, module_type_store, ['StringIO'], [StringIO])

# SSA branch for the else part of an if statement (line 8)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))

# 'from cStringIO import StringIO' statement (line 11)
try:
    from cStringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])

# SSA join for if statement (line 8)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
import_129678 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_129678) is not StypyTypeError):

    if (import_129678 != 'pyd_module'):
        __import__(import_129678)
        sys_modules_129679 = sys.modules[import_129678]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_129679.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_129678)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal, assert_' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
import_129680 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.testing')

if (type(import_129680) is not StypyTypeError):

    if (import_129680 != 'pyd_module'):
        __import__(import_129680)
        sys_modules_129681 = sys.modules[import_129680]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.testing', sys_modules_129681.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_array_equal', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_129681, sys_modules_129681.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_array_equal', 'assert_equal', 'assert_'], [assert_array_almost_equal, assert_array_equal, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.testing', import_129680)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'import pytest' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
import_129682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest')

if (type(import_129682) is not StypyTypeError):

    if (import_129682 != 'pyd_module'):
        __import__(import_129682)
        sys_modules_129683 = sys.modules[import_129682]
        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', sys_modules_129683.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'pytest', import_129682)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from pytest import assert_raises' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
import_129684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'pytest')

if (type(import_129684) is not StypyTypeError):

    if (import_129684 != 'pyd_module'):
        __import__(import_129684)
        sys_modules_129685 = sys.modules[import_129684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'pytest', sys_modules_129685.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_129685, sys_modules_129685.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'pytest', import_129684)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.io.arff.arffread import loadarff' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
import_129686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.arff.arffread')

if (type(import_129686) is not StypyTypeError):

    if (import_129686 != 'pyd_module'):
        __import__(import_129686)
        sys_modules_129687 = sys.modules[import_129686]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.arff.arffread', sys_modules_129687.module_type_store, module_type_store, ['loadarff'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_129687, sys_modules_129687.module_type_store, module_type_store)
    else:
        from scipy.io.arff.arffread import loadarff

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.arff.arffread', None, module_type_store, ['loadarff'], [loadarff])

else:
    # Assigning a type to the variable 'scipy.io.arff.arffread' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.io.arff.arffread', import_129686)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from scipy.io.arff.arffread import read_header, parse_type, ParseArffError' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
import_129688 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.io.arff.arffread')

if (type(import_129688) is not StypyTypeError):

    if (import_129688 != 'pyd_module'):
        __import__(import_129688)
        sys_modules_129689 = sys.modules[import_129688]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.io.arff.arffread', sys_modules_129689.module_type_store, module_type_store, ['read_header', 'parse_type', 'ParseArffError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_129689, sys_modules_129689.module_type_store, module_type_store)
    else:
        from scipy.io.arff.arffread import read_header, parse_type, ParseArffError

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.io.arff.arffread', None, module_type_store, ['read_header', 'parse_type', 'ParseArffError'], [read_header, parse_type, ParseArffError])

else:
    # Assigning a type to the variable 'scipy.io.arff.arffread' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'scipy.io.arff.arffread', import_129688)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/tests/')


# Assigning a Call to a Name (line 24):

# Assigning a Call to a Name (line 24):

# Call to pjoin(...): (line 24)
# Processing the call arguments (line 24)

# Call to dirname(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of '__file__' (line 24)
file___129694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), '__file__', False)
# Processing the call keyword arguments (line 24)
kwargs_129695 = {}
# Getting the type of 'os' (line 24)
os_129691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'os', False)
# Obtaining the member 'path' of a type (line 24)
path_129692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 18), os_129691, 'path')
# Obtaining the member 'dirname' of a type (line 24)
dirname_129693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 18), path_129692, 'dirname')
# Calling dirname(args, kwargs) (line 24)
dirname_call_result_129696 = invoke(stypy.reporting.localization.Localization(__file__, 24, 18), dirname_129693, *[file___129694], **kwargs_129695)

str_129697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 45), 'str', 'data')
# Processing the call keyword arguments (line 24)
kwargs_129698 = {}
# Getting the type of 'pjoin' (line 24)
pjoin_129690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 24)
pjoin_call_result_129699 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), pjoin_129690, *[dirname_call_result_129696, str_129697], **kwargs_129698)

# Assigning a type to the variable 'data_path' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'data_path', pjoin_call_result_129699)

# Assigning a Call to a Name (line 26):

# Assigning a Call to a Name (line 26):

# Call to pjoin(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'data_path' (line 26)
data_path_129701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'data_path', False)
str_129702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'str', 'test1.arff')
# Processing the call keyword arguments (line 26)
kwargs_129703 = {}
# Getting the type of 'pjoin' (line 26)
pjoin_129700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 26)
pjoin_call_result_129704 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), pjoin_129700, *[data_path_129701, str_129702], **kwargs_129703)

# Assigning a type to the variable 'test1' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'test1', pjoin_call_result_129704)

# Assigning a Call to a Name (line 27):

# Assigning a Call to a Name (line 27):

# Call to pjoin(...): (line 27)
# Processing the call arguments (line 27)
# Getting the type of 'data_path' (line 27)
data_path_129706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'data_path', False)
str_129707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'str', 'test2.arff')
# Processing the call keyword arguments (line 27)
kwargs_129708 = {}
# Getting the type of 'pjoin' (line 27)
pjoin_129705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 27)
pjoin_call_result_129709 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), pjoin_129705, *[data_path_129706, str_129707], **kwargs_129708)

# Assigning a type to the variable 'test2' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'test2', pjoin_call_result_129709)

# Assigning a Call to a Name (line 28):

# Assigning a Call to a Name (line 28):

# Call to pjoin(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of 'data_path' (line 28)
data_path_129711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'data_path', False)
str_129712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', 'test3.arff')
# Processing the call keyword arguments (line 28)
kwargs_129713 = {}
# Getting the type of 'pjoin' (line 28)
pjoin_129710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 28)
pjoin_call_result_129714 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), pjoin_129710, *[data_path_129711, str_129712], **kwargs_129713)

# Assigning a type to the variable 'test3' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'test3', pjoin_call_result_129714)

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to pjoin(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'data_path' (line 30)
data_path_129716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'data_path', False)
str_129717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'str', 'test4.arff')
# Processing the call keyword arguments (line 30)
kwargs_129718 = {}
# Getting the type of 'pjoin' (line 30)
pjoin_129715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 30)
pjoin_call_result_129719 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), pjoin_129715, *[data_path_129716, str_129717], **kwargs_129718)

# Assigning a type to the variable 'test4' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'test4', pjoin_call_result_129719)

# Assigning a Call to a Name (line 31):

# Assigning a Call to a Name (line 31):

# Call to pjoin(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'data_path' (line 31)
data_path_129721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'data_path', False)
str_129722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'str', 'test5.arff')
# Processing the call keyword arguments (line 31)
kwargs_129723 = {}
# Getting the type of 'pjoin' (line 31)
pjoin_129720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 31)
pjoin_call_result_129724 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), pjoin_129720, *[data_path_129721, str_129722], **kwargs_129723)

# Assigning a type to the variable 'test5' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'test5', pjoin_call_result_129724)

# Assigning a Call to a Name (line 32):

# Assigning a Call to a Name (line 32):

# Call to pjoin(...): (line 32)
# Processing the call arguments (line 32)
# Getting the type of 'data_path' (line 32)
data_path_129726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'data_path', False)
str_129727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'str', 'test6.arff')
# Processing the call keyword arguments (line 32)
kwargs_129728 = {}
# Getting the type of 'pjoin' (line 32)
pjoin_129725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 32)
pjoin_call_result_129729 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), pjoin_129725, *[data_path_129726, str_129727], **kwargs_129728)

# Assigning a type to the variable 'test6' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'test6', pjoin_call_result_129729)

# Assigning a Call to a Name (line 33):

# Assigning a Call to a Name (line 33):

# Call to pjoin(...): (line 33)
# Processing the call arguments (line 33)
# Getting the type of 'data_path' (line 33)
data_path_129731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'data_path', False)
str_129732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'str', 'test7.arff')
# Processing the call keyword arguments (line 33)
kwargs_129733 = {}
# Getting the type of 'pjoin' (line 33)
pjoin_129730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 33)
pjoin_call_result_129734 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), pjoin_129730, *[data_path_129731, str_129732], **kwargs_129733)

# Assigning a type to the variable 'test7' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'test7', pjoin_call_result_129734)

# Assigning a Call to a Name (line 34):

# Assigning a Call to a Name (line 34):

# Call to pjoin(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of 'data_path' (line 34)
data_path_129736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'data_path', False)
str_129737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', 'test8.arff')
# Processing the call keyword arguments (line 34)
kwargs_129738 = {}
# Getting the type of 'pjoin' (line 34)
pjoin_129735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 34)
pjoin_call_result_129739 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), pjoin_129735, *[data_path_129736, str_129737], **kwargs_129738)

# Assigning a type to the variable 'test8' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'test8', pjoin_call_result_129739)

# Assigning a List to a Name (line 35):

# Assigning a List to a Name (line 35):

# Obtaining an instance of the builtin type 'list' (line 35)
list_129740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 35)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_129741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
float_129742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), tuple_129741, float_129742)
# Adding element type (line 35)
float_129743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), tuple_129741, float_129743)
# Adding element type (line 35)
float_129744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), tuple_129741, float_129744)
# Adding element type (line 35)
float_129745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), tuple_129741, float_129745)
# Adding element type (line 35)
str_129746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 37), 'str', 'class1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), tuple_129741, str_129746)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), list_129740, tuple_129741)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_129747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
float_129748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), tuple_129747, float_129748)
# Adding element type (line 36)
float_129749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), tuple_129747, float_129749)
# Adding element type (line 36)
float_129750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), tuple_129747, float_129750)
# Adding element type (line 36)
float_129751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), tuple_129747, float_129751)
# Adding element type (line 36)
str_129752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 41), 'str', 'class2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), tuple_129747, str_129752)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), list_129740, tuple_129747)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_129753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
int_129754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), tuple_129753, int_129754)
# Adding element type (line 37)
int_129755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), tuple_129753, int_129755)
# Adding element type (line 37)
int_129756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), tuple_129753, int_129756)
# Adding element type (line 37)
int_129757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), tuple_129753, int_129757)
# Adding element type (line 37)
str_129758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'str', 'class3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), tuple_129753, str_129758)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), list_129740, tuple_129753)

# Assigning a type to the variable 'expect4_data' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'expect4_data', list_129740)

# Assigning a List to a Name (line 38):

# Assigning a List to a Name (line 38):

# Obtaining an instance of the builtin type 'list' (line 38)
list_129759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
str_129760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'str', 'numeric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 17), list_129759, str_129760)
# Adding element type (line 38)
str_129761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 29), 'str', 'numeric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 17), list_129759, str_129761)
# Adding element type (line 38)
str_129762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'str', 'numeric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 17), list_129759, str_129762)
# Adding element type (line 38)
str_129763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 51), 'str', 'numeric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 17), list_129759, str_129763)
# Adding element type (line 38)
str_129764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 62), 'str', 'nominal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 17), list_129759, str_129764)

# Assigning a type to the variable 'expected_types' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'expected_types', list_129759)

# Assigning a Call to a Name (line 40):

# Assigning a Call to a Name (line 40):

# Call to pjoin(...): (line 40)
# Processing the call arguments (line 40)
# Getting the type of 'data_path' (line 40)
data_path_129766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'data_path', False)
str_129767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'str', 'missing.arff')
# Processing the call keyword arguments (line 40)
kwargs_129768 = {}
# Getting the type of 'pjoin' (line 40)
pjoin_129765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 40)
pjoin_call_result_129769 = invoke(stypy.reporting.localization.Localization(__file__, 40, 10), pjoin_129765, *[data_path_129766, str_129767], **kwargs_129768)

# Assigning a type to the variable 'missing' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'missing', pjoin_call_result_129769)

# Assigning a Call to a Name (line 41):

# Assigning a Call to a Name (line 41):

# Call to array(...): (line 41)
# Processing the call arguments (line 41)

# Obtaining an instance of the builtin type 'list' (line 41)
list_129772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)

# Obtaining an instance of the builtin type 'list' (line 41)
list_129773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
int_129774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 31), list_129773, int_129774)
# Adding element type (line 41)
int_129775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 31), list_129773, int_129775)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 30), list_129772, list_129773)
# Adding element type (line 41)

# Obtaining an instance of the builtin type 'list' (line 41)
list_129776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
int_129777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), list_129776, int_129777)
# Adding element type (line 41)
int_129778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 39), list_129776, int_129778)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 30), list_129772, list_129776)
# Adding element type (line 41)

# Obtaining an instance of the builtin type 'list' (line 41)
list_129779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 47), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
# Getting the type of 'np' (line 41)
np_129780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 48), 'np', False)
# Obtaining the member 'nan' of a type (line 41)
nan_129781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 48), np_129780, 'nan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 47), list_129779, nan_129781)
# Adding element type (line 41)
# Getting the type of 'np' (line 41)
np_129782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 56), 'np', False)
# Obtaining the member 'nan' of a type (line 41)
nan_129783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 56), np_129782, 'nan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 47), list_129779, nan_129783)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 30), list_129772, list_129779)

# Processing the call keyword arguments (line 41)
kwargs_129784 = {}
# Getting the type of 'np' (line 41)
np_129770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'np', False)
# Obtaining the member 'array' of a type (line 41)
array_129771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 21), np_129770, 'array')
# Calling array(args, kwargs) (line 41)
array_call_result_129785 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), array_129771, *[list_129772], **kwargs_129784)

# Assigning a type to the variable 'expect_missing_raw' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'expect_missing_raw', array_call_result_129785)

# Assigning a Call to a Name (line 42):

# Assigning a Call to a Name (line 42):

# Call to empty(...): (line 42)
# Processing the call arguments (line 42)
int_129788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'int')

# Obtaining an instance of the builtin type 'list' (line 42)
list_129789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_129790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
str_129791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'str', 'yop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 31), tuple_129790, str_129791)
# Adding element type (line 42)
# Getting the type of 'float' (line 42)
float_129792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 31), tuple_129790, float_129792)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 29), list_129789, tuple_129790)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_129793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 47), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
str_129794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 47), 'str', 'yap')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 47), tuple_129793, str_129794)
# Adding element type (line 42)
# Getting the type of 'float' (line 42)
float_129795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 54), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 47), tuple_129793, float_129795)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 29), list_129789, tuple_129793)

# Processing the call keyword arguments (line 42)
kwargs_129796 = {}
# Getting the type of 'np' (line 42)
np_129786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'np', False)
# Obtaining the member 'empty' of a type (line 42)
empty_129787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 17), np_129786, 'empty')
# Calling empty(args, kwargs) (line 42)
empty_call_result_129797 = invoke(stypy.reporting.localization.Localization(__file__, 42, 17), empty_129787, *[int_129788, list_129789], **kwargs_129796)

# Assigning a type to the variable 'expect_missing' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'expect_missing', empty_call_result_129797)

# Assigning a Subscript to a Subscript (line 43):

# Assigning a Subscript to a Subscript (line 43):

# Obtaining the type of the subscript
slice_129798 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 43, 24), None, None, None)
int_129799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 46), 'int')
# Getting the type of 'expect_missing_raw' (line 43)
expect_missing_raw_129800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'expect_missing_raw')
# Obtaining the member '__getitem__' of a type (line 43)
getitem___129801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), expect_missing_raw_129800, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 43)
subscript_call_result_129802 = invoke(stypy.reporting.localization.Localization(__file__, 43, 24), getitem___129801, (slice_129798, int_129799))

# Getting the type of 'expect_missing' (line 43)
expect_missing_129803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'expect_missing')
str_129804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'str', 'yop')
# Storing an element on a container (line 43)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 0), expect_missing_129803, (str_129804, subscript_call_result_129802))

# Assigning a Subscript to a Subscript (line 44):

# Assigning a Subscript to a Subscript (line 44):

# Obtaining the type of the subscript
slice_129805 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 44, 24), None, None, None)
int_129806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 46), 'int')
# Getting the type of 'expect_missing_raw' (line 44)
expect_missing_raw_129807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'expect_missing_raw')
# Obtaining the member '__getitem__' of a type (line 44)
getitem___129808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), expect_missing_raw_129807, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 44)
subscript_call_result_129809 = invoke(stypy.reporting.localization.Localization(__file__, 44, 24), getitem___129808, (slice_129805, int_129806))

# Getting the type of 'expect_missing' (line 44)
expect_missing_129810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'expect_missing')
str_129811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 15), 'str', 'yap')
# Storing an element on a container (line 44)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 0), expect_missing_129810, (str_129811, subscript_call_result_129809))
# Declaration of the 'TestData' class

class TestData(object, ):

    @norecursion
    def test1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test1'
        module_type_store = module_type_store.open_function_context('test1', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestData.test1.__dict__.__setitem__('stypy_localization', localization)
        TestData.test1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestData.test1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestData.test1.__dict__.__setitem__('stypy_function_name', 'TestData.test1')
        TestData.test1.__dict__.__setitem__('stypy_param_names_list', [])
        TestData.test1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestData.test1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestData.test1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestData.test1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestData.test1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestData.test1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestData.test1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test1(...)' code ##################

        
        # Call to _test(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'test4' (line 50)
        test4_129814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'test4', False)
        # Processing the call keyword arguments (line 50)
        kwargs_129815 = {}
        # Getting the type of 'self' (line 50)
        self_129812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self', False)
        # Obtaining the member '_test' of a type (line 50)
        _test_129813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_129812, '_test')
        # Calling _test(args, kwargs) (line 50)
        _test_call_result_129816 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), _test_129813, *[test4_129814], **kwargs_129815)
        
        
        # ################# End of 'test1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test1' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_129817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test1'
        return stypy_return_type_129817


    @norecursion
    def test2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test2'
        module_type_store = module_type_store.open_function_context('test2', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestData.test2.__dict__.__setitem__('stypy_localization', localization)
        TestData.test2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestData.test2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestData.test2.__dict__.__setitem__('stypy_function_name', 'TestData.test2')
        TestData.test2.__dict__.__setitem__('stypy_param_names_list', [])
        TestData.test2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestData.test2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestData.test2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestData.test2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestData.test2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestData.test2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestData.test2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test2(...)' code ##################

        
        # Call to _test(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'test5' (line 54)
        test5_129820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'test5', False)
        # Processing the call keyword arguments (line 54)
        kwargs_129821 = {}
        # Getting the type of 'self' (line 54)
        self_129818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member '_test' of a type (line 54)
        _test_129819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_129818, '_test')
        # Calling _test(args, kwargs) (line 54)
        _test_call_result_129822 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), _test_129819, *[test5_129820], **kwargs_129821)
        
        
        # ################# End of 'test2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test2' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_129823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test2'
        return stypy_return_type_129823


    @norecursion
    def test3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test3'
        module_type_store = module_type_store.open_function_context('test3', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestData.test3.__dict__.__setitem__('stypy_localization', localization)
        TestData.test3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestData.test3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestData.test3.__dict__.__setitem__('stypy_function_name', 'TestData.test3')
        TestData.test3.__dict__.__setitem__('stypy_param_names_list', [])
        TestData.test3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestData.test3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestData.test3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestData.test3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestData.test3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestData.test3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestData.test3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test3(...)' code ##################

        
        # Call to _test(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'test6' (line 58)
        test6_129826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'test6', False)
        # Processing the call keyword arguments (line 58)
        kwargs_129827 = {}
        # Getting the type of 'self' (line 58)
        self_129824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', False)
        # Obtaining the member '_test' of a type (line 58)
        _test_129825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_129824, '_test')
        # Calling _test(args, kwargs) (line 58)
        _test_call_result_129828 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), _test_129825, *[test6_129826], **kwargs_129827)
        
        
        # ################# End of 'test3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test3' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_129829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129829)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test3'
        return stypy_return_type_129829


    @norecursion
    def _test(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_test'
        module_type_store = module_type_store.open_function_context('_test', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestData._test.__dict__.__setitem__('stypy_localization', localization)
        TestData._test.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestData._test.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestData._test.__dict__.__setitem__('stypy_function_name', 'TestData._test')
        TestData._test.__dict__.__setitem__('stypy_param_names_list', ['test_file'])
        TestData._test.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestData._test.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestData._test.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestData._test.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestData._test.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestData._test.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestData._test', ['test_file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_test', localization, ['test_file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_test(...)' code ##################

        
        # Assigning a Call to a Tuple (line 61):
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_129830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        
        # Call to loadarff(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'test_file' (line 61)
        test_file_129832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'test_file', False)
        # Processing the call keyword arguments (line 61)
        kwargs_129833 = {}
        # Getting the type of 'loadarff' (line 61)
        loadarff_129831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 61)
        loadarff_call_result_129834 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), loadarff_129831, *[test_file_129832], **kwargs_129833)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___129835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), loadarff_call_result_129834, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_129836 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___129835, int_129830)
        
        # Assigning a type to the variable 'tuple_var_assignment_129642' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_129642', subscript_call_result_129836)
        
        # Assigning a Subscript to a Name (line 61):
        
        # Obtaining the type of the subscript
        int_129837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
        
        # Call to loadarff(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'test_file' (line 61)
        test_file_129839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'test_file', False)
        # Processing the call keyword arguments (line 61)
        kwargs_129840 = {}
        # Getting the type of 'loadarff' (line 61)
        loadarff_129838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 61)
        loadarff_call_result_129841 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), loadarff_129838, *[test_file_129839], **kwargs_129840)
        
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___129842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), loadarff_call_result_129841, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_129843 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___129842, int_129837)
        
        # Assigning a type to the variable 'tuple_var_assignment_129643' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_129643', subscript_call_result_129843)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'tuple_var_assignment_129642' (line 61)
        tuple_var_assignment_129642_129844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_129642')
        # Assigning a type to the variable 'data' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'data', tuple_var_assignment_129642_129844)
        
        # Assigning a Name to a Name (line 61):
        # Getting the type of 'tuple_var_assignment_129643' (line 61)
        tuple_var_assignment_129643_129845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_129643')
        # Assigning a type to the variable 'meta' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'meta', tuple_var_assignment_129643_129845)
        
        
        # Call to range(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to len(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'data' (line 62)
        data_129848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'data', False)
        # Processing the call keyword arguments (line 62)
        kwargs_129849 = {}
        # Getting the type of 'len' (line 62)
        len_129847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'len', False)
        # Calling len(args, kwargs) (line 62)
        len_call_result_129850 = invoke(stypy.reporting.localization.Localization(__file__, 62, 23), len_129847, *[data_129848], **kwargs_129849)
        
        # Processing the call keyword arguments (line 62)
        kwargs_129851 = {}
        # Getting the type of 'range' (line 62)
        range_129846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'range', False)
        # Calling range(args, kwargs) (line 62)
        range_call_result_129852 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), range_129846, *[len_call_result_129850], **kwargs_129851)
        
        # Testing the type of a for loop iterable (line 62)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 8), range_call_result_129852)
        # Getting the type of the for loop variable (line 62)
        for_loop_var_129853 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 8), range_call_result_129852)
        # Assigning a type to the variable 'i' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'i', for_loop_var_129853)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 63)
        # Processing the call arguments (line 63)
        int_129855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_129856 = {}
        # Getting the type of 'range' (line 63)
        range_129854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'range', False)
        # Calling range(args, kwargs) (line 63)
        range_call_result_129857 = invoke(stypy.reporting.localization.Localization(__file__, 63, 21), range_129854, *[int_129855], **kwargs_129856)
        
        # Testing the type of a for loop iterable (line 63)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 12), range_call_result_129857)
        # Getting the type of the for loop variable (line 63)
        for_loop_var_129858 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 12), range_call_result_129857)
        # Assigning a type to the variable 'j' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'j', for_loop_var_129858)
        # SSA begins for a for statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_almost_equal(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 64)
        j_129860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 58), 'j', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 64)
        i_129861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 55), 'i', False)
        # Getting the type of 'expect4_data' (line 64)
        expect4_data_129862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 42), 'expect4_data', False)
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___129863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 42), expect4_data_129862, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_129864 = invoke(stypy.reporting.localization.Localization(__file__, 64, 42), getitem___129863, i_129861)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___129865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 42), subscript_call_result_129864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_129866 = invoke(stypy.reporting.localization.Localization(__file__, 64, 42), getitem___129865, j_129860)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 64)
        j_129867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 70), 'j', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 64)
        i_129868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 67), 'i', False)
        # Getting the type of 'data' (line 64)
        data_129869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 62), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___129870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 62), data_129869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_129871 = invoke(stypy.reporting.localization.Localization(__file__, 64, 62), getitem___129870, i_129868)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___129872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 62), subscript_call_result_129871, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_129873 = invoke(stypy.reporting.localization.Localization(__file__, 64, 62), getitem___129872, j_129867)
        
        # Processing the call keyword arguments (line 64)
        kwargs_129874 = {}
        # Getting the type of 'assert_array_almost_equal' (line 64)
        assert_array_almost_equal_129859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 64)
        assert_array_almost_equal_call_result_129875 = invoke(stypy.reporting.localization.Localization(__file__, 64, 16), assert_array_almost_equal_129859, *[subscript_call_result_129866, subscript_call_result_129873], **kwargs_129874)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Call to types(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_129879 = {}
        # Getting the type of 'meta' (line 65)
        meta_129877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'meta', False)
        # Obtaining the member 'types' of a type (line 65)
        types_129878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), meta_129877, 'types')
        # Calling types(args, kwargs) (line 65)
        types_call_result_129880 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), types_129878, *[], **kwargs_129879)
        
        # Getting the type of 'expected_types' (line 65)
        expected_types_129881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'expected_types', False)
        # Processing the call keyword arguments (line 65)
        kwargs_129882 = {}
        # Getting the type of 'assert_equal' (line 65)
        assert_equal_129876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 65)
        assert_equal_call_result_129883 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_equal_129876, *[types_call_result_129880, expected_types_129881], **kwargs_129882)
        
        
        # ################# End of '_test(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_test' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_129884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129884)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_test'
        return stypy_return_type_129884


    @norecursion
    def test_filelike(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_filelike'
        module_type_store = module_type_store.open_function_context('test_filelike', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestData.test_filelike.__dict__.__setitem__('stypy_localization', localization)
        TestData.test_filelike.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestData.test_filelike.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestData.test_filelike.__dict__.__setitem__('stypy_function_name', 'TestData.test_filelike')
        TestData.test_filelike.__dict__.__setitem__('stypy_param_names_list', [])
        TestData.test_filelike.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestData.test_filelike.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestData.test_filelike.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestData.test_filelike.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestData.test_filelike.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestData.test_filelike.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestData.test_filelike', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_filelike', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_filelike(...)' code ##################

        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to open(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'test1' (line 69)
        test1_129886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'test1', False)
        # Processing the call keyword arguments (line 69)
        kwargs_129887 = {}
        # Getting the type of 'open' (line 69)
        open_129885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'open', False)
        # Calling open(args, kwargs) (line 69)
        open_call_result_129888 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), open_129885, *[test1_129886], **kwargs_129887)
        
        # Assigning a type to the variable 'f1' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'f1', open_call_result_129888)
        
        # Assigning a Call to a Tuple (line 70):
        
        # Assigning a Subscript to a Name (line 70):
        
        # Obtaining the type of the subscript
        int_129889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 8), 'int')
        
        # Call to loadarff(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'f1' (line 70)
        f1_129891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'f1', False)
        # Processing the call keyword arguments (line 70)
        kwargs_129892 = {}
        # Getting the type of 'loadarff' (line 70)
        loadarff_129890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 70)
        loadarff_call_result_129893 = invoke(stypy.reporting.localization.Localization(__file__, 70, 23), loadarff_129890, *[f1_129891], **kwargs_129892)
        
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___129894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), loadarff_call_result_129893, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_129895 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), getitem___129894, int_129889)
        
        # Assigning a type to the variable 'tuple_var_assignment_129644' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'tuple_var_assignment_129644', subscript_call_result_129895)
        
        # Assigning a Subscript to a Name (line 70):
        
        # Obtaining the type of the subscript
        int_129896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 8), 'int')
        
        # Call to loadarff(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'f1' (line 70)
        f1_129898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'f1', False)
        # Processing the call keyword arguments (line 70)
        kwargs_129899 = {}
        # Getting the type of 'loadarff' (line 70)
        loadarff_129897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 70)
        loadarff_call_result_129900 = invoke(stypy.reporting.localization.Localization(__file__, 70, 23), loadarff_129897, *[f1_129898], **kwargs_129899)
        
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___129901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), loadarff_call_result_129900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_129902 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), getitem___129901, int_129896)
        
        # Assigning a type to the variable 'tuple_var_assignment_129645' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'tuple_var_assignment_129645', subscript_call_result_129902)
        
        # Assigning a Name to a Name (line 70):
        # Getting the type of 'tuple_var_assignment_129644' (line 70)
        tuple_var_assignment_129644_129903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'tuple_var_assignment_129644')
        # Assigning a type to the variable 'data1' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'data1', tuple_var_assignment_129644_129903)
        
        # Assigning a Name to a Name (line 70):
        # Getting the type of 'tuple_var_assignment_129645' (line 70)
        tuple_var_assignment_129645_129904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'tuple_var_assignment_129645')
        # Assigning a type to the variable 'meta1' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'meta1', tuple_var_assignment_129645_129904)
        
        # Call to close(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_129907 = {}
        # Getting the type of 'f1' (line 71)
        f1_129905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'f1', False)
        # Obtaining the member 'close' of a type (line 71)
        close_129906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), f1_129905, 'close')
        # Calling close(args, kwargs) (line 71)
        close_call_result_129908 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), close_129906, *[], **kwargs_129907)
        
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to open(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'test1' (line 72)
        test1_129910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'test1', False)
        # Processing the call keyword arguments (line 72)
        kwargs_129911 = {}
        # Getting the type of 'open' (line 72)
        open_129909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'open', False)
        # Calling open(args, kwargs) (line 72)
        open_call_result_129912 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), open_129909, *[test1_129910], **kwargs_129911)
        
        # Assigning a type to the variable 'f2' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'f2', open_call_result_129912)
        
        # Assigning a Call to a Tuple (line 73):
        
        # Assigning a Subscript to a Name (line 73):
        
        # Obtaining the type of the subscript
        int_129913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'int')
        
        # Call to loadarff(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to StringIO(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to read(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_129918 = {}
        # Getting the type of 'f2' (line 73)
        f2_129916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 41), 'f2', False)
        # Obtaining the member 'read' of a type (line 73)
        read_129917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 41), f2_129916, 'read')
        # Calling read(args, kwargs) (line 73)
        read_call_result_129919 = invoke(stypy.reporting.localization.Localization(__file__, 73, 41), read_129917, *[], **kwargs_129918)
        
        # Processing the call keyword arguments (line 73)
        kwargs_129920 = {}
        # Getting the type of 'StringIO' (line 73)
        StringIO_129915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 73)
        StringIO_call_result_129921 = invoke(stypy.reporting.localization.Localization(__file__, 73, 32), StringIO_129915, *[read_call_result_129919], **kwargs_129920)
        
        # Processing the call keyword arguments (line 73)
        kwargs_129922 = {}
        # Getting the type of 'loadarff' (line 73)
        loadarff_129914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 73)
        loadarff_call_result_129923 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), loadarff_129914, *[StringIO_call_result_129921], **kwargs_129922)
        
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___129924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), loadarff_call_result_129923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_129925 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), getitem___129924, int_129913)
        
        # Assigning a type to the variable 'tuple_var_assignment_129646' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_var_assignment_129646', subscript_call_result_129925)
        
        # Assigning a Subscript to a Name (line 73):
        
        # Obtaining the type of the subscript
        int_129926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'int')
        
        # Call to loadarff(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to StringIO(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to read(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_129931 = {}
        # Getting the type of 'f2' (line 73)
        f2_129929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 41), 'f2', False)
        # Obtaining the member 'read' of a type (line 73)
        read_129930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 41), f2_129929, 'read')
        # Calling read(args, kwargs) (line 73)
        read_call_result_129932 = invoke(stypy.reporting.localization.Localization(__file__, 73, 41), read_129930, *[], **kwargs_129931)
        
        # Processing the call keyword arguments (line 73)
        kwargs_129933 = {}
        # Getting the type of 'StringIO' (line 73)
        StringIO_129928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 32), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 73)
        StringIO_call_result_129934 = invoke(stypy.reporting.localization.Localization(__file__, 73, 32), StringIO_129928, *[read_call_result_129932], **kwargs_129933)
        
        # Processing the call keyword arguments (line 73)
        kwargs_129935 = {}
        # Getting the type of 'loadarff' (line 73)
        loadarff_129927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 73)
        loadarff_call_result_129936 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), loadarff_129927, *[StringIO_call_result_129934], **kwargs_129935)
        
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___129937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), loadarff_call_result_129936, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_129938 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), getitem___129937, int_129926)
        
        # Assigning a type to the variable 'tuple_var_assignment_129647' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_var_assignment_129647', subscript_call_result_129938)
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'tuple_var_assignment_129646' (line 73)
        tuple_var_assignment_129646_129939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_var_assignment_129646')
        # Assigning a type to the variable 'data2' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'data2', tuple_var_assignment_129646_129939)
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'tuple_var_assignment_129647' (line 73)
        tuple_var_assignment_129647_129940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_var_assignment_129647')
        # Assigning a type to the variable 'meta2' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'meta2', tuple_var_assignment_129647_129940)
        
        # Call to close(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_129943 = {}
        # Getting the type of 'f2' (line 74)
        f2_129941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'f2', False)
        # Obtaining the member 'close' of a type (line 74)
        close_129942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), f2_129941, 'close')
        # Calling close(args, kwargs) (line 74)
        close_call_result_129944 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), close_129942, *[], **kwargs_129943)
        
        
        # Call to assert_(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Getting the type of 'data1' (line 75)
        data1_129946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'data1', False)
        # Getting the type of 'data2' (line 75)
        data2_129947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'data2', False)
        # Applying the binary operator '==' (line 75)
        result_eq_129948 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '==', data1_129946, data2_129947)
        
        # Processing the call keyword arguments (line 75)
        kwargs_129949 = {}
        # Getting the type of 'assert_' (line 75)
        assert__129945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 75)
        assert__call_result_129950 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), assert__129945, *[result_eq_129948], **kwargs_129949)
        
        
        # Call to assert_(...): (line 76)
        # Processing the call arguments (line 76)
        
        
        # Call to repr(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'meta1' (line 76)
        meta1_129953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 21), 'meta1', False)
        # Processing the call keyword arguments (line 76)
        kwargs_129954 = {}
        # Getting the type of 'repr' (line 76)
        repr_129952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'repr', False)
        # Calling repr(args, kwargs) (line 76)
        repr_call_result_129955 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), repr_129952, *[meta1_129953], **kwargs_129954)
        
        
        # Call to repr(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'meta2' (line 76)
        meta2_129957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'meta2', False)
        # Processing the call keyword arguments (line 76)
        kwargs_129958 = {}
        # Getting the type of 'repr' (line 76)
        repr_129956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'repr', False)
        # Calling repr(args, kwargs) (line 76)
        repr_call_result_129959 = invoke(stypy.reporting.localization.Localization(__file__, 76, 31), repr_129956, *[meta2_129957], **kwargs_129958)
        
        # Applying the binary operator '==' (line 76)
        result_eq_129960 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 16), '==', repr_call_result_129955, repr_call_result_129959)
        
        # Processing the call keyword arguments (line 76)
        kwargs_129961 = {}
        # Getting the type of 'assert_' (line 76)
        assert__129951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 76)
        assert__call_result_129962 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert__129951, *[result_eq_129960], **kwargs_129961)
        
        
        # ################# End of 'test_filelike(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_filelike' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_129963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_filelike'
        return stypy_return_type_129963


    @norecursion
    def test_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_path'
        module_type_store = module_type_store.open_function_context('test_path', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestData.test_path.__dict__.__setitem__('stypy_localization', localization)
        TestData.test_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestData.test_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestData.test_path.__dict__.__setitem__('stypy_function_name', 'TestData.test_path')
        TestData.test_path.__dict__.__setitem__('stypy_param_names_list', [])
        TestData.test_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestData.test_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestData.test_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestData.test_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestData.test_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestData.test_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestData.test_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_path(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 82, 8))
        
        # 'from pathlib import Path' statement (line 82)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
        import_129964 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 82, 8), 'pathlib')

        if (type(import_129964) is not StypyTypeError):

            if (import_129964 != 'pyd_module'):
                __import__(import_129964)
                sys_modules_129965 = sys.modules[import_129964]
                import_from_module(stypy.reporting.localization.Localization(__file__, 82, 8), 'pathlib', sys_modules_129965.module_type_store, module_type_store, ['Path'])
                nest_module(stypy.reporting.localization.Localization(__file__, 82, 8), __file__, sys_modules_129965, sys_modules_129965.module_type_store, module_type_store)
            else:
                from pathlib import Path

                import_from_module(stypy.reporting.localization.Localization(__file__, 82, 8), 'pathlib', None, module_type_store, ['Path'], [Path])

        else:
            # Assigning a type to the variable 'pathlib' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'pathlib', import_129964)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/tests/')
        
        
        # Call to open(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'test1' (line 84)
        test1_129967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'test1', False)
        # Processing the call keyword arguments (line 84)
        kwargs_129968 = {}
        # Getting the type of 'open' (line 84)
        open_129966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'open', False)
        # Calling open(args, kwargs) (line 84)
        open_call_result_129969 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), open_129966, *[test1_129967], **kwargs_129968)
        
        with_129970 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 84, 13), open_call_result_129969, 'with parameter', '__enter__', '__exit__')

        if with_129970:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 84)
            enter___129971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), open_call_result_129969, '__enter__')
            with_enter_129972 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), enter___129971)
            # Assigning a type to the variable 'f1' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'f1', with_enter_129972)
            
            # Assigning a Call to a Tuple (line 85):
            
            # Assigning a Subscript to a Name (line 85):
            
            # Obtaining the type of the subscript
            int_129973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'int')
            
            # Call to loadarff(...): (line 85)
            # Processing the call arguments (line 85)
            # Getting the type of 'f1' (line 85)
            f1_129975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'f1', False)
            # Processing the call keyword arguments (line 85)
            kwargs_129976 = {}
            # Getting the type of 'loadarff' (line 85)
            loadarff_129974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'loadarff', False)
            # Calling loadarff(args, kwargs) (line 85)
            loadarff_call_result_129977 = invoke(stypy.reporting.localization.Localization(__file__, 85, 27), loadarff_129974, *[f1_129975], **kwargs_129976)
            
            # Obtaining the member '__getitem__' of a type (line 85)
            getitem___129978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), loadarff_call_result_129977, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 85)
            subscript_call_result_129979 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), getitem___129978, int_129973)
            
            # Assigning a type to the variable 'tuple_var_assignment_129648' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'tuple_var_assignment_129648', subscript_call_result_129979)
            
            # Assigning a Subscript to a Name (line 85):
            
            # Obtaining the type of the subscript
            int_129980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'int')
            
            # Call to loadarff(...): (line 85)
            # Processing the call arguments (line 85)
            # Getting the type of 'f1' (line 85)
            f1_129982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'f1', False)
            # Processing the call keyword arguments (line 85)
            kwargs_129983 = {}
            # Getting the type of 'loadarff' (line 85)
            loadarff_129981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'loadarff', False)
            # Calling loadarff(args, kwargs) (line 85)
            loadarff_call_result_129984 = invoke(stypy.reporting.localization.Localization(__file__, 85, 27), loadarff_129981, *[f1_129982], **kwargs_129983)
            
            # Obtaining the member '__getitem__' of a type (line 85)
            getitem___129985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), loadarff_call_result_129984, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 85)
            subscript_call_result_129986 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), getitem___129985, int_129980)
            
            # Assigning a type to the variable 'tuple_var_assignment_129649' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'tuple_var_assignment_129649', subscript_call_result_129986)
            
            # Assigning a Name to a Name (line 85):
            # Getting the type of 'tuple_var_assignment_129648' (line 85)
            tuple_var_assignment_129648_129987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'tuple_var_assignment_129648')
            # Assigning a type to the variable 'data1' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'data1', tuple_var_assignment_129648_129987)
            
            # Assigning a Name to a Name (line 85):
            # Getting the type of 'tuple_var_assignment_129649' (line 85)
            tuple_var_assignment_129649_129988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'tuple_var_assignment_129649')
            # Assigning a type to the variable 'meta1' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'meta1', tuple_var_assignment_129649_129988)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 84)
            exit___129989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), open_call_result_129969, '__exit__')
            with_exit_129990 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), exit___129989, None, None, None)

        
        # Assigning a Call to a Tuple (line 87):
        
        # Assigning a Subscript to a Name (line 87):
        
        # Obtaining the type of the subscript
        int_129991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'int')
        
        # Call to loadarff(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to Path(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'test1' (line 87)
        test1_129994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'test1', False)
        # Processing the call keyword arguments (line 87)
        kwargs_129995 = {}
        # Getting the type of 'Path' (line 87)
        Path_129993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'Path', False)
        # Calling Path(args, kwargs) (line 87)
        Path_call_result_129996 = invoke(stypy.reporting.localization.Localization(__file__, 87, 32), Path_129993, *[test1_129994], **kwargs_129995)
        
        # Processing the call keyword arguments (line 87)
        kwargs_129997 = {}
        # Getting the type of 'loadarff' (line 87)
        loadarff_129992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 87)
        loadarff_call_result_129998 = invoke(stypy.reporting.localization.Localization(__file__, 87, 23), loadarff_129992, *[Path_call_result_129996], **kwargs_129997)
        
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___129999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), loadarff_call_result_129998, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_130000 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), getitem___129999, int_129991)
        
        # Assigning a type to the variable 'tuple_var_assignment_129650' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'tuple_var_assignment_129650', subscript_call_result_130000)
        
        # Assigning a Subscript to a Name (line 87):
        
        # Obtaining the type of the subscript
        int_130001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'int')
        
        # Call to loadarff(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to Path(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'test1' (line 87)
        test1_130004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'test1', False)
        # Processing the call keyword arguments (line 87)
        kwargs_130005 = {}
        # Getting the type of 'Path' (line 87)
        Path_130003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'Path', False)
        # Calling Path(args, kwargs) (line 87)
        Path_call_result_130006 = invoke(stypy.reporting.localization.Localization(__file__, 87, 32), Path_130003, *[test1_130004], **kwargs_130005)
        
        # Processing the call keyword arguments (line 87)
        kwargs_130007 = {}
        # Getting the type of 'loadarff' (line 87)
        loadarff_130002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 87)
        loadarff_call_result_130008 = invoke(stypy.reporting.localization.Localization(__file__, 87, 23), loadarff_130002, *[Path_call_result_130006], **kwargs_130007)
        
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___130009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), loadarff_call_result_130008, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_130010 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), getitem___130009, int_130001)
        
        # Assigning a type to the variable 'tuple_var_assignment_129651' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'tuple_var_assignment_129651', subscript_call_result_130010)
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of 'tuple_var_assignment_129650' (line 87)
        tuple_var_assignment_129650_130011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'tuple_var_assignment_129650')
        # Assigning a type to the variable 'data2' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'data2', tuple_var_assignment_129650_130011)
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of 'tuple_var_assignment_129651' (line 87)
        tuple_var_assignment_129651_130012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'tuple_var_assignment_129651')
        # Assigning a type to the variable 'meta2' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'meta2', tuple_var_assignment_129651_130012)
        
        # Call to assert_(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Getting the type of 'data1' (line 89)
        data1_130014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'data1', False)
        # Getting the type of 'data2' (line 89)
        data2_130015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'data2', False)
        # Applying the binary operator '==' (line 89)
        result_eq_130016 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 16), '==', data1_130014, data2_130015)
        
        # Processing the call keyword arguments (line 89)
        kwargs_130017 = {}
        # Getting the type of 'assert_' (line 89)
        assert__130013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 89)
        assert__call_result_130018 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert__130013, *[result_eq_130016], **kwargs_130017)
        
        
        # Call to assert_(...): (line 90)
        # Processing the call arguments (line 90)
        
        
        # Call to repr(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'meta1' (line 90)
        meta1_130021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'meta1', False)
        # Processing the call keyword arguments (line 90)
        kwargs_130022 = {}
        # Getting the type of 'repr' (line 90)
        repr_130020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'repr', False)
        # Calling repr(args, kwargs) (line 90)
        repr_call_result_130023 = invoke(stypy.reporting.localization.Localization(__file__, 90, 16), repr_130020, *[meta1_130021], **kwargs_130022)
        
        
        # Call to repr(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'meta2' (line 90)
        meta2_130025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 36), 'meta2', False)
        # Processing the call keyword arguments (line 90)
        kwargs_130026 = {}
        # Getting the type of 'repr' (line 90)
        repr_130024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'repr', False)
        # Calling repr(args, kwargs) (line 90)
        repr_call_result_130027 = invoke(stypy.reporting.localization.Localization(__file__, 90, 31), repr_130024, *[meta2_130025], **kwargs_130026)
        
        # Applying the binary operator '==' (line 90)
        result_eq_130028 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 16), '==', repr_call_result_130023, repr_call_result_130027)
        
        # Processing the call keyword arguments (line 90)
        kwargs_130029 = {}
        # Getting the type of 'assert_' (line 90)
        assert__130019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 90)
        assert__call_result_130030 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert__130019, *[result_eq_130028], **kwargs_130029)
        
        
        # ################# End of 'test_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_path' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_130031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130031)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_path'
        return stypy_return_type_130031


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 47, 0, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestData.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestData' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'TestData', TestData)
# Declaration of the 'TestMissingData' class

class TestMissingData(object, ):

    @norecursion
    def test_missing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_missing'
        module_type_store = module_type_store.open_function_context('test_missing', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMissingData.test_missing.__dict__.__setitem__('stypy_localization', localization)
        TestMissingData.test_missing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMissingData.test_missing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMissingData.test_missing.__dict__.__setitem__('stypy_function_name', 'TestMissingData.test_missing')
        TestMissingData.test_missing.__dict__.__setitem__('stypy_param_names_list', [])
        TestMissingData.test_missing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMissingData.test_missing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMissingData.test_missing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMissingData.test_missing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMissingData.test_missing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMissingData.test_missing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMissingData.test_missing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_missing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_missing(...)' code ##################

        
        # Assigning a Call to a Tuple (line 94):
        
        # Assigning a Subscript to a Name (line 94):
        
        # Obtaining the type of the subscript
        int_130032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'int')
        
        # Call to loadarff(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'missing' (line 94)
        missing_130034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), 'missing', False)
        # Processing the call keyword arguments (line 94)
        kwargs_130035 = {}
        # Getting the type of 'loadarff' (line 94)
        loadarff_130033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 94)
        loadarff_call_result_130036 = invoke(stypy.reporting.localization.Localization(__file__, 94, 21), loadarff_130033, *[missing_130034], **kwargs_130035)
        
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___130037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), loadarff_call_result_130036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_130038 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), getitem___130037, int_130032)
        
        # Assigning a type to the variable 'tuple_var_assignment_129652' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_129652', subscript_call_result_130038)
        
        # Assigning a Subscript to a Name (line 94):
        
        # Obtaining the type of the subscript
        int_130039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'int')
        
        # Call to loadarff(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'missing' (line 94)
        missing_130041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), 'missing', False)
        # Processing the call keyword arguments (line 94)
        kwargs_130042 = {}
        # Getting the type of 'loadarff' (line 94)
        loadarff_130040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 94)
        loadarff_call_result_130043 = invoke(stypy.reporting.localization.Localization(__file__, 94, 21), loadarff_130040, *[missing_130041], **kwargs_130042)
        
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___130044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), loadarff_call_result_130043, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_130045 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), getitem___130044, int_130039)
        
        # Assigning a type to the variable 'tuple_var_assignment_129653' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_129653', subscript_call_result_130045)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_var_assignment_129652' (line 94)
        tuple_var_assignment_129652_130046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_129652')
        # Assigning a type to the variable 'data' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'data', tuple_var_assignment_129652_130046)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_var_assignment_129653' (line 94)
        tuple_var_assignment_129653_130047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_129653')
        # Assigning a type to the variable 'meta' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'meta', tuple_var_assignment_129653_130047)
        
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_130048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        str_130049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 18), 'str', 'yop')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 17), list_130048, str_130049)
        # Adding element type (line 95)
        str_130050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'str', 'yap')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 17), list_130048, str_130050)
        
        # Testing the type of a for loop iterable (line 95)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 8), list_130048)
        # Getting the type of the for loop variable (line 95)
        for_loop_var_130051 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 8), list_130048)
        # Assigning a type to the variable 'i' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'i', for_loop_var_130051)
        # SSA begins for a for statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_almost_equal(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 96)
        i_130053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'i', False)
        # Getting the type of 'data' (line 96)
        data_130054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___130055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), data_130054, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_130056 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), getitem___130055, i_130053)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 96)
        i_130057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 62), 'i', False)
        # Getting the type of 'expect_missing' (line 96)
        expect_missing_130058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 47), 'expect_missing', False)
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___130059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 47), expect_missing_130058, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_130060 = invoke(stypy.reporting.localization.Localization(__file__, 96, 47), getitem___130059, i_130057)
        
        # Processing the call keyword arguments (line 96)
        kwargs_130061 = {}
        # Getting the type of 'assert_array_almost_equal' (line 96)
        assert_array_almost_equal_130052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 96)
        assert_array_almost_equal_call_result_130062 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), assert_array_almost_equal_130052, *[subscript_call_result_130056, subscript_call_result_130060], **kwargs_130061)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_missing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_missing' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_130063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_missing'
        return stypy_return_type_130063


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 92, 0, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMissingData.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMissingData' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'TestMissingData', TestMissingData)
# Declaration of the 'TestNoData' class

class TestNoData(object, ):

    @norecursion
    def test_nodata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_nodata'
        module_type_store = module_type_store.open_function_context('test_nodata', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNoData.test_nodata.__dict__.__setitem__('stypy_localization', localization)
        TestNoData.test_nodata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNoData.test_nodata.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNoData.test_nodata.__dict__.__setitem__('stypy_function_name', 'TestNoData.test_nodata')
        TestNoData.test_nodata.__dict__.__setitem__('stypy_param_names_list', [])
        TestNoData.test_nodata.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNoData.test_nodata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNoData.test_nodata.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNoData.test_nodata.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNoData.test_nodata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNoData.test_nodata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNoData.test_nodata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_nodata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_nodata(...)' code ##################

        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to join(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'data_path' (line 103)
        data_path_130067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'data_path', False)
        str_130068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 50), 'str', 'nodata.arff')
        # Processing the call keyword arguments (line 103)
        kwargs_130069 = {}
        # Getting the type of 'os' (line 103)
        os_130064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 103)
        path_130065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), os_130064, 'path')
        # Obtaining the member 'join' of a type (line 103)
        join_130066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), path_130065, 'join')
        # Calling join(args, kwargs) (line 103)
        join_call_result_130070 = invoke(stypy.reporting.localization.Localization(__file__, 103, 26), join_130066, *[data_path_130067, str_130068], **kwargs_130069)
        
        # Assigning a type to the variable 'nodata_filename' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'nodata_filename', join_call_result_130070)
        
        # Assigning a Call to a Tuple (line 104):
        
        # Assigning a Subscript to a Name (line 104):
        
        # Obtaining the type of the subscript
        int_130071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'int')
        
        # Call to loadarff(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'nodata_filename' (line 104)
        nodata_filename_130073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'nodata_filename', False)
        # Processing the call keyword arguments (line 104)
        kwargs_130074 = {}
        # Getting the type of 'loadarff' (line 104)
        loadarff_130072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 104)
        loadarff_call_result_130075 = invoke(stypy.reporting.localization.Localization(__file__, 104, 21), loadarff_130072, *[nodata_filename_130073], **kwargs_130074)
        
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___130076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), loadarff_call_result_130075, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_130077 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), getitem___130076, int_130071)
        
        # Assigning a type to the variable 'tuple_var_assignment_129654' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tuple_var_assignment_129654', subscript_call_result_130077)
        
        # Assigning a Subscript to a Name (line 104):
        
        # Obtaining the type of the subscript
        int_130078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'int')
        
        # Call to loadarff(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'nodata_filename' (line 104)
        nodata_filename_130080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'nodata_filename', False)
        # Processing the call keyword arguments (line 104)
        kwargs_130081 = {}
        # Getting the type of 'loadarff' (line 104)
        loadarff_130079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 104)
        loadarff_call_result_130082 = invoke(stypy.reporting.localization.Localization(__file__, 104, 21), loadarff_130079, *[nodata_filename_130080], **kwargs_130081)
        
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___130083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), loadarff_call_result_130082, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_130084 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), getitem___130083, int_130078)
        
        # Assigning a type to the variable 'tuple_var_assignment_129655' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tuple_var_assignment_129655', subscript_call_result_130084)
        
        # Assigning a Name to a Name (line 104):
        # Getting the type of 'tuple_var_assignment_129654' (line 104)
        tuple_var_assignment_129654_130085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tuple_var_assignment_129654')
        # Assigning a type to the variable 'data' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'data', tuple_var_assignment_129654_130085)
        
        # Assigning a Name to a Name (line 104):
        # Getting the type of 'tuple_var_assignment_129655' (line 104)
        tuple_var_assignment_129655_130086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tuple_var_assignment_129655')
        # Assigning a type to the variable 'meta' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'meta', tuple_var_assignment_129655_130086)
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to dtype(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_130089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_130090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        str_130091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 36), 'str', 'sepallength')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 36), tuple_130090, str_130091)
        # Adding element type (line 105)
        str_130092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 51), 'str', '<f8')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 36), tuple_130090, str_130092)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), list_130089, tuple_130090)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_130093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        str_130094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 36), 'str', 'sepalwidth')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 36), tuple_130093, str_130094)
        # Adding element type (line 106)
        str_130095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 50), 'str', '<f8')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 36), tuple_130093, str_130095)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), list_130089, tuple_130093)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 107)
        tuple_130096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 107)
        # Adding element type (line 107)
        str_130097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 36), 'str', 'petallength')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 36), tuple_130096, str_130097)
        # Adding element type (line 107)
        str_130098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 51), 'str', '<f8')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 36), tuple_130096, str_130098)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), list_130089, tuple_130096)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_130099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        str_130100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'str', 'petalwidth')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 36), tuple_130099, str_130100)
        # Adding element type (line 108)
        str_130101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 50), 'str', '<f8')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 36), tuple_130099, str_130101)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), list_130089, tuple_130099)
        # Adding element type (line 105)
        
        # Obtaining an instance of the builtin type 'tuple' (line 109)
        tuple_130102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 109)
        # Adding element type (line 109)
        str_130103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'str', 'class')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 36), tuple_130102, str_130103)
        # Adding element type (line 109)
        str_130104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 45), 'str', 'S15')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 36), tuple_130102, str_130104)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 34), list_130089, tuple_130102)
        
        # Processing the call keyword arguments (line 105)
        kwargs_130105 = {}
        # Getting the type of 'np' (line 105)
        np_130087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'np', False)
        # Obtaining the member 'dtype' of a type (line 105)
        dtype_130088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), np_130087, 'dtype')
        # Calling dtype(args, kwargs) (line 105)
        dtype_call_result_130106 = invoke(stypy.reporting.localization.Localization(__file__, 105, 25), dtype_130088, *[list_130089], **kwargs_130105)
        
        # Assigning a type to the variable 'expected_dtype' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'expected_dtype', dtype_call_result_130106)
        
        # Call to assert_equal(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'data' (line 110)
        data_130108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'data', False)
        # Obtaining the member 'dtype' of a type (line 110)
        dtype_130109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 21), data_130108, 'dtype')
        # Getting the type of 'expected_dtype' (line 110)
        expected_dtype_130110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 33), 'expected_dtype', False)
        # Processing the call keyword arguments (line 110)
        kwargs_130111 = {}
        # Getting the type of 'assert_equal' (line 110)
        assert_equal_130107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 110)
        assert_equal_call_result_130112 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assert_equal_130107, *[dtype_130109, expected_dtype_130110], **kwargs_130111)
        
        
        # Call to assert_equal(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'data' (line 111)
        data_130114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'data', False)
        # Obtaining the member 'size' of a type (line 111)
        size_130115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 21), data_130114, 'size')
        int_130116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 32), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_130117 = {}
        # Getting the type of 'assert_equal' (line 111)
        assert_equal_130113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 111)
        assert_equal_call_result_130118 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assert_equal_130113, *[size_130115, int_130116], **kwargs_130117)
        
        
        # ################# End of 'test_nodata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_nodata' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_130119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130119)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_nodata'
        return stypy_return_type_130119


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 99, 0, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNoData.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestNoData' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'TestNoData', TestNoData)
# Declaration of the 'TestHeader' class

class TestHeader(object, ):

    @norecursion
    def test_type_parsing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_type_parsing'
        module_type_store = module_type_store.open_function_context('test_type_parsing', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_localization', localization)
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_function_name', 'TestHeader.test_type_parsing')
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_param_names_list', [])
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHeader.test_type_parsing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHeader.test_type_parsing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_type_parsing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_type_parsing(...)' code ##################

        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to open(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'test2' (line 117)
        test2_130121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'test2', False)
        # Processing the call keyword arguments (line 117)
        kwargs_130122 = {}
        # Getting the type of 'open' (line 117)
        open_130120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'open', False)
        # Calling open(args, kwargs) (line 117)
        open_call_result_130123 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), open_130120, *[test2_130121], **kwargs_130122)
        
        # Assigning a type to the variable 'ofile' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'ofile', open_call_result_130123)
        
        # Assigning a Call to a Tuple (line 118):
        
        # Assigning a Subscript to a Name (line 118):
        
        # Obtaining the type of the subscript
        int_130124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 8), 'int')
        
        # Call to read_header(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'ofile' (line 118)
        ofile_130126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'ofile', False)
        # Processing the call keyword arguments (line 118)
        kwargs_130127 = {}
        # Getting the type of 'read_header' (line 118)
        read_header_130125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 118)
        read_header_call_result_130128 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), read_header_130125, *[ofile_130126], **kwargs_130127)
        
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___130129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), read_header_call_result_130128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_130130 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), getitem___130129, int_130124)
        
        # Assigning a type to the variable 'tuple_var_assignment_129656' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_129656', subscript_call_result_130130)
        
        # Assigning a Subscript to a Name (line 118):
        
        # Obtaining the type of the subscript
        int_130131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 8), 'int')
        
        # Call to read_header(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'ofile' (line 118)
        ofile_130133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'ofile', False)
        # Processing the call keyword arguments (line 118)
        kwargs_130134 = {}
        # Getting the type of 'read_header' (line 118)
        read_header_130132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 118)
        read_header_call_result_130135 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), read_header_130132, *[ofile_130133], **kwargs_130134)
        
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___130136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), read_header_call_result_130135, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_130137 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), getitem___130136, int_130131)
        
        # Assigning a type to the variable 'tuple_var_assignment_129657' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_129657', subscript_call_result_130137)
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'tuple_var_assignment_129656' (line 118)
        tuple_var_assignment_129656_130138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_129656')
        # Assigning a type to the variable 'rel' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'rel', tuple_var_assignment_129656_130138)
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'tuple_var_assignment_129657' (line 118)
        tuple_var_assignment_129657_130139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tuple_var_assignment_129657')
        # Assigning a type to the variable 'attrs' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 13), 'attrs', tuple_var_assignment_129657_130139)
        
        # Call to close(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_130142 = {}
        # Getting the type of 'ofile' (line 119)
        ofile_130140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'ofile', False)
        # Obtaining the member 'close' of a type (line 119)
        close_130141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), ofile_130140, 'close')
        # Calling close(args, kwargs) (line 119)
        close_call_result_130143 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), close_130141, *[], **kwargs_130142)
        
        
        # Assigning a List to a Name (line 121):
        
        # Assigning a List to a Name (line 121):
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_130144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        str_130145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 20), 'str', 'numeric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130145)
        # Adding element type (line 121)
        str_130146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 31), 'str', 'numeric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130146)
        # Adding element type (line 121)
        str_130147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 42), 'str', 'numeric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130147)
        # Adding element type (line 121)
        str_130148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 53), 'str', 'numeric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130148)
        # Adding element type (line 121)
        str_130149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 64), 'str', 'numeric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130149)
        # Adding element type (line 121)
        str_130150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'str', 'numeric')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130150)
        # Adding element type (line 121)
        str_130151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 31), 'str', 'string')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130151)
        # Adding element type (line 121)
        str_130152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 41), 'str', 'string')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130152)
        # Adding element type (line 121)
        str_130153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 51), 'str', 'nominal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130153)
        # Adding element type (line 121)
        str_130154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 62), 'str', 'nominal')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_130144, str_130154)
        
        # Assigning a type to the variable 'expected' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'expected', list_130144)
        
        
        # Call to range(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Call to len(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'attrs' (line 124)
        attrs_130157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'attrs', False)
        # Processing the call keyword arguments (line 124)
        kwargs_130158 = {}
        # Getting the type of 'len' (line 124)
        len_130156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'len', False)
        # Calling len(args, kwargs) (line 124)
        len_call_result_130159 = invoke(stypy.reporting.localization.Localization(__file__, 124, 23), len_130156, *[attrs_130157], **kwargs_130158)
        
        # Processing the call keyword arguments (line 124)
        kwargs_130160 = {}
        # Getting the type of 'range' (line 124)
        range_130155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'range', False)
        # Calling range(args, kwargs) (line 124)
        range_call_result_130161 = invoke(stypy.reporting.localization.Localization(__file__, 124, 17), range_130155, *[len_call_result_130159], **kwargs_130160)
        
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 8), range_call_result_130161)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_130162 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 8), range_call_result_130161)
        # Assigning a type to the variable 'i' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'i', for_loop_var_130162)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_(...): (line 125)
        # Processing the call arguments (line 125)
        
        
        # Call to parse_type(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining the type of the subscript
        int_130165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 40), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 125)
        i_130166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'i', False)
        # Getting the type of 'attrs' (line 125)
        attrs_130167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___130168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 31), attrs_130167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_130169 = invoke(stypy.reporting.localization.Localization(__file__, 125, 31), getitem___130168, i_130166)
        
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___130170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 31), subscript_call_result_130169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_130171 = invoke(stypy.reporting.localization.Localization(__file__, 125, 31), getitem___130170, int_130165)
        
        # Processing the call keyword arguments (line 125)
        kwargs_130172 = {}
        # Getting the type of 'parse_type' (line 125)
        parse_type_130164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'parse_type', False)
        # Calling parse_type(args, kwargs) (line 125)
        parse_type_call_result_130173 = invoke(stypy.reporting.localization.Localization(__file__, 125, 20), parse_type_130164, *[subscript_call_result_130171], **kwargs_130172)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 125)
        i_130174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 56), 'i', False)
        # Getting the type of 'expected' (line 125)
        expected_130175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'expected', False)
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___130176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 47), expected_130175, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_130177 = invoke(stypy.reporting.localization.Localization(__file__, 125, 47), getitem___130176, i_130174)
        
        # Applying the binary operator '==' (line 125)
        result_eq_130178 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 20), '==', parse_type_call_result_130173, subscript_call_result_130177)
        
        # Processing the call keyword arguments (line 125)
        kwargs_130179 = {}
        # Getting the type of 'assert_' (line 125)
        assert__130163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 125)
        assert__call_result_130180 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), assert__130163, *[result_eq_130178], **kwargs_130179)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_type_parsing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_type_parsing' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_130181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_type_parsing'
        return stypy_return_type_130181


    @norecursion
    def test_badtype_parsing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_badtype_parsing'
        module_type_store = module_type_store.open_function_context('test_badtype_parsing', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_localization', localization)
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_function_name', 'TestHeader.test_badtype_parsing')
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_param_names_list', [])
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHeader.test_badtype_parsing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHeader.test_badtype_parsing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_badtype_parsing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_badtype_parsing(...)' code ##################

        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to open(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'test3' (line 129)
        test3_130183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'test3', False)
        # Processing the call keyword arguments (line 129)
        kwargs_130184 = {}
        # Getting the type of 'open' (line 129)
        open_130182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'open', False)
        # Calling open(args, kwargs) (line 129)
        open_call_result_130185 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), open_130182, *[test3_130183], **kwargs_130184)
        
        # Assigning a type to the variable 'ofile' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'ofile', open_call_result_130185)
        
        # Assigning a Call to a Tuple (line 130):
        
        # Assigning a Subscript to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_130186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        
        # Call to read_header(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'ofile' (line 130)
        ofile_130188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'ofile', False)
        # Processing the call keyword arguments (line 130)
        kwargs_130189 = {}
        # Getting the type of 'read_header' (line 130)
        read_header_130187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 130)
        read_header_call_result_130190 = invoke(stypy.reporting.localization.Localization(__file__, 130, 21), read_header_130187, *[ofile_130188], **kwargs_130189)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___130191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), read_header_call_result_130190, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_130192 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), getitem___130191, int_130186)
        
        # Assigning a type to the variable 'tuple_var_assignment_129658' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_129658', subscript_call_result_130192)
        
        # Assigning a Subscript to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_130193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        
        # Call to read_header(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'ofile' (line 130)
        ofile_130195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'ofile', False)
        # Processing the call keyword arguments (line 130)
        kwargs_130196 = {}
        # Getting the type of 'read_header' (line 130)
        read_header_130194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 130)
        read_header_call_result_130197 = invoke(stypy.reporting.localization.Localization(__file__, 130, 21), read_header_130194, *[ofile_130195], **kwargs_130196)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___130198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), read_header_call_result_130197, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_130199 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), getitem___130198, int_130193)
        
        # Assigning a type to the variable 'tuple_var_assignment_129659' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_129659', subscript_call_result_130199)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'tuple_var_assignment_129658' (line 130)
        tuple_var_assignment_129658_130200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_129658')
        # Assigning a type to the variable 'rel' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'rel', tuple_var_assignment_129658_130200)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'tuple_var_assignment_129659' (line 130)
        tuple_var_assignment_129659_130201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_129659')
        # Assigning a type to the variable 'attrs' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'attrs', tuple_var_assignment_129659_130201)
        
        # Call to close(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_130204 = {}
        # Getting the type of 'ofile' (line 131)
        ofile_130202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'ofile', False)
        # Obtaining the member 'close' of a type (line 131)
        close_130203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), ofile_130202, 'close')
        # Calling close(args, kwargs) (line 131)
        close_call_result_130205 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), close_130203, *[], **kwargs_130204)
        
        
        # Getting the type of 'attrs' (line 133)
        attrs_130206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), 'attrs')
        # Testing the type of a for loop iterable (line 133)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 133, 8), attrs_130206)
        # Getting the type of the for loop variable (line 133)
        for_loop_var_130207 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 133, 8), attrs_130206)
        # Assigning a type to the variable 'name' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 8), for_loop_var_130207))
        # Assigning a type to the variable 'value' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 8), for_loop_var_130207))
        # SSA begins for a for statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_raises(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'ParseArffError' (line 134)
        ParseArffError_130209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'ParseArffError', False)
        # Getting the type of 'parse_type' (line 134)
        parse_type_130210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'parse_type', False)
        # Getting the type of 'value' (line 134)
        value_130211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 54), 'value', False)
        # Processing the call keyword arguments (line 134)
        kwargs_130212 = {}
        # Getting the type of 'assert_raises' (line 134)
        assert_raises_130208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 134)
        assert_raises_call_result_130213 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), assert_raises_130208, *[ParseArffError_130209, parse_type_130210, value_130211], **kwargs_130212)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_badtype_parsing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_badtype_parsing' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_130214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_badtype_parsing'
        return stypy_return_type_130214


    @norecursion
    def test_fullheader1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fullheader1'
        module_type_store = module_type_store.open_function_context('test_fullheader1', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_localization', localization)
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_function_name', 'TestHeader.test_fullheader1')
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_param_names_list', [])
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHeader.test_fullheader1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHeader.test_fullheader1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fullheader1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fullheader1(...)' code ##################

        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to open(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'test1' (line 138)
        test1_130216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'test1', False)
        # Processing the call keyword arguments (line 138)
        kwargs_130217 = {}
        # Getting the type of 'open' (line 138)
        open_130215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'open', False)
        # Calling open(args, kwargs) (line 138)
        open_call_result_130218 = invoke(stypy.reporting.localization.Localization(__file__, 138, 16), open_130215, *[test1_130216], **kwargs_130217)
        
        # Assigning a type to the variable 'ofile' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'ofile', open_call_result_130218)
        
        # Assigning a Call to a Tuple (line 139):
        
        # Assigning a Subscript to a Name (line 139):
        
        # Obtaining the type of the subscript
        int_130219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 8), 'int')
        
        # Call to read_header(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'ofile' (line 139)
        ofile_130221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'ofile', False)
        # Processing the call keyword arguments (line 139)
        kwargs_130222 = {}
        # Getting the type of 'read_header' (line 139)
        read_header_130220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 139)
        read_header_call_result_130223 = invoke(stypy.reporting.localization.Localization(__file__, 139, 21), read_header_130220, *[ofile_130221], **kwargs_130222)
        
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___130224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), read_header_call_result_130223, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_130225 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), getitem___130224, int_130219)
        
        # Assigning a type to the variable 'tuple_var_assignment_129660' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'tuple_var_assignment_129660', subscript_call_result_130225)
        
        # Assigning a Subscript to a Name (line 139):
        
        # Obtaining the type of the subscript
        int_130226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 8), 'int')
        
        # Call to read_header(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'ofile' (line 139)
        ofile_130228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'ofile', False)
        # Processing the call keyword arguments (line 139)
        kwargs_130229 = {}
        # Getting the type of 'read_header' (line 139)
        read_header_130227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 139)
        read_header_call_result_130230 = invoke(stypy.reporting.localization.Localization(__file__, 139, 21), read_header_130227, *[ofile_130228], **kwargs_130229)
        
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___130231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), read_header_call_result_130230, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_130232 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), getitem___130231, int_130226)
        
        # Assigning a type to the variable 'tuple_var_assignment_129661' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'tuple_var_assignment_129661', subscript_call_result_130232)
        
        # Assigning a Name to a Name (line 139):
        # Getting the type of 'tuple_var_assignment_129660' (line 139)
        tuple_var_assignment_129660_130233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'tuple_var_assignment_129660')
        # Assigning a type to the variable 'rel' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'rel', tuple_var_assignment_129660_130233)
        
        # Assigning a Name to a Name (line 139):
        # Getting the type of 'tuple_var_assignment_129661' (line 139)
        tuple_var_assignment_129661_130234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'tuple_var_assignment_129661')
        # Assigning a type to the variable 'attrs' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 13), 'attrs', tuple_var_assignment_129661_130234)
        
        # Call to close(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_130237 = {}
        # Getting the type of 'ofile' (line 140)
        ofile_130235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'ofile', False)
        # Obtaining the member 'close' of a type (line 140)
        close_130236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), ofile_130235, 'close')
        # Calling close(args, kwargs) (line 140)
        close_call_result_130238 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), close_130236, *[], **kwargs_130237)
        
        
        # Call to assert_(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Getting the type of 'rel' (line 143)
        rel_130240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'rel', False)
        str_130241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 23), 'str', 'test1')
        # Applying the binary operator '==' (line 143)
        result_eq_130242 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 16), '==', rel_130240, str_130241)
        
        # Processing the call keyword arguments (line 143)
        kwargs_130243 = {}
        # Getting the type of 'assert_' (line 143)
        assert__130239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 143)
        assert__call_result_130244 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assert__130239, *[result_eq_130242], **kwargs_130243)
        
        
        # Call to assert_(...): (line 146)
        # Processing the call arguments (line 146)
        
        
        # Call to len(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'attrs' (line 146)
        attrs_130247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'attrs', False)
        # Processing the call keyword arguments (line 146)
        kwargs_130248 = {}
        # Getting the type of 'len' (line 146)
        len_130246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'len', False)
        # Calling len(args, kwargs) (line 146)
        len_call_result_130249 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), len_130246, *[attrs_130247], **kwargs_130248)
        
        int_130250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 30), 'int')
        # Applying the binary operator '==' (line 146)
        result_eq_130251 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 16), '==', len_call_result_130249, int_130250)
        
        # Processing the call keyword arguments (line 146)
        kwargs_130252 = {}
        # Getting the type of 'assert_' (line 146)
        assert__130245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 146)
        assert__call_result_130253 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), assert__130245, *[result_eq_130251], **kwargs_130252)
        
        
        
        # Call to range(...): (line 147)
        # Processing the call arguments (line 147)
        int_130255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'int')
        # Processing the call keyword arguments (line 147)
        kwargs_130256 = {}
        # Getting the type of 'range' (line 147)
        range_130254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'range', False)
        # Calling range(args, kwargs) (line 147)
        range_call_result_130257 = invoke(stypy.reporting.localization.Localization(__file__, 147, 17), range_130254, *[int_130255], **kwargs_130256)
        
        # Testing the type of a for loop iterable (line 147)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 8), range_call_result_130257)
        # Getting the type of the for loop variable (line 147)
        for_loop_var_130258 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 8), range_call_result_130257)
        # Assigning a type to the variable 'i' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'i', for_loop_var_130258)
        # SSA begins for a for statement (line 147)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_(...): (line 148)
        # Processing the call arguments (line 148)
        
        
        # Obtaining the type of the subscript
        int_130260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 29), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 148)
        i_130261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'i', False)
        # Getting the type of 'attrs' (line 148)
        attrs_130262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___130263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), attrs_130262, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_130264 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), getitem___130263, i_130261)
        
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___130265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), subscript_call_result_130264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_130266 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), getitem___130265, int_130260)
        
        str_130267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 35), 'str', 'attr%d')
        # Getting the type of 'i' (line 148)
        i_130268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 46), 'i', False)
        # Applying the binary operator '%' (line 148)
        result_mod_130269 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 35), '%', str_130267, i_130268)
        
        # Applying the binary operator '==' (line 148)
        result_eq_130270 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 20), '==', subscript_call_result_130266, result_mod_130269)
        
        # Processing the call keyword arguments (line 148)
        kwargs_130271 = {}
        # Getting the type of 'assert_' (line 148)
        assert__130259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 148)
        assert__call_result_130272 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), assert__130259, *[result_eq_130270], **kwargs_130271)
        
        
        # Call to assert_(...): (line 149)
        # Processing the call arguments (line 149)
        
        
        # Obtaining the type of the subscript
        int_130274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 149)
        i_130275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'i', False)
        # Getting the type of 'attrs' (line 149)
        attrs_130276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___130277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), attrs_130276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_130278 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), getitem___130277, i_130275)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___130279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), subscript_call_result_130278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_130280 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), getitem___130279, int_130274)
        
        str_130281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'str', 'REAL')
        # Applying the binary operator '==' (line 149)
        result_eq_130282 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 20), '==', subscript_call_result_130280, str_130281)
        
        # Processing the call keyword arguments (line 149)
        kwargs_130283 = {}
        # Getting the type of 'assert_' (line 149)
        assert__130273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 149)
        assert__call_result_130284 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), assert__130273, *[result_eq_130282], **kwargs_130283)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 152)
        # Processing the call arguments (line 152)
        
        
        # Obtaining the type of the subscript
        int_130286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'int')
        # Getting the type of 'attrs' (line 152)
        attrs_130288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___130289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), attrs_130288, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_130290 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), getitem___130289, int_130287)
        
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___130291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), subscript_call_result_130290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_130292 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), getitem___130291, int_130286)
        
        str_130293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 31), 'str', 'class')
        # Applying the binary operator '==' (line 152)
        result_eq_130294 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 16), '==', subscript_call_result_130292, str_130293)
        
        # Processing the call keyword arguments (line 152)
        kwargs_130295 = {}
        # Getting the type of 'assert_' (line 152)
        assert__130285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 152)
        assert__call_result_130296 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), assert__130285, *[result_eq_130294], **kwargs_130295)
        
        
        # Call to assert_(...): (line 153)
        # Processing the call arguments (line 153)
        
        
        # Obtaining the type of the subscript
        int_130298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'int')
        # Getting the type of 'attrs' (line 153)
        attrs_130300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___130301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), attrs_130300, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_130302 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), getitem___130301, int_130299)
        
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___130303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), subscript_call_result_130302, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_130304 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), getitem___130303, int_130298)
        
        str_130305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 31), 'str', '{class0, class1, class2, class3}')
        # Applying the binary operator '==' (line 153)
        result_eq_130306 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 16), '==', subscript_call_result_130304, str_130305)
        
        # Processing the call keyword arguments (line 153)
        kwargs_130307 = {}
        # Getting the type of 'assert_' (line 153)
        assert__130297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 153)
        assert__call_result_130308 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert__130297, *[result_eq_130306], **kwargs_130307)
        
        
        # ################# End of 'test_fullheader1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fullheader1' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_130309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130309)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fullheader1'
        return stypy_return_type_130309


    @norecursion
    def test_dateheader(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dateheader'
        module_type_store = module_type_store.open_function_context('test_dateheader', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_localization', localization)
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_function_name', 'TestHeader.test_dateheader')
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_param_names_list', [])
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHeader.test_dateheader.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHeader.test_dateheader', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dateheader', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dateheader(...)' code ##################

        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to open(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'test7' (line 156)
        test7_130311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'test7', False)
        # Processing the call keyword arguments (line 156)
        kwargs_130312 = {}
        # Getting the type of 'open' (line 156)
        open_130310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'open', False)
        # Calling open(args, kwargs) (line 156)
        open_call_result_130313 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), open_130310, *[test7_130311], **kwargs_130312)
        
        # Assigning a type to the variable 'ofile' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'ofile', open_call_result_130313)
        
        # Assigning a Call to a Tuple (line 157):
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_130314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
        
        # Call to read_header(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'ofile' (line 157)
        ofile_130316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 33), 'ofile', False)
        # Processing the call keyword arguments (line 157)
        kwargs_130317 = {}
        # Getting the type of 'read_header' (line 157)
        read_header_130315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 157)
        read_header_call_result_130318 = invoke(stypy.reporting.localization.Localization(__file__, 157, 21), read_header_130315, *[ofile_130316], **kwargs_130317)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___130319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), read_header_call_result_130318, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_130320 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___130319, int_130314)
        
        # Assigning a type to the variable 'tuple_var_assignment_129662' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_129662', subscript_call_result_130320)
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_130321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
        
        # Call to read_header(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'ofile' (line 157)
        ofile_130323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 33), 'ofile', False)
        # Processing the call keyword arguments (line 157)
        kwargs_130324 = {}
        # Getting the type of 'read_header' (line 157)
        read_header_130322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 157)
        read_header_call_result_130325 = invoke(stypy.reporting.localization.Localization(__file__, 157, 21), read_header_130322, *[ofile_130323], **kwargs_130324)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___130326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), read_header_call_result_130325, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_130327 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___130326, int_130321)
        
        # Assigning a type to the variable 'tuple_var_assignment_129663' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_129663', subscript_call_result_130327)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_129662' (line 157)
        tuple_var_assignment_129662_130328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_129662')
        # Assigning a type to the variable 'rel' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'rel', tuple_var_assignment_129662_130328)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_129663' (line 157)
        tuple_var_assignment_129663_130329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_129663')
        # Assigning a type to the variable 'attrs' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'attrs', tuple_var_assignment_129663_130329)
        
        # Call to close(...): (line 158)
        # Processing the call keyword arguments (line 158)
        kwargs_130332 = {}
        # Getting the type of 'ofile' (line 158)
        ofile_130330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'ofile', False)
        # Obtaining the member 'close' of a type (line 158)
        close_130331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), ofile_130330, 'close')
        # Calling close(args, kwargs) (line 158)
        close_call_result_130333 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), close_130331, *[], **kwargs_130332)
        
        
        # Call to assert_(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Getting the type of 'rel' (line 160)
        rel_130335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'rel', False)
        str_130336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 23), 'str', 'test7')
        # Applying the binary operator '==' (line 160)
        result_eq_130337 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 16), '==', rel_130335, str_130336)
        
        # Processing the call keyword arguments (line 160)
        kwargs_130338 = {}
        # Getting the type of 'assert_' (line 160)
        assert__130334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 160)
        assert__call_result_130339 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), assert__130334, *[result_eq_130337], **kwargs_130338)
        
        
        # Call to assert_(...): (line 162)
        # Processing the call arguments (line 162)
        
        
        # Call to len(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'attrs' (line 162)
        attrs_130342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'attrs', False)
        # Processing the call keyword arguments (line 162)
        kwargs_130343 = {}
        # Getting the type of 'len' (line 162)
        len_130341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'len', False)
        # Calling len(args, kwargs) (line 162)
        len_call_result_130344 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), len_130341, *[attrs_130342], **kwargs_130343)
        
        int_130345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 30), 'int')
        # Applying the binary operator '==' (line 162)
        result_eq_130346 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 16), '==', len_call_result_130344, int_130345)
        
        # Processing the call keyword arguments (line 162)
        kwargs_130347 = {}
        # Getting the type of 'assert_' (line 162)
        assert__130340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 162)
        assert__call_result_130348 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), assert__130340, *[result_eq_130346], **kwargs_130347)
        
        
        # Call to assert_(...): (line 164)
        # Processing the call arguments (line 164)
        
        
        # Obtaining the type of the subscript
        int_130350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 22), 'int')
        # Getting the type of 'attrs' (line 164)
        attrs_130352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___130353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), attrs_130352, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_130354 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), getitem___130353, int_130351)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___130355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 16), subscript_call_result_130354, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_130356 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), getitem___130355, int_130350)
        
        str_130357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 31), 'str', 'attr_year')
        # Applying the binary operator '==' (line 164)
        result_eq_130358 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 16), '==', subscript_call_result_130356, str_130357)
        
        # Processing the call keyword arguments (line 164)
        kwargs_130359 = {}
        # Getting the type of 'assert_' (line 164)
        assert__130349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 164)
        assert__call_result_130360 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), assert__130349, *[result_eq_130358], **kwargs_130359)
        
        
        # Call to assert_(...): (line 165)
        # Processing the call arguments (line 165)
        
        
        # Obtaining the type of the subscript
        int_130362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 22), 'int')
        # Getting the type of 'attrs' (line 165)
        attrs_130364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___130365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), attrs_130364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_130366 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), getitem___130365, int_130363)
        
        # Obtaining the member '__getitem__' of a type (line 165)
        getitem___130367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), subscript_call_result_130366, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 165)
        subscript_call_result_130368 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), getitem___130367, int_130362)
        
        str_130369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 31), 'str', 'DATE yyyy')
        # Applying the binary operator '==' (line 165)
        result_eq_130370 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 16), '==', subscript_call_result_130368, str_130369)
        
        # Processing the call keyword arguments (line 165)
        kwargs_130371 = {}
        # Getting the type of 'assert_' (line 165)
        assert__130361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 165)
        assert__call_result_130372 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert__130361, *[result_eq_130370], **kwargs_130371)
        
        
        # Call to assert_(...): (line 167)
        # Processing the call arguments (line 167)
        
        
        # Obtaining the type of the subscript
        int_130374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 22), 'int')
        # Getting the type of 'attrs' (line 167)
        attrs_130376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___130377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), attrs_130376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_130378 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), getitem___130377, int_130375)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___130379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), subscript_call_result_130378, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_130380 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), getitem___130379, int_130374)
        
        str_130381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 31), 'str', 'attr_month')
        # Applying the binary operator '==' (line 167)
        result_eq_130382 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 16), '==', subscript_call_result_130380, str_130381)
        
        # Processing the call keyword arguments (line 167)
        kwargs_130383 = {}
        # Getting the type of 'assert_' (line 167)
        assert__130373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 167)
        assert__call_result_130384 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), assert__130373, *[result_eq_130382], **kwargs_130383)
        
        
        # Call to assert_(...): (line 168)
        # Processing the call arguments (line 168)
        
        
        # Obtaining the type of the subscript
        int_130386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 22), 'int')
        # Getting the type of 'attrs' (line 168)
        attrs_130388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___130389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), attrs_130388, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_130390 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), getitem___130389, int_130387)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___130391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 16), subscript_call_result_130390, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_130392 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), getitem___130391, int_130386)
        
        str_130393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 31), 'str', 'DATE yyyy-MM')
        # Applying the binary operator '==' (line 168)
        result_eq_130394 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 16), '==', subscript_call_result_130392, str_130393)
        
        # Processing the call keyword arguments (line 168)
        kwargs_130395 = {}
        # Getting the type of 'assert_' (line 168)
        assert__130385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 168)
        assert__call_result_130396 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), assert__130385, *[result_eq_130394], **kwargs_130395)
        
        
        # Call to assert_(...): (line 170)
        # Processing the call arguments (line 170)
        
        
        # Obtaining the type of the subscript
        int_130398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 22), 'int')
        # Getting the type of 'attrs' (line 170)
        attrs_130400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___130401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), attrs_130400, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_130402 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), getitem___130401, int_130399)
        
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___130403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), subscript_call_result_130402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_130404 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), getitem___130403, int_130398)
        
        str_130405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'str', 'attr_date')
        # Applying the binary operator '==' (line 170)
        result_eq_130406 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 16), '==', subscript_call_result_130404, str_130405)
        
        # Processing the call keyword arguments (line 170)
        kwargs_130407 = {}
        # Getting the type of 'assert_' (line 170)
        assert__130397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 170)
        assert__call_result_130408 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), assert__130397, *[result_eq_130406], **kwargs_130407)
        
        
        # Call to assert_(...): (line 171)
        # Processing the call arguments (line 171)
        
        
        # Obtaining the type of the subscript
        int_130410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 22), 'int')
        # Getting the type of 'attrs' (line 171)
        attrs_130412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___130413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), attrs_130412, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_130414 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), getitem___130413, int_130411)
        
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___130415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), subscript_call_result_130414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_130416 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), getitem___130415, int_130410)
        
        str_130417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 31), 'str', 'DATE yyyy-MM-dd')
        # Applying the binary operator '==' (line 171)
        result_eq_130418 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 16), '==', subscript_call_result_130416, str_130417)
        
        # Processing the call keyword arguments (line 171)
        kwargs_130419 = {}
        # Getting the type of 'assert_' (line 171)
        assert__130409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 171)
        assert__call_result_130420 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assert__130409, *[result_eq_130418], **kwargs_130419)
        
        
        # Call to assert_(...): (line 173)
        # Processing the call arguments (line 173)
        
        
        # Obtaining the type of the subscript
        int_130422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 22), 'int')
        # Getting the type of 'attrs' (line 173)
        attrs_130424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___130425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), attrs_130424, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_130426 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), getitem___130425, int_130423)
        
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___130427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), subscript_call_result_130426, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_130428 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), getitem___130427, int_130422)
        
        str_130429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 31), 'str', 'attr_datetime_local')
        # Applying the binary operator '==' (line 173)
        result_eq_130430 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '==', subscript_call_result_130428, str_130429)
        
        # Processing the call keyword arguments (line 173)
        kwargs_130431 = {}
        # Getting the type of 'assert_' (line 173)
        assert__130421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 173)
        assert__call_result_130432 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), assert__130421, *[result_eq_130430], **kwargs_130431)
        
        
        # Call to assert_(...): (line 174)
        # Processing the call arguments (line 174)
        
        
        # Obtaining the type of the subscript
        int_130434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 22), 'int')
        # Getting the type of 'attrs' (line 174)
        attrs_130436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___130437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), attrs_130436, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_130438 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), getitem___130437, int_130435)
        
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___130439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), subscript_call_result_130438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_130440 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), getitem___130439, int_130434)
        
        str_130441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 31), 'str', 'DATE "yyyy-MM-dd HH:mm"')
        # Applying the binary operator '==' (line 174)
        result_eq_130442 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 16), '==', subscript_call_result_130440, str_130441)
        
        # Processing the call keyword arguments (line 174)
        kwargs_130443 = {}
        # Getting the type of 'assert_' (line 174)
        assert__130433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 174)
        assert__call_result_130444 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), assert__130433, *[result_eq_130442], **kwargs_130443)
        
        
        # Call to assert_(...): (line 176)
        # Processing the call arguments (line 176)
        
        
        # Obtaining the type of the subscript
        int_130446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 22), 'int')
        # Getting the type of 'attrs' (line 176)
        attrs_130448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___130449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), attrs_130448, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_130450 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), getitem___130449, int_130447)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___130451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), subscript_call_result_130450, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_130452 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), getitem___130451, int_130446)
        
        str_130453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 31), 'str', 'attr_datetime_missing')
        # Applying the binary operator '==' (line 176)
        result_eq_130454 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 16), '==', subscript_call_result_130452, str_130453)
        
        # Processing the call keyword arguments (line 176)
        kwargs_130455 = {}
        # Getting the type of 'assert_' (line 176)
        assert__130445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 176)
        assert__call_result_130456 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assert__130445, *[result_eq_130454], **kwargs_130455)
        
        
        # Call to assert_(...): (line 177)
        # Processing the call arguments (line 177)
        
        
        # Obtaining the type of the subscript
        int_130458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 22), 'int')
        # Getting the type of 'attrs' (line 177)
        attrs_130460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___130461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 16), attrs_130460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_130462 = invoke(stypy.reporting.localization.Localization(__file__, 177, 16), getitem___130461, int_130459)
        
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___130463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 16), subscript_call_result_130462, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_130464 = invoke(stypy.reporting.localization.Localization(__file__, 177, 16), getitem___130463, int_130458)
        
        str_130465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'str', 'DATE "yyyy-MM-dd HH:mm"')
        # Applying the binary operator '==' (line 177)
        result_eq_130466 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 16), '==', subscript_call_result_130464, str_130465)
        
        # Processing the call keyword arguments (line 177)
        kwargs_130467 = {}
        # Getting the type of 'assert_' (line 177)
        assert__130457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 177)
        assert__call_result_130468 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assert__130457, *[result_eq_130466], **kwargs_130467)
        
        
        # ################# End of 'test_dateheader(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dateheader' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_130469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130469)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dateheader'
        return stypy_return_type_130469


    @norecursion
    def test_dateheader_unsupported(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dateheader_unsupported'
        module_type_store = module_type_store.open_function_context('test_dateheader_unsupported', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_localization', localization)
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_function_name', 'TestHeader.test_dateheader_unsupported')
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_param_names_list', [])
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHeader.test_dateheader_unsupported.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHeader.test_dateheader_unsupported', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dateheader_unsupported', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dateheader_unsupported(...)' code ##################

        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to open(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'test8' (line 180)
        test8_130471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'test8', False)
        # Processing the call keyword arguments (line 180)
        kwargs_130472 = {}
        # Getting the type of 'open' (line 180)
        open_130470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'open', False)
        # Calling open(args, kwargs) (line 180)
        open_call_result_130473 = invoke(stypy.reporting.localization.Localization(__file__, 180, 16), open_130470, *[test8_130471], **kwargs_130472)
        
        # Assigning a type to the variable 'ofile' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'ofile', open_call_result_130473)
        
        # Assigning a Call to a Tuple (line 181):
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        int_130474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'int')
        
        # Call to read_header(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'ofile' (line 181)
        ofile_130476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'ofile', False)
        # Processing the call keyword arguments (line 181)
        kwargs_130477 = {}
        # Getting the type of 'read_header' (line 181)
        read_header_130475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 181)
        read_header_call_result_130478 = invoke(stypy.reporting.localization.Localization(__file__, 181, 21), read_header_130475, *[ofile_130476], **kwargs_130477)
        
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___130479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), read_header_call_result_130478, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_130480 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), getitem___130479, int_130474)
        
        # Assigning a type to the variable 'tuple_var_assignment_129664' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_129664', subscript_call_result_130480)
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        int_130481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 8), 'int')
        
        # Call to read_header(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'ofile' (line 181)
        ofile_130483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'ofile', False)
        # Processing the call keyword arguments (line 181)
        kwargs_130484 = {}
        # Getting the type of 'read_header' (line 181)
        read_header_130482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'read_header', False)
        # Calling read_header(args, kwargs) (line 181)
        read_header_call_result_130485 = invoke(stypy.reporting.localization.Localization(__file__, 181, 21), read_header_130482, *[ofile_130483], **kwargs_130484)
        
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___130486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), read_header_call_result_130485, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_130487 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), getitem___130486, int_130481)
        
        # Assigning a type to the variable 'tuple_var_assignment_129665' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_129665', subscript_call_result_130487)
        
        # Assigning a Name to a Name (line 181):
        # Getting the type of 'tuple_var_assignment_129664' (line 181)
        tuple_var_assignment_129664_130488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_129664')
        # Assigning a type to the variable 'rel' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'rel', tuple_var_assignment_129664_130488)
        
        # Assigning a Name to a Name (line 181):
        # Getting the type of 'tuple_var_assignment_129665' (line 181)
        tuple_var_assignment_129665_130489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'tuple_var_assignment_129665')
        # Assigning a type to the variable 'attrs' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'attrs', tuple_var_assignment_129665_130489)
        
        # Call to close(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_130492 = {}
        # Getting the type of 'ofile' (line 182)
        ofile_130490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'ofile', False)
        # Obtaining the member 'close' of a type (line 182)
        close_130491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), ofile_130490, 'close')
        # Calling close(args, kwargs) (line 182)
        close_call_result_130493 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), close_130491, *[], **kwargs_130492)
        
        
        # Call to assert_(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Getting the type of 'rel' (line 184)
        rel_130495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'rel', False)
        str_130496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 23), 'str', 'test8')
        # Applying the binary operator '==' (line 184)
        result_eq_130497 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 16), '==', rel_130495, str_130496)
        
        # Processing the call keyword arguments (line 184)
        kwargs_130498 = {}
        # Getting the type of 'assert_' (line 184)
        assert__130494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 184)
        assert__call_result_130499 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), assert__130494, *[result_eq_130497], **kwargs_130498)
        
        
        # Call to assert_(...): (line 186)
        # Processing the call arguments (line 186)
        
        
        # Call to len(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'attrs' (line 186)
        attrs_130502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'attrs', False)
        # Processing the call keyword arguments (line 186)
        kwargs_130503 = {}
        # Getting the type of 'len' (line 186)
        len_130501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'len', False)
        # Calling len(args, kwargs) (line 186)
        len_call_result_130504 = invoke(stypy.reporting.localization.Localization(__file__, 186, 16), len_130501, *[attrs_130502], **kwargs_130503)
        
        int_130505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 30), 'int')
        # Applying the binary operator '==' (line 186)
        result_eq_130506 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 16), '==', len_call_result_130504, int_130505)
        
        # Processing the call keyword arguments (line 186)
        kwargs_130507 = {}
        # Getting the type of 'assert_' (line 186)
        assert__130500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 186)
        assert__call_result_130508 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), assert__130500, *[result_eq_130506], **kwargs_130507)
        
        
        # Call to assert_(...): (line 187)
        # Processing the call arguments (line 187)
        
        
        # Obtaining the type of the subscript
        int_130510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 22), 'int')
        # Getting the type of 'attrs' (line 187)
        attrs_130512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___130513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), attrs_130512, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_130514 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), getitem___130513, int_130511)
        
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___130515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), subscript_call_result_130514, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_130516 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), getitem___130515, int_130510)
        
        str_130517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 31), 'str', 'attr_datetime_utc')
        # Applying the binary operator '==' (line 187)
        result_eq_130518 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 16), '==', subscript_call_result_130516, str_130517)
        
        # Processing the call keyword arguments (line 187)
        kwargs_130519 = {}
        # Getting the type of 'assert_' (line 187)
        assert__130509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 187)
        assert__call_result_130520 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), assert__130509, *[result_eq_130518], **kwargs_130519)
        
        
        # Call to assert_(...): (line 188)
        # Processing the call arguments (line 188)
        
        
        # Obtaining the type of the subscript
        int_130522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 22), 'int')
        # Getting the type of 'attrs' (line 188)
        attrs_130524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___130525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 16), attrs_130524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_130526 = invoke(stypy.reporting.localization.Localization(__file__, 188, 16), getitem___130525, int_130523)
        
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___130527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 16), subscript_call_result_130526, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_130528 = invoke(stypy.reporting.localization.Localization(__file__, 188, 16), getitem___130527, int_130522)
        
        str_130529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 31), 'str', 'DATE "yyyy-MM-dd HH:mm Z"')
        # Applying the binary operator '==' (line 188)
        result_eq_130530 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 16), '==', subscript_call_result_130528, str_130529)
        
        # Processing the call keyword arguments (line 188)
        kwargs_130531 = {}
        # Getting the type of 'assert_' (line 188)
        assert__130521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 188)
        assert__call_result_130532 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), assert__130521, *[result_eq_130530], **kwargs_130531)
        
        
        # Call to assert_(...): (line 190)
        # Processing the call arguments (line 190)
        
        
        # Obtaining the type of the subscript
        int_130534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 22), 'int')
        # Getting the type of 'attrs' (line 190)
        attrs_130536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___130537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), attrs_130536, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_130538 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), getitem___130537, int_130535)
        
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___130539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), subscript_call_result_130538, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_130540 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), getitem___130539, int_130534)
        
        str_130541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 31), 'str', 'attr_datetime_full')
        # Applying the binary operator '==' (line 190)
        result_eq_130542 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 16), '==', subscript_call_result_130540, str_130541)
        
        # Processing the call keyword arguments (line 190)
        kwargs_130543 = {}
        # Getting the type of 'assert_' (line 190)
        assert__130533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 190)
        assert__call_result_130544 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), assert__130533, *[result_eq_130542], **kwargs_130543)
        
        
        # Call to assert_(...): (line 191)
        # Processing the call arguments (line 191)
        
        
        # Obtaining the type of the subscript
        int_130546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 25), 'int')
        
        # Obtaining the type of the subscript
        int_130547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 22), 'int')
        # Getting the type of 'attrs' (line 191)
        attrs_130548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'attrs', False)
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___130549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 16), attrs_130548, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_130550 = invoke(stypy.reporting.localization.Localization(__file__, 191, 16), getitem___130549, int_130547)
        
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___130551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 16), subscript_call_result_130550, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_130552 = invoke(stypy.reporting.localization.Localization(__file__, 191, 16), getitem___130551, int_130546)
        
        str_130553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 31), 'str', 'DATE "yy-MM-dd HH:mm:ss z"')
        # Applying the binary operator '==' (line 191)
        result_eq_130554 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 16), '==', subscript_call_result_130552, str_130553)
        
        # Processing the call keyword arguments (line 191)
        kwargs_130555 = {}
        # Getting the type of 'assert_' (line 191)
        assert__130545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 191)
        assert__call_result_130556 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), assert__130545, *[result_eq_130554], **kwargs_130555)
        
        
        # ################# End of 'test_dateheader_unsupported(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dateheader_unsupported' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_130557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130557)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dateheader_unsupported'
        return stypy_return_type_130557


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 114, 0, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHeader.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHeader' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'TestHeader', TestHeader)
# Declaration of the 'TestDateAttribute' class

class TestDateAttribute(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_function_name', 'TestDateAttribute.setup_method')
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDateAttribute.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDateAttribute.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a Call to a Tuple (line 196):
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        int_130558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'int')
        
        # Call to loadarff(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'test7' (line 196)
        test7_130560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 40), 'test7', False)
        # Processing the call keyword arguments (line 196)
        kwargs_130561 = {}
        # Getting the type of 'loadarff' (line 196)
        loadarff_130559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 196)
        loadarff_call_result_130562 = invoke(stypy.reporting.localization.Localization(__file__, 196, 31), loadarff_130559, *[test7_130560], **kwargs_130561)
        
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___130563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), loadarff_call_result_130562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_130564 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), getitem___130563, int_130558)
        
        # Assigning a type to the variable 'tuple_var_assignment_129666' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_129666', subscript_call_result_130564)
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        int_130565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 8), 'int')
        
        # Call to loadarff(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'test7' (line 196)
        test7_130567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 40), 'test7', False)
        # Processing the call keyword arguments (line 196)
        kwargs_130568 = {}
        # Getting the type of 'loadarff' (line 196)
        loadarff_130566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'loadarff', False)
        # Calling loadarff(args, kwargs) (line 196)
        loadarff_call_result_130569 = invoke(stypy.reporting.localization.Localization(__file__, 196, 31), loadarff_130566, *[test7_130567], **kwargs_130568)
        
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___130570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), loadarff_call_result_130569, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_130571 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), getitem___130570, int_130565)
        
        # Assigning a type to the variable 'tuple_var_assignment_129667' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_129667', subscript_call_result_130571)
        
        # Assigning a Name to a Attribute (line 196):
        # Getting the type of 'tuple_var_assignment_129666' (line 196)
        tuple_var_assignment_129666_130572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_129666')
        # Getting the type of 'self' (line 196)
        self_130573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member 'data' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_130573, 'data', tuple_var_assignment_129666_130572)
        
        # Assigning a Name to a Attribute (line 196):
        # Getting the type of 'tuple_var_assignment_129667' (line 196)
        tuple_var_assignment_129667_130574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'tuple_var_assignment_129667')
        # Getting the type of 'self' (line 196)
        self_130575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'self')
        # Setting the type of the member 'meta' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 19), self_130575, 'meta', tuple_var_assignment_129667_130574)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_130576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130576)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_130576


    @norecursion
    def test_year_attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_year_attribute'
        module_type_store = module_type_store.open_function_context('test_year_attribute', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_localization', localization)
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_function_name', 'TestDateAttribute.test_year_attribute')
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_param_names_list', [])
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDateAttribute.test_year_attribute.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDateAttribute.test_year_attribute', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_year_attribute', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_year_attribute(...)' code ##################

        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to array(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_130579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        str_130580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 12), 'str', '1999')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 28), list_130579, str_130580)
        # Adding element type (line 199)
        str_130581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 12), 'str', '2004')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 28), list_130579, str_130581)
        # Adding element type (line 199)
        str_130582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 12), 'str', '1817')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 28), list_130579, str_130582)
        # Adding element type (line 199)
        str_130583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 12), 'str', '2100')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 28), list_130579, str_130583)
        # Adding element type (line 199)
        str_130584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 12), 'str', '2013')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 28), list_130579, str_130584)
        # Adding element type (line 199)
        str_130585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 12), 'str', '1631')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 28), list_130579, str_130585)
        
        # Processing the call keyword arguments (line 199)
        str_130586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 17), 'str', 'datetime64[Y]')
        keyword_130587 = str_130586
        kwargs_130588 = {'dtype': keyword_130587}
        # Getting the type of 'np' (line 199)
        np_130577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 199)
        array_130578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 19), np_130577, 'array')
        # Calling array(args, kwargs) (line 199)
        array_call_result_130589 = invoke(stypy.reporting.localization.Localization(__file__, 199, 19), array_130578, *[list_130579], **kwargs_130588)
        
        # Assigning a type to the variable 'expected' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'expected', array_call_result_130589)
        
        # Call to assert_array_equal(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Obtaining the type of the subscript
        str_130591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 37), 'str', 'attr_year')
        # Getting the type of 'self' (line 208)
        self_130592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 208)
        data_130593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 27), self_130592, 'data')
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___130594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 27), data_130593, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_130595 = invoke(stypy.reporting.localization.Localization(__file__, 208, 27), getitem___130594, str_130591)
        
        # Getting the type of 'expected' (line 208)
        expected_130596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 51), 'expected', False)
        # Processing the call keyword arguments (line 208)
        kwargs_130597 = {}
        # Getting the type of 'assert_array_equal' (line 208)
        assert_array_equal_130590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 208)
        assert_array_equal_call_result_130598 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), assert_array_equal_130590, *[subscript_call_result_130595, expected_130596], **kwargs_130597)
        
        
        # ################# End of 'test_year_attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_year_attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_130599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130599)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_year_attribute'
        return stypy_return_type_130599


    @norecursion
    def test_month_attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_month_attribute'
        module_type_store = module_type_store.open_function_context('test_month_attribute', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_localization', localization)
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_function_name', 'TestDateAttribute.test_month_attribute')
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_param_names_list', [])
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDateAttribute.test_month_attribute.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDateAttribute.test_month_attribute', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_month_attribute', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_month_attribute(...)' code ##################

        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to array(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_130602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        str_130603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 12), 'str', '1999-01')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 28), list_130602, str_130603)
        # Adding element type (line 211)
        str_130604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 12), 'str', '2004-12')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 28), list_130602, str_130604)
        # Adding element type (line 211)
        str_130605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 12), 'str', '1817-04')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 28), list_130602, str_130605)
        # Adding element type (line 211)
        str_130606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 12), 'str', '2100-09')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 28), list_130602, str_130606)
        # Adding element type (line 211)
        str_130607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'str', '2013-11')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 28), list_130602, str_130607)
        # Adding element type (line 211)
        str_130608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 12), 'str', '1631-10')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 28), list_130602, str_130608)
        
        # Processing the call keyword arguments (line 211)
        str_130609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 17), 'str', 'datetime64[M]')
        keyword_130610 = str_130609
        kwargs_130611 = {'dtype': keyword_130610}
        # Getting the type of 'np' (line 211)
        np_130600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 211)
        array_130601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 19), np_130600, 'array')
        # Calling array(args, kwargs) (line 211)
        array_call_result_130612 = invoke(stypy.reporting.localization.Localization(__file__, 211, 19), array_130601, *[list_130602], **kwargs_130611)
        
        # Assigning a type to the variable 'expected' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'expected', array_call_result_130612)
        
        # Call to assert_array_equal(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Obtaining the type of the subscript
        str_130614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 37), 'str', 'attr_month')
        # Getting the type of 'self' (line 220)
        self_130615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 220)
        data_130616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 27), self_130615, 'data')
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___130617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 27), data_130616, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_130618 = invoke(stypy.reporting.localization.Localization(__file__, 220, 27), getitem___130617, str_130614)
        
        # Getting the type of 'expected' (line 220)
        expected_130619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 52), 'expected', False)
        # Processing the call keyword arguments (line 220)
        kwargs_130620 = {}
        # Getting the type of 'assert_array_equal' (line 220)
        assert_array_equal_130613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 220)
        assert_array_equal_call_result_130621 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), assert_array_equal_130613, *[subscript_call_result_130618, expected_130619], **kwargs_130620)
        
        
        # ################# End of 'test_month_attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_month_attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_130622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_month_attribute'
        return stypy_return_type_130622


    @norecursion
    def test_date_attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_date_attribute'
        module_type_store = module_type_store.open_function_context('test_date_attribute', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_localization', localization)
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_function_name', 'TestDateAttribute.test_date_attribute')
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_param_names_list', [])
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDateAttribute.test_date_attribute.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDateAttribute.test_date_attribute', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_date_attribute', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_date_attribute(...)' code ##################

        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to array(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_130625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        str_130626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 12), 'str', '1999-01-31')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 28), list_130625, str_130626)
        # Adding element type (line 223)
        str_130627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 12), 'str', '2004-12-01')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 28), list_130625, str_130627)
        # Adding element type (line 223)
        str_130628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 12), 'str', '1817-04-28')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 28), list_130625, str_130628)
        # Adding element type (line 223)
        str_130629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 12), 'str', '2100-09-10')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 28), list_130625, str_130629)
        # Adding element type (line 223)
        str_130630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 12), 'str', '2013-11-30')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 28), list_130625, str_130630)
        # Adding element type (line 223)
        str_130631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 12), 'str', '1631-10-15')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 28), list_130625, str_130631)
        
        # Processing the call keyword arguments (line 223)
        str_130632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 17), 'str', 'datetime64[D]')
        keyword_130633 = str_130632
        kwargs_130634 = {'dtype': keyword_130633}
        # Getting the type of 'np' (line 223)
        np_130623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 223)
        array_130624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 19), np_130623, 'array')
        # Calling array(args, kwargs) (line 223)
        array_call_result_130635 = invoke(stypy.reporting.localization.Localization(__file__, 223, 19), array_130624, *[list_130625], **kwargs_130634)
        
        # Assigning a type to the variable 'expected' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'expected', array_call_result_130635)
        
        # Call to assert_array_equal(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Obtaining the type of the subscript
        str_130637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 37), 'str', 'attr_date')
        # Getting the type of 'self' (line 232)
        self_130638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 232)
        data_130639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 27), self_130638, 'data')
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___130640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 27), data_130639, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_130641 = invoke(stypy.reporting.localization.Localization(__file__, 232, 27), getitem___130640, str_130637)
        
        # Getting the type of 'expected' (line 232)
        expected_130642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 51), 'expected', False)
        # Processing the call keyword arguments (line 232)
        kwargs_130643 = {}
        # Getting the type of 'assert_array_equal' (line 232)
        assert_array_equal_130636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 232)
        assert_array_equal_call_result_130644 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assert_array_equal_130636, *[subscript_call_result_130641, expected_130642], **kwargs_130643)
        
        
        # ################# End of 'test_date_attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_date_attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_130645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130645)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_date_attribute'
        return stypy_return_type_130645


    @norecursion
    def test_datetime_local_attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_datetime_local_attribute'
        module_type_store = module_type_store.open_function_context('test_datetime_local_attribute', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_localization', localization)
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_function_name', 'TestDateAttribute.test_datetime_local_attribute')
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_param_names_list', [])
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDateAttribute.test_datetime_local_attribute.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDateAttribute.test_datetime_local_attribute', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_datetime_local_attribute', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_datetime_local_attribute(...)' code ##################

        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to array(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_130648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        
        # Call to datetime(...): (line 236)
        # Processing the call keyword arguments (line 236)
        int_130651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 35), 'int')
        keyword_130652 = int_130651
        int_130653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 47), 'int')
        keyword_130654 = int_130653
        int_130655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 54), 'int')
        keyword_130656 = int_130655
        int_130657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 63), 'int')
        keyword_130658 = int_130657
        int_130659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 73), 'int')
        keyword_130660 = int_130659
        kwargs_130661 = {'month': keyword_130654, 'minute': keyword_130660, 'day': keyword_130656, 'hour': keyword_130658, 'year': keyword_130652}
        # Getting the type of 'datetime' (line 236)
        datetime_130649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 236)
        datetime_130650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), datetime_130649, 'datetime')
        # Calling datetime(args, kwargs) (line 236)
        datetime_call_result_130662 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), datetime_130650, *[], **kwargs_130661)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 28), list_130648, datetime_call_result_130662)
        # Adding element type (line 235)
        
        # Call to datetime(...): (line 237)
        # Processing the call keyword arguments (line 237)
        int_130665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 35), 'int')
        keyword_130666 = int_130665
        int_130667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 47), 'int')
        keyword_130668 = int_130667
        int_130669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 55), 'int')
        keyword_130670 = int_130669
        int_130671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 63), 'int')
        keyword_130672 = int_130671
        int_130673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 74), 'int')
        keyword_130674 = int_130673
        kwargs_130675 = {'month': keyword_130668, 'minute': keyword_130674, 'day': keyword_130670, 'hour': keyword_130672, 'year': keyword_130666}
        # Getting the type of 'datetime' (line 237)
        datetime_130663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 237)
        datetime_130664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), datetime_130663, 'datetime')
        # Calling datetime(args, kwargs) (line 237)
        datetime_call_result_130676 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), datetime_130664, *[], **kwargs_130675)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 28), list_130648, datetime_call_result_130676)
        # Adding element type (line 235)
        
        # Call to datetime(...): (line 238)
        # Processing the call keyword arguments (line 238)
        int_130679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 35), 'int')
        keyword_130680 = int_130679
        int_130681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 47), 'int')
        keyword_130682 = int_130681
        int_130683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 54), 'int')
        keyword_130684 = int_130683
        int_130685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 63), 'int')
        keyword_130686 = int_130685
        int_130687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 74), 'int')
        keyword_130688 = int_130687
        kwargs_130689 = {'month': keyword_130682, 'minute': keyword_130688, 'day': keyword_130684, 'hour': keyword_130686, 'year': keyword_130680}
        # Getting the type of 'datetime' (line 238)
        datetime_130677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 238)
        datetime_130678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), datetime_130677, 'datetime')
        # Calling datetime(args, kwargs) (line 238)
        datetime_call_result_130690 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), datetime_130678, *[], **kwargs_130689)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 28), list_130648, datetime_call_result_130690)
        # Adding element type (line 235)
        
        # Call to datetime(...): (line 239)
        # Processing the call keyword arguments (line 239)
        int_130693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 35), 'int')
        keyword_130694 = int_130693
        int_130695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 47), 'int')
        keyword_130696 = int_130695
        int_130697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 54), 'int')
        keyword_130698 = int_130697
        int_130699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 63), 'int')
        keyword_130700 = int_130699
        int_130701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 74), 'int')
        keyword_130702 = int_130701
        kwargs_130703 = {'month': keyword_130696, 'minute': keyword_130702, 'day': keyword_130698, 'hour': keyword_130700, 'year': keyword_130694}
        # Getting the type of 'datetime' (line 239)
        datetime_130691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 239)
        datetime_130692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), datetime_130691, 'datetime')
        # Calling datetime(args, kwargs) (line 239)
        datetime_call_result_130704 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), datetime_130692, *[], **kwargs_130703)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 28), list_130648, datetime_call_result_130704)
        # Adding element type (line 235)
        
        # Call to datetime(...): (line 240)
        # Processing the call keyword arguments (line 240)
        int_130707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 35), 'int')
        keyword_130708 = int_130707
        int_130709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 47), 'int')
        keyword_130710 = int_130709
        int_130711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 55), 'int')
        keyword_130712 = int_130711
        int_130713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 64), 'int')
        keyword_130714 = int_130713
        int_130715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 74), 'int')
        keyword_130716 = int_130715
        kwargs_130717 = {'month': keyword_130710, 'minute': keyword_130716, 'day': keyword_130712, 'hour': keyword_130714, 'year': keyword_130708}
        # Getting the type of 'datetime' (line 240)
        datetime_130705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 240)
        datetime_130706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), datetime_130705, 'datetime')
        # Calling datetime(args, kwargs) (line 240)
        datetime_call_result_130718 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), datetime_130706, *[], **kwargs_130717)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 28), list_130648, datetime_call_result_130718)
        # Adding element type (line 235)
        
        # Call to datetime(...): (line 241)
        # Processing the call keyword arguments (line 241)
        int_130721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 35), 'int')
        keyword_130722 = int_130721
        int_130723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 47), 'int')
        keyword_130724 = int_130723
        int_130725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 55), 'int')
        keyword_130726 = int_130725
        int_130727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 64), 'int')
        keyword_130728 = int_130727
        int_130729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 75), 'int')
        keyword_130730 = int_130729
        kwargs_130731 = {'month': keyword_130724, 'minute': keyword_130730, 'day': keyword_130726, 'hour': keyword_130728, 'year': keyword_130722}
        # Getting the type of 'datetime' (line 241)
        datetime_130719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 241)
        datetime_130720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), datetime_130719, 'datetime')
        # Calling datetime(args, kwargs) (line 241)
        datetime_call_result_130732 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), datetime_130720, *[], **kwargs_130731)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 28), list_130648, datetime_call_result_130732)
        
        # Processing the call keyword arguments (line 235)
        str_130733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 17), 'str', 'datetime64[m]')
        keyword_130734 = str_130733
        kwargs_130735 = {'dtype': keyword_130734}
        # Getting the type of 'np' (line 235)
        np_130646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 235)
        array_130647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 19), np_130646, 'array')
        # Calling array(args, kwargs) (line 235)
        array_call_result_130736 = invoke(stypy.reporting.localization.Localization(__file__, 235, 19), array_130647, *[list_130648], **kwargs_130735)
        
        # Assigning a type to the variable 'expected' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'expected', array_call_result_130736)
        
        # Call to assert_array_equal(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Obtaining the type of the subscript
        str_130738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 37), 'str', 'attr_datetime_local')
        # Getting the type of 'self' (line 244)
        self_130739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 244)
        data_130740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 27), self_130739, 'data')
        # Obtaining the member '__getitem__' of a type (line 244)
        getitem___130741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 27), data_130740, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 244)
        subscript_call_result_130742 = invoke(stypy.reporting.localization.Localization(__file__, 244, 27), getitem___130741, str_130738)
        
        # Getting the type of 'expected' (line 244)
        expected_130743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 61), 'expected', False)
        # Processing the call keyword arguments (line 244)
        kwargs_130744 = {}
        # Getting the type of 'assert_array_equal' (line 244)
        assert_array_equal_130737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 244)
        assert_array_equal_call_result_130745 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), assert_array_equal_130737, *[subscript_call_result_130742, expected_130743], **kwargs_130744)
        
        
        # ################# End of 'test_datetime_local_attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_datetime_local_attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_130746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130746)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_datetime_local_attribute'
        return stypy_return_type_130746


    @norecursion
    def test_datetime_missing(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_datetime_missing'
        module_type_store = module_type_store.open_function_context('test_datetime_missing', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_localization', localization)
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_function_name', 'TestDateAttribute.test_datetime_missing')
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_param_names_list', [])
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDateAttribute.test_datetime_missing.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDateAttribute.test_datetime_missing', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_datetime_missing', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_datetime_missing(...)' code ##################

        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to array(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_130749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        str_130750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 12), 'str', 'nat')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 28), list_130749, str_130750)
        # Adding element type (line 247)
        str_130751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'str', '2004-12-01T23:59')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 28), list_130749, str_130751)
        # Adding element type (line 247)
        str_130752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 12), 'str', 'nat')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 28), list_130749, str_130752)
        # Adding element type (line 247)
        str_130753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 12), 'str', 'nat')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 28), list_130749, str_130753)
        # Adding element type (line 247)
        str_130754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 12), 'str', '2013-11-30T04:55')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 28), list_130749, str_130754)
        # Adding element type (line 247)
        str_130755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 12), 'str', '1631-10-15T20:04')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 28), list_130749, str_130755)
        
        # Processing the call keyword arguments (line 247)
        str_130756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 17), 'str', 'datetime64[m]')
        keyword_130757 = str_130756
        kwargs_130758 = {'dtype': keyword_130757}
        # Getting the type of 'np' (line 247)
        np_130747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 247)
        array_130748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 19), np_130747, 'array')
        # Calling array(args, kwargs) (line 247)
        array_call_result_130759 = invoke(stypy.reporting.localization.Localization(__file__, 247, 19), array_130748, *[list_130749], **kwargs_130758)
        
        # Assigning a type to the variable 'expected' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'expected', array_call_result_130759)
        
        # Call to assert_array_equal(...): (line 256)
        # Processing the call arguments (line 256)
        
        # Obtaining the type of the subscript
        str_130761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 37), 'str', 'attr_datetime_missing')
        # Getting the type of 'self' (line 256)
        self_130762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 256)
        data_130763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 27), self_130762, 'data')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___130764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 27), data_130763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_130765 = invoke(stypy.reporting.localization.Localization(__file__, 256, 27), getitem___130764, str_130761)
        
        # Getting the type of 'expected' (line 256)
        expected_130766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 63), 'expected', False)
        # Processing the call keyword arguments (line 256)
        kwargs_130767 = {}
        # Getting the type of 'assert_array_equal' (line 256)
        assert_array_equal_130760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 256)
        assert_array_equal_call_result_130768 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), assert_array_equal_130760, *[subscript_call_result_130765, expected_130766], **kwargs_130767)
        
        
        # ################# End of 'test_datetime_missing(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_datetime_missing' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_130769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130769)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_datetime_missing'
        return stypy_return_type_130769


    @norecursion
    def test_datetime_timezone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_datetime_timezone'
        module_type_store = module_type_store.open_function_context('test_datetime_timezone', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_localization', localization)
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_function_name', 'TestDateAttribute.test_datetime_timezone')
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_param_names_list', [])
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDateAttribute.test_datetime_timezone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDateAttribute.test_datetime_timezone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_datetime_timezone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_datetime_timezone(...)' code ##################

        
        # Call to assert_raises(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'ValueError' (line 259)
        ValueError_130771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 22), 'ValueError', False)
        # Getting the type of 'loadarff' (line 259)
        loadarff_130772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'loadarff', False)
        # Getting the type of 'test8' (line 259)
        test8_130773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 44), 'test8', False)
        # Processing the call keyword arguments (line 259)
        kwargs_130774 = {}
        # Getting the type of 'assert_raises' (line 259)
        assert_raises_130770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 259)
        assert_raises_call_result_130775 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), assert_raises_130770, *[ValueError_130771, loadarff_130772, test8_130773], **kwargs_130774)
        
        
        # ################# End of 'test_datetime_timezone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_datetime_timezone' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_130776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130776)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_datetime_timezone'
        return stypy_return_type_130776


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 194, 0, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDateAttribute.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDateAttribute' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'TestDateAttribute', TestDateAttribute)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_equal, assert_array_equal
5: 
6: from scipy.stats import rankdata, tiecorrect
7: 
8: 
9: class TestTieCorrect(object):
10: 
11:     def test_empty(self):
12:         '''An empty array requires no correction, should return 1.0.'''
13:         ranks = np.array([], dtype=np.float64)
14:         c = tiecorrect(ranks)
15:         assert_equal(c, 1.0)
16: 
17:     def test_one(self):
18:         '''A single element requires no correction, should return 1.0.'''
19:         ranks = np.array([1.0], dtype=np.float64)
20:         c = tiecorrect(ranks)
21:         assert_equal(c, 1.0)
22: 
23:     def test_no_correction(self):
24:         '''Arrays with no ties require no correction.'''
25:         ranks = np.arange(2.0)
26:         c = tiecorrect(ranks)
27:         assert_equal(c, 1.0)
28:         ranks = np.arange(3.0)
29:         c = tiecorrect(ranks)
30:         assert_equal(c, 1.0)
31: 
32:     def test_basic(self):
33:         '''Check a few basic examples of the tie correction factor.'''
34:         # One tie of two elements
35:         ranks = np.array([1.0, 2.5, 2.5])
36:         c = tiecorrect(ranks)
37:         T = 2.0
38:         N = ranks.size
39:         expected = 1.0 - (T**3 - T) / (N**3 - N)
40:         assert_equal(c, expected)
41: 
42:         # One tie of two elements (same as above, but tie is not at the end)
43:         ranks = np.array([1.5, 1.5, 3.0])
44:         c = tiecorrect(ranks)
45:         T = 2.0
46:         N = ranks.size
47:         expected = 1.0 - (T**3 - T) / (N**3 - N)
48:         assert_equal(c, expected)
49: 
50:         # One tie of three elements
51:         ranks = np.array([1.0, 3.0, 3.0, 3.0])
52:         c = tiecorrect(ranks)
53:         T = 3.0
54:         N = ranks.size
55:         expected = 1.0 - (T**3 - T) / (N**3 - N)
56:         assert_equal(c, expected)
57: 
58:         # Two ties, lengths 2 and 3.
59:         ranks = np.array([1.5, 1.5, 4.0, 4.0, 4.0])
60:         c = tiecorrect(ranks)
61:         T1 = 2.0
62:         T2 = 3.0
63:         N = ranks.size
64:         expected = 1.0 - ((T1**3 - T1) + (T2**3 - T2)) / (N**3 - N)
65:         assert_equal(c, expected)
66: 
67:     def test_overflow(self):
68:         ntie, k = 2000, 5
69:         a = np.repeat(np.arange(k), ntie)
70:         n = a.size  # ntie * k
71:         out = tiecorrect(rankdata(a))
72:         assert_equal(out, 1.0 - k * (ntie**3 - ntie) / float(n**3 - n))
73: 
74: 
75: class TestRankData(object):
76: 
77:     def test_empty(self):
78:         '''stats.rankdata([]) should return an empty array.'''
79:         a = np.array([], dtype=int)
80:         r = rankdata(a)
81:         assert_array_equal(r, np.array([], dtype=np.float64))
82:         r = rankdata([])
83:         assert_array_equal(r, np.array([], dtype=np.float64))
84: 
85:     def test_one(self):
86:         '''Check stats.rankdata with an array of length 1.'''
87:         data = [100]
88:         a = np.array(data, dtype=int)
89:         r = rankdata(a)
90:         assert_array_equal(r, np.array([1.0], dtype=np.float64))
91:         r = rankdata(data)
92:         assert_array_equal(r, np.array([1.0], dtype=np.float64))
93: 
94:     def test_basic(self):
95:         '''Basic tests of stats.rankdata.'''
96:         data = [100, 10, 50]
97:         expected = np.array([3.0, 1.0, 2.0], dtype=np.float64)
98:         a = np.array(data, dtype=int)
99:         r = rankdata(a)
100:         assert_array_equal(r, expected)
101:         r = rankdata(data)
102:         assert_array_equal(r, expected)
103: 
104:         data = [40, 10, 30, 10, 50]
105:         expected = np.array([4.0, 1.5, 3.0, 1.5, 5.0], dtype=np.float64)
106:         a = np.array(data, dtype=int)
107:         r = rankdata(a)
108:         assert_array_equal(r, expected)
109:         r = rankdata(data)
110:         assert_array_equal(r, expected)
111: 
112:         data = [20, 20, 20, 10, 10, 10]
113:         expected = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0], dtype=np.float64)
114:         a = np.array(data, dtype=int)
115:         r = rankdata(a)
116:         assert_array_equal(r, expected)
117:         r = rankdata(data)
118:         assert_array_equal(r, expected)
119:         # The docstring states explicitly that the argument is flattened.
120:         a2d = a.reshape(2, 3)
121:         r = rankdata(a2d)
122:         assert_array_equal(r, expected)
123: 
124:     def test_rankdata_object_string(self):
125:         min_rank = lambda a: [1 + sum(i < j for i in a) for j in a]
126:         max_rank = lambda a: [sum(i <= j for i in a) for j in a]
127:         ordinal_rank = lambda a: min_rank([(x, i) for i, x in enumerate(a)])
128: 
129:         def average_rank(a):
130:             return [(i + j) / 2.0 for i, j in zip(min_rank(a), max_rank(a))]
131: 
132:         def dense_rank(a):
133:             b = np.unique(a)
134:             return [1 + sum(i < j for i in b) for j in a]
135: 
136:         rankf = dict(min=min_rank, max=max_rank, ordinal=ordinal_rank,
137:                      average=average_rank, dense=dense_rank)
138: 
139:         def check_ranks(a):
140:             for method in 'min', 'max', 'dense', 'ordinal', 'average':
141:                 out = rankdata(a, method=method)
142:                 assert_array_equal(out, rankf[method](a))
143: 
144:         val = ['foo', 'bar', 'qux', 'xyz', 'abc', 'efg', 'ace', 'qwe', 'qaz']
145:         check_ranks(np.random.choice(val, 200))
146:         check_ranks(np.random.choice(val, 200).astype('object'))
147: 
148:         val = np.array([0, 1, 2, 2.718, 3, 3.141], dtype='object')
149:         check_ranks(np.random.choice(val, 200).astype('object'))
150: 
151:     def test_large_int(self):
152:         data = np.array([2**60, 2**60+1], dtype=np.uint64)
153:         r = rankdata(data)
154:         assert_array_equal(r, [1.0, 2.0])
155: 
156:         data = np.array([2**60, 2**60+1], dtype=np.int64)
157:         r = rankdata(data)
158:         assert_array_equal(r, [1.0, 2.0])
159: 
160:         data = np.array([2**60, -2**60+1], dtype=np.int64)
161:         r = rankdata(data)
162:         assert_array_equal(r, [2.0, 1.0])
163: 
164:     def test_big_tie(self):
165:         for n in [10000, 100000, 1000000]:
166:             data = np.ones(n, dtype=int)
167:             r = rankdata(data)
168:             expected_rank = 0.5 * (n + 1)
169:             assert_array_equal(r, expected_rank * data,
170:                                "test failed with n=%d" % n)
171: 
172: 
173: _cases = (
174:     # values, method, expected
175:     ([], 'average', []),
176:     ([], 'min', []),
177:     ([], 'max', []),
178:     ([], 'dense', []),
179:     ([], 'ordinal', []),
180:     #
181:     ([100], 'average', [1.0]),
182:     ([100], 'min', [1.0]),
183:     ([100], 'max', [1.0]),
184:     ([100], 'dense', [1.0]),
185:     ([100], 'ordinal', [1.0]),
186:     #
187:     ([100, 100, 100], 'average', [2.0, 2.0, 2.0]),
188:     ([100, 100, 100], 'min', [1.0, 1.0, 1.0]),
189:     ([100, 100, 100], 'max', [3.0, 3.0, 3.0]),
190:     ([100, 100, 100], 'dense', [1.0, 1.0, 1.0]),
191:     ([100, 100, 100], 'ordinal', [1.0, 2.0, 3.0]),
192:     #
193:     ([100, 300, 200], 'average', [1.0, 3.0, 2.0]),
194:     ([100, 300, 200], 'min', [1.0, 3.0, 2.0]),
195:     ([100, 300, 200], 'max', [1.0, 3.0, 2.0]),
196:     ([100, 300, 200], 'dense', [1.0, 3.0, 2.0]),
197:     ([100, 300, 200], 'ordinal', [1.0, 3.0, 2.0]),
198:     #
199:     ([100, 200, 300, 200], 'average', [1.0, 2.5, 4.0, 2.5]),
200:     ([100, 200, 300, 200], 'min', [1.0, 2.0, 4.0, 2.0]),
201:     ([100, 200, 300, 200], 'max', [1.0, 3.0, 4.0, 3.0]),
202:     ([100, 200, 300, 200], 'dense', [1.0, 2.0, 3.0, 2.0]),
203:     ([100, 200, 300, 200], 'ordinal', [1.0, 2.0, 4.0, 3.0]),
204:     #
205:     ([100, 200, 300, 200, 100], 'average', [1.5, 3.5, 5.0, 3.5, 1.5]),
206:     ([100, 200, 300, 200, 100], 'min', [1.0, 3.0, 5.0, 3.0, 1.0]),
207:     ([100, 200, 300, 200, 100], 'max', [2.0, 4.0, 5.0, 4.0, 2.0]),
208:     ([100, 200, 300, 200, 100], 'dense', [1.0, 2.0, 3.0, 2.0, 1.0]),
209:     ([100, 200, 300, 200, 100], 'ordinal', [1.0, 3.0, 5.0, 4.0, 2.0]),
210:     #
211:     ([10] * 30, 'ordinal', np.arange(1.0, 31.0)),
212: )
213: 
214: 
215: def test_cases():
216:     for values, method, expected in _cases:
217:         r = rankdata(values, method=method)
218:         assert_array_equal(r, expected)
219: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_678925 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_678925) is not StypyTypeError):

    if (import_678925 != 'pyd_module'):
        __import__(import_678925)
        sys_modules_678926 = sys.modules[import_678925]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_678926.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_678925)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_array_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_678927 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_678927) is not StypyTypeError):

    if (import_678927 != 'pyd_module'):
        __import__(import_678927)
        sys_modules_678928 = sys.modules[import_678927]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_678928.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_678928, sys_modules_678928.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal'], [assert_equal, assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_678927)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.stats import rankdata, tiecorrect' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_678929 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.stats')

if (type(import_678929) is not StypyTypeError):

    if (import_678929 != 'pyd_module'):
        __import__(import_678929)
        sys_modules_678930 = sys.modules[import_678929]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.stats', sys_modules_678930.module_type_store, module_type_store, ['rankdata', 'tiecorrect'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_678930, sys_modules_678930.module_type_store, module_type_store)
    else:
        from scipy.stats import rankdata, tiecorrect

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.stats', None, module_type_store, ['rankdata', 'tiecorrect'], [rankdata, tiecorrect])

else:
    # Assigning a type to the variable 'scipy.stats' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.stats', import_678929)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

# Declaration of the 'TestTieCorrect' class

class TestTieCorrect(object, ):

    @norecursion
    def test_empty(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty'
        module_type_store = module_type_store.open_function_context('test_empty', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_localization', localization)
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_function_name', 'TestTieCorrect.test_empty')
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_param_names_list', [])
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTieCorrect.test_empty.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTieCorrect.test_empty', [], None, None, defaults, varargs, kwargs)

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

        str_678931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'str', 'An empty array requires no correction, should return 1.0.')
        
        # Assigning a Call to a Name (line 13):
        
        # Assigning a Call to a Name (line 13):
        
        # Call to array(...): (line 13)
        # Processing the call arguments (line 13)
        
        # Obtaining an instance of the builtin type 'list' (line 13)
        list_678934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 13)
        
        # Processing the call keyword arguments (line 13)
        # Getting the type of 'np' (line 13)
        np_678935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 35), 'np', False)
        # Obtaining the member 'float64' of a type (line 13)
        float64_678936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 35), np_678935, 'float64')
        keyword_678937 = float64_678936
        kwargs_678938 = {'dtype': keyword_678937}
        # Getting the type of 'np' (line 13)
        np_678932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 13)
        array_678933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), np_678932, 'array')
        # Calling array(args, kwargs) (line 13)
        array_call_result_678939 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), array_678933, *[list_678934], **kwargs_678938)
        
        # Assigning a type to the variable 'ranks' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'ranks', array_call_result_678939)
        
        # Assigning a Call to a Name (line 14):
        
        # Assigning a Call to a Name (line 14):
        
        # Call to tiecorrect(...): (line 14)
        # Processing the call arguments (line 14)
        # Getting the type of 'ranks' (line 14)
        ranks_678941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'ranks', False)
        # Processing the call keyword arguments (line 14)
        kwargs_678942 = {}
        # Getting the type of 'tiecorrect' (line 14)
        tiecorrect_678940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 14)
        tiecorrect_call_result_678943 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), tiecorrect_678940, *[ranks_678941], **kwargs_678942)
        
        # Assigning a type to the variable 'c' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'c', tiecorrect_call_result_678943)
        
        # Call to assert_equal(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'c' (line 15)
        c_678945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'c', False)
        float_678946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'float')
        # Processing the call keyword arguments (line 15)
        kwargs_678947 = {}
        # Getting the type of 'assert_equal' (line 15)
        assert_equal_678944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 15)
        assert_equal_call_result_678948 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), assert_equal_678944, *[c_678945, float_678946], **kwargs_678947)
        
        
        # ################# End of 'test_empty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_678949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_678949)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty'
        return stypy_return_type_678949


    @norecursion
    def test_one(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_one'
        module_type_store = module_type_store.open_function_context('test_one', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_localization', localization)
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_function_name', 'TestTieCorrect.test_one')
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_param_names_list', [])
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTieCorrect.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTieCorrect.test_one', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_one', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_one(...)' code ##################

        str_678950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'A single element requires no correction, should return 1.0.')
        
        # Assigning a Call to a Name (line 19):
        
        # Assigning a Call to a Name (line 19):
        
        # Call to array(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_678953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        float_678954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_678953, float_678954)
        
        # Processing the call keyword arguments (line 19)
        # Getting the type of 'np' (line 19)
        np_678955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'np', False)
        # Obtaining the member 'float64' of a type (line 19)
        float64_678956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 38), np_678955, 'float64')
        keyword_678957 = float64_678956
        kwargs_678958 = {'dtype': keyword_678957}
        # Getting the type of 'np' (line 19)
        np_678951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 19)
        array_678952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 16), np_678951, 'array')
        # Calling array(args, kwargs) (line 19)
        array_call_result_678959 = invoke(stypy.reporting.localization.Localization(__file__, 19, 16), array_678952, *[list_678953], **kwargs_678958)
        
        # Assigning a type to the variable 'ranks' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'ranks', array_call_result_678959)
        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to tiecorrect(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'ranks' (line 20)
        ranks_678961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'ranks', False)
        # Processing the call keyword arguments (line 20)
        kwargs_678962 = {}
        # Getting the type of 'tiecorrect' (line 20)
        tiecorrect_678960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 20)
        tiecorrect_call_result_678963 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), tiecorrect_678960, *[ranks_678961], **kwargs_678962)
        
        # Assigning a type to the variable 'c' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'c', tiecorrect_call_result_678963)
        
        # Call to assert_equal(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'c' (line 21)
        c_678965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'c', False)
        float_678966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'float')
        # Processing the call keyword arguments (line 21)
        kwargs_678967 = {}
        # Getting the type of 'assert_equal' (line 21)
        assert_equal_678964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 21)
        assert_equal_call_result_678968 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert_equal_678964, *[c_678965, float_678966], **kwargs_678967)
        
        
        # ################# End of 'test_one(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_one' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_678969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_678969)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_one'
        return stypy_return_type_678969


    @norecursion
    def test_no_correction(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_no_correction'
        module_type_store = module_type_store.open_function_context('test_no_correction', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_localization', localization)
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_function_name', 'TestTieCorrect.test_no_correction')
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_param_names_list', [])
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTieCorrect.test_no_correction.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTieCorrect.test_no_correction', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_no_correction', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_no_correction(...)' code ##################

        str_678970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'str', 'Arrays with no ties require no correction.')
        
        # Assigning a Call to a Name (line 25):
        
        # Assigning a Call to a Name (line 25):
        
        # Call to arange(...): (line 25)
        # Processing the call arguments (line 25)
        float_678973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'float')
        # Processing the call keyword arguments (line 25)
        kwargs_678974 = {}
        # Getting the type of 'np' (line 25)
        np_678971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 25)
        arange_678972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), np_678971, 'arange')
        # Calling arange(args, kwargs) (line 25)
        arange_call_result_678975 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), arange_678972, *[float_678973], **kwargs_678974)
        
        # Assigning a type to the variable 'ranks' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'ranks', arange_call_result_678975)
        
        # Assigning a Call to a Name (line 26):
        
        # Assigning a Call to a Name (line 26):
        
        # Call to tiecorrect(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'ranks' (line 26)
        ranks_678977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'ranks', False)
        # Processing the call keyword arguments (line 26)
        kwargs_678978 = {}
        # Getting the type of 'tiecorrect' (line 26)
        tiecorrect_678976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 26)
        tiecorrect_call_result_678979 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), tiecorrect_678976, *[ranks_678977], **kwargs_678978)
        
        # Assigning a type to the variable 'c' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'c', tiecorrect_call_result_678979)
        
        # Call to assert_equal(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'c' (line 27)
        c_678981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'c', False)
        float_678982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'float')
        # Processing the call keyword arguments (line 27)
        kwargs_678983 = {}
        # Getting the type of 'assert_equal' (line 27)
        assert_equal_678980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 27)
        assert_equal_call_result_678984 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_equal_678980, *[c_678981, float_678982], **kwargs_678983)
        
        
        # Assigning a Call to a Name (line 28):
        
        # Assigning a Call to a Name (line 28):
        
        # Call to arange(...): (line 28)
        # Processing the call arguments (line 28)
        float_678987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 26), 'float')
        # Processing the call keyword arguments (line 28)
        kwargs_678988 = {}
        # Getting the type of 'np' (line 28)
        np_678985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'np', False)
        # Obtaining the member 'arange' of a type (line 28)
        arange_678986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), np_678985, 'arange')
        # Calling arange(args, kwargs) (line 28)
        arange_call_result_678989 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), arange_678986, *[float_678987], **kwargs_678988)
        
        # Assigning a type to the variable 'ranks' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'ranks', arange_call_result_678989)
        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Call to tiecorrect(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'ranks' (line 29)
        ranks_678991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'ranks', False)
        # Processing the call keyword arguments (line 29)
        kwargs_678992 = {}
        # Getting the type of 'tiecorrect' (line 29)
        tiecorrect_678990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 29)
        tiecorrect_call_result_678993 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), tiecorrect_678990, *[ranks_678991], **kwargs_678992)
        
        # Assigning a type to the variable 'c' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'c', tiecorrect_call_result_678993)
        
        # Call to assert_equal(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'c' (line 30)
        c_678995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'c', False)
        float_678996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'float')
        # Processing the call keyword arguments (line 30)
        kwargs_678997 = {}
        # Getting the type of 'assert_equal' (line 30)
        assert_equal_678994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 30)
        assert_equal_call_result_678998 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_equal_678994, *[c_678995, float_678996], **kwargs_678997)
        
        
        # ################# End of 'test_no_correction(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_no_correction' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_678999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_678999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_no_correction'
        return stypy_return_type_678999


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_function_name', 'TestTieCorrect.test_basic')
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTieCorrect.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTieCorrect.test_basic', [], None, None, defaults, varargs, kwargs)

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

        str_679000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'str', 'Check a few basic examples of the tie correction factor.')
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to array(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_679003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        float_679004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_679003, float_679004)
        # Adding element type (line 35)
        float_679005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_679003, float_679005)
        # Adding element type (line 35)
        float_679006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 25), list_679003, float_679006)
        
        # Processing the call keyword arguments (line 35)
        kwargs_679007 = {}
        # Getting the type of 'np' (line 35)
        np_679001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 35)
        array_679002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), np_679001, 'array')
        # Calling array(args, kwargs) (line 35)
        array_call_result_679008 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), array_679002, *[list_679003], **kwargs_679007)
        
        # Assigning a type to the variable 'ranks' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'ranks', array_call_result_679008)
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to tiecorrect(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'ranks' (line 36)
        ranks_679010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'ranks', False)
        # Processing the call keyword arguments (line 36)
        kwargs_679011 = {}
        # Getting the type of 'tiecorrect' (line 36)
        tiecorrect_679009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 36)
        tiecorrect_call_result_679012 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), tiecorrect_679009, *[ranks_679010], **kwargs_679011)
        
        # Assigning a type to the variable 'c' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'c', tiecorrect_call_result_679012)
        
        # Assigning a Num to a Name (line 37):
        
        # Assigning a Num to a Name (line 37):
        float_679013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 12), 'float')
        # Assigning a type to the variable 'T' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'T', float_679013)
        
        # Assigning a Attribute to a Name (line 38):
        
        # Assigning a Attribute to a Name (line 38):
        # Getting the type of 'ranks' (line 38)
        ranks_679014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'ranks')
        # Obtaining the member 'size' of a type (line 38)
        size_679015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), ranks_679014, 'size')
        # Assigning a type to the variable 'N' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'N', size_679015)
        
        # Assigning a BinOp to a Name (line 39):
        
        # Assigning a BinOp to a Name (line 39):
        float_679016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'float')
        # Getting the type of 'T' (line 39)
        T_679017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'T')
        int_679018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'int')
        # Applying the binary operator '**' (line 39)
        result_pow_679019 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 26), '**', T_679017, int_679018)
        
        # Getting the type of 'T' (line 39)
        T_679020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'T')
        # Applying the binary operator '-' (line 39)
        result_sub_679021 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 26), '-', result_pow_679019, T_679020)
        
        # Getting the type of 'N' (line 39)
        N_679022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'N')
        int_679023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 42), 'int')
        # Applying the binary operator '**' (line 39)
        result_pow_679024 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 39), '**', N_679022, int_679023)
        
        # Getting the type of 'N' (line 39)
        N_679025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 46), 'N')
        # Applying the binary operator '-' (line 39)
        result_sub_679026 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 39), '-', result_pow_679024, N_679025)
        
        # Applying the binary operator 'div' (line 39)
        result_div_679027 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 25), 'div', result_sub_679021, result_sub_679026)
        
        # Applying the binary operator '-' (line 39)
        result_sub_679028 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 19), '-', float_679016, result_div_679027)
        
        # Assigning a type to the variable 'expected' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'expected', result_sub_679028)
        
        # Call to assert_equal(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'c' (line 40)
        c_679030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'c', False)
        # Getting the type of 'expected' (line 40)
        expected_679031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'expected', False)
        # Processing the call keyword arguments (line 40)
        kwargs_679032 = {}
        # Getting the type of 'assert_equal' (line 40)
        assert_equal_679029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 40)
        assert_equal_call_result_679033 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assert_equal_679029, *[c_679030, expected_679031], **kwargs_679032)
        
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to array(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_679036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        float_679037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 25), list_679036, float_679037)
        # Adding element type (line 43)
        float_679038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 25), list_679036, float_679038)
        # Adding element type (line 43)
        float_679039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 25), list_679036, float_679039)
        
        # Processing the call keyword arguments (line 43)
        kwargs_679040 = {}
        # Getting the type of 'np' (line 43)
        np_679034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 43)
        array_679035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 16), np_679034, 'array')
        # Calling array(args, kwargs) (line 43)
        array_call_result_679041 = invoke(stypy.reporting.localization.Localization(__file__, 43, 16), array_679035, *[list_679036], **kwargs_679040)
        
        # Assigning a type to the variable 'ranks' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'ranks', array_call_result_679041)
        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to tiecorrect(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'ranks' (line 44)
        ranks_679043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'ranks', False)
        # Processing the call keyword arguments (line 44)
        kwargs_679044 = {}
        # Getting the type of 'tiecorrect' (line 44)
        tiecorrect_679042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 44)
        tiecorrect_call_result_679045 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), tiecorrect_679042, *[ranks_679043], **kwargs_679044)
        
        # Assigning a type to the variable 'c' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'c', tiecorrect_call_result_679045)
        
        # Assigning a Num to a Name (line 45):
        
        # Assigning a Num to a Name (line 45):
        float_679046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'float')
        # Assigning a type to the variable 'T' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'T', float_679046)
        
        # Assigning a Attribute to a Name (line 46):
        
        # Assigning a Attribute to a Name (line 46):
        # Getting the type of 'ranks' (line 46)
        ranks_679047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'ranks')
        # Obtaining the member 'size' of a type (line 46)
        size_679048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), ranks_679047, 'size')
        # Assigning a type to the variable 'N' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'N', size_679048)
        
        # Assigning a BinOp to a Name (line 47):
        
        # Assigning a BinOp to a Name (line 47):
        float_679049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'float')
        # Getting the type of 'T' (line 47)
        T_679050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'T')
        int_679051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'int')
        # Applying the binary operator '**' (line 47)
        result_pow_679052 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 26), '**', T_679050, int_679051)
        
        # Getting the type of 'T' (line 47)
        T_679053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'T')
        # Applying the binary operator '-' (line 47)
        result_sub_679054 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 26), '-', result_pow_679052, T_679053)
        
        # Getting the type of 'N' (line 47)
        N_679055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 39), 'N')
        int_679056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 42), 'int')
        # Applying the binary operator '**' (line 47)
        result_pow_679057 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 39), '**', N_679055, int_679056)
        
        # Getting the type of 'N' (line 47)
        N_679058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 46), 'N')
        # Applying the binary operator '-' (line 47)
        result_sub_679059 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 39), '-', result_pow_679057, N_679058)
        
        # Applying the binary operator 'div' (line 47)
        result_div_679060 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 25), 'div', result_sub_679054, result_sub_679059)
        
        # Applying the binary operator '-' (line 47)
        result_sub_679061 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), '-', float_679049, result_div_679060)
        
        # Assigning a type to the variable 'expected' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'expected', result_sub_679061)
        
        # Call to assert_equal(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'c' (line 48)
        c_679063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'c', False)
        # Getting the type of 'expected' (line 48)
        expected_679064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'expected', False)
        # Processing the call keyword arguments (line 48)
        kwargs_679065 = {}
        # Getting the type of 'assert_equal' (line 48)
        assert_equal_679062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 48)
        assert_equal_call_result_679066 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_equal_679062, *[c_679063, expected_679064], **kwargs_679065)
        
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to array(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_679069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        float_679070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), list_679069, float_679070)
        # Adding element type (line 51)
        float_679071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), list_679069, float_679071)
        # Adding element type (line 51)
        float_679072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), list_679069, float_679072)
        # Adding element type (line 51)
        float_679073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), list_679069, float_679073)
        
        # Processing the call keyword arguments (line 51)
        kwargs_679074 = {}
        # Getting the type of 'np' (line 51)
        np_679067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 51)
        array_679068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), np_679067, 'array')
        # Calling array(args, kwargs) (line 51)
        array_call_result_679075 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), array_679068, *[list_679069], **kwargs_679074)
        
        # Assigning a type to the variable 'ranks' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'ranks', array_call_result_679075)
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to tiecorrect(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'ranks' (line 52)
        ranks_679077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'ranks', False)
        # Processing the call keyword arguments (line 52)
        kwargs_679078 = {}
        # Getting the type of 'tiecorrect' (line 52)
        tiecorrect_679076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 52)
        tiecorrect_call_result_679079 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), tiecorrect_679076, *[ranks_679077], **kwargs_679078)
        
        # Assigning a type to the variable 'c' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'c', tiecorrect_call_result_679079)
        
        # Assigning a Num to a Name (line 53):
        
        # Assigning a Num to a Name (line 53):
        float_679080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 12), 'float')
        # Assigning a type to the variable 'T' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'T', float_679080)
        
        # Assigning a Attribute to a Name (line 54):
        
        # Assigning a Attribute to a Name (line 54):
        # Getting the type of 'ranks' (line 54)
        ranks_679081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'ranks')
        # Obtaining the member 'size' of a type (line 54)
        size_679082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), ranks_679081, 'size')
        # Assigning a type to the variable 'N' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'N', size_679082)
        
        # Assigning a BinOp to a Name (line 55):
        
        # Assigning a BinOp to a Name (line 55):
        float_679083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'float')
        # Getting the type of 'T' (line 55)
        T_679084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 26), 'T')
        int_679085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'int')
        # Applying the binary operator '**' (line 55)
        result_pow_679086 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 26), '**', T_679084, int_679085)
        
        # Getting the type of 'T' (line 55)
        T_679087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'T')
        # Applying the binary operator '-' (line 55)
        result_sub_679088 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 26), '-', result_pow_679086, T_679087)
        
        # Getting the type of 'N' (line 55)
        N_679089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 39), 'N')
        int_679090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'int')
        # Applying the binary operator '**' (line 55)
        result_pow_679091 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 39), '**', N_679089, int_679090)
        
        # Getting the type of 'N' (line 55)
        N_679092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'N')
        # Applying the binary operator '-' (line 55)
        result_sub_679093 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 39), '-', result_pow_679091, N_679092)
        
        # Applying the binary operator 'div' (line 55)
        result_div_679094 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 25), 'div', result_sub_679088, result_sub_679093)
        
        # Applying the binary operator '-' (line 55)
        result_sub_679095 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 19), '-', float_679083, result_div_679094)
        
        # Assigning a type to the variable 'expected' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'expected', result_sub_679095)
        
        # Call to assert_equal(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'c' (line 56)
        c_679097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'c', False)
        # Getting the type of 'expected' (line 56)
        expected_679098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'expected', False)
        # Processing the call keyword arguments (line 56)
        kwargs_679099 = {}
        # Getting the type of 'assert_equal' (line 56)
        assert_equal_679096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 56)
        assert_equal_call_result_679100 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assert_equal_679096, *[c_679097, expected_679098], **kwargs_679099)
        
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to array(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_679103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        float_679104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 25), list_679103, float_679104)
        # Adding element type (line 59)
        float_679105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 25), list_679103, float_679105)
        # Adding element type (line 59)
        float_679106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 25), list_679103, float_679106)
        # Adding element type (line 59)
        float_679107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 25), list_679103, float_679107)
        # Adding element type (line 59)
        float_679108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 25), list_679103, float_679108)
        
        # Processing the call keyword arguments (line 59)
        kwargs_679109 = {}
        # Getting the type of 'np' (line 59)
        np_679101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 59)
        array_679102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), np_679101, 'array')
        # Calling array(args, kwargs) (line 59)
        array_call_result_679110 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), array_679102, *[list_679103], **kwargs_679109)
        
        # Assigning a type to the variable 'ranks' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'ranks', array_call_result_679110)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to tiecorrect(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'ranks' (line 60)
        ranks_679112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'ranks', False)
        # Processing the call keyword arguments (line 60)
        kwargs_679113 = {}
        # Getting the type of 'tiecorrect' (line 60)
        tiecorrect_679111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 60)
        tiecorrect_call_result_679114 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), tiecorrect_679111, *[ranks_679112], **kwargs_679113)
        
        # Assigning a type to the variable 'c' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'c', tiecorrect_call_result_679114)
        
        # Assigning a Num to a Name (line 61):
        
        # Assigning a Num to a Name (line 61):
        float_679115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 13), 'float')
        # Assigning a type to the variable 'T1' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'T1', float_679115)
        
        # Assigning a Num to a Name (line 62):
        
        # Assigning a Num to a Name (line 62):
        float_679116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 13), 'float')
        # Assigning a type to the variable 'T2' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'T2', float_679116)
        
        # Assigning a Attribute to a Name (line 63):
        
        # Assigning a Attribute to a Name (line 63):
        # Getting the type of 'ranks' (line 63)
        ranks_679117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'ranks')
        # Obtaining the member 'size' of a type (line 63)
        size_679118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), ranks_679117, 'size')
        # Assigning a type to the variable 'N' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'N', size_679118)
        
        # Assigning a BinOp to a Name (line 64):
        
        # Assigning a BinOp to a Name (line 64):
        float_679119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'float')
        # Getting the type of 'T1' (line 64)
        T1_679120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'T1')
        int_679121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'int')
        # Applying the binary operator '**' (line 64)
        result_pow_679122 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 27), '**', T1_679120, int_679121)
        
        # Getting the type of 'T1' (line 64)
        T1_679123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 'T1')
        # Applying the binary operator '-' (line 64)
        result_sub_679124 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 27), '-', result_pow_679122, T1_679123)
        
        # Getting the type of 'T2' (line 64)
        T2_679125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 42), 'T2')
        int_679126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 46), 'int')
        # Applying the binary operator '**' (line 64)
        result_pow_679127 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 42), '**', T2_679125, int_679126)
        
        # Getting the type of 'T2' (line 64)
        T2_679128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 50), 'T2')
        # Applying the binary operator '-' (line 64)
        result_sub_679129 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 42), '-', result_pow_679127, T2_679128)
        
        # Applying the binary operator '+' (line 64)
        result_add_679130 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 26), '+', result_sub_679124, result_sub_679129)
        
        # Getting the type of 'N' (line 64)
        N_679131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 58), 'N')
        int_679132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 61), 'int')
        # Applying the binary operator '**' (line 64)
        result_pow_679133 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 58), '**', N_679131, int_679132)
        
        # Getting the type of 'N' (line 64)
        N_679134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 65), 'N')
        # Applying the binary operator '-' (line 64)
        result_sub_679135 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 58), '-', result_pow_679133, N_679134)
        
        # Applying the binary operator 'div' (line 64)
        result_div_679136 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 25), 'div', result_add_679130, result_sub_679135)
        
        # Applying the binary operator '-' (line 64)
        result_sub_679137 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 19), '-', float_679119, result_div_679136)
        
        # Assigning a type to the variable 'expected' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'expected', result_sub_679137)
        
        # Call to assert_equal(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'c' (line 65)
        c_679139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'c', False)
        # Getting the type of 'expected' (line 65)
        expected_679140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'expected', False)
        # Processing the call keyword arguments (line 65)
        kwargs_679141 = {}
        # Getting the type of 'assert_equal' (line 65)
        assert_equal_679138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 65)
        assert_equal_call_result_679142 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_equal_679138, *[c_679139, expected_679140], **kwargs_679141)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_679143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679143)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_679143


    @norecursion
    def test_overflow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_overflow'
        module_type_store = module_type_store.open_function_context('test_overflow', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_localization', localization)
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_function_name', 'TestTieCorrect.test_overflow')
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_param_names_list', [])
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTieCorrect.test_overflow.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTieCorrect.test_overflow', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_overflow', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_overflow(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 68):
        
        # Assigning a Num to a Name (line 68):
        int_679144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 18), 'int')
        # Assigning a type to the variable 'tuple_assignment_678923' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_assignment_678923', int_679144)
        
        # Assigning a Num to a Name (line 68):
        int_679145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 24), 'int')
        # Assigning a type to the variable 'tuple_assignment_678924' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_assignment_678924', int_679145)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_assignment_678923' (line 68)
        tuple_assignment_678923_679146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_assignment_678923')
        # Assigning a type to the variable 'ntie' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'ntie', tuple_assignment_678923_679146)
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'tuple_assignment_678924' (line 68)
        tuple_assignment_678924_679147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_assignment_678924')
        # Assigning a type to the variable 'k' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'k', tuple_assignment_678924_679147)
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to repeat(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to arange(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'k' (line 69)
        k_679152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 32), 'k', False)
        # Processing the call keyword arguments (line 69)
        kwargs_679153 = {}
        # Getting the type of 'np' (line 69)
        np_679150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'np', False)
        # Obtaining the member 'arange' of a type (line 69)
        arange_679151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 22), np_679150, 'arange')
        # Calling arange(args, kwargs) (line 69)
        arange_call_result_679154 = invoke(stypy.reporting.localization.Localization(__file__, 69, 22), arange_679151, *[k_679152], **kwargs_679153)
        
        # Getting the type of 'ntie' (line 69)
        ntie_679155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 36), 'ntie', False)
        # Processing the call keyword arguments (line 69)
        kwargs_679156 = {}
        # Getting the type of 'np' (line 69)
        np_679148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'np', False)
        # Obtaining the member 'repeat' of a type (line 69)
        repeat_679149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), np_679148, 'repeat')
        # Calling repeat(args, kwargs) (line 69)
        repeat_call_result_679157 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), repeat_679149, *[arange_call_result_679154, ntie_679155], **kwargs_679156)
        
        # Assigning a type to the variable 'a' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'a', repeat_call_result_679157)
        
        # Assigning a Attribute to a Name (line 70):
        
        # Assigning a Attribute to a Name (line 70):
        # Getting the type of 'a' (line 70)
        a_679158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'a')
        # Obtaining the member 'size' of a type (line 70)
        size_679159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), a_679158, 'size')
        # Assigning a type to the variable 'n' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'n', size_679159)
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to tiecorrect(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to rankdata(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'a' (line 71)
        a_679162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'a', False)
        # Processing the call keyword arguments (line 71)
        kwargs_679163 = {}
        # Getting the type of 'rankdata' (line 71)
        rankdata_679161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 71)
        rankdata_call_result_679164 = invoke(stypy.reporting.localization.Localization(__file__, 71, 25), rankdata_679161, *[a_679162], **kwargs_679163)
        
        # Processing the call keyword arguments (line 71)
        kwargs_679165 = {}
        # Getting the type of 'tiecorrect' (line 71)
        tiecorrect_679160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'tiecorrect', False)
        # Calling tiecorrect(args, kwargs) (line 71)
        tiecorrect_call_result_679166 = invoke(stypy.reporting.localization.Localization(__file__, 71, 14), tiecorrect_679160, *[rankdata_call_result_679164], **kwargs_679165)
        
        # Assigning a type to the variable 'out' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'out', tiecorrect_call_result_679166)
        
        # Call to assert_equal(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'out' (line 72)
        out_679168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'out', False)
        float_679169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'float')
        # Getting the type of 'k' (line 72)
        k_679170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'k', False)
        # Getting the type of 'ntie' (line 72)
        ntie_679171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 37), 'ntie', False)
        int_679172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 43), 'int')
        # Applying the binary operator '**' (line 72)
        result_pow_679173 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 37), '**', ntie_679171, int_679172)
        
        # Getting the type of 'ntie' (line 72)
        ntie_679174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 47), 'ntie', False)
        # Applying the binary operator '-' (line 72)
        result_sub_679175 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 37), '-', result_pow_679173, ntie_679174)
        
        # Applying the binary operator '*' (line 72)
        result_mul_679176 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 32), '*', k_679170, result_sub_679175)
        
        
        # Call to float(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'n' (line 72)
        n_679178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 61), 'n', False)
        int_679179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 64), 'int')
        # Applying the binary operator '**' (line 72)
        result_pow_679180 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 61), '**', n_679178, int_679179)
        
        # Getting the type of 'n' (line 72)
        n_679181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 68), 'n', False)
        # Applying the binary operator '-' (line 72)
        result_sub_679182 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 61), '-', result_pow_679180, n_679181)
        
        # Processing the call keyword arguments (line 72)
        kwargs_679183 = {}
        # Getting the type of 'float' (line 72)
        float_679177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 55), 'float', False)
        # Calling float(args, kwargs) (line 72)
        float_call_result_679184 = invoke(stypy.reporting.localization.Localization(__file__, 72, 55), float_679177, *[result_sub_679182], **kwargs_679183)
        
        # Applying the binary operator 'div' (line 72)
        result_div_679185 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 53), 'div', result_mul_679176, float_call_result_679184)
        
        # Applying the binary operator '-' (line 72)
        result_sub_679186 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 26), '-', float_679169, result_div_679185)
        
        # Processing the call keyword arguments (line 72)
        kwargs_679187 = {}
        # Getting the type of 'assert_equal' (line 72)
        assert_equal_679167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 72)
        assert_equal_call_result_679188 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert_equal_679167, *[out_679168, result_sub_679186], **kwargs_679187)
        
        
        # ################# End of 'test_overflow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_overflow' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_679189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_overflow'
        return stypy_return_type_679189


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTieCorrect.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTieCorrect' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'TestTieCorrect', TestTieCorrect)
# Declaration of the 'TestRankData' class

class TestRankData(object, ):

    @norecursion
    def test_empty(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_empty'
        module_type_store = module_type_store.open_function_context('test_empty', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRankData.test_empty.__dict__.__setitem__('stypy_localization', localization)
        TestRankData.test_empty.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRankData.test_empty.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRankData.test_empty.__dict__.__setitem__('stypy_function_name', 'TestRankData.test_empty')
        TestRankData.test_empty.__dict__.__setitem__('stypy_param_names_list', [])
        TestRankData.test_empty.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRankData.test_empty.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRankData.test_empty.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRankData.test_empty.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRankData.test_empty.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRankData.test_empty.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRankData.test_empty', [], None, None, defaults, varargs, kwargs)

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

        str_679190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'str', 'stats.rankdata([]) should return an empty array.')
        
        # Assigning a Call to a Name (line 79):
        
        # Assigning a Call to a Name (line 79):
        
        # Call to array(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_679193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        
        # Processing the call keyword arguments (line 79)
        # Getting the type of 'int' (line 79)
        int_679194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'int', False)
        keyword_679195 = int_679194
        kwargs_679196 = {'dtype': keyword_679195}
        # Getting the type of 'np' (line 79)
        np_679191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 79)
        array_679192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), np_679191, 'array')
        # Calling array(args, kwargs) (line 79)
        array_call_result_679197 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), array_679192, *[list_679193], **kwargs_679196)
        
        # Assigning a type to the variable 'a' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'a', array_call_result_679197)
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to rankdata(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'a' (line 80)
        a_679199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'a', False)
        # Processing the call keyword arguments (line 80)
        kwargs_679200 = {}
        # Getting the type of 'rankdata' (line 80)
        rankdata_679198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 80)
        rankdata_call_result_679201 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), rankdata_679198, *[a_679199], **kwargs_679200)
        
        # Assigning a type to the variable 'r' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'r', rankdata_call_result_679201)
        
        # Call to assert_array_equal(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'r' (line 81)
        r_679203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 27), 'r', False)
        
        # Call to array(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_679206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        
        # Processing the call keyword arguments (line 81)
        # Getting the type of 'np' (line 81)
        np_679207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 49), 'np', False)
        # Obtaining the member 'float64' of a type (line 81)
        float64_679208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 49), np_679207, 'float64')
        keyword_679209 = float64_679208
        kwargs_679210 = {'dtype': keyword_679209}
        # Getting the type of 'np' (line 81)
        np_679204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'np', False)
        # Obtaining the member 'array' of a type (line 81)
        array_679205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 30), np_679204, 'array')
        # Calling array(args, kwargs) (line 81)
        array_call_result_679211 = invoke(stypy.reporting.localization.Localization(__file__, 81, 30), array_679205, *[list_679206], **kwargs_679210)
        
        # Processing the call keyword arguments (line 81)
        kwargs_679212 = {}
        # Getting the type of 'assert_array_equal' (line 81)
        assert_array_equal_679202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 81)
        assert_array_equal_call_result_679213 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), assert_array_equal_679202, *[r_679203, array_call_result_679211], **kwargs_679212)
        
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to rankdata(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_679215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        
        # Processing the call keyword arguments (line 82)
        kwargs_679216 = {}
        # Getting the type of 'rankdata' (line 82)
        rankdata_679214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 82)
        rankdata_call_result_679217 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), rankdata_679214, *[list_679215], **kwargs_679216)
        
        # Assigning a type to the variable 'r' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'r', rankdata_call_result_679217)
        
        # Call to assert_array_equal(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'r' (line 83)
        r_679219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'r', False)
        
        # Call to array(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_679222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        
        # Processing the call keyword arguments (line 83)
        # Getting the type of 'np' (line 83)
        np_679223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'np', False)
        # Obtaining the member 'float64' of a type (line 83)
        float64_679224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 49), np_679223, 'float64')
        keyword_679225 = float64_679224
        kwargs_679226 = {'dtype': keyword_679225}
        # Getting the type of 'np' (line 83)
        np_679220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'np', False)
        # Obtaining the member 'array' of a type (line 83)
        array_679221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), np_679220, 'array')
        # Calling array(args, kwargs) (line 83)
        array_call_result_679227 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), array_679221, *[list_679222], **kwargs_679226)
        
        # Processing the call keyword arguments (line 83)
        kwargs_679228 = {}
        # Getting the type of 'assert_array_equal' (line 83)
        assert_array_equal_679218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 83)
        assert_array_equal_call_result_679229 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assert_array_equal_679218, *[r_679219, array_call_result_679227], **kwargs_679228)
        
        
        # ################# End of 'test_empty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_empty' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_679230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_empty'
        return stypy_return_type_679230


    @norecursion
    def test_one(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_one'
        module_type_store = module_type_store.open_function_context('test_one', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRankData.test_one.__dict__.__setitem__('stypy_localization', localization)
        TestRankData.test_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRankData.test_one.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRankData.test_one.__dict__.__setitem__('stypy_function_name', 'TestRankData.test_one')
        TestRankData.test_one.__dict__.__setitem__('stypy_param_names_list', [])
        TestRankData.test_one.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRankData.test_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRankData.test_one.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRankData.test_one.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRankData.test_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRankData.test_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRankData.test_one', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_one', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_one(...)' code ##################

        str_679231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'str', 'Check stats.rankdata with an array of length 1.')
        
        # Assigning a List to a Name (line 87):
        
        # Assigning a List to a Name (line 87):
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_679232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        int_679233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 15), list_679232, int_679233)
        
        # Assigning a type to the variable 'data' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'data', list_679232)
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to array(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'data' (line 88)
        data_679236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'data', False)
        # Processing the call keyword arguments (line 88)
        # Getting the type of 'int' (line 88)
        int_679237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'int', False)
        keyword_679238 = int_679237
        kwargs_679239 = {'dtype': keyword_679238}
        # Getting the type of 'np' (line 88)
        np_679234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 88)
        array_679235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), np_679234, 'array')
        # Calling array(args, kwargs) (line 88)
        array_call_result_679240 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), array_679235, *[data_679236], **kwargs_679239)
        
        # Assigning a type to the variable 'a' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'a', array_call_result_679240)
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to rankdata(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'a' (line 89)
        a_679242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 21), 'a', False)
        # Processing the call keyword arguments (line 89)
        kwargs_679243 = {}
        # Getting the type of 'rankdata' (line 89)
        rankdata_679241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 89)
        rankdata_call_result_679244 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), rankdata_679241, *[a_679242], **kwargs_679243)
        
        # Assigning a type to the variable 'r' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'r', rankdata_call_result_679244)
        
        # Call to assert_array_equal(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'r' (line 90)
        r_679246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 'r', False)
        
        # Call to array(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_679249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        float_679250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 39), list_679249, float_679250)
        
        # Processing the call keyword arguments (line 90)
        # Getting the type of 'np' (line 90)
        np_679251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 52), 'np', False)
        # Obtaining the member 'float64' of a type (line 90)
        float64_679252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 52), np_679251, 'float64')
        keyword_679253 = float64_679252
        kwargs_679254 = {'dtype': keyword_679253}
        # Getting the type of 'np' (line 90)
        np_679247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'np', False)
        # Obtaining the member 'array' of a type (line 90)
        array_679248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 30), np_679247, 'array')
        # Calling array(args, kwargs) (line 90)
        array_call_result_679255 = invoke(stypy.reporting.localization.Localization(__file__, 90, 30), array_679248, *[list_679249], **kwargs_679254)
        
        # Processing the call keyword arguments (line 90)
        kwargs_679256 = {}
        # Getting the type of 'assert_array_equal' (line 90)
        assert_array_equal_679245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 90)
        assert_array_equal_call_result_679257 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_array_equal_679245, *[r_679246, array_call_result_679255], **kwargs_679256)
        
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to rankdata(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'data' (line 91)
        data_679259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'data', False)
        # Processing the call keyword arguments (line 91)
        kwargs_679260 = {}
        # Getting the type of 'rankdata' (line 91)
        rankdata_679258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 91)
        rankdata_call_result_679261 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), rankdata_679258, *[data_679259], **kwargs_679260)
        
        # Assigning a type to the variable 'r' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'r', rankdata_call_result_679261)
        
        # Call to assert_array_equal(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'r' (line 92)
        r_679263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'r', False)
        
        # Call to array(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_679266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        float_679267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 39), list_679266, float_679267)
        
        # Processing the call keyword arguments (line 92)
        # Getting the type of 'np' (line 92)
        np_679268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 52), 'np', False)
        # Obtaining the member 'float64' of a type (line 92)
        float64_679269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 52), np_679268, 'float64')
        keyword_679270 = float64_679269
        kwargs_679271 = {'dtype': keyword_679270}
        # Getting the type of 'np' (line 92)
        np_679264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'np', False)
        # Obtaining the member 'array' of a type (line 92)
        array_679265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 30), np_679264, 'array')
        # Calling array(args, kwargs) (line 92)
        array_call_result_679272 = invoke(stypy.reporting.localization.Localization(__file__, 92, 30), array_679265, *[list_679266], **kwargs_679271)
        
        # Processing the call keyword arguments (line 92)
        kwargs_679273 = {}
        # Getting the type of 'assert_array_equal' (line 92)
        assert_array_equal_679262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 92)
        assert_array_equal_call_result_679274 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), assert_array_equal_679262, *[r_679263, array_call_result_679272], **kwargs_679273)
        
        
        # ################# End of 'test_one(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_one' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_679275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679275)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_one'
        return stypy_return_type_679275


    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRankData.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestRankData.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRankData.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRankData.test_basic.__dict__.__setitem__('stypy_function_name', 'TestRankData.test_basic')
        TestRankData.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestRankData.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRankData.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRankData.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRankData.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRankData.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRankData.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRankData.test_basic', [], None, None, defaults, varargs, kwargs)

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

        str_679276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'str', 'Basic tests of stats.rankdata.')
        
        # Assigning a List to a Name (line 96):
        
        # Assigning a List to a Name (line 96):
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_679277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        int_679278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 15), list_679277, int_679278)
        # Adding element type (line 96)
        int_679279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 15), list_679277, int_679279)
        # Adding element type (line 96)
        int_679280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 15), list_679277, int_679280)
        
        # Assigning a type to the variable 'data' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'data', list_679277)
        
        # Assigning a Call to a Name (line 97):
        
        # Assigning a Call to a Name (line 97):
        
        # Call to array(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_679283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        float_679284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_679283, float_679284)
        # Adding element type (line 97)
        float_679285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_679283, float_679285)
        # Adding element type (line 97)
        float_679286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_679283, float_679286)
        
        # Processing the call keyword arguments (line 97)
        # Getting the type of 'np' (line 97)
        np_679287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 51), 'np', False)
        # Obtaining the member 'float64' of a type (line 97)
        float64_679288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 51), np_679287, 'float64')
        keyword_679289 = float64_679288
        kwargs_679290 = {'dtype': keyword_679289}
        # Getting the type of 'np' (line 97)
        np_679281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 97)
        array_679282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), np_679281, 'array')
        # Calling array(args, kwargs) (line 97)
        array_call_result_679291 = invoke(stypy.reporting.localization.Localization(__file__, 97, 19), array_679282, *[list_679283], **kwargs_679290)
        
        # Assigning a type to the variable 'expected' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'expected', array_call_result_679291)
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to array(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'data' (line 98)
        data_679294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 21), 'data', False)
        # Processing the call keyword arguments (line 98)
        # Getting the type of 'int' (line 98)
        int_679295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'int', False)
        keyword_679296 = int_679295
        kwargs_679297 = {'dtype': keyword_679296}
        # Getting the type of 'np' (line 98)
        np_679292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 98)
        array_679293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), np_679292, 'array')
        # Calling array(args, kwargs) (line 98)
        array_call_result_679298 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), array_679293, *[data_679294], **kwargs_679297)
        
        # Assigning a type to the variable 'a' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'a', array_call_result_679298)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to rankdata(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'a' (line 99)
        a_679300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'a', False)
        # Processing the call keyword arguments (line 99)
        kwargs_679301 = {}
        # Getting the type of 'rankdata' (line 99)
        rankdata_679299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 99)
        rankdata_call_result_679302 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), rankdata_679299, *[a_679300], **kwargs_679301)
        
        # Assigning a type to the variable 'r' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'r', rankdata_call_result_679302)
        
        # Call to assert_array_equal(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'r' (line 100)
        r_679304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'r', False)
        # Getting the type of 'expected' (line 100)
        expected_679305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'expected', False)
        # Processing the call keyword arguments (line 100)
        kwargs_679306 = {}
        # Getting the type of 'assert_array_equal' (line 100)
        assert_array_equal_679303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 100)
        assert_array_equal_call_result_679307 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), assert_array_equal_679303, *[r_679304, expected_679305], **kwargs_679306)
        
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to rankdata(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'data' (line 101)
        data_679309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'data', False)
        # Processing the call keyword arguments (line 101)
        kwargs_679310 = {}
        # Getting the type of 'rankdata' (line 101)
        rankdata_679308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 101)
        rankdata_call_result_679311 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), rankdata_679308, *[data_679309], **kwargs_679310)
        
        # Assigning a type to the variable 'r' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'r', rankdata_call_result_679311)
        
        # Call to assert_array_equal(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'r' (line 102)
        r_679313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'r', False)
        # Getting the type of 'expected' (line 102)
        expected_679314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'expected', False)
        # Processing the call keyword arguments (line 102)
        kwargs_679315 = {}
        # Getting the type of 'assert_array_equal' (line 102)
        assert_array_equal_679312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 102)
        assert_array_equal_call_result_679316 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_array_equal_679312, *[r_679313, expected_679314], **kwargs_679315)
        
        
        # Assigning a List to a Name (line 104):
        
        # Assigning a List to a Name (line 104):
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_679317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        int_679318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_679317, int_679318)
        # Adding element type (line 104)
        int_679319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_679317, int_679319)
        # Adding element type (line 104)
        int_679320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_679317, int_679320)
        # Adding element type (line 104)
        int_679321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_679317, int_679321)
        # Adding element type (line 104)
        int_679322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 15), list_679317, int_679322)
        
        # Assigning a type to the variable 'data' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'data', list_679317)
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to array(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_679325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        float_679326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 28), list_679325, float_679326)
        # Adding element type (line 105)
        float_679327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 28), list_679325, float_679327)
        # Adding element type (line 105)
        float_679328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 28), list_679325, float_679328)
        # Adding element type (line 105)
        float_679329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 28), list_679325, float_679329)
        # Adding element type (line 105)
        float_679330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 28), list_679325, float_679330)
        
        # Processing the call keyword arguments (line 105)
        # Getting the type of 'np' (line 105)
        np_679331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 61), 'np', False)
        # Obtaining the member 'float64' of a type (line 105)
        float64_679332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 61), np_679331, 'float64')
        keyword_679333 = float64_679332
        kwargs_679334 = {'dtype': keyword_679333}
        # Getting the type of 'np' (line 105)
        np_679323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 105)
        array_679324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), np_679323, 'array')
        # Calling array(args, kwargs) (line 105)
        array_call_result_679335 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), array_679324, *[list_679325], **kwargs_679334)
        
        # Assigning a type to the variable 'expected' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'expected', array_call_result_679335)
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to array(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'data' (line 106)
        data_679338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'data', False)
        # Processing the call keyword arguments (line 106)
        # Getting the type of 'int' (line 106)
        int_679339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'int', False)
        keyword_679340 = int_679339
        kwargs_679341 = {'dtype': keyword_679340}
        # Getting the type of 'np' (line 106)
        np_679336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 106)
        array_679337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), np_679336, 'array')
        # Calling array(args, kwargs) (line 106)
        array_call_result_679342 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), array_679337, *[data_679338], **kwargs_679341)
        
        # Assigning a type to the variable 'a' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'a', array_call_result_679342)
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to rankdata(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'a' (line 107)
        a_679344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'a', False)
        # Processing the call keyword arguments (line 107)
        kwargs_679345 = {}
        # Getting the type of 'rankdata' (line 107)
        rankdata_679343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 107)
        rankdata_call_result_679346 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), rankdata_679343, *[a_679344], **kwargs_679345)
        
        # Assigning a type to the variable 'r' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'r', rankdata_call_result_679346)
        
        # Call to assert_array_equal(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'r' (line 108)
        r_679348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'r', False)
        # Getting the type of 'expected' (line 108)
        expected_679349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'expected', False)
        # Processing the call keyword arguments (line 108)
        kwargs_679350 = {}
        # Getting the type of 'assert_array_equal' (line 108)
        assert_array_equal_679347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 108)
        assert_array_equal_call_result_679351 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assert_array_equal_679347, *[r_679348, expected_679349], **kwargs_679350)
        
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to rankdata(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'data' (line 109)
        data_679353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'data', False)
        # Processing the call keyword arguments (line 109)
        kwargs_679354 = {}
        # Getting the type of 'rankdata' (line 109)
        rankdata_679352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 109)
        rankdata_call_result_679355 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), rankdata_679352, *[data_679353], **kwargs_679354)
        
        # Assigning a type to the variable 'r' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'r', rankdata_call_result_679355)
        
        # Call to assert_array_equal(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'r' (line 110)
        r_679357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'r', False)
        # Getting the type of 'expected' (line 110)
        expected_679358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 30), 'expected', False)
        # Processing the call keyword arguments (line 110)
        kwargs_679359 = {}
        # Getting the type of 'assert_array_equal' (line 110)
        assert_array_equal_679356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 110)
        assert_array_equal_call_result_679360 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assert_array_equal_679356, *[r_679357, expected_679358], **kwargs_679359)
        
        
        # Assigning a List to a Name (line 112):
        
        # Assigning a List to a Name (line 112):
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_679361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        int_679362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_679361, int_679362)
        # Adding element type (line 112)
        int_679363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_679361, int_679363)
        # Adding element type (line 112)
        int_679364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_679361, int_679364)
        # Adding element type (line 112)
        int_679365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_679361, int_679365)
        # Adding element type (line 112)
        int_679366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_679361, int_679366)
        # Adding element type (line 112)
        int_679367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), list_679361, int_679367)
        
        # Assigning a type to the variable 'data' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'data', list_679361)
        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to array(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_679370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        float_679371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 28), list_679370, float_679371)
        # Adding element type (line 113)
        float_679372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 28), list_679370, float_679372)
        # Adding element type (line 113)
        float_679373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 28), list_679370, float_679373)
        # Adding element type (line 113)
        float_679374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 28), list_679370, float_679374)
        # Adding element type (line 113)
        float_679375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 28), list_679370, float_679375)
        # Adding element type (line 113)
        float_679376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 28), list_679370, float_679376)
        
        # Processing the call keyword arguments (line 113)
        # Getting the type of 'np' (line 113)
        np_679377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 66), 'np', False)
        # Obtaining the member 'float64' of a type (line 113)
        float64_679378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 66), np_679377, 'float64')
        keyword_679379 = float64_679378
        kwargs_679380 = {'dtype': keyword_679379}
        # Getting the type of 'np' (line 113)
        np_679368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 113)
        array_679369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), np_679368, 'array')
        # Calling array(args, kwargs) (line 113)
        array_call_result_679381 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), array_679369, *[list_679370], **kwargs_679380)
        
        # Assigning a type to the variable 'expected' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'expected', array_call_result_679381)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to array(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'data' (line 114)
        data_679384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'data', False)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'int' (line 114)
        int_679385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'int', False)
        keyword_679386 = int_679385
        kwargs_679387 = {'dtype': keyword_679386}
        # Getting the type of 'np' (line 114)
        np_679382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 114)
        array_679383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), np_679382, 'array')
        # Calling array(args, kwargs) (line 114)
        array_call_result_679388 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), array_679383, *[data_679384], **kwargs_679387)
        
        # Assigning a type to the variable 'a' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'a', array_call_result_679388)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to rankdata(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'a' (line 115)
        a_679390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'a', False)
        # Processing the call keyword arguments (line 115)
        kwargs_679391 = {}
        # Getting the type of 'rankdata' (line 115)
        rankdata_679389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 115)
        rankdata_call_result_679392 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), rankdata_679389, *[a_679390], **kwargs_679391)
        
        # Assigning a type to the variable 'r' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'r', rankdata_call_result_679392)
        
        # Call to assert_array_equal(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'r' (line 116)
        r_679394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 27), 'r', False)
        # Getting the type of 'expected' (line 116)
        expected_679395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'expected', False)
        # Processing the call keyword arguments (line 116)
        kwargs_679396 = {}
        # Getting the type of 'assert_array_equal' (line 116)
        assert_array_equal_679393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 116)
        assert_array_equal_call_result_679397 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert_array_equal_679393, *[r_679394, expected_679395], **kwargs_679396)
        
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to rankdata(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'data' (line 117)
        data_679399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'data', False)
        # Processing the call keyword arguments (line 117)
        kwargs_679400 = {}
        # Getting the type of 'rankdata' (line 117)
        rankdata_679398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 117)
        rankdata_call_result_679401 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), rankdata_679398, *[data_679399], **kwargs_679400)
        
        # Assigning a type to the variable 'r' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'r', rankdata_call_result_679401)
        
        # Call to assert_array_equal(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'r' (line 118)
        r_679403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 27), 'r', False)
        # Getting the type of 'expected' (line 118)
        expected_679404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'expected', False)
        # Processing the call keyword arguments (line 118)
        kwargs_679405 = {}
        # Getting the type of 'assert_array_equal' (line 118)
        assert_array_equal_679402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 118)
        assert_array_equal_call_result_679406 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), assert_array_equal_679402, *[r_679403, expected_679404], **kwargs_679405)
        
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to reshape(...): (line 120)
        # Processing the call arguments (line 120)
        int_679409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 24), 'int')
        int_679410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 27), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_679411 = {}
        # Getting the type of 'a' (line 120)
        a_679407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'a', False)
        # Obtaining the member 'reshape' of a type (line 120)
        reshape_679408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 14), a_679407, 'reshape')
        # Calling reshape(args, kwargs) (line 120)
        reshape_call_result_679412 = invoke(stypy.reporting.localization.Localization(__file__, 120, 14), reshape_679408, *[int_679409, int_679410], **kwargs_679411)
        
        # Assigning a type to the variable 'a2d' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'a2d', reshape_call_result_679412)
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to rankdata(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'a2d' (line 121)
        a2d_679414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'a2d', False)
        # Processing the call keyword arguments (line 121)
        kwargs_679415 = {}
        # Getting the type of 'rankdata' (line 121)
        rankdata_679413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 121)
        rankdata_call_result_679416 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), rankdata_679413, *[a2d_679414], **kwargs_679415)
        
        # Assigning a type to the variable 'r' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'r', rankdata_call_result_679416)
        
        # Call to assert_array_equal(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'r' (line 122)
        r_679418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 27), 'r', False)
        # Getting the type of 'expected' (line 122)
        expected_679419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'expected', False)
        # Processing the call keyword arguments (line 122)
        kwargs_679420 = {}
        # Getting the type of 'assert_array_equal' (line 122)
        assert_array_equal_679417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 122)
        assert_array_equal_call_result_679421 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assert_array_equal_679417, *[r_679418, expected_679419], **kwargs_679420)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_679422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_679422


    @norecursion
    def test_rankdata_object_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_rankdata_object_string'
        module_type_store = module_type_store.open_function_context('test_rankdata_object_string', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_localization', localization)
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_function_name', 'TestRankData.test_rankdata_object_string')
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_param_names_list', [])
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRankData.test_rankdata_object_string.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRankData.test_rankdata_object_string', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_rankdata_object_string', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_rankdata_object_string(...)' code ##################

        
        # Assigning a Lambda to a Name (line 125):
        
        # Assigning a Lambda to a Name (line 125):

        @norecursion
        def _stypy_temp_lambda_565(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_565'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_565', 125, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_565.stypy_localization = localization
            _stypy_temp_lambda_565.stypy_type_of_self = None
            _stypy_temp_lambda_565.stypy_type_store = module_type_store
            _stypy_temp_lambda_565.stypy_function_name = '_stypy_temp_lambda_565'
            _stypy_temp_lambda_565.stypy_param_names_list = ['a']
            _stypy_temp_lambda_565.stypy_varargs_param_name = None
            _stypy_temp_lambda_565.stypy_kwargs_param_name = None
            _stypy_temp_lambda_565.stypy_call_defaults = defaults
            _stypy_temp_lambda_565.stypy_call_varargs = varargs
            _stypy_temp_lambda_565.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_565', ['a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_565', ['a'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'a' (line 125)
            a_679434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 65), 'a')
            comprehension_679435 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 30), a_679434)
            # Assigning a type to the variable 'j' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 30), 'j', comprehension_679435)
            int_679423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 30), 'int')
            
            # Call to sum(...): (line 125)
            # Processing the call arguments (line 125)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 125, 38, True)
            # Calculating comprehension expression
            # Getting the type of 'a' (line 125)
            a_679428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'a', False)
            comprehension_679429 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 38), a_679428)
            # Assigning a type to the variable 'i' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'i', comprehension_679429)
            
            # Getting the type of 'i' (line 125)
            i_679425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'i', False)
            # Getting the type of 'j' (line 125)
            j_679426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 42), 'j', False)
            # Applying the binary operator '<' (line 125)
            result_lt_679427 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 38), '<', i_679425, j_679426)
            
            list_679430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 38), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 38), list_679430, result_lt_679427)
            # Processing the call keyword arguments (line 125)
            kwargs_679431 = {}
            # Getting the type of 'sum' (line 125)
            sum_679424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'sum', False)
            # Calling sum(args, kwargs) (line 125)
            sum_call_result_679432 = invoke(stypy.reporting.localization.Localization(__file__, 125, 34), sum_679424, *[list_679430], **kwargs_679431)
            
            # Applying the binary operator '+' (line 125)
            result_add_679433 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 30), '+', int_679423, sum_call_result_679432)
            
            list_679436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 30), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 30), list_679436, result_add_679433)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'stypy_return_type', list_679436)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_565' in the type store
            # Getting the type of 'stypy_return_type' (line 125)
            stypy_return_type_679437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_679437)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_565'
            return stypy_return_type_679437

        # Assigning a type to the variable '_stypy_temp_lambda_565' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), '_stypy_temp_lambda_565', _stypy_temp_lambda_565)
        # Getting the type of '_stypy_temp_lambda_565' (line 125)
        _stypy_temp_lambda_565_679438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), '_stypy_temp_lambda_565')
        # Assigning a type to the variable 'min_rank' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'min_rank', _stypy_temp_lambda_565_679438)
        
        # Assigning a Lambda to a Name (line 126):
        
        # Assigning a Lambda to a Name (line 126):

        @norecursion
        def _stypy_temp_lambda_566(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_566'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_566', 126, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_566.stypy_localization = localization
            _stypy_temp_lambda_566.stypy_type_of_self = None
            _stypy_temp_lambda_566.stypy_type_store = module_type_store
            _stypy_temp_lambda_566.stypy_function_name = '_stypy_temp_lambda_566'
            _stypy_temp_lambda_566.stypy_param_names_list = ['a']
            _stypy_temp_lambda_566.stypy_varargs_param_name = None
            _stypy_temp_lambda_566.stypy_kwargs_param_name = None
            _stypy_temp_lambda_566.stypy_call_defaults = defaults
            _stypy_temp_lambda_566.stypy_call_varargs = varargs
            _stypy_temp_lambda_566.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_566', ['a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_566', ['a'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'a' (line 126)
            a_679448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 62), 'a')
            comprehension_679449 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 30), a_679448)
            # Assigning a type to the variable 'j' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'j', comprehension_679449)
            
            # Call to sum(...): (line 126)
            # Processing the call arguments (line 126)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 126, 34, True)
            # Calculating comprehension expression
            # Getting the type of 'a' (line 126)
            a_679443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 50), 'a', False)
            comprehension_679444 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 34), a_679443)
            # Assigning a type to the variable 'i' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), 'i', comprehension_679444)
            
            # Getting the type of 'i' (line 126)
            i_679440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 34), 'i', False)
            # Getting the type of 'j' (line 126)
            j_679441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'j', False)
            # Applying the binary operator '<=' (line 126)
            result_le_679442 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 34), '<=', i_679440, j_679441)
            
            list_679445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 34), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 34), list_679445, result_le_679442)
            # Processing the call keyword arguments (line 126)
            kwargs_679446 = {}
            # Getting the type of 'sum' (line 126)
            sum_679439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'sum', False)
            # Calling sum(args, kwargs) (line 126)
            sum_call_result_679447 = invoke(stypy.reporting.localization.Localization(__file__, 126, 30), sum_679439, *[list_679445], **kwargs_679446)
            
            list_679450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 30), list_679450, sum_call_result_679447)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'stypy_return_type', list_679450)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_566' in the type store
            # Getting the type of 'stypy_return_type' (line 126)
            stypy_return_type_679451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_679451)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_566'
            return stypy_return_type_679451

        # Assigning a type to the variable '_stypy_temp_lambda_566' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), '_stypy_temp_lambda_566', _stypy_temp_lambda_566)
        # Getting the type of '_stypy_temp_lambda_566' (line 126)
        _stypy_temp_lambda_566_679452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), '_stypy_temp_lambda_566')
        # Assigning a type to the variable 'max_rank' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'max_rank', _stypy_temp_lambda_566_679452)
        
        # Assigning a Lambda to a Name (line 127):
        
        # Assigning a Lambda to a Name (line 127):

        @norecursion
        def _stypy_temp_lambda_567(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_567'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_567', 127, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_567.stypy_localization = localization
            _stypy_temp_lambda_567.stypy_type_of_self = None
            _stypy_temp_lambda_567.stypy_type_store = module_type_store
            _stypy_temp_lambda_567.stypy_function_name = '_stypy_temp_lambda_567'
            _stypy_temp_lambda_567.stypy_param_names_list = ['a']
            _stypy_temp_lambda_567.stypy_varargs_param_name = None
            _stypy_temp_lambda_567.stypy_kwargs_param_name = None
            _stypy_temp_lambda_567.stypy_call_defaults = defaults
            _stypy_temp_lambda_567.stypy_call_varargs = varargs
            _stypy_temp_lambda_567.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_567', ['a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_567', ['a'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to min_rank(...): (line 127)
            # Processing the call arguments (line 127)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to enumerate(...): (line 127)
            # Processing the call arguments (line 127)
            # Getting the type of 'a' (line 127)
            a_679458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 72), 'a', False)
            # Processing the call keyword arguments (line 127)
            kwargs_679459 = {}
            # Getting the type of 'enumerate' (line 127)
            enumerate_679457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 62), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 127)
            enumerate_call_result_679460 = invoke(stypy.reporting.localization.Localization(__file__, 127, 62), enumerate_679457, *[a_679458], **kwargs_679459)
            
            comprehension_679461 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 43), enumerate_call_result_679460)
            # Assigning a type to the variable 'i' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 43), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 43), comprehension_679461))
            # Assigning a type to the variable 'x' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 43), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 43), comprehension_679461))
            
            # Obtaining an instance of the builtin type 'tuple' (line 127)
            tuple_679454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 44), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 127)
            # Adding element type (line 127)
            # Getting the type of 'x' (line 127)
            x_679455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 44), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), tuple_679454, x_679455)
            # Adding element type (line 127)
            # Getting the type of 'i' (line 127)
            i_679456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'i', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 44), tuple_679454, i_679456)
            
            list_679462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 43), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 43), list_679462, tuple_679454)
            # Processing the call keyword arguments (line 127)
            kwargs_679463 = {}
            # Getting the type of 'min_rank' (line 127)
            min_rank_679453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'min_rank', False)
            # Calling min_rank(args, kwargs) (line 127)
            min_rank_call_result_679464 = invoke(stypy.reporting.localization.Localization(__file__, 127, 33), min_rank_679453, *[list_679462], **kwargs_679463)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'stypy_return_type', min_rank_call_result_679464)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_567' in the type store
            # Getting the type of 'stypy_return_type' (line 127)
            stypy_return_type_679465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_679465)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_567'
            return stypy_return_type_679465

        # Assigning a type to the variable '_stypy_temp_lambda_567' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), '_stypy_temp_lambda_567', _stypy_temp_lambda_567)
        # Getting the type of '_stypy_temp_lambda_567' (line 127)
        _stypy_temp_lambda_567_679466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), '_stypy_temp_lambda_567')
        # Assigning a type to the variable 'ordinal_rank' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'ordinal_rank', _stypy_temp_lambda_567_679466)

        @norecursion
        def average_rank(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'average_rank'
            module_type_store = module_type_store.open_function_context('average_rank', 129, 8, False)
            
            # Passed parameters checking function
            average_rank.stypy_localization = localization
            average_rank.stypy_type_of_self = None
            average_rank.stypy_type_store = module_type_store
            average_rank.stypy_function_name = 'average_rank'
            average_rank.stypy_param_names_list = ['a']
            average_rank.stypy_varargs_param_name = None
            average_rank.stypy_kwargs_param_name = None
            average_rank.stypy_call_defaults = defaults
            average_rank.stypy_call_varargs = varargs
            average_rank.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'average_rank', ['a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'average_rank', localization, ['a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'average_rank(...)' code ##################

            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to zip(...): (line 130)
            # Processing the call arguments (line 130)
            
            # Call to min_rank(...): (line 130)
            # Processing the call arguments (line 130)
            # Getting the type of 'a' (line 130)
            a_679474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 59), 'a', False)
            # Processing the call keyword arguments (line 130)
            kwargs_679475 = {}
            # Getting the type of 'min_rank' (line 130)
            min_rank_679473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 50), 'min_rank', False)
            # Calling min_rank(args, kwargs) (line 130)
            min_rank_call_result_679476 = invoke(stypy.reporting.localization.Localization(__file__, 130, 50), min_rank_679473, *[a_679474], **kwargs_679475)
            
            
            # Call to max_rank(...): (line 130)
            # Processing the call arguments (line 130)
            # Getting the type of 'a' (line 130)
            a_679478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 72), 'a', False)
            # Processing the call keyword arguments (line 130)
            kwargs_679479 = {}
            # Getting the type of 'max_rank' (line 130)
            max_rank_679477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 63), 'max_rank', False)
            # Calling max_rank(args, kwargs) (line 130)
            max_rank_call_result_679480 = invoke(stypy.reporting.localization.Localization(__file__, 130, 63), max_rank_679477, *[a_679478], **kwargs_679479)
            
            # Processing the call keyword arguments (line 130)
            kwargs_679481 = {}
            # Getting the type of 'zip' (line 130)
            zip_679472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 46), 'zip', False)
            # Calling zip(args, kwargs) (line 130)
            zip_call_result_679482 = invoke(stypy.reporting.localization.Localization(__file__, 130, 46), zip_679472, *[min_rank_call_result_679476, max_rank_call_result_679480], **kwargs_679481)
            
            comprehension_679483 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 20), zip_call_result_679482)
            # Assigning a type to the variable 'i' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 20), comprehension_679483))
            # Assigning a type to the variable 'j' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 20), comprehension_679483))
            # Getting the type of 'i' (line 130)
            i_679467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'i')
            # Getting the type of 'j' (line 130)
            j_679468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'j')
            # Applying the binary operator '+' (line 130)
            result_add_679469 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 21), '+', i_679467, j_679468)
            
            float_679470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 30), 'float')
            # Applying the binary operator 'div' (line 130)
            result_div_679471 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 20), 'div', result_add_679469, float_679470)
            
            list_679484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 20), list_679484, result_div_679471)
            # Assigning a type to the variable 'stypy_return_type' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'stypy_return_type', list_679484)
            
            # ################# End of 'average_rank(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'average_rank' in the type store
            # Getting the type of 'stypy_return_type' (line 129)
            stypy_return_type_679485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_679485)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'average_rank'
            return stypy_return_type_679485

        # Assigning a type to the variable 'average_rank' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'average_rank', average_rank)

        @norecursion
        def dense_rank(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'dense_rank'
            module_type_store = module_type_store.open_function_context('dense_rank', 132, 8, False)
            
            # Passed parameters checking function
            dense_rank.stypy_localization = localization
            dense_rank.stypy_type_of_self = None
            dense_rank.stypy_type_store = module_type_store
            dense_rank.stypy_function_name = 'dense_rank'
            dense_rank.stypy_param_names_list = ['a']
            dense_rank.stypy_varargs_param_name = None
            dense_rank.stypy_kwargs_param_name = None
            dense_rank.stypy_call_defaults = defaults
            dense_rank.stypy_call_varargs = varargs
            dense_rank.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'dense_rank', ['a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'dense_rank', localization, ['a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'dense_rank(...)' code ##################

            
            # Assigning a Call to a Name (line 133):
            
            # Assigning a Call to a Name (line 133):
            
            # Call to unique(...): (line 133)
            # Processing the call arguments (line 133)
            # Getting the type of 'a' (line 133)
            a_679488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 26), 'a', False)
            # Processing the call keyword arguments (line 133)
            kwargs_679489 = {}
            # Getting the type of 'np' (line 133)
            np_679486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'np', False)
            # Obtaining the member 'unique' of a type (line 133)
            unique_679487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), np_679486, 'unique')
            # Calling unique(args, kwargs) (line 133)
            unique_call_result_679490 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), unique_679487, *[a_679488], **kwargs_679489)
            
            # Assigning a type to the variable 'b' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'b', unique_call_result_679490)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'a' (line 134)
            a_679502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 55), 'a')
            comprehension_679503 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 20), a_679502)
            # Assigning a type to the variable 'j' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'j', comprehension_679503)
            int_679491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 20), 'int')
            
            # Call to sum(...): (line 134)
            # Processing the call arguments (line 134)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 134, 28, True)
            # Calculating comprehension expression
            # Getting the type of 'b' (line 134)
            b_679496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 43), 'b', False)
            comprehension_679497 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 28), b_679496)
            # Assigning a type to the variable 'i' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 28), 'i', comprehension_679497)
            
            # Getting the type of 'i' (line 134)
            i_679493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 28), 'i', False)
            # Getting the type of 'j' (line 134)
            j_679494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 32), 'j', False)
            # Applying the binary operator '<' (line 134)
            result_lt_679495 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 28), '<', i_679493, j_679494)
            
            list_679498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 28), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 28), list_679498, result_lt_679495)
            # Processing the call keyword arguments (line 134)
            kwargs_679499 = {}
            # Getting the type of 'sum' (line 134)
            sum_679492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'sum', False)
            # Calling sum(args, kwargs) (line 134)
            sum_call_result_679500 = invoke(stypy.reporting.localization.Localization(__file__, 134, 24), sum_679492, *[list_679498], **kwargs_679499)
            
            # Applying the binary operator '+' (line 134)
            result_add_679501 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 20), '+', int_679491, sum_call_result_679500)
            
            list_679504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 20), list_679504, result_add_679501)
            # Assigning a type to the variable 'stypy_return_type' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'stypy_return_type', list_679504)
            
            # ################# End of 'dense_rank(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'dense_rank' in the type store
            # Getting the type of 'stypy_return_type' (line 132)
            stypy_return_type_679505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_679505)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'dense_rank'
            return stypy_return_type_679505

        # Assigning a type to the variable 'dense_rank' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'dense_rank', dense_rank)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to dict(...): (line 136)
        # Processing the call keyword arguments (line 136)
        # Getting the type of 'min_rank' (line 136)
        min_rank_679507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 'min_rank', False)
        keyword_679508 = min_rank_679507
        # Getting the type of 'max_rank' (line 136)
        max_rank_679509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 39), 'max_rank', False)
        keyword_679510 = max_rank_679509
        # Getting the type of 'ordinal_rank' (line 136)
        ordinal_rank_679511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 57), 'ordinal_rank', False)
        keyword_679512 = ordinal_rank_679511
        # Getting the type of 'average_rank' (line 137)
        average_rank_679513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'average_rank', False)
        keyword_679514 = average_rank_679513
        # Getting the type of 'dense_rank' (line 137)
        dense_rank_679515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 49), 'dense_rank', False)
        keyword_679516 = dense_rank_679515
        kwargs_679517 = {'ordinal': keyword_679512, 'max': keyword_679510, 'average': keyword_679514, 'dense': keyword_679516, 'min': keyword_679508}
        # Getting the type of 'dict' (line 136)
        dict_679506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'dict', False)
        # Calling dict(args, kwargs) (line 136)
        dict_call_result_679518 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), dict_679506, *[], **kwargs_679517)
        
        # Assigning a type to the variable 'rankf' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'rankf', dict_call_result_679518)

        @norecursion
        def check_ranks(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'check_ranks'
            module_type_store = module_type_store.open_function_context('check_ranks', 139, 8, False)
            
            # Passed parameters checking function
            check_ranks.stypy_localization = localization
            check_ranks.stypy_type_of_self = None
            check_ranks.stypy_type_store = module_type_store
            check_ranks.stypy_function_name = 'check_ranks'
            check_ranks.stypy_param_names_list = ['a']
            check_ranks.stypy_varargs_param_name = None
            check_ranks.stypy_kwargs_param_name = None
            check_ranks.stypy_call_defaults = defaults
            check_ranks.stypy_call_varargs = varargs
            check_ranks.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'check_ranks', ['a'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'check_ranks', localization, ['a'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'check_ranks(...)' code ##################

            
            
            # Obtaining an instance of the builtin type 'tuple' (line 140)
            tuple_679519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 26), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 140)
            # Adding element type (line 140)
            str_679520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 26), 'str', 'min')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 26), tuple_679519, str_679520)
            # Adding element type (line 140)
            str_679521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'str', 'max')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 26), tuple_679519, str_679521)
            # Adding element type (line 140)
            str_679522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 40), 'str', 'dense')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 26), tuple_679519, str_679522)
            # Adding element type (line 140)
            str_679523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 49), 'str', 'ordinal')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 26), tuple_679519, str_679523)
            # Adding element type (line 140)
            str_679524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 60), 'str', 'average')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 26), tuple_679519, str_679524)
            
            # Testing the type of a for loop iterable (line 140)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 12), tuple_679519)
            # Getting the type of the for loop variable (line 140)
            for_loop_var_679525 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 12), tuple_679519)
            # Assigning a type to the variable 'method' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'method', for_loop_var_679525)
            # SSA begins for a for statement (line 140)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 141):
            
            # Assigning a Call to a Name (line 141):
            
            # Call to rankdata(...): (line 141)
            # Processing the call arguments (line 141)
            # Getting the type of 'a' (line 141)
            a_679527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 31), 'a', False)
            # Processing the call keyword arguments (line 141)
            # Getting the type of 'method' (line 141)
            method_679528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'method', False)
            keyword_679529 = method_679528
            kwargs_679530 = {'method': keyword_679529}
            # Getting the type of 'rankdata' (line 141)
            rankdata_679526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'rankdata', False)
            # Calling rankdata(args, kwargs) (line 141)
            rankdata_call_result_679531 = invoke(stypy.reporting.localization.Localization(__file__, 141, 22), rankdata_679526, *[a_679527], **kwargs_679530)
            
            # Assigning a type to the variable 'out' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'out', rankdata_call_result_679531)
            
            # Call to assert_array_equal(...): (line 142)
            # Processing the call arguments (line 142)
            # Getting the type of 'out' (line 142)
            out_679533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 35), 'out', False)
            
            # Call to (...): (line 142)
            # Processing the call arguments (line 142)
            # Getting the type of 'a' (line 142)
            a_679538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 54), 'a', False)
            # Processing the call keyword arguments (line 142)
            kwargs_679539 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'method' (line 142)
            method_679534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 46), 'method', False)
            # Getting the type of 'rankf' (line 142)
            rankf_679535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 40), 'rankf', False)
            # Obtaining the member '__getitem__' of a type (line 142)
            getitem___679536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 40), rankf_679535, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 142)
            subscript_call_result_679537 = invoke(stypy.reporting.localization.Localization(__file__, 142, 40), getitem___679536, method_679534)
            
            # Calling (args, kwargs) (line 142)
            _call_result_679540 = invoke(stypy.reporting.localization.Localization(__file__, 142, 40), subscript_call_result_679537, *[a_679538], **kwargs_679539)
            
            # Processing the call keyword arguments (line 142)
            kwargs_679541 = {}
            # Getting the type of 'assert_array_equal' (line 142)
            assert_array_equal_679532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'assert_array_equal', False)
            # Calling assert_array_equal(args, kwargs) (line 142)
            assert_array_equal_call_result_679542 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), assert_array_equal_679532, *[out_679533, _call_result_679540], **kwargs_679541)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'check_ranks(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'check_ranks' in the type store
            # Getting the type of 'stypy_return_type' (line 139)
            stypy_return_type_679543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_679543)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'check_ranks'
            return stypy_return_type_679543

        # Assigning a type to the variable 'check_ranks' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'check_ranks', check_ranks)
        
        # Assigning a List to a Name (line 144):
        
        # Assigning a List to a Name (line 144):
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_679544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        str_679545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679545)
        # Adding element type (line 144)
        str_679546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 22), 'str', 'bar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679546)
        # Adding element type (line 144)
        str_679547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 29), 'str', 'qux')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679547)
        # Adding element type (line 144)
        str_679548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 36), 'str', 'xyz')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679548)
        # Adding element type (line 144)
        str_679549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 43), 'str', 'abc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679549)
        # Adding element type (line 144)
        str_679550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 50), 'str', 'efg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679550)
        # Adding element type (line 144)
        str_679551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 57), 'str', 'ace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679551)
        # Adding element type (line 144)
        str_679552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 64), 'str', 'qwe')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679552)
        # Adding element type (line 144)
        str_679553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 71), 'str', 'qaz')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), list_679544, str_679553)
        
        # Assigning a type to the variable 'val' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'val', list_679544)
        
        # Call to check_ranks(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Call to choice(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'val' (line 145)
        val_679558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 37), 'val', False)
        int_679559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 42), 'int')
        # Processing the call keyword arguments (line 145)
        kwargs_679560 = {}
        # Getting the type of 'np' (line 145)
        np_679555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 145)
        random_679556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 20), np_679555, 'random')
        # Obtaining the member 'choice' of a type (line 145)
        choice_679557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 20), random_679556, 'choice')
        # Calling choice(args, kwargs) (line 145)
        choice_call_result_679561 = invoke(stypy.reporting.localization.Localization(__file__, 145, 20), choice_679557, *[val_679558, int_679559], **kwargs_679560)
        
        # Processing the call keyword arguments (line 145)
        kwargs_679562 = {}
        # Getting the type of 'check_ranks' (line 145)
        check_ranks_679554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'check_ranks', False)
        # Calling check_ranks(args, kwargs) (line 145)
        check_ranks_call_result_679563 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), check_ranks_679554, *[choice_call_result_679561], **kwargs_679562)
        
        
        # Call to check_ranks(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to astype(...): (line 146)
        # Processing the call arguments (line 146)
        str_679573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 54), 'str', 'object')
        # Processing the call keyword arguments (line 146)
        kwargs_679574 = {}
        
        # Call to choice(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'val' (line 146)
        val_679568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 37), 'val', False)
        int_679569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 42), 'int')
        # Processing the call keyword arguments (line 146)
        kwargs_679570 = {}
        # Getting the type of 'np' (line 146)
        np_679565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 146)
        random_679566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), np_679565, 'random')
        # Obtaining the member 'choice' of a type (line 146)
        choice_679567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), random_679566, 'choice')
        # Calling choice(args, kwargs) (line 146)
        choice_call_result_679571 = invoke(stypy.reporting.localization.Localization(__file__, 146, 20), choice_679567, *[val_679568, int_679569], **kwargs_679570)
        
        # Obtaining the member 'astype' of a type (line 146)
        astype_679572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), choice_call_result_679571, 'astype')
        # Calling astype(args, kwargs) (line 146)
        astype_call_result_679575 = invoke(stypy.reporting.localization.Localization(__file__, 146, 20), astype_679572, *[str_679573], **kwargs_679574)
        
        # Processing the call keyword arguments (line 146)
        kwargs_679576 = {}
        # Getting the type of 'check_ranks' (line 146)
        check_ranks_679564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'check_ranks', False)
        # Calling check_ranks(args, kwargs) (line 146)
        check_ranks_call_result_679577 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), check_ranks_679564, *[astype_call_result_679575], **kwargs_679576)
        
        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to array(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_679580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        int_679581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), list_679580, int_679581)
        # Adding element type (line 148)
        int_679582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), list_679580, int_679582)
        # Adding element type (line 148)
        int_679583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), list_679580, int_679583)
        # Adding element type (line 148)
        float_679584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), list_679580, float_679584)
        # Adding element type (line 148)
        int_679585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), list_679580, int_679585)
        # Adding element type (line 148)
        float_679586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), list_679580, float_679586)
        
        # Processing the call keyword arguments (line 148)
        str_679587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 57), 'str', 'object')
        keyword_679588 = str_679587
        kwargs_679589 = {'dtype': keyword_679588}
        # Getting the type of 'np' (line 148)
        np_679578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'np', False)
        # Obtaining the member 'array' of a type (line 148)
        array_679579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 14), np_679578, 'array')
        # Calling array(args, kwargs) (line 148)
        array_call_result_679590 = invoke(stypy.reporting.localization.Localization(__file__, 148, 14), array_679579, *[list_679580], **kwargs_679589)
        
        # Assigning a type to the variable 'val' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'val', array_call_result_679590)
        
        # Call to check_ranks(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to astype(...): (line 149)
        # Processing the call arguments (line 149)
        str_679600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 54), 'str', 'object')
        # Processing the call keyword arguments (line 149)
        kwargs_679601 = {}
        
        # Call to choice(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'val' (line 149)
        val_679595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'val', False)
        int_679596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 42), 'int')
        # Processing the call keyword arguments (line 149)
        kwargs_679597 = {}
        # Getting the type of 'np' (line 149)
        np_679592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 149)
        random_679593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), np_679592, 'random')
        # Obtaining the member 'choice' of a type (line 149)
        choice_679594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), random_679593, 'choice')
        # Calling choice(args, kwargs) (line 149)
        choice_call_result_679598 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), choice_679594, *[val_679595, int_679596], **kwargs_679597)
        
        # Obtaining the member 'astype' of a type (line 149)
        astype_679599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), choice_call_result_679598, 'astype')
        # Calling astype(args, kwargs) (line 149)
        astype_call_result_679602 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), astype_679599, *[str_679600], **kwargs_679601)
        
        # Processing the call keyword arguments (line 149)
        kwargs_679603 = {}
        # Getting the type of 'check_ranks' (line 149)
        check_ranks_679591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'check_ranks', False)
        # Calling check_ranks(args, kwargs) (line 149)
        check_ranks_call_result_679604 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), check_ranks_679591, *[astype_call_result_679602], **kwargs_679603)
        
        
        # ################# End of 'test_rankdata_object_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_rankdata_object_string' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_679605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_rankdata_object_string'
        return stypy_return_type_679605


    @norecursion
    def test_large_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_large_int'
        module_type_store = module_type_store.open_function_context('test_large_int', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRankData.test_large_int.__dict__.__setitem__('stypy_localization', localization)
        TestRankData.test_large_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRankData.test_large_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRankData.test_large_int.__dict__.__setitem__('stypy_function_name', 'TestRankData.test_large_int')
        TestRankData.test_large_int.__dict__.__setitem__('stypy_param_names_list', [])
        TestRankData.test_large_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRankData.test_large_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRankData.test_large_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRankData.test_large_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRankData.test_large_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRankData.test_large_int.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRankData.test_large_int', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_large_int', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_large_int(...)' code ##################

        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to array(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_679608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        int_679609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'int')
        int_679610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'int')
        # Applying the binary operator '**' (line 152)
        result_pow_679611 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 25), '**', int_679609, int_679610)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 24), list_679608, result_pow_679611)
        # Adding element type (line 152)
        int_679612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'int')
        int_679613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 35), 'int')
        # Applying the binary operator '**' (line 152)
        result_pow_679614 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 32), '**', int_679612, int_679613)
        
        int_679615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 38), 'int')
        # Applying the binary operator '+' (line 152)
        result_add_679616 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 32), '+', result_pow_679614, int_679615)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 24), list_679608, result_add_679616)
        
        # Processing the call keyword arguments (line 152)
        # Getting the type of 'np' (line 152)
        np_679617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 48), 'np', False)
        # Obtaining the member 'uint64' of a type (line 152)
        uint64_679618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 48), np_679617, 'uint64')
        keyword_679619 = uint64_679618
        kwargs_679620 = {'dtype': keyword_679619}
        # Getting the type of 'np' (line 152)
        np_679606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 152)
        array_679607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 15), np_679606, 'array')
        # Calling array(args, kwargs) (line 152)
        array_call_result_679621 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), array_679607, *[list_679608], **kwargs_679620)
        
        # Assigning a type to the variable 'data' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'data', array_call_result_679621)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to rankdata(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'data' (line 153)
        data_679623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'data', False)
        # Processing the call keyword arguments (line 153)
        kwargs_679624 = {}
        # Getting the type of 'rankdata' (line 153)
        rankdata_679622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 153)
        rankdata_call_result_679625 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), rankdata_679622, *[data_679623], **kwargs_679624)
        
        # Assigning a type to the variable 'r' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'r', rankdata_call_result_679625)
        
        # Call to assert_array_equal(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'r' (line 154)
        r_679627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'r', False)
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_679628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        float_679629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 30), list_679628, float_679629)
        # Adding element type (line 154)
        float_679630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 30), list_679628, float_679630)
        
        # Processing the call keyword arguments (line 154)
        kwargs_679631 = {}
        # Getting the type of 'assert_array_equal' (line 154)
        assert_array_equal_679626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 154)
        assert_array_equal_call_result_679632 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), assert_array_equal_679626, *[r_679627, list_679628], **kwargs_679631)
        
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to array(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_679635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        int_679636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 25), 'int')
        int_679637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'int')
        # Applying the binary operator '**' (line 156)
        result_pow_679638 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 25), '**', int_679636, int_679637)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 24), list_679635, result_pow_679638)
        # Adding element type (line 156)
        int_679639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 32), 'int')
        int_679640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 35), 'int')
        # Applying the binary operator '**' (line 156)
        result_pow_679641 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 32), '**', int_679639, int_679640)
        
        int_679642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 38), 'int')
        # Applying the binary operator '+' (line 156)
        result_add_679643 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 32), '+', result_pow_679641, int_679642)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 24), list_679635, result_add_679643)
        
        # Processing the call keyword arguments (line 156)
        # Getting the type of 'np' (line 156)
        np_679644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 48), 'np', False)
        # Obtaining the member 'int64' of a type (line 156)
        int64_679645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 48), np_679644, 'int64')
        keyword_679646 = int64_679645
        kwargs_679647 = {'dtype': keyword_679646}
        # Getting the type of 'np' (line 156)
        np_679633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 156)
        array_679634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 15), np_679633, 'array')
        # Calling array(args, kwargs) (line 156)
        array_call_result_679648 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), array_679634, *[list_679635], **kwargs_679647)
        
        # Assigning a type to the variable 'data' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'data', array_call_result_679648)
        
        # Assigning a Call to a Name (line 157):
        
        # Assigning a Call to a Name (line 157):
        
        # Call to rankdata(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'data' (line 157)
        data_679650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'data', False)
        # Processing the call keyword arguments (line 157)
        kwargs_679651 = {}
        # Getting the type of 'rankdata' (line 157)
        rankdata_679649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 157)
        rankdata_call_result_679652 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), rankdata_679649, *[data_679650], **kwargs_679651)
        
        # Assigning a type to the variable 'r' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'r', rankdata_call_result_679652)
        
        # Call to assert_array_equal(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'r' (line 158)
        r_679654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 'r', False)
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_679655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        float_679656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 30), list_679655, float_679656)
        # Adding element type (line 158)
        float_679657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 30), list_679655, float_679657)
        
        # Processing the call keyword arguments (line 158)
        kwargs_679658 = {}
        # Getting the type of 'assert_array_equal' (line 158)
        assert_array_equal_679653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 158)
        assert_array_equal_call_result_679659 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), assert_array_equal_679653, *[r_679654, list_679655], **kwargs_679658)
        
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to array(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_679662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        int_679663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'int')
        int_679664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'int')
        # Applying the binary operator '**' (line 160)
        result_pow_679665 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 25), '**', int_679663, int_679664)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 24), list_679662, result_pow_679665)
        # Adding element type (line 160)
        
        int_679666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'int')
        int_679667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'int')
        # Applying the binary operator '**' (line 160)
        result_pow_679668 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 33), '**', int_679666, int_679667)
        
        # Applying the 'usub' unary operator (line 160)
        result___neg___679669 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 32), 'usub', result_pow_679668)
        
        int_679670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 39), 'int')
        # Applying the binary operator '+' (line 160)
        result_add_679671 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 32), '+', result___neg___679669, int_679670)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 24), list_679662, result_add_679671)
        
        # Processing the call keyword arguments (line 160)
        # Getting the type of 'np' (line 160)
        np_679672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 49), 'np', False)
        # Obtaining the member 'int64' of a type (line 160)
        int64_679673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 49), np_679672, 'int64')
        keyword_679674 = int64_679673
        kwargs_679675 = {'dtype': keyword_679674}
        # Getting the type of 'np' (line 160)
        np_679660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 160)
        array_679661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 15), np_679660, 'array')
        # Calling array(args, kwargs) (line 160)
        array_call_result_679676 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), array_679661, *[list_679662], **kwargs_679675)
        
        # Assigning a type to the variable 'data' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'data', array_call_result_679676)
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to rankdata(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'data' (line 161)
        data_679678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 'data', False)
        # Processing the call keyword arguments (line 161)
        kwargs_679679 = {}
        # Getting the type of 'rankdata' (line 161)
        rankdata_679677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 161)
        rankdata_call_result_679680 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), rankdata_679677, *[data_679678], **kwargs_679679)
        
        # Assigning a type to the variable 'r' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'r', rankdata_call_result_679680)
        
        # Call to assert_array_equal(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'r' (line 162)
        r_679682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'r', False)
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_679683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        float_679684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 30), list_679683, float_679684)
        # Adding element type (line 162)
        float_679685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 30), list_679683, float_679685)
        
        # Processing the call keyword arguments (line 162)
        kwargs_679686 = {}
        # Getting the type of 'assert_array_equal' (line 162)
        assert_array_equal_679681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 162)
        assert_array_equal_call_result_679687 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), assert_array_equal_679681, *[r_679682, list_679683], **kwargs_679686)
        
        
        # ################# End of 'test_large_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_large_int' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_679688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679688)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_large_int'
        return stypy_return_type_679688


    @norecursion
    def test_big_tie(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_big_tie'
        module_type_store = module_type_store.open_function_context('test_big_tie', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_localization', localization)
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_function_name', 'TestRankData.test_big_tie')
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_param_names_list', [])
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRankData.test_big_tie.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRankData.test_big_tie', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_big_tie', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_big_tie(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_679689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        int_679690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 17), list_679689, int_679690)
        # Adding element type (line 165)
        int_679691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 17), list_679689, int_679691)
        # Adding element type (line 165)
        int_679692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 17), list_679689, int_679692)
        
        # Testing the type of a for loop iterable (line 165)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 165, 8), list_679689)
        # Getting the type of the for loop variable (line 165)
        for_loop_var_679693 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 165, 8), list_679689)
        # Assigning a type to the variable 'n' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'n', for_loop_var_679693)
        # SSA begins for a for statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to ones(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'n' (line 166)
        n_679696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 'n', False)
        # Processing the call keyword arguments (line 166)
        # Getting the type of 'int' (line 166)
        int_679697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), 'int', False)
        keyword_679698 = int_679697
        kwargs_679699 = {'dtype': keyword_679698}
        # Getting the type of 'np' (line 166)
        np_679694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 166)
        ones_679695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 19), np_679694, 'ones')
        # Calling ones(args, kwargs) (line 166)
        ones_call_result_679700 = invoke(stypy.reporting.localization.Localization(__file__, 166, 19), ones_679695, *[n_679696], **kwargs_679699)
        
        # Assigning a type to the variable 'data' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'data', ones_call_result_679700)
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to rankdata(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'data' (line 167)
        data_679702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'data', False)
        # Processing the call keyword arguments (line 167)
        kwargs_679703 = {}
        # Getting the type of 'rankdata' (line 167)
        rankdata_679701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'rankdata', False)
        # Calling rankdata(args, kwargs) (line 167)
        rankdata_call_result_679704 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), rankdata_679701, *[data_679702], **kwargs_679703)
        
        # Assigning a type to the variable 'r' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'r', rankdata_call_result_679704)
        
        # Assigning a BinOp to a Name (line 168):
        
        # Assigning a BinOp to a Name (line 168):
        float_679705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 28), 'float')
        # Getting the type of 'n' (line 168)
        n_679706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'n')
        int_679707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 39), 'int')
        # Applying the binary operator '+' (line 168)
        result_add_679708 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 35), '+', n_679706, int_679707)
        
        # Applying the binary operator '*' (line 168)
        result_mul_679709 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 28), '*', float_679705, result_add_679708)
        
        # Assigning a type to the variable 'expected_rank' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'expected_rank', result_mul_679709)
        
        # Call to assert_array_equal(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'r' (line 169)
        r_679711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'r', False)
        # Getting the type of 'expected_rank' (line 169)
        expected_rank_679712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'expected_rank', False)
        # Getting the type of 'data' (line 169)
        data_679713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 50), 'data', False)
        # Applying the binary operator '*' (line 169)
        result_mul_679714 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 34), '*', expected_rank_679712, data_679713)
        
        str_679715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 31), 'str', 'test failed with n=%d')
        # Getting the type of 'n' (line 170)
        n_679716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 57), 'n', False)
        # Applying the binary operator '%' (line 170)
        result_mod_679717 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 31), '%', str_679715, n_679716)
        
        # Processing the call keyword arguments (line 169)
        kwargs_679718 = {}
        # Getting the type of 'assert_array_equal' (line 169)
        assert_array_equal_679710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 169)
        assert_array_equal_call_result_679719 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), assert_array_equal_679710, *[r_679711, result_mul_679714, result_mod_679717], **kwargs_679718)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_big_tie(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_big_tie' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_679720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_679720)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_big_tie'
        return stypy_return_type_679720


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 75, 0, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRankData.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestRankData' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'TestRankData', TestRankData)

# Assigning a Tuple to a Name (line 173):

# Assigning a Tuple to a Name (line 173):

# Obtaining an instance of the builtin type 'tuple' (line 175)
tuple_679721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 4), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 175)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 175)
tuple_679722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 175)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'list' (line 175)
list_679723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 175)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 5), tuple_679722, list_679723)
# Adding element type (line 175)
str_679724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 9), 'str', 'average')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 5), tuple_679722, str_679724)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'list' (line 175)
list_679725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 175)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 5), tuple_679722, list_679725)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679722)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 176)
tuple_679726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 176)
# Adding element type (line 176)

# Obtaining an instance of the builtin type 'list' (line 176)
list_679727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 176)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 5), tuple_679726, list_679727)
# Adding element type (line 176)
str_679728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 9), 'str', 'min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 5), tuple_679726, str_679728)
# Adding element type (line 176)

# Obtaining an instance of the builtin type 'list' (line 176)
list_679729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 176)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 5), tuple_679726, list_679729)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679726)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 177)
tuple_679730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 177)
# Adding element type (line 177)

# Obtaining an instance of the builtin type 'list' (line 177)
list_679731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 177)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 5), tuple_679730, list_679731)
# Adding element type (line 177)
str_679732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 9), 'str', 'max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 5), tuple_679730, str_679732)
# Adding element type (line 177)

# Obtaining an instance of the builtin type 'list' (line 177)
list_679733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 177)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 5), tuple_679730, list_679733)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679730)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 178)
tuple_679734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 178)
# Adding element type (line 178)

# Obtaining an instance of the builtin type 'list' (line 178)
list_679735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 178)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 5), tuple_679734, list_679735)
# Adding element type (line 178)
str_679736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 9), 'str', 'dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 5), tuple_679734, str_679736)
# Adding element type (line 178)

# Obtaining an instance of the builtin type 'list' (line 178)
list_679737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 178)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 5), tuple_679734, list_679737)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679734)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 179)
tuple_679738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 179)
# Adding element type (line 179)

# Obtaining an instance of the builtin type 'list' (line 179)
list_679739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 179)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 5), tuple_679738, list_679739)
# Adding element type (line 179)
str_679740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 9), 'str', 'ordinal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 5), tuple_679738, str_679740)
# Adding element type (line 179)

# Obtaining an instance of the builtin type 'list' (line 179)
list_679741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 179)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 5), tuple_679738, list_679741)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679738)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 181)
tuple_679742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 181)
# Adding element type (line 181)

# Obtaining an instance of the builtin type 'list' (line 181)
list_679743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 181)
# Adding element type (line 181)
int_679744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 5), list_679743, int_679744)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 5), tuple_679742, list_679743)
# Adding element type (line 181)
str_679745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 12), 'str', 'average')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 5), tuple_679742, str_679745)
# Adding element type (line 181)

# Obtaining an instance of the builtin type 'list' (line 181)
list_679746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 181)
# Adding element type (line 181)
float_679747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 23), list_679746, float_679747)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 5), tuple_679742, list_679746)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679742)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 182)
tuple_679748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 182)
# Adding element type (line 182)

# Obtaining an instance of the builtin type 'list' (line 182)
list_679749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 182)
# Adding element type (line 182)
int_679750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 5), list_679749, int_679750)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 5), tuple_679748, list_679749)
# Adding element type (line 182)
str_679751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 12), 'str', 'min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 5), tuple_679748, str_679751)
# Adding element type (line 182)

# Obtaining an instance of the builtin type 'list' (line 182)
list_679752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 182)
# Adding element type (line 182)
float_679753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 19), list_679752, float_679753)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 5), tuple_679748, list_679752)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679748)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 183)
tuple_679754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 183)
# Adding element type (line 183)

# Obtaining an instance of the builtin type 'list' (line 183)
list_679755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 183)
# Adding element type (line 183)
int_679756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 5), list_679755, int_679756)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 5), tuple_679754, list_679755)
# Adding element type (line 183)
str_679757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 12), 'str', 'max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 5), tuple_679754, str_679757)
# Adding element type (line 183)

# Obtaining an instance of the builtin type 'list' (line 183)
list_679758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 183)
# Adding element type (line 183)
float_679759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 19), list_679758, float_679759)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 5), tuple_679754, list_679758)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679754)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 184)
tuple_679760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 184)
# Adding element type (line 184)

# Obtaining an instance of the builtin type 'list' (line 184)
list_679761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 184)
# Adding element type (line 184)
int_679762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 5), list_679761, int_679762)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 5), tuple_679760, list_679761)
# Adding element type (line 184)
str_679763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 12), 'str', 'dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 5), tuple_679760, str_679763)
# Adding element type (line 184)

# Obtaining an instance of the builtin type 'list' (line 184)
list_679764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 184)
# Adding element type (line 184)
float_679765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 22), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 21), list_679764, float_679765)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 5), tuple_679760, list_679764)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679760)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 185)
tuple_679766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 185)
# Adding element type (line 185)

# Obtaining an instance of the builtin type 'list' (line 185)
list_679767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 185)
# Adding element type (line 185)
int_679768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 5), list_679767, int_679768)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 5), tuple_679766, list_679767)
# Adding element type (line 185)
str_679769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 12), 'str', 'ordinal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 5), tuple_679766, str_679769)
# Adding element type (line 185)

# Obtaining an instance of the builtin type 'list' (line 185)
list_679770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 185)
# Adding element type (line 185)
float_679771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 23), list_679770, float_679771)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 5), tuple_679766, list_679770)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679766)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 187)
tuple_679772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 187)
# Adding element type (line 187)

# Obtaining an instance of the builtin type 'list' (line 187)
list_679773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 187)
# Adding element type (line 187)
int_679774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 5), list_679773, int_679774)
# Adding element type (line 187)
int_679775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 5), list_679773, int_679775)
# Adding element type (line 187)
int_679776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 5), list_679773, int_679776)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 5), tuple_679772, list_679773)
# Adding element type (line 187)
str_679777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 22), 'str', 'average')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 5), tuple_679772, str_679777)
# Adding element type (line 187)

# Obtaining an instance of the builtin type 'list' (line 187)
list_679778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 187)
# Adding element type (line 187)
float_679779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 33), list_679778, float_679779)
# Adding element type (line 187)
float_679780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 33), list_679778, float_679780)
# Adding element type (line 187)
float_679781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 33), list_679778, float_679781)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 5), tuple_679772, list_679778)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679772)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 188)
tuple_679782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 188)
# Adding element type (line 188)

# Obtaining an instance of the builtin type 'list' (line 188)
list_679783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 188)
# Adding element type (line 188)
int_679784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 5), list_679783, int_679784)
# Adding element type (line 188)
int_679785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 5), list_679783, int_679785)
# Adding element type (line 188)
int_679786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 5), list_679783, int_679786)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 5), tuple_679782, list_679783)
# Adding element type (line 188)
str_679787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 22), 'str', 'min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 5), tuple_679782, str_679787)
# Adding element type (line 188)

# Obtaining an instance of the builtin type 'list' (line 188)
list_679788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 188)
# Adding element type (line 188)
float_679789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 29), list_679788, float_679789)
# Adding element type (line 188)
float_679790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 29), list_679788, float_679790)
# Adding element type (line 188)
float_679791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 29), list_679788, float_679791)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 5), tuple_679782, list_679788)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679782)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 189)
tuple_679792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 189)
# Adding element type (line 189)

# Obtaining an instance of the builtin type 'list' (line 189)
list_679793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 189)
# Adding element type (line 189)
int_679794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 5), list_679793, int_679794)
# Adding element type (line 189)
int_679795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 5), list_679793, int_679795)
# Adding element type (line 189)
int_679796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 5), list_679793, int_679796)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 5), tuple_679792, list_679793)
# Adding element type (line 189)
str_679797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 22), 'str', 'max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 5), tuple_679792, str_679797)
# Adding element type (line 189)

# Obtaining an instance of the builtin type 'list' (line 189)
list_679798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 189)
# Adding element type (line 189)
float_679799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 29), list_679798, float_679799)
# Adding element type (line 189)
float_679800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 29), list_679798, float_679800)
# Adding element type (line 189)
float_679801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 29), list_679798, float_679801)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 5), tuple_679792, list_679798)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679792)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 190)
tuple_679802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 190)
# Adding element type (line 190)

# Obtaining an instance of the builtin type 'list' (line 190)
list_679803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 190)
# Adding element type (line 190)
int_679804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 5), list_679803, int_679804)
# Adding element type (line 190)
int_679805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 5), list_679803, int_679805)
# Adding element type (line 190)
int_679806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 5), list_679803, int_679806)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 5), tuple_679802, list_679803)
# Adding element type (line 190)
str_679807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 22), 'str', 'dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 5), tuple_679802, str_679807)
# Adding element type (line 190)

# Obtaining an instance of the builtin type 'list' (line 190)
list_679808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 190)
# Adding element type (line 190)
float_679809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 31), list_679808, float_679809)
# Adding element type (line 190)
float_679810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 31), list_679808, float_679810)
# Adding element type (line 190)
float_679811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 31), list_679808, float_679811)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 5), tuple_679802, list_679808)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679802)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 191)
tuple_679812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 191)
# Adding element type (line 191)

# Obtaining an instance of the builtin type 'list' (line 191)
list_679813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 191)
# Adding element type (line 191)
int_679814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 5), list_679813, int_679814)
# Adding element type (line 191)
int_679815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 5), list_679813, int_679815)
# Adding element type (line 191)
int_679816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 5), list_679813, int_679816)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 5), tuple_679812, list_679813)
# Adding element type (line 191)
str_679817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 22), 'str', 'ordinal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 5), tuple_679812, str_679817)
# Adding element type (line 191)

# Obtaining an instance of the builtin type 'list' (line 191)
list_679818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 191)
# Adding element type (line 191)
float_679819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 33), list_679818, float_679819)
# Adding element type (line 191)
float_679820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 33), list_679818, float_679820)
# Adding element type (line 191)
float_679821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 33), list_679818, float_679821)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 5), tuple_679812, list_679818)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679812)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 193)
tuple_679822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 193)
# Adding element type (line 193)

# Obtaining an instance of the builtin type 'list' (line 193)
list_679823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 193)
# Adding element type (line 193)
int_679824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 5), list_679823, int_679824)
# Adding element type (line 193)
int_679825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 5), list_679823, int_679825)
# Adding element type (line 193)
int_679826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 5), list_679823, int_679826)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 5), tuple_679822, list_679823)
# Adding element type (line 193)
str_679827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 22), 'str', 'average')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 5), tuple_679822, str_679827)
# Adding element type (line 193)

# Obtaining an instance of the builtin type 'list' (line 193)
list_679828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 193)
# Adding element type (line 193)
float_679829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 33), list_679828, float_679829)
# Adding element type (line 193)
float_679830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 33), list_679828, float_679830)
# Adding element type (line 193)
float_679831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 33), list_679828, float_679831)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 5), tuple_679822, list_679828)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679822)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 194)
tuple_679832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 194)
# Adding element type (line 194)

# Obtaining an instance of the builtin type 'list' (line 194)
list_679833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 194)
# Adding element type (line 194)
int_679834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 5), list_679833, int_679834)
# Adding element type (line 194)
int_679835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 5), list_679833, int_679835)
# Adding element type (line 194)
int_679836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 5), list_679833, int_679836)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 5), tuple_679832, list_679833)
# Adding element type (line 194)
str_679837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 22), 'str', 'min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 5), tuple_679832, str_679837)
# Adding element type (line 194)

# Obtaining an instance of the builtin type 'list' (line 194)
list_679838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 194)
# Adding element type (line 194)
float_679839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 29), list_679838, float_679839)
# Adding element type (line 194)
float_679840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 29), list_679838, float_679840)
# Adding element type (line 194)
float_679841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 29), list_679838, float_679841)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 5), tuple_679832, list_679838)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679832)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 195)
tuple_679842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 195)
# Adding element type (line 195)

# Obtaining an instance of the builtin type 'list' (line 195)
list_679843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 195)
# Adding element type (line 195)
int_679844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 5), list_679843, int_679844)
# Adding element type (line 195)
int_679845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 5), list_679843, int_679845)
# Adding element type (line 195)
int_679846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 5), list_679843, int_679846)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 5), tuple_679842, list_679843)
# Adding element type (line 195)
str_679847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 22), 'str', 'max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 5), tuple_679842, str_679847)
# Adding element type (line 195)

# Obtaining an instance of the builtin type 'list' (line 195)
list_679848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 195)
# Adding element type (line 195)
float_679849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 29), list_679848, float_679849)
# Adding element type (line 195)
float_679850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 29), list_679848, float_679850)
# Adding element type (line 195)
float_679851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 29), list_679848, float_679851)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 5), tuple_679842, list_679848)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679842)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 196)
tuple_679852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 196)
# Adding element type (line 196)

# Obtaining an instance of the builtin type 'list' (line 196)
list_679853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 196)
# Adding element type (line 196)
int_679854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 5), list_679853, int_679854)
# Adding element type (line 196)
int_679855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 5), list_679853, int_679855)
# Adding element type (line 196)
int_679856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 5), list_679853, int_679856)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 5), tuple_679852, list_679853)
# Adding element type (line 196)
str_679857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 22), 'str', 'dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 5), tuple_679852, str_679857)
# Adding element type (line 196)

# Obtaining an instance of the builtin type 'list' (line 196)
list_679858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 196)
# Adding element type (line 196)
float_679859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 31), list_679858, float_679859)
# Adding element type (line 196)
float_679860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 31), list_679858, float_679860)
# Adding element type (line 196)
float_679861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 31), list_679858, float_679861)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 5), tuple_679852, list_679858)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679852)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 197)
tuple_679862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 197)
# Adding element type (line 197)

# Obtaining an instance of the builtin type 'list' (line 197)
list_679863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 197)
# Adding element type (line 197)
int_679864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 5), list_679863, int_679864)
# Adding element type (line 197)
int_679865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 5), list_679863, int_679865)
# Adding element type (line 197)
int_679866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 5), list_679863, int_679866)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 5), tuple_679862, list_679863)
# Adding element type (line 197)
str_679867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'str', 'ordinal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 5), tuple_679862, str_679867)
# Adding element type (line 197)

# Obtaining an instance of the builtin type 'list' (line 197)
list_679868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 197)
# Adding element type (line 197)
float_679869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 33), list_679868, float_679869)
# Adding element type (line 197)
float_679870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 33), list_679868, float_679870)
# Adding element type (line 197)
float_679871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 33), list_679868, float_679871)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 5), tuple_679862, list_679868)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679862)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 199)
tuple_679872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 199)
# Adding element type (line 199)

# Obtaining an instance of the builtin type 'list' (line 199)
list_679873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 199)
# Adding element type (line 199)
int_679874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 5), list_679873, int_679874)
# Adding element type (line 199)
int_679875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 5), list_679873, int_679875)
# Adding element type (line 199)
int_679876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 5), list_679873, int_679876)
# Adding element type (line 199)
int_679877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 5), list_679873, int_679877)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 5), tuple_679872, list_679873)
# Adding element type (line 199)
str_679878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 27), 'str', 'average')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 5), tuple_679872, str_679878)
# Adding element type (line 199)

# Obtaining an instance of the builtin type 'list' (line 199)
list_679879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 38), 'list')
# Adding type elements to the builtin type 'list' instance (line 199)
# Adding element type (line 199)
float_679880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_679879, float_679880)
# Adding element type (line 199)
float_679881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_679879, float_679881)
# Adding element type (line 199)
float_679882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_679879, float_679882)
# Adding element type (line 199)
float_679883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 38), list_679879, float_679883)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 5), tuple_679872, list_679879)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679872)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 200)
tuple_679884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 200)
# Adding element type (line 200)

# Obtaining an instance of the builtin type 'list' (line 200)
list_679885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 200)
# Adding element type (line 200)
int_679886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 5), list_679885, int_679886)
# Adding element type (line 200)
int_679887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 5), list_679885, int_679887)
# Adding element type (line 200)
int_679888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 5), list_679885, int_679888)
# Adding element type (line 200)
int_679889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 5), list_679885, int_679889)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 5), tuple_679884, list_679885)
# Adding element type (line 200)
str_679890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'str', 'min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 5), tuple_679884, str_679890)
# Adding element type (line 200)

# Obtaining an instance of the builtin type 'list' (line 200)
list_679891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 200)
# Adding element type (line 200)
float_679892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), list_679891, float_679892)
# Adding element type (line 200)
float_679893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), list_679891, float_679893)
# Adding element type (line 200)
float_679894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), list_679891, float_679894)
# Adding element type (line 200)
float_679895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), list_679891, float_679895)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 5), tuple_679884, list_679891)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679884)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 201)
tuple_679896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 201)
# Adding element type (line 201)

# Obtaining an instance of the builtin type 'list' (line 201)
list_679897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 201)
# Adding element type (line 201)
int_679898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), list_679897, int_679898)
# Adding element type (line 201)
int_679899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), list_679897, int_679899)
# Adding element type (line 201)
int_679900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), list_679897, int_679900)
# Adding element type (line 201)
int_679901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), list_679897, int_679901)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), tuple_679896, list_679897)
# Adding element type (line 201)
str_679902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 27), 'str', 'max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), tuple_679896, str_679902)
# Adding element type (line 201)

# Obtaining an instance of the builtin type 'list' (line 201)
list_679903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 201)
# Adding element type (line 201)
float_679904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 34), list_679903, float_679904)
# Adding element type (line 201)
float_679905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 34), list_679903, float_679905)
# Adding element type (line 201)
float_679906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 34), list_679903, float_679906)
# Adding element type (line 201)
float_679907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 34), list_679903, float_679907)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 5), tuple_679896, list_679903)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679896)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 202)
tuple_679908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 202)
# Adding element type (line 202)

# Obtaining an instance of the builtin type 'list' (line 202)
list_679909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 202)
# Adding element type (line 202)
int_679910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 5), list_679909, int_679910)
# Adding element type (line 202)
int_679911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 5), list_679909, int_679911)
# Adding element type (line 202)
int_679912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 5), list_679909, int_679912)
# Adding element type (line 202)
int_679913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 5), list_679909, int_679913)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 5), tuple_679908, list_679909)
# Adding element type (line 202)
str_679914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 27), 'str', 'dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 5), tuple_679908, str_679914)
# Adding element type (line 202)

# Obtaining an instance of the builtin type 'list' (line 202)
list_679915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 202)
# Adding element type (line 202)
float_679916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 36), list_679915, float_679916)
# Adding element type (line 202)
float_679917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 36), list_679915, float_679917)
# Adding element type (line 202)
float_679918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 36), list_679915, float_679918)
# Adding element type (line 202)
float_679919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 52), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 36), list_679915, float_679919)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 5), tuple_679908, list_679915)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679908)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 203)
tuple_679920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 203)
# Adding element type (line 203)

# Obtaining an instance of the builtin type 'list' (line 203)
list_679921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 203)
# Adding element type (line 203)
int_679922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 5), list_679921, int_679922)
# Adding element type (line 203)
int_679923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 5), list_679921, int_679923)
# Adding element type (line 203)
int_679924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 5), list_679921, int_679924)
# Adding element type (line 203)
int_679925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 5), list_679921, int_679925)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 5), tuple_679920, list_679921)
# Adding element type (line 203)
str_679926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 27), 'str', 'ordinal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 5), tuple_679920, str_679926)
# Adding element type (line 203)

# Obtaining an instance of the builtin type 'list' (line 203)
list_679927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 38), 'list')
# Adding type elements to the builtin type 'list' instance (line 203)
# Adding element type (line 203)
float_679928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 38), list_679927, float_679928)
# Adding element type (line 203)
float_679929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 38), list_679927, float_679929)
# Adding element type (line 203)
float_679930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 38), list_679927, float_679930)
# Adding element type (line 203)
float_679931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 38), list_679927, float_679931)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 5), tuple_679920, list_679927)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679920)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 205)
tuple_679932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 205)
# Adding element type (line 205)

# Obtaining an instance of the builtin type 'list' (line 205)
list_679933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 205)
# Adding element type (line 205)
int_679934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 5), list_679933, int_679934)
# Adding element type (line 205)
int_679935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 5), list_679933, int_679935)
# Adding element type (line 205)
int_679936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 5), list_679933, int_679936)
# Adding element type (line 205)
int_679937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 5), list_679933, int_679937)
# Adding element type (line 205)
int_679938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 5), list_679933, int_679938)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 5), tuple_679932, list_679933)
# Adding element type (line 205)
str_679939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 32), 'str', 'average')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 5), tuple_679932, str_679939)
# Adding element type (line 205)

# Obtaining an instance of the builtin type 'list' (line 205)
list_679940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 43), 'list')
# Adding type elements to the builtin type 'list' instance (line 205)
# Adding element type (line 205)
float_679941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 43), list_679940, float_679941)
# Adding element type (line 205)
float_679942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 43), list_679940, float_679942)
# Adding element type (line 205)
float_679943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 43), list_679940, float_679943)
# Adding element type (line 205)
float_679944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 59), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 43), list_679940, float_679944)
# Adding element type (line 205)
float_679945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 64), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 43), list_679940, float_679945)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 5), tuple_679932, list_679940)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679932)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 206)
tuple_679946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 206)
# Adding element type (line 206)

# Obtaining an instance of the builtin type 'list' (line 206)
list_679947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 206)
# Adding element type (line 206)
int_679948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 5), list_679947, int_679948)
# Adding element type (line 206)
int_679949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 5), list_679947, int_679949)
# Adding element type (line 206)
int_679950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 5), list_679947, int_679950)
# Adding element type (line 206)
int_679951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 5), list_679947, int_679951)
# Adding element type (line 206)
int_679952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 5), list_679947, int_679952)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 5), tuple_679946, list_679947)
# Adding element type (line 206)
str_679953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 32), 'str', 'min')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 5), tuple_679946, str_679953)
# Adding element type (line 206)

# Obtaining an instance of the builtin type 'list' (line 206)
list_679954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 206)
# Adding element type (line 206)
float_679955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 39), list_679954, float_679955)
# Adding element type (line 206)
float_679956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 39), list_679954, float_679956)
# Adding element type (line 206)
float_679957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 39), list_679954, float_679957)
# Adding element type (line 206)
float_679958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 39), list_679954, float_679958)
# Adding element type (line 206)
float_679959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 60), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 39), list_679954, float_679959)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 5), tuple_679946, list_679954)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679946)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 207)
tuple_679960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 207)
# Adding element type (line 207)

# Obtaining an instance of the builtin type 'list' (line 207)
list_679961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 207)
# Adding element type (line 207)
int_679962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), list_679961, int_679962)
# Adding element type (line 207)
int_679963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), list_679961, int_679963)
# Adding element type (line 207)
int_679964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), list_679961, int_679964)
# Adding element type (line 207)
int_679965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), list_679961, int_679965)
# Adding element type (line 207)
int_679966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), list_679961, int_679966)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), tuple_679960, list_679961)
# Adding element type (line 207)
str_679967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 32), 'str', 'max')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), tuple_679960, str_679967)
# Adding element type (line 207)

# Obtaining an instance of the builtin type 'list' (line 207)
list_679968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 207)
# Adding element type (line 207)
float_679969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 39), list_679968, float_679969)
# Adding element type (line 207)
float_679970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 39), list_679968, float_679970)
# Adding element type (line 207)
float_679971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 39), list_679968, float_679971)
# Adding element type (line 207)
float_679972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 39), list_679968, float_679972)
# Adding element type (line 207)
float_679973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 60), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 39), list_679968, float_679973)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), tuple_679960, list_679968)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679960)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 208)
tuple_679974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 208)
# Adding element type (line 208)

# Obtaining an instance of the builtin type 'list' (line 208)
list_679975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 208)
# Adding element type (line 208)
int_679976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 5), list_679975, int_679976)
# Adding element type (line 208)
int_679977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 5), list_679975, int_679977)
# Adding element type (line 208)
int_679978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 5), list_679975, int_679978)
# Adding element type (line 208)
int_679979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 5), list_679975, int_679979)
# Adding element type (line 208)
int_679980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 5), list_679975, int_679980)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 5), tuple_679974, list_679975)
# Adding element type (line 208)
str_679981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 32), 'str', 'dense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 5), tuple_679974, str_679981)
# Adding element type (line 208)

# Obtaining an instance of the builtin type 'list' (line 208)
list_679982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 41), 'list')
# Adding type elements to the builtin type 'list' instance (line 208)
# Adding element type (line 208)
float_679983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 41), list_679982, float_679983)
# Adding element type (line 208)
float_679984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 41), list_679982, float_679984)
# Adding element type (line 208)
float_679985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 52), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 41), list_679982, float_679985)
# Adding element type (line 208)
float_679986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 41), list_679982, float_679986)
# Adding element type (line 208)
float_679987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 41), list_679982, float_679987)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 5), tuple_679974, list_679982)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679974)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 209)
tuple_679988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 209)
# Adding element type (line 209)

# Obtaining an instance of the builtin type 'list' (line 209)
list_679989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 209)
# Adding element type (line 209)
int_679990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 5), list_679989, int_679990)
# Adding element type (line 209)
int_679991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 5), list_679989, int_679991)
# Adding element type (line 209)
int_679992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 5), list_679989, int_679992)
# Adding element type (line 209)
int_679993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 5), list_679989, int_679993)
# Adding element type (line 209)
int_679994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 5), list_679989, int_679994)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 5), tuple_679988, list_679989)
# Adding element type (line 209)
str_679995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 32), 'str', 'ordinal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 5), tuple_679988, str_679995)
# Adding element type (line 209)

# Obtaining an instance of the builtin type 'list' (line 209)
list_679996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 43), 'list')
# Adding type elements to the builtin type 'list' instance (line 209)
# Adding element type (line 209)
float_679997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 43), list_679996, float_679997)
# Adding element type (line 209)
float_679998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 43), list_679996, float_679998)
# Adding element type (line 209)
float_679999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 43), list_679996, float_679999)
# Adding element type (line 209)
float_680000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 59), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 43), list_679996, float_680000)
# Adding element type (line 209)
float_680001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 64), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 43), list_679996, float_680001)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 5), tuple_679988, list_679996)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_679988)
# Adding element type (line 175)

# Obtaining an instance of the builtin type 'tuple' (line 211)
tuple_680002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 211)
# Adding element type (line 211)

# Obtaining an instance of the builtin type 'list' (line 211)
list_680003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 211)
# Adding element type (line 211)
int_680004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 6), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 5), list_680003, int_680004)

int_680005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
# Applying the binary operator '*' (line 211)
result_mul_680006 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 5), '*', list_680003, int_680005)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 5), tuple_680002, result_mul_680006)
# Adding element type (line 211)
str_680007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 16), 'str', 'ordinal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 5), tuple_680002, str_680007)
# Adding element type (line 211)

# Call to arange(...): (line 211)
# Processing the call arguments (line 211)
float_680010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 37), 'float')
float_680011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 42), 'float')
# Processing the call keyword arguments (line 211)
kwargs_680012 = {}
# Getting the type of 'np' (line 211)
np_680008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'np', False)
# Obtaining the member 'arange' of a type (line 211)
arange_680009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 27), np_680008, 'arange')
# Calling arange(args, kwargs) (line 211)
arange_call_result_680013 = invoke(stypy.reporting.localization.Localization(__file__, 211, 27), arange_680009, *[float_680010, float_680011], **kwargs_680012)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 5), tuple_680002, arange_call_result_680013)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 4), tuple_679721, tuple_680002)

# Assigning a type to the variable '_cases' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), '_cases', tuple_679721)

@norecursion
def test_cases(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_cases'
    module_type_store = module_type_store.open_function_context('test_cases', 215, 0, False)
    
    # Passed parameters checking function
    test_cases.stypy_localization = localization
    test_cases.stypy_type_of_self = None
    test_cases.stypy_type_store = module_type_store
    test_cases.stypy_function_name = 'test_cases'
    test_cases.stypy_param_names_list = []
    test_cases.stypy_varargs_param_name = None
    test_cases.stypy_kwargs_param_name = None
    test_cases.stypy_call_defaults = defaults
    test_cases.stypy_call_varargs = varargs
    test_cases.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_cases', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_cases', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_cases(...)' code ##################

    
    # Getting the type of '_cases' (line 216)
    _cases_680014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 36), '_cases')
    # Testing the type of a for loop iterable (line 216)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 4), _cases_680014)
    # Getting the type of the for loop variable (line 216)
    for_loop_var_680015 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 4), _cases_680014)
    # Assigning a type to the variable 'values' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'values', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 4), for_loop_var_680015))
    # Assigning a type to the variable 'method' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'method', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 4), for_loop_var_680015))
    # Assigning a type to the variable 'expected' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'expected', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 4), for_loop_var_680015))
    # SSA begins for a for statement (line 216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to rankdata(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'values' (line 217)
    values_680017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'values', False)
    # Processing the call keyword arguments (line 217)
    # Getting the type of 'method' (line 217)
    method_680018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 36), 'method', False)
    keyword_680019 = method_680018
    kwargs_680020 = {'method': keyword_680019}
    # Getting the type of 'rankdata' (line 217)
    rankdata_680016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'rankdata', False)
    # Calling rankdata(args, kwargs) (line 217)
    rankdata_call_result_680021 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), rankdata_680016, *[values_680017], **kwargs_680020)
    
    # Assigning a type to the variable 'r' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'r', rankdata_call_result_680021)
    
    # Call to assert_array_equal(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'r' (line 218)
    r_680023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'r', False)
    # Getting the type of 'expected' (line 218)
    expected_680024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 30), 'expected', False)
    # Processing the call keyword arguments (line 218)
    kwargs_680025 = {}
    # Getting the type of 'assert_array_equal' (line 218)
    assert_array_equal_680022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 218)
    assert_array_equal_call_result_680026 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), assert_array_equal_680022, *[r_680023, expected_680024], **kwargs_680025)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_cases(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_cases' in the type store
    # Getting the type of 'stypy_return_type' (line 215)
    stypy_return_type_680027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_680027)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_cases'
    return stypy_return_type_680027

# Assigning a type to the variable 'test_cases' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'test_cases', test_cases)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

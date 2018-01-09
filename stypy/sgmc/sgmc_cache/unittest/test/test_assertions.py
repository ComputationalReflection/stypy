
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import datetime
2: 
3: import unittest
4: 
5: 
6: class Test_Assertions(unittest.TestCase):
7:     def test_AlmostEqual(self):
8:         self.assertAlmostEqual(1.00000001, 1.0)
9:         self.assertNotAlmostEqual(1.0000001, 1.0)
10:         self.assertRaises(self.failureException,
11:                           self.assertAlmostEqual, 1.0000001, 1.0)
12:         self.assertRaises(self.failureException,
13:                           self.assertNotAlmostEqual, 1.00000001, 1.0)
14: 
15:         self.assertAlmostEqual(1.1, 1.0, places=0)
16:         self.assertRaises(self.failureException,
17:                           self.assertAlmostEqual, 1.1, 1.0, places=1)
18: 
19:         self.assertAlmostEqual(0, .1+.1j, places=0)
20:         self.assertNotAlmostEqual(0, .1+.1j, places=1)
21:         self.assertRaises(self.failureException,
22:                           self.assertAlmostEqual, 0, .1+.1j, places=1)
23:         self.assertRaises(self.failureException,
24:                           self.assertNotAlmostEqual, 0, .1+.1j, places=0)
25: 
26:         self.assertAlmostEqual(float('inf'), float('inf'))
27:         self.assertRaises(self.failureException, self.assertNotAlmostEqual,
28:                           float('inf'), float('inf'))
29: 
30:     def test_AmostEqualWithDelta(self):
31:         self.assertAlmostEqual(1.1, 1.0, delta=0.5)
32:         self.assertAlmostEqual(1.0, 1.1, delta=0.5)
33:         self.assertNotAlmostEqual(1.1, 1.0, delta=0.05)
34:         self.assertNotAlmostEqual(1.0, 1.1, delta=0.05)
35: 
36:         self.assertAlmostEqual(1.0, 1.0, delta=0.5)
37:         self.assertRaises(self.failureException, self.assertNotAlmostEqual,
38:                           1.0, 1.0, delta=0.5)
39: 
40:         self.assertRaises(self.failureException, self.assertAlmostEqual,
41:                           1.1, 1.0, delta=0.05)
42:         self.assertRaises(self.failureException, self.assertNotAlmostEqual,
43:                           1.1, 1.0, delta=0.5)
44: 
45:         self.assertRaises(TypeError, self.assertAlmostEqual,
46:                           1.1, 1.0, places=2, delta=2)
47:         self.assertRaises(TypeError, self.assertNotAlmostEqual,
48:                           1.1, 1.0, places=2, delta=2)
49: 
50:         first = datetime.datetime.now()
51:         second = first + datetime.timedelta(seconds=10)
52:         self.assertAlmostEqual(first, second,
53:                                delta=datetime.timedelta(seconds=20))
54:         self.assertNotAlmostEqual(first, second,
55:                                   delta=datetime.timedelta(seconds=5))
56: 
57:     def test_assertRaises(self):
58:         def _raise(e):
59:             raise e
60:         self.assertRaises(KeyError, _raise, KeyError)
61:         self.assertRaises(KeyError, _raise, KeyError("key"))
62:         try:
63:             self.assertRaises(KeyError, lambda: None)
64:         except self.failureException as e:
65:             self.assertIn("KeyError not raised", e.args)
66:         else:
67:             self.fail("assertRaises() didn't fail")
68:         try:
69:             self.assertRaises(KeyError, _raise, ValueError)
70:         except ValueError:
71:             pass
72:         else:
73:             self.fail("assertRaises() didn't let exception pass through")
74:         with self.assertRaises(KeyError) as cm:
75:             try:
76:                 raise KeyError
77:             except Exception, e:
78:                 raise
79:         self.assertIs(cm.exception, e)
80: 
81:         with self.assertRaises(KeyError):
82:             raise KeyError("key")
83:         try:
84:             with self.assertRaises(KeyError):
85:                 pass
86:         except self.failureException as e:
87:             self.assertIn("KeyError not raised", e.args)
88:         else:
89:             self.fail("assertRaises() didn't fail")
90:         try:
91:             with self.assertRaises(KeyError):
92:                 raise ValueError
93:         except ValueError:
94:             pass
95:         else:
96:             self.fail("assertRaises() didn't let exception pass through")
97: 
98:     def testAssertNotRegexpMatches(self):
99:         self.assertNotRegexpMatches('Ala ma kota', r'r+')
100:         try:
101:             self.assertNotRegexpMatches('Ala ma kota', r'k.t', 'Message')
102:         except self.failureException, e:
103:             self.assertIn("'kot'", e.args[0])
104:             self.assertIn('Message', e.args[0])
105:         else:
106:             self.fail('assertNotRegexpMatches should have failed.')
107: 
108: 
109: class TestLongMessage(unittest.TestCase):
110:     '''Test that the individual asserts honour longMessage.
111:     This actually tests all the message behaviour for
112:     asserts that use longMessage.'''
113: 
114:     def setUp(self):
115:         class TestableTestFalse(unittest.TestCase):
116:             longMessage = False
117:             failureException = self.failureException
118: 
119:             def testTest(self):
120:                 pass
121: 
122:         class TestableTestTrue(unittest.TestCase):
123:             longMessage = True
124:             failureException = self.failureException
125: 
126:             def testTest(self):
127:                 pass
128: 
129:         self.testableTrue = TestableTestTrue('testTest')
130:         self.testableFalse = TestableTestFalse('testTest')
131: 
132:     def testDefault(self):
133:         self.assertFalse(unittest.TestCase.longMessage)
134: 
135:     def test_formatMsg(self):
136:         self.assertEqual(self.testableFalse._formatMessage(None, "foo"), "foo")
137:         self.assertEqual(self.testableFalse._formatMessage("foo", "bar"), "foo")
138: 
139:         self.assertEqual(self.testableTrue._formatMessage(None, "foo"), "foo")
140:         self.assertEqual(self.testableTrue._formatMessage("foo", "bar"), "bar : foo")
141: 
142:         # This blows up if _formatMessage uses string concatenation
143:         self.testableTrue._formatMessage(object(), 'foo')
144: 
145:     def test_formatMessage_unicode_error(self):
146:         one = ''.join(chr(i) for i in range(255))
147:         # this used to cause a UnicodeDecodeError constructing msg
148:         self.testableTrue._formatMessage(one, u'\uFFFD')
149: 
150:     def assertMessages(self, methodName, args, errors):
151:         def getMethod(i):
152:             useTestableFalse  = i < 2
153:             if useTestableFalse:
154:                 test = self.testableFalse
155:             else:
156:                 test = self.testableTrue
157:             return getattr(test, methodName)
158: 
159:         for i, expected_regexp in enumerate(errors):
160:             testMethod = getMethod(i)
161:             kwargs = {}
162:             withMsg = i % 2
163:             if withMsg:
164:                 kwargs = {"msg": "oops"}
165: 
166:             with self.assertRaisesRegexp(self.failureException,
167:                                          expected_regexp=expected_regexp):
168:                 testMethod(*args, **kwargs)
169: 
170:     def testAssertTrue(self):
171:         self.assertMessages('assertTrue', (False,),
172:                             ["^False is not true$", "^oops$", "^False is not true$",
173:                              "^False is not true : oops$"])
174: 
175:     def testAssertFalse(self):
176:         self.assertMessages('assertFalse', (True,),
177:                             ["^True is not false$", "^oops$", "^True is not false$",
178:                              "^True is not false : oops$"])
179: 
180:     def testNotEqual(self):
181:         self.assertMessages('assertNotEqual', (1, 1),
182:                             ["^1 == 1$", "^oops$", "^1 == 1$",
183:                              "^1 == 1 : oops$"])
184: 
185:     def testAlmostEqual(self):
186:         self.assertMessages('assertAlmostEqual', (1, 2),
187:                             ["^1 != 2 within 7 places$", "^oops$",
188:                              "^1 != 2 within 7 places$", "^1 != 2 within 7 places : oops$"])
189: 
190:     def testNotAlmostEqual(self):
191:         self.assertMessages('assertNotAlmostEqual', (1, 1),
192:                             ["^1 == 1 within 7 places$", "^oops$",
193:                              "^1 == 1 within 7 places$", "^1 == 1 within 7 places : oops$"])
194: 
195:     def test_baseAssertEqual(self):
196:         self.assertMessages('_baseAssertEqual', (1, 2),
197:                             ["^1 != 2$", "^oops$", "^1 != 2$", "^1 != 2 : oops$"])
198: 
199:     def testAssertSequenceEqual(self):
200:         # Error messages are multiline so not testing on full message
201:         # assertTupleEqual and assertListEqual delegate to this method
202:         self.assertMessages('assertSequenceEqual', ([], [None]),
203:                             ["\+ \[None\]$", "^oops$", r"\+ \[None\]$",
204:                              r"\+ \[None\] : oops$"])
205: 
206:     def testAssertSetEqual(self):
207:         self.assertMessages('assertSetEqual', (set(), set([None])),
208:                             ["None$", "^oops$", "None$",
209:                              "None : oops$"])
210: 
211:     def testAssertIn(self):
212:         self.assertMessages('assertIn', (None, []),
213:                             ['^None not found in \[\]$', "^oops$",
214:                              '^None not found in \[\]$',
215:                              '^None not found in \[\] : oops$'])
216: 
217:     def testAssertNotIn(self):
218:         self.assertMessages('assertNotIn', (None, [None]),
219:                             ['^None unexpectedly found in \[None\]$', "^oops$",
220:                              '^None unexpectedly found in \[None\]$',
221:                              '^None unexpectedly found in \[None\] : oops$'])
222: 
223:     def testAssertDictEqual(self):
224:         self.assertMessages('assertDictEqual', ({}, {'key': 'value'}),
225:                             [r"\+ \{'key': 'value'\}$", "^oops$",
226:                              "\+ \{'key': 'value'\}$",
227:                              "\+ \{'key': 'value'\} : oops$"])
228: 
229:     def testAssertDictContainsSubset(self):
230:         self.assertMessages('assertDictContainsSubset', ({'key': 'value'}, {}),
231:                             ["^Missing: 'key'$", "^oops$",
232:                              "^Missing: 'key'$",
233:                              "^Missing: 'key' : oops$"])
234: 
235:     def testAssertMultiLineEqual(self):
236:         self.assertMessages('assertMultiLineEqual', ("", "foo"),
237:                             [r"\+ foo$", "^oops$",
238:                              r"\+ foo$",
239:                              r"\+ foo : oops$"])
240: 
241:     def testAssertLess(self):
242:         self.assertMessages('assertLess', (2, 1),
243:                             ["^2 not less than 1$", "^oops$",
244:                              "^2 not less than 1$", "^2 not less than 1 : oops$"])
245: 
246:     def testAssertLessEqual(self):
247:         self.assertMessages('assertLessEqual', (2, 1),
248:                             ["^2 not less than or equal to 1$", "^oops$",
249:                              "^2 not less than or equal to 1$",
250:                              "^2 not less than or equal to 1 : oops$"])
251: 
252:     def testAssertGreater(self):
253:         self.assertMessages('assertGreater', (1, 2),
254:                             ["^1 not greater than 2$", "^oops$",
255:                              "^1 not greater than 2$",
256:                              "^1 not greater than 2 : oops$"])
257: 
258:     def testAssertGreaterEqual(self):
259:         self.assertMessages('assertGreaterEqual', (1, 2),
260:                             ["^1 not greater than or equal to 2$", "^oops$",
261:                              "^1 not greater than or equal to 2$",
262:                              "^1 not greater than or equal to 2 : oops$"])
263: 
264:     def testAssertIsNone(self):
265:         self.assertMessages('assertIsNone', ('not None',),
266:                             ["^'not None' is not None$", "^oops$",
267:                              "^'not None' is not None$",
268:                              "^'not None' is not None : oops$"])
269: 
270:     def testAssertIsNotNone(self):
271:         self.assertMessages('assertIsNotNone', (None,),
272:                             ["^unexpectedly None$", "^oops$",
273:                              "^unexpectedly None$",
274:                              "^unexpectedly None : oops$"])
275: 
276:     def testAssertIs(self):
277:         self.assertMessages('assertIs', (None, 'foo'),
278:                             ["^None is not 'foo'$", "^oops$",
279:                              "^None is not 'foo'$",
280:                              "^None is not 'foo' : oops$"])
281: 
282:     def testAssertIsNot(self):
283:         self.assertMessages('assertIsNot', (None, None),
284:                             ["^unexpectedly identical: None$", "^oops$",
285:                              "^unexpectedly identical: None$",
286:                              "^unexpectedly identical: None : oops$"])
287: 
288: 
289: if __name__ == '__main__':
290:     unittest.main()
291: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import datetime' statement (line 1)
import datetime

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'datetime', datetime, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)

# Declaration of the 'Test_Assertions' class
# Getting the type of 'unittest' (line 6)
unittest_193409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 22), 'unittest')
# Obtaining the member 'TestCase' of a type (line 6)
TestCase_193410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 22), unittest_193409, 'TestCase')

class Test_Assertions(TestCase_193410, ):

    @norecursion
    def test_AlmostEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_AlmostEqual'
        module_type_store = module_type_store.open_function_context('test_AlmostEqual', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_localization', localization)
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_function_name', 'Test_Assertions.test_AlmostEqual')
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Assertions.test_AlmostEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Assertions.test_AlmostEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_AlmostEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_AlmostEqual(...)' code ##################

        
        # Call to assertAlmostEqual(...): (line 8)
        # Processing the call arguments (line 8)
        float_193413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 31), 'float')
        float_193414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 43), 'float')
        # Processing the call keyword arguments (line 8)
        kwargs_193415 = {}
        # Getting the type of 'self' (line 8)
        self_193411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 8)
        assertAlmostEqual_193412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_193411, 'assertAlmostEqual')
        # Calling assertAlmostEqual(args, kwargs) (line 8)
        assertAlmostEqual_call_result_193416 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), assertAlmostEqual_193412, *[float_193413, float_193414], **kwargs_193415)
        
        
        # Call to assertNotAlmostEqual(...): (line 9)
        # Processing the call arguments (line 9)
        float_193419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 34), 'float')
        float_193420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 45), 'float')
        # Processing the call keyword arguments (line 9)
        kwargs_193421 = {}
        # Getting the type of 'self' (line 9)
        self_193417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 9)
        assertNotAlmostEqual_193418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), self_193417, 'assertNotAlmostEqual')
        # Calling assertNotAlmostEqual(args, kwargs) (line 9)
        assertNotAlmostEqual_call_result_193422 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), assertNotAlmostEqual_193418, *[float_193419, float_193420], **kwargs_193421)
        
        
        # Call to assertRaises(...): (line 10)
        # Processing the call arguments (line 10)
        # Getting the type of 'self' (line 10)
        self_193425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 10)
        failureException_193426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 26), self_193425, 'failureException')
        # Getting the type of 'self' (line 11)
        self_193427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 11)
        assertAlmostEqual_193428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 26), self_193427, 'assertAlmostEqual')
        float_193429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 50), 'float')
        float_193430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 61), 'float')
        # Processing the call keyword arguments (line 10)
        kwargs_193431 = {}
        # Getting the type of 'self' (line 10)
        self_193423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 10)
        assertRaises_193424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), self_193423, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 10)
        assertRaises_call_result_193432 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), assertRaises_193424, *[failureException_193426, assertAlmostEqual_193428, float_193429, float_193430], **kwargs_193431)
        
        
        # Call to assertRaises(...): (line 12)
        # Processing the call arguments (line 12)
        # Getting the type of 'self' (line 12)
        self_193435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 12)
        failureException_193436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 26), self_193435, 'failureException')
        # Getting the type of 'self' (line 13)
        self_193437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 26), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 13)
        assertNotAlmostEqual_193438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 26), self_193437, 'assertNotAlmostEqual')
        float_193439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 53), 'float')
        float_193440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 65), 'float')
        # Processing the call keyword arguments (line 12)
        kwargs_193441 = {}
        # Getting the type of 'self' (line 12)
        self_193433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 12)
        assertRaises_193434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), self_193433, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 12)
        assertRaises_call_result_193442 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), assertRaises_193434, *[failureException_193436, assertNotAlmostEqual_193438, float_193439, float_193440], **kwargs_193441)
        
        
        # Call to assertAlmostEqual(...): (line 15)
        # Processing the call arguments (line 15)
        float_193445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 31), 'float')
        float_193446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 36), 'float')
        # Processing the call keyword arguments (line 15)
        int_193447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'int')
        keyword_193448 = int_193447
        kwargs_193449 = {'places': keyword_193448}
        # Getting the type of 'self' (line 15)
        self_193443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 15)
        assertAlmostEqual_193444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_193443, 'assertAlmostEqual')
        # Calling assertAlmostEqual(args, kwargs) (line 15)
        assertAlmostEqual_call_result_193450 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), assertAlmostEqual_193444, *[float_193445, float_193446], **kwargs_193449)
        
        
        # Call to assertRaises(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'self' (line 16)
        self_193453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 16)
        failureException_193454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 26), self_193453, 'failureException')
        # Getting the type of 'self' (line 17)
        self_193455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 26), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 17)
        assertAlmostEqual_193456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 26), self_193455, 'assertAlmostEqual')
        float_193457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 50), 'float')
        float_193458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 55), 'float')
        # Processing the call keyword arguments (line 16)
        int_193459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 67), 'int')
        keyword_193460 = int_193459
        kwargs_193461 = {'places': keyword_193460}
        # Getting the type of 'self' (line 16)
        self_193451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 16)
        assertRaises_193452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_193451, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 16)
        assertRaises_call_result_193462 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), assertRaises_193452, *[failureException_193454, assertAlmostEqual_193456, float_193457, float_193458], **kwargs_193461)
        
        
        # Call to assertAlmostEqual(...): (line 19)
        # Processing the call arguments (line 19)
        int_193465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
        float_193466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'float')
        complex_193467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 37), 'complex')
        # Applying the binary operator '+' (line 19)
        result_add_193468 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 34), '+', float_193466, complex_193467)
        
        # Processing the call keyword arguments (line 19)
        int_193469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 49), 'int')
        keyword_193470 = int_193469
        kwargs_193471 = {'places': keyword_193470}
        # Getting the type of 'self' (line 19)
        self_193463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 19)
        assertAlmostEqual_193464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_193463, 'assertAlmostEqual')
        # Calling assertAlmostEqual(args, kwargs) (line 19)
        assertAlmostEqual_call_result_193472 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), assertAlmostEqual_193464, *[int_193465, result_add_193468], **kwargs_193471)
        
        
        # Call to assertNotAlmostEqual(...): (line 20)
        # Processing the call arguments (line 20)
        int_193475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'int')
        float_193476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 37), 'float')
        complex_193477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 40), 'complex')
        # Applying the binary operator '+' (line 20)
        result_add_193478 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 37), '+', float_193476, complex_193477)
        
        # Processing the call keyword arguments (line 20)
        int_193479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 52), 'int')
        keyword_193480 = int_193479
        kwargs_193481 = {'places': keyword_193480}
        # Getting the type of 'self' (line 20)
        self_193473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 20)
        assertNotAlmostEqual_193474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_193473, 'assertNotAlmostEqual')
        # Calling assertNotAlmostEqual(args, kwargs) (line 20)
        assertNotAlmostEqual_call_result_193482 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), assertNotAlmostEqual_193474, *[int_193475, result_add_193478], **kwargs_193481)
        
        
        # Call to assertRaises(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'self' (line 21)
        self_193485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 21)
        failureException_193486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 26), self_193485, 'failureException')
        # Getting the type of 'self' (line 22)
        self_193487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 22)
        assertAlmostEqual_193488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 26), self_193487, 'assertAlmostEqual')
        int_193489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 50), 'int')
        float_193490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 53), 'float')
        complex_193491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 56), 'complex')
        # Applying the binary operator '+' (line 22)
        result_add_193492 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 53), '+', float_193490, complex_193491)
        
        # Processing the call keyword arguments (line 21)
        int_193493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 68), 'int')
        keyword_193494 = int_193493
        kwargs_193495 = {'places': keyword_193494}
        # Getting the type of 'self' (line 21)
        self_193483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 21)
        assertRaises_193484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_193483, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 21)
        assertRaises_call_result_193496 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assertRaises_193484, *[failureException_193486, assertAlmostEqual_193488, int_193489, result_add_193492], **kwargs_193495)
        
        
        # Call to assertRaises(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'self' (line 23)
        self_193499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 23)
        failureException_193500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 26), self_193499, 'failureException')
        # Getting the type of 'self' (line 24)
        self_193501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 24)
        assertNotAlmostEqual_193502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 26), self_193501, 'assertNotAlmostEqual')
        int_193503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 53), 'int')
        float_193504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 56), 'float')
        complex_193505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 59), 'complex')
        # Applying the binary operator '+' (line 24)
        result_add_193506 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 56), '+', float_193504, complex_193505)
        
        # Processing the call keyword arguments (line 23)
        int_193507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 71), 'int')
        keyword_193508 = int_193507
        kwargs_193509 = {'places': keyword_193508}
        # Getting the type of 'self' (line 23)
        self_193497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 23)
        assertRaises_193498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_193497, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 23)
        assertRaises_call_result_193510 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), assertRaises_193498, *[failureException_193500, assertNotAlmostEqual_193502, int_193503, result_add_193506], **kwargs_193509)
        
        
        # Call to assertAlmostEqual(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to float(...): (line 26)
        # Processing the call arguments (line 26)
        str_193514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 37), 'str', 'inf')
        # Processing the call keyword arguments (line 26)
        kwargs_193515 = {}
        # Getting the type of 'float' (line 26)
        float_193513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'float', False)
        # Calling float(args, kwargs) (line 26)
        float_call_result_193516 = invoke(stypy.reporting.localization.Localization(__file__, 26, 31), float_193513, *[str_193514], **kwargs_193515)
        
        
        # Call to float(...): (line 26)
        # Processing the call arguments (line 26)
        str_193518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 51), 'str', 'inf')
        # Processing the call keyword arguments (line 26)
        kwargs_193519 = {}
        # Getting the type of 'float' (line 26)
        float_193517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 45), 'float', False)
        # Calling float(args, kwargs) (line 26)
        float_call_result_193520 = invoke(stypy.reporting.localization.Localization(__file__, 26, 45), float_193517, *[str_193518], **kwargs_193519)
        
        # Processing the call keyword arguments (line 26)
        kwargs_193521 = {}
        # Getting the type of 'self' (line 26)
        self_193511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 26)
        assertAlmostEqual_193512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_193511, 'assertAlmostEqual')
        # Calling assertAlmostEqual(args, kwargs) (line 26)
        assertAlmostEqual_call_result_193522 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), assertAlmostEqual_193512, *[float_call_result_193516, float_call_result_193520], **kwargs_193521)
        
        
        # Call to assertRaises(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_193525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 27)
        failureException_193526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 26), self_193525, 'failureException')
        # Getting the type of 'self' (line 27)
        self_193527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 49), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 27)
        assertNotAlmostEqual_193528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 49), self_193527, 'assertNotAlmostEqual')
        
        # Call to float(...): (line 28)
        # Processing the call arguments (line 28)
        str_193530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'str', 'inf')
        # Processing the call keyword arguments (line 28)
        kwargs_193531 = {}
        # Getting the type of 'float' (line 28)
        float_193529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'float', False)
        # Calling float(args, kwargs) (line 28)
        float_call_result_193532 = invoke(stypy.reporting.localization.Localization(__file__, 28, 26), float_193529, *[str_193530], **kwargs_193531)
        
        
        # Call to float(...): (line 28)
        # Processing the call arguments (line 28)
        str_193534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 46), 'str', 'inf')
        # Processing the call keyword arguments (line 28)
        kwargs_193535 = {}
        # Getting the type of 'float' (line 28)
        float_193533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'float', False)
        # Calling float(args, kwargs) (line 28)
        float_call_result_193536 = invoke(stypy.reporting.localization.Localization(__file__, 28, 40), float_193533, *[str_193534], **kwargs_193535)
        
        # Processing the call keyword arguments (line 27)
        kwargs_193537 = {}
        # Getting the type of 'self' (line 27)
        self_193523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 27)
        assertRaises_193524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_193523, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 27)
        assertRaises_call_result_193538 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assertRaises_193524, *[failureException_193526, assertNotAlmostEqual_193528, float_call_result_193532, float_call_result_193536], **kwargs_193537)
        
        
        # ################# End of 'test_AlmostEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_AlmostEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_193539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193539)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_AlmostEqual'
        return stypy_return_type_193539


    @norecursion
    def test_AmostEqualWithDelta(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_AmostEqualWithDelta'
        module_type_store = module_type_store.open_function_context('test_AmostEqualWithDelta', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_localization', localization)
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_function_name', 'Test_Assertions.test_AmostEqualWithDelta')
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Assertions.test_AmostEqualWithDelta.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Assertions.test_AmostEqualWithDelta', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_AmostEqualWithDelta', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_AmostEqualWithDelta(...)' code ##################

        
        # Call to assertAlmostEqual(...): (line 31)
        # Processing the call arguments (line 31)
        float_193542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'float')
        float_193543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'float')
        # Processing the call keyword arguments (line 31)
        float_193544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 47), 'float')
        keyword_193545 = float_193544
        kwargs_193546 = {'delta': keyword_193545}
        # Getting the type of 'self' (line 31)
        self_193540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 31)
        assertAlmostEqual_193541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_193540, 'assertAlmostEqual')
        # Calling assertAlmostEqual(args, kwargs) (line 31)
        assertAlmostEqual_call_result_193547 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assertAlmostEqual_193541, *[float_193542, float_193543], **kwargs_193546)
        
        
        # Call to assertAlmostEqual(...): (line 32)
        # Processing the call arguments (line 32)
        float_193550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'float')
        float_193551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'float')
        # Processing the call keyword arguments (line 32)
        float_193552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 47), 'float')
        keyword_193553 = float_193552
        kwargs_193554 = {'delta': keyword_193553}
        # Getting the type of 'self' (line 32)
        self_193548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 32)
        assertAlmostEqual_193549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_193548, 'assertAlmostEqual')
        # Calling assertAlmostEqual(args, kwargs) (line 32)
        assertAlmostEqual_call_result_193555 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), assertAlmostEqual_193549, *[float_193550, float_193551], **kwargs_193554)
        
        
        # Call to assertNotAlmostEqual(...): (line 33)
        # Processing the call arguments (line 33)
        float_193558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'float')
        float_193559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 39), 'float')
        # Processing the call keyword arguments (line 33)
        float_193560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 50), 'float')
        keyword_193561 = float_193560
        kwargs_193562 = {'delta': keyword_193561}
        # Getting the type of 'self' (line 33)
        self_193556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 33)
        assertNotAlmostEqual_193557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_193556, 'assertNotAlmostEqual')
        # Calling assertNotAlmostEqual(args, kwargs) (line 33)
        assertNotAlmostEqual_call_result_193563 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assertNotAlmostEqual_193557, *[float_193558, float_193559], **kwargs_193562)
        
        
        # Call to assertNotAlmostEqual(...): (line 34)
        # Processing the call arguments (line 34)
        float_193566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'float')
        float_193567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 39), 'float')
        # Processing the call keyword arguments (line 34)
        float_193568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 50), 'float')
        keyword_193569 = float_193568
        kwargs_193570 = {'delta': keyword_193569}
        # Getting the type of 'self' (line 34)
        self_193564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 34)
        assertNotAlmostEqual_193565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_193564, 'assertNotAlmostEqual')
        # Calling assertNotAlmostEqual(args, kwargs) (line 34)
        assertNotAlmostEqual_call_result_193571 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assertNotAlmostEqual_193565, *[float_193566, float_193567], **kwargs_193570)
        
        
        # Call to assertAlmostEqual(...): (line 36)
        # Processing the call arguments (line 36)
        float_193574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'float')
        float_193575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'float')
        # Processing the call keyword arguments (line 36)
        float_193576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 47), 'float')
        keyword_193577 = float_193576
        kwargs_193578 = {'delta': keyword_193577}
        # Getting the type of 'self' (line 36)
        self_193572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 36)
        assertAlmostEqual_193573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_193572, 'assertAlmostEqual')
        # Calling assertAlmostEqual(args, kwargs) (line 36)
        assertAlmostEqual_call_result_193579 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assertAlmostEqual_193573, *[float_193574, float_193575], **kwargs_193578)
        
        
        # Call to assertRaises(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'self' (line 37)
        self_193582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 37)
        failureException_193583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 26), self_193582, 'failureException')
        # Getting the type of 'self' (line 37)
        self_193584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 49), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 37)
        assertNotAlmostEqual_193585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 49), self_193584, 'assertNotAlmostEqual')
        float_193586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'float')
        float_193587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 31), 'float')
        # Processing the call keyword arguments (line 37)
        float_193588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 42), 'float')
        keyword_193589 = float_193588
        kwargs_193590 = {'delta': keyword_193589}
        # Getting the type of 'self' (line 37)
        self_193580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 37)
        assertRaises_193581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_193580, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 37)
        assertRaises_call_result_193591 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assertRaises_193581, *[failureException_193583, assertNotAlmostEqual_193585, float_193586, float_193587], **kwargs_193590)
        
        
        # Call to assertRaises(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_193594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 40)
        failureException_193595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 26), self_193594, 'failureException')
        # Getting the type of 'self' (line 40)
        self_193596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 49), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 40)
        assertAlmostEqual_193597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 49), self_193596, 'assertAlmostEqual')
        float_193598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'float')
        float_193599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'float')
        # Processing the call keyword arguments (line 40)
        float_193600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 42), 'float')
        keyword_193601 = float_193600
        kwargs_193602 = {'delta': keyword_193601}
        # Getting the type of 'self' (line 40)
        self_193592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 40)
        assertRaises_193593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_193592, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 40)
        assertRaises_call_result_193603 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), assertRaises_193593, *[failureException_193595, assertAlmostEqual_193597, float_193598, float_193599], **kwargs_193602)
        
        
        # Call to assertRaises(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_193606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'self', False)
        # Obtaining the member 'failureException' of a type (line 42)
        failureException_193607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 26), self_193606, 'failureException')
        # Getting the type of 'self' (line 42)
        self_193608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 49), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 42)
        assertNotAlmostEqual_193609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 49), self_193608, 'assertNotAlmostEqual')
        float_193610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'float')
        float_193611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'float')
        # Processing the call keyword arguments (line 42)
        float_193612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 42), 'float')
        keyword_193613 = float_193612
        kwargs_193614 = {'delta': keyword_193613}
        # Getting the type of 'self' (line 42)
        self_193604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 42)
        assertRaises_193605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_193604, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 42)
        assertRaises_call_result_193615 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assertRaises_193605, *[failureException_193607, assertNotAlmostEqual_193609, float_193610, float_193611], **kwargs_193614)
        
        
        # Call to assertRaises(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'TypeError' (line 45)
        TypeError_193618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'TypeError', False)
        # Getting the type of 'self' (line 45)
        self_193619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 37), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 45)
        assertAlmostEqual_193620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 37), self_193619, 'assertAlmostEqual')
        float_193621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'float')
        float_193622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 31), 'float')
        # Processing the call keyword arguments (line 45)
        int_193623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 43), 'int')
        keyword_193624 = int_193623
        int_193625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 52), 'int')
        keyword_193626 = int_193625
        kwargs_193627 = {'places': keyword_193624, 'delta': keyword_193626}
        # Getting the type of 'self' (line 45)
        self_193616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 45)
        assertRaises_193617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_193616, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 45)
        assertRaises_call_result_193628 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assertRaises_193617, *[TypeError_193618, assertAlmostEqual_193620, float_193621, float_193622], **kwargs_193627)
        
        
        # Call to assertRaises(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'TypeError' (line 47)
        TypeError_193631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'TypeError', False)
        # Getting the type of 'self' (line 47)
        self_193632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 47)
        assertNotAlmostEqual_193633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 37), self_193632, 'assertNotAlmostEqual')
        float_193634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 26), 'float')
        float_193635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 31), 'float')
        # Processing the call keyword arguments (line 47)
        int_193636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 43), 'int')
        keyword_193637 = int_193636
        int_193638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 52), 'int')
        keyword_193639 = int_193638
        kwargs_193640 = {'places': keyword_193637, 'delta': keyword_193639}
        # Getting the type of 'self' (line 47)
        self_193629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 47)
        assertRaises_193630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_193629, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 47)
        assertRaises_call_result_193641 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assertRaises_193630, *[TypeError_193631, assertNotAlmostEqual_193633, float_193634, float_193635], **kwargs_193640)
        
        
        # Assigning a Call to a Name (line 50):
        
        # Call to now(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_193645 = {}
        # Getting the type of 'datetime' (line 50)
        datetime_193642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'datetime', False)
        # Obtaining the member 'datetime' of a type (line 50)
        datetime_193643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 16), datetime_193642, 'datetime')
        # Obtaining the member 'now' of a type (line 50)
        now_193644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 16), datetime_193643, 'now')
        # Calling now(args, kwargs) (line 50)
        now_call_result_193646 = invoke(stypy.reporting.localization.Localization(__file__, 50, 16), now_193644, *[], **kwargs_193645)
        
        # Assigning a type to the variable 'first' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'first', now_call_result_193646)
        
        # Assigning a BinOp to a Name (line 51):
        # Getting the type of 'first' (line 51)
        first_193647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'first')
        
        # Call to timedelta(...): (line 51)
        # Processing the call keyword arguments (line 51)
        int_193650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 52), 'int')
        keyword_193651 = int_193650
        kwargs_193652 = {'seconds': keyword_193651}
        # Getting the type of 'datetime' (line 51)
        datetime_193648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 51)
        timedelta_193649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), datetime_193648, 'timedelta')
        # Calling timedelta(args, kwargs) (line 51)
        timedelta_call_result_193653 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), timedelta_193649, *[], **kwargs_193652)
        
        # Applying the binary operator '+' (line 51)
        result_add_193654 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 17), '+', first_193647, timedelta_call_result_193653)
        
        # Assigning a type to the variable 'second' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'second', result_add_193654)
        
        # Call to assertAlmostEqual(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'first' (line 52)
        first_193657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'first', False)
        # Getting the type of 'second' (line 52)
        second_193658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'second', False)
        # Processing the call keyword arguments (line 52)
        
        # Call to timedelta(...): (line 53)
        # Processing the call keyword arguments (line 53)
        int_193661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 64), 'int')
        keyword_193662 = int_193661
        kwargs_193663 = {'seconds': keyword_193662}
        # Getting the type of 'datetime' (line 53)
        datetime_193659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 53)
        timedelta_193660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 37), datetime_193659, 'timedelta')
        # Calling timedelta(args, kwargs) (line 53)
        timedelta_call_result_193664 = invoke(stypy.reporting.localization.Localization(__file__, 53, 37), timedelta_193660, *[], **kwargs_193663)
        
        keyword_193665 = timedelta_call_result_193664
        kwargs_193666 = {'delta': keyword_193665}
        # Getting the type of 'self' (line 52)
        self_193655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'assertAlmostEqual' of a type (line 52)
        assertAlmostEqual_193656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_193655, 'assertAlmostEqual')
        # Calling assertAlmostEqual(args, kwargs) (line 52)
        assertAlmostEqual_call_result_193667 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assertAlmostEqual_193656, *[first_193657, second_193658], **kwargs_193666)
        
        
        # Call to assertNotAlmostEqual(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'first' (line 54)
        first_193670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'first', False)
        # Getting the type of 'second' (line 54)
        second_193671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 41), 'second', False)
        # Processing the call keyword arguments (line 54)
        
        # Call to timedelta(...): (line 55)
        # Processing the call keyword arguments (line 55)
        int_193674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 67), 'int')
        keyword_193675 = int_193674
        kwargs_193676 = {'seconds': keyword_193675}
        # Getting the type of 'datetime' (line 55)
        datetime_193672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'datetime', False)
        # Obtaining the member 'timedelta' of a type (line 55)
        timedelta_193673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 40), datetime_193672, 'timedelta')
        # Calling timedelta(args, kwargs) (line 55)
        timedelta_call_result_193677 = invoke(stypy.reporting.localization.Localization(__file__, 55, 40), timedelta_193673, *[], **kwargs_193676)
        
        keyword_193678 = timedelta_call_result_193677
        kwargs_193679 = {'delta': keyword_193678}
        # Getting the type of 'self' (line 54)
        self_193668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', False)
        # Obtaining the member 'assertNotAlmostEqual' of a type (line 54)
        assertNotAlmostEqual_193669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_193668, 'assertNotAlmostEqual')
        # Calling assertNotAlmostEqual(args, kwargs) (line 54)
        assertNotAlmostEqual_call_result_193680 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assertNotAlmostEqual_193669, *[first_193670, second_193671], **kwargs_193679)
        
        
        # ################# End of 'test_AmostEqualWithDelta(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_AmostEqualWithDelta' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_193681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193681)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_AmostEqualWithDelta'
        return stypy_return_type_193681


    @norecursion
    def test_assertRaises(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_assertRaises'
        module_type_store = module_type_store.open_function_context('test_assertRaises', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_localization', localization)
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_function_name', 'Test_Assertions.test_assertRaises')
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Assertions.test_assertRaises.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Assertions.test_assertRaises', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_assertRaises', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_assertRaises(...)' code ##################


        @norecursion
        def _raise(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_raise'
            module_type_store = module_type_store.open_function_context('_raise', 58, 8, False)
            
            # Passed parameters checking function
            _raise.stypy_localization = localization
            _raise.stypy_type_of_self = None
            _raise.stypy_type_store = module_type_store
            _raise.stypy_function_name = '_raise'
            _raise.stypy_param_names_list = ['e']
            _raise.stypy_varargs_param_name = None
            _raise.stypy_kwargs_param_name = None
            _raise.stypy_call_defaults = defaults
            _raise.stypy_call_varargs = varargs
            _raise.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_raise', ['e'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_raise', localization, ['e'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_raise(...)' code ##################

            # Getting the type of 'e' (line 59)
            e_193682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'e')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 59, 12), e_193682, 'raise parameter', BaseException)
            
            # ################# End of '_raise(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_raise' in the type store
            # Getting the type of 'stypy_return_type' (line 58)
            stypy_return_type_193683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_193683)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_raise'
            return stypy_return_type_193683

        # Assigning a type to the variable '_raise' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), '_raise', _raise)
        
        # Call to assertRaises(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'KeyError' (line 60)
        KeyError_193686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'KeyError', False)
        # Getting the type of '_raise' (line 60)
        _raise_193687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), '_raise', False)
        # Getting the type of 'KeyError' (line 60)
        KeyError_193688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'KeyError', False)
        # Processing the call keyword arguments (line 60)
        kwargs_193689 = {}
        # Getting the type of 'self' (line 60)
        self_193684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 60)
        assertRaises_193685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_193684, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 60)
        assertRaises_call_result_193690 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assertRaises_193685, *[KeyError_193686, _raise_193687, KeyError_193688], **kwargs_193689)
        
        
        # Call to assertRaises(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'KeyError' (line 61)
        KeyError_193693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'KeyError', False)
        # Getting the type of '_raise' (line 61)
        _raise_193694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 36), '_raise', False)
        
        # Call to KeyError(...): (line 61)
        # Processing the call arguments (line 61)
        str_193696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 53), 'str', 'key')
        # Processing the call keyword arguments (line 61)
        kwargs_193697 = {}
        # Getting the type of 'KeyError' (line 61)
        KeyError_193695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 44), 'KeyError', False)
        # Calling KeyError(args, kwargs) (line 61)
        KeyError_call_result_193698 = invoke(stypy.reporting.localization.Localization(__file__, 61, 44), KeyError_193695, *[str_193696], **kwargs_193697)
        
        # Processing the call keyword arguments (line 61)
        kwargs_193699 = {}
        # Getting the type of 'self' (line 61)
        self_193691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 61)
        assertRaises_193692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_193691, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 61)
        assertRaises_call_result_193700 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), assertRaises_193692, *[KeyError_193693, _raise_193694, KeyError_call_result_193698], **kwargs_193699)
        
        
        
        # SSA begins for try-except statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to assertRaises(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'KeyError' (line 63)
        KeyError_193703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'KeyError', False)

        @norecursion
        def _stypy_temp_lambda_54(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_54'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_54', 63, 40, True)
            # Passed parameters checking function
            _stypy_temp_lambda_54.stypy_localization = localization
            _stypy_temp_lambda_54.stypy_type_of_self = None
            _stypy_temp_lambda_54.stypy_type_store = module_type_store
            _stypy_temp_lambda_54.stypy_function_name = '_stypy_temp_lambda_54'
            _stypy_temp_lambda_54.stypy_param_names_list = []
            _stypy_temp_lambda_54.stypy_varargs_param_name = None
            _stypy_temp_lambda_54.stypy_kwargs_param_name = None
            _stypy_temp_lambda_54.stypy_call_defaults = defaults
            _stypy_temp_lambda_54.stypy_call_varargs = varargs
            _stypy_temp_lambda_54.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_54', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_54', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'None' (line 63)
            None_193704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 48), 'None', False)
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'stypy_return_type', None_193704)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_54' in the type store
            # Getting the type of 'stypy_return_type' (line 63)
            stypy_return_type_193705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_193705)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_54'
            return stypy_return_type_193705

        # Assigning a type to the variable '_stypy_temp_lambda_54' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), '_stypy_temp_lambda_54', _stypy_temp_lambda_54)
        # Getting the type of '_stypy_temp_lambda_54' (line 63)
        _stypy_temp_lambda_54_193706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 40), '_stypy_temp_lambda_54')
        # Processing the call keyword arguments (line 63)
        kwargs_193707 = {}
        # Getting the type of 'self' (line 63)
        self_193701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 63)
        assertRaises_193702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_193701, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 63)
        assertRaises_call_result_193708 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), assertRaises_193702, *[KeyError_193703, _stypy_temp_lambda_54_193706], **kwargs_193707)
        
        # SSA branch for the except part of a try statement (line 62)
        # SSA branch for the except 'Attribute' branch of a try statement (line 62)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'self' (line 64)
        self_193709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'self')
        # Obtaining the member 'failureException' of a type (line 64)
        failureException_193710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), self_193709, 'failureException')
        # Assigning a type to the variable 'e' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'e', failureException_193710)
        
        # Call to assertIn(...): (line 65)
        # Processing the call arguments (line 65)
        str_193713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'str', 'KeyError not raised')
        # Getting the type of 'e' (line 65)
        e_193714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 49), 'e', False)
        # Obtaining the member 'args' of a type (line 65)
        args_193715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 49), e_193714, 'args')
        # Processing the call keyword arguments (line 65)
        kwargs_193716 = {}
        # Getting the type of 'self' (line 65)
        self_193711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 65)
        assertIn_193712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), self_193711, 'assertIn')
        # Calling assertIn(args, kwargs) (line 65)
        assertIn_call_result_193717 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), assertIn_193712, *[str_193713, args_193715], **kwargs_193716)
        
        # SSA branch for the else branch of a try statement (line 62)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 67)
        # Processing the call arguments (line 67)
        str_193720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'str', "assertRaises() didn't fail")
        # Processing the call keyword arguments (line 67)
        kwargs_193721 = {}
        # Getting the type of 'self' (line 67)
        self_193718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 67)
        fail_193719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), self_193718, 'fail')
        # Calling fail(args, kwargs) (line 67)
        fail_call_result_193722 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), fail_193719, *[str_193720], **kwargs_193721)
        
        # SSA join for try-except statement (line 62)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to assertRaises(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'KeyError' (line 69)
        KeyError_193725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'KeyError', False)
        # Getting the type of '_raise' (line 69)
        _raise_193726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 40), '_raise', False)
        # Getting the type of 'ValueError' (line 69)
        ValueError_193727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 48), 'ValueError', False)
        # Processing the call keyword arguments (line 69)
        kwargs_193728 = {}
        # Getting the type of 'self' (line 69)
        self_193723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 69)
        assertRaises_193724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), self_193723, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 69)
        assertRaises_call_result_193729 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), assertRaises_193724, *[KeyError_193725, _raise_193726, ValueError_193727], **kwargs_193728)
        
        # SSA branch for the except part of a try statement (line 68)
        # SSA branch for the except 'ValueError' branch of a try statement (line 68)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 68)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 73)
        # Processing the call arguments (line 73)
        str_193732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'str', "assertRaises() didn't let exception pass through")
        # Processing the call keyword arguments (line 73)
        kwargs_193733 = {}
        # Getting the type of 'self' (line 73)
        self_193730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 73)
        fail_193731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 12), self_193730, 'fail')
        # Calling fail(args, kwargs) (line 73)
        fail_call_result_193734 = invoke(stypy.reporting.localization.Localization(__file__, 73, 12), fail_193731, *[str_193732], **kwargs_193733)
        
        # SSA join for try-except statement (line 68)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertRaises(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'KeyError' (line 74)
        KeyError_193737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 31), 'KeyError', False)
        # Processing the call keyword arguments (line 74)
        kwargs_193738 = {}
        # Getting the type of 'self' (line 74)
        self_193735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 74)
        assertRaises_193736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), self_193735, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 74)
        assertRaises_call_result_193739 = invoke(stypy.reporting.localization.Localization(__file__, 74, 13), assertRaises_193736, *[KeyError_193737], **kwargs_193738)
        
        with_193740 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 74, 13), assertRaises_call_result_193739, 'with parameter', '__enter__', '__exit__')

        if with_193740:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 74)
            enter___193741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), assertRaises_call_result_193739, '__enter__')
            with_enter_193742 = invoke(stypy.reporting.localization.Localization(__file__, 74, 13), enter___193741)
            # Assigning a type to the variable 'cm' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'cm', with_enter_193742)
            
            
            # SSA begins for try-except statement (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            # Getting the type of 'KeyError' (line 76)
            KeyError_193743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'KeyError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 76, 16), KeyError_193743, 'raise parameter', BaseException)
            # SSA branch for the except part of a try statement (line 75)
            # SSA branch for the except 'Exception' branch of a try statement (line 75)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'Exception' (line 77)
            Exception_193744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'Exception')
            # Assigning a type to the variable 'e' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'e', Exception_193744)
            # SSA join for try-except statement (line 75)
            module_type_store = module_type_store.join_ssa_context()
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 74)
            exit___193745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 13), assertRaises_call_result_193739, '__exit__')
            with_exit_193746 = invoke(stypy.reporting.localization.Localization(__file__, 74, 13), exit___193745, None, None, None)

        
        # Call to assertIs(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'cm' (line 79)
        cm_193749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'cm', False)
        # Obtaining the member 'exception' of a type (line 79)
        exception_193750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 22), cm_193749, 'exception')
        # Getting the type of 'e' (line 79)
        e_193751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 36), 'e', False)
        # Processing the call keyword arguments (line 79)
        kwargs_193752 = {}
        # Getting the type of 'self' (line 79)
        self_193747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', False)
        # Obtaining the member 'assertIs' of a type (line 79)
        assertIs_193748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_193747, 'assertIs')
        # Calling assertIs(args, kwargs) (line 79)
        assertIs_call_result_193753 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assertIs_193748, *[exception_193750, e_193751], **kwargs_193752)
        
        
        # Call to assertRaises(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'KeyError' (line 81)
        KeyError_193756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'KeyError', False)
        # Processing the call keyword arguments (line 81)
        kwargs_193757 = {}
        # Getting the type of 'self' (line 81)
        self_193754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 81)
        assertRaises_193755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 13), self_193754, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 81)
        assertRaises_call_result_193758 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), assertRaises_193755, *[KeyError_193756], **kwargs_193757)
        
        with_193759 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 81, 13), assertRaises_call_result_193758, 'with parameter', '__enter__', '__exit__')

        if with_193759:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 81)
            enter___193760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 13), assertRaises_call_result_193758, '__enter__')
            with_enter_193761 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), enter___193760)
            
            # Call to KeyError(...): (line 82)
            # Processing the call arguments (line 82)
            str_193763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'str', 'key')
            # Processing the call keyword arguments (line 82)
            kwargs_193764 = {}
            # Getting the type of 'KeyError' (line 82)
            KeyError_193762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'KeyError', False)
            # Calling KeyError(args, kwargs) (line 82)
            KeyError_call_result_193765 = invoke(stypy.reporting.localization.Localization(__file__, 82, 18), KeyError_193762, *[str_193763], **kwargs_193764)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 82, 12), KeyError_call_result_193765, 'raise parameter', BaseException)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 81)
            exit___193766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 13), assertRaises_call_result_193758, '__exit__')
            with_exit_193767 = invoke(stypy.reporting.localization.Localization(__file__, 81, 13), exit___193766, None, None, None)

        
        
        # SSA begins for try-except statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to assertRaises(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'KeyError' (line 84)
        KeyError_193770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 35), 'KeyError', False)
        # Processing the call keyword arguments (line 84)
        kwargs_193771 = {}
        # Getting the type of 'self' (line 84)
        self_193768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 84)
        assertRaises_193769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 17), self_193768, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 84)
        assertRaises_call_result_193772 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), assertRaises_193769, *[KeyError_193770], **kwargs_193771)
        
        with_193773 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 84, 17), assertRaises_call_result_193772, 'with parameter', '__enter__', '__exit__')

        if with_193773:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 84)
            enter___193774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 17), assertRaises_call_result_193772, '__enter__')
            with_enter_193775 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), enter___193774)
            pass
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 84)
            exit___193776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 17), assertRaises_call_result_193772, '__exit__')
            with_exit_193777 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), exit___193776, None, None, None)

        # SSA branch for the except part of a try statement (line 83)
        # SSA branch for the except 'Attribute' branch of a try statement (line 83)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'self' (line 86)
        self_193778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self')
        # Obtaining the member 'failureException' of a type (line 86)
        failureException_193779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), self_193778, 'failureException')
        # Assigning a type to the variable 'e' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'e', failureException_193779)
        
        # Call to assertIn(...): (line 87)
        # Processing the call arguments (line 87)
        str_193782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 26), 'str', 'KeyError not raised')
        # Getting the type of 'e' (line 87)
        e_193783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 49), 'e', False)
        # Obtaining the member 'args' of a type (line 87)
        args_193784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 49), e_193783, 'args')
        # Processing the call keyword arguments (line 87)
        kwargs_193785 = {}
        # Getting the type of 'self' (line 87)
        self_193780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 87)
        assertIn_193781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), self_193780, 'assertIn')
        # Calling assertIn(args, kwargs) (line 87)
        assertIn_call_result_193786 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), assertIn_193781, *[str_193782, args_193784], **kwargs_193785)
        
        # SSA branch for the else branch of a try statement (line 83)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 89)
        # Processing the call arguments (line 89)
        str_193789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'str', "assertRaises() didn't fail")
        # Processing the call keyword arguments (line 89)
        kwargs_193790 = {}
        # Getting the type of 'self' (line 89)
        self_193787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 89)
        fail_193788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), self_193787, 'fail')
        # Calling fail(args, kwargs) (line 89)
        fail_call_result_193791 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), fail_193788, *[str_193789], **kwargs_193790)
        
        # SSA join for try-except statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to assertRaises(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'KeyError' (line 91)
        KeyError_193794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 35), 'KeyError', False)
        # Processing the call keyword arguments (line 91)
        kwargs_193795 = {}
        # Getting the type of 'self' (line 91)
        self_193792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 17), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 91)
        assertRaises_193793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 17), self_193792, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 91)
        assertRaises_call_result_193796 = invoke(stypy.reporting.localization.Localization(__file__, 91, 17), assertRaises_193793, *[KeyError_193794], **kwargs_193795)
        
        with_193797 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 91, 17), assertRaises_call_result_193796, 'with parameter', '__enter__', '__exit__')

        if with_193797:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 91)
            enter___193798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 17), assertRaises_call_result_193796, '__enter__')
            with_enter_193799 = invoke(stypy.reporting.localization.Localization(__file__, 91, 17), enter___193798)
            # Getting the type of 'ValueError' (line 92)
            ValueError_193800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'ValueError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 92, 16), ValueError_193800, 'raise parameter', BaseException)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 91)
            exit___193801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 17), assertRaises_call_result_193796, '__exit__')
            with_exit_193802 = invoke(stypy.reporting.localization.Localization(__file__, 91, 17), exit___193801, None, None, None)

        # SSA branch for the except part of a try statement (line 90)
        # SSA branch for the except 'ValueError' branch of a try statement (line 90)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 90)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 96)
        # Processing the call arguments (line 96)
        str_193805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'str', "assertRaises() didn't let exception pass through")
        # Processing the call keyword arguments (line 96)
        kwargs_193806 = {}
        # Getting the type of 'self' (line 96)
        self_193803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 96)
        fail_193804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), self_193803, 'fail')
        # Calling fail(args, kwargs) (line 96)
        fail_call_result_193807 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), fail_193804, *[str_193805], **kwargs_193806)
        
        # SSA join for try-except statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_assertRaises(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_assertRaises' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_193808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193808)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_assertRaises'
        return stypy_return_type_193808


    @norecursion
    def testAssertNotRegexpMatches(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertNotRegexpMatches'
        module_type_store = module_type_store.open_function_context('testAssertNotRegexpMatches', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_localization', localization)
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_function_name', 'Test_Assertions.testAssertNotRegexpMatches')
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_param_names_list', [])
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_Assertions.testAssertNotRegexpMatches.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Assertions.testAssertNotRegexpMatches', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertNotRegexpMatches', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertNotRegexpMatches(...)' code ##################

        
        # Call to assertNotRegexpMatches(...): (line 99)
        # Processing the call arguments (line 99)
        str_193811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 36), 'str', 'Ala ma kota')
        str_193812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 51), 'str', 'r+')
        # Processing the call keyword arguments (line 99)
        kwargs_193813 = {}
        # Getting the type of 'self' (line 99)
        self_193809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self', False)
        # Obtaining the member 'assertNotRegexpMatches' of a type (line 99)
        assertNotRegexpMatches_193810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_193809, 'assertNotRegexpMatches')
        # Calling assertNotRegexpMatches(args, kwargs) (line 99)
        assertNotRegexpMatches_call_result_193814 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), assertNotRegexpMatches_193810, *[str_193811, str_193812], **kwargs_193813)
        
        
        
        # SSA begins for try-except statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to assertNotRegexpMatches(...): (line 101)
        # Processing the call arguments (line 101)
        str_193817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 40), 'str', 'Ala ma kota')
        str_193818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 55), 'str', 'k.t')
        str_193819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 63), 'str', 'Message')
        # Processing the call keyword arguments (line 101)
        kwargs_193820 = {}
        # Getting the type of 'self' (line 101)
        self_193815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self', False)
        # Obtaining the member 'assertNotRegexpMatches' of a type (line 101)
        assertNotRegexpMatches_193816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_193815, 'assertNotRegexpMatches')
        # Calling assertNotRegexpMatches(args, kwargs) (line 101)
        assertNotRegexpMatches_call_result_193821 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), assertNotRegexpMatches_193816, *[str_193817, str_193818, str_193819], **kwargs_193820)
        
        # SSA branch for the except part of a try statement (line 100)
        # SSA branch for the except 'Attribute' branch of a try statement (line 100)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'self' (line 102)
        self_193822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'self')
        # Obtaining the member 'failureException' of a type (line 102)
        failureException_193823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), self_193822, 'failureException')
        # Assigning a type to the variable 'e' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'e', failureException_193823)
        
        # Call to assertIn(...): (line 103)
        # Processing the call arguments (line 103)
        str_193826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 26), 'str', "'kot'")
        
        # Obtaining the type of the subscript
        int_193827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 42), 'int')
        # Getting the type of 'e' (line 103)
        e_193828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 35), 'e', False)
        # Obtaining the member 'args' of a type (line 103)
        args_193829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 35), e_193828, 'args')
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___193830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 35), args_193829, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_193831 = invoke(stypy.reporting.localization.Localization(__file__, 103, 35), getitem___193830, int_193827)
        
        # Processing the call keyword arguments (line 103)
        kwargs_193832 = {}
        # Getting the type of 'self' (line 103)
        self_193824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 103)
        assertIn_193825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_193824, 'assertIn')
        # Calling assertIn(args, kwargs) (line 103)
        assertIn_call_result_193833 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), assertIn_193825, *[str_193826, subscript_call_result_193831], **kwargs_193832)
        
        
        # Call to assertIn(...): (line 104)
        # Processing the call arguments (line 104)
        str_193836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 26), 'str', 'Message')
        
        # Obtaining the type of the subscript
        int_193837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 44), 'int')
        # Getting the type of 'e' (line 104)
        e_193838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 37), 'e', False)
        # Obtaining the member 'args' of a type (line 104)
        args_193839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 37), e_193838, 'args')
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___193840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 37), args_193839, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_193841 = invoke(stypy.reporting.localization.Localization(__file__, 104, 37), getitem___193840, int_193837)
        
        # Processing the call keyword arguments (line 104)
        kwargs_193842 = {}
        # Getting the type of 'self' (line 104)
        self_193834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 104)
        assertIn_193835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_193834, 'assertIn')
        # Calling assertIn(args, kwargs) (line 104)
        assertIn_call_result_193843 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), assertIn_193835, *[str_193836, subscript_call_result_193841], **kwargs_193842)
        
        # SSA branch for the else branch of a try statement (line 100)
        module_type_store.open_ssa_branch('except else')
        
        # Call to fail(...): (line 106)
        # Processing the call arguments (line 106)
        str_193846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 22), 'str', 'assertNotRegexpMatches should have failed.')
        # Processing the call keyword arguments (line 106)
        kwargs_193847 = {}
        # Getting the type of 'self' (line 106)
        self_193844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self', False)
        # Obtaining the member 'fail' of a type (line 106)
        fail_193845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_193844, 'fail')
        # Calling fail(args, kwargs) (line 106)
        fail_call_result_193848 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), fail_193845, *[str_193846], **kwargs_193847)
        
        # SSA join for try-except statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'testAssertNotRegexpMatches(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertNotRegexpMatches' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_193849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertNotRegexpMatches'
        return stypy_return_type_193849


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 0, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_Assertions.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_Assertions' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Test_Assertions', Test_Assertions)
# Declaration of the 'TestLongMessage' class
# Getting the type of 'unittest' (line 109)
unittest_193850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'unittest')
# Obtaining the member 'TestCase' of a type (line 109)
TestCase_193851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 22), unittest_193850, 'TestCase')

class TestLongMessage(TestCase_193851, ):
    str_193852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, (-1)), 'str', 'Test that the individual asserts honour longMessage.\n    This actually tests all the message behaviour for\n    asserts that use longMessage.')

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.setUp.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.setUp.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.setUp')
        TestLongMessage.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUp(...)' code ##################

        # Declaration of the 'TestableTestFalse' class
        # Getting the type of 'unittest' (line 115)
        unittest_193853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 115)
        TestCase_193854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 32), unittest_193853, 'TestCase')

        class TestableTestFalse(TestCase_193854, ):
            
            # Assigning a Name to a Name (line 116):
            # Getting the type of 'False' (line 116)
            False_193855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'False')
            # Assigning a type to the variable 'longMessage' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'longMessage', False_193855)
            
            # Assigning a Attribute to a Name (line 117):
            # Getting the type of 'self' (line 117)
            self_193856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 31), 'self')
            # Obtaining the member 'failureException' of a type (line 117)
            failureException_193857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 31), self_193856, 'failureException')
            # Assigning a type to the variable 'failureException' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'failureException', failureException_193857)

            @norecursion
            def testTest(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testTest'
                module_type_store = module_type_store.open_function_context('testTest', 119, 12, False)
                # Assigning a type to the variable 'self' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_localization', localization)
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_function_name', 'TestableTestFalse.testTest')
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTestFalse.testTest.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTestFalse.testTest', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testTest', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testTest(...)' code ##################

                pass
                
                # ################# End of 'testTest(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testTest' in the type store
                # Getting the type of 'stypy_return_type' (line 119)
                stypy_return_type_193858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_193858)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testTest'
                return stypy_return_type_193858

        
        # Assigning a type to the variable 'TestableTestFalse' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'TestableTestFalse', TestableTestFalse)
        # Declaration of the 'TestableTestTrue' class
        # Getting the type of 'unittest' (line 122)
        unittest_193859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'unittest')
        # Obtaining the member 'TestCase' of a type (line 122)
        TestCase_193860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 31), unittest_193859, 'TestCase')

        class TestableTestTrue(TestCase_193860, ):
            
            # Assigning a Name to a Name (line 123):
            # Getting the type of 'True' (line 123)
            True_193861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'True')
            # Assigning a type to the variable 'longMessage' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'longMessage', True_193861)
            
            # Assigning a Attribute to a Name (line 124):
            # Getting the type of 'self' (line 124)
            self_193862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'self')
            # Obtaining the member 'failureException' of a type (line 124)
            failureException_193863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 31), self_193862, 'failureException')
            # Assigning a type to the variable 'failureException' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'failureException', failureException_193863)

            @norecursion
            def testTest(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'testTest'
                module_type_store = module_type_store.open_function_context('testTest', 126, 12, False)
                # Assigning a type to the variable 'self' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_localization', localization)
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_type_store', module_type_store)
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_function_name', 'TestableTestTrue.testTest')
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_param_names_list', [])
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_varargs_param_name', None)
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_kwargs_param_name', None)
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_call_defaults', defaults)
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_call_varargs', varargs)
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                TestableTestTrue.testTest.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestableTestTrue.testTest', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'testTest', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'testTest(...)' code ##################

                pass
                
                # ################# End of 'testTest(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'testTest' in the type store
                # Getting the type of 'stypy_return_type' (line 126)
                stypy_return_type_193864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_193864)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'testTest'
                return stypy_return_type_193864

        
        # Assigning a type to the variable 'TestableTestTrue' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'TestableTestTrue', TestableTestTrue)
        
        # Assigning a Call to a Attribute (line 129):
        
        # Call to TestableTestTrue(...): (line 129)
        # Processing the call arguments (line 129)
        str_193866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 45), 'str', 'testTest')
        # Processing the call keyword arguments (line 129)
        kwargs_193867 = {}
        # Getting the type of 'TestableTestTrue' (line 129)
        TestableTestTrue_193865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'TestableTestTrue', False)
        # Calling TestableTestTrue(args, kwargs) (line 129)
        TestableTestTrue_call_result_193868 = invoke(stypy.reporting.localization.Localization(__file__, 129, 28), TestableTestTrue_193865, *[str_193866], **kwargs_193867)
        
        # Getting the type of 'self' (line 129)
        self_193869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self')
        # Setting the type of the member 'testableTrue' of a type (line 129)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_193869, 'testableTrue', TestableTestTrue_call_result_193868)
        
        # Assigning a Call to a Attribute (line 130):
        
        # Call to TestableTestFalse(...): (line 130)
        # Processing the call arguments (line 130)
        str_193871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 47), 'str', 'testTest')
        # Processing the call keyword arguments (line 130)
        kwargs_193872 = {}
        # Getting the type of 'TestableTestFalse' (line 130)
        TestableTestFalse_193870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 29), 'TestableTestFalse', False)
        # Calling TestableTestFalse(args, kwargs) (line 130)
        TestableTestFalse_call_result_193873 = invoke(stypy.reporting.localization.Localization(__file__, 130, 29), TestableTestFalse_193870, *[str_193871], **kwargs_193872)
        
        # Getting the type of 'self' (line 130)
        self_193874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self')
        # Setting the type of the member 'testableFalse' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_193874, 'testableFalse', TestableTestFalse_call_result_193873)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_193875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_193875


    @norecursion
    def testDefault(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testDefault'
        module_type_store = module_type_store.open_function_context('testDefault', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testDefault')
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testDefault.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testDefault', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testDefault', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testDefault(...)' code ##################

        
        # Call to assertFalse(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'unittest' (line 133)
        unittest_193878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'unittest', False)
        # Obtaining the member 'TestCase' of a type (line 133)
        TestCase_193879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 25), unittest_193878, 'TestCase')
        # Obtaining the member 'longMessage' of a type (line 133)
        longMessage_193880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 25), TestCase_193879, 'longMessage')
        # Processing the call keyword arguments (line 133)
        kwargs_193881 = {}
        # Getting the type of 'self' (line 133)
        self_193876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 133)
        assertFalse_193877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), self_193876, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 133)
        assertFalse_call_result_193882 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assertFalse_193877, *[longMessage_193880], **kwargs_193881)
        
        
        # ################# End of 'testDefault(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testDefault' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_193883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193883)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testDefault'
        return stypy_return_type_193883


    @norecursion
    def test_formatMsg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_formatMsg'
        module_type_store = module_type_store.open_function_context('test_formatMsg', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.test_formatMsg')
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.test_formatMsg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.test_formatMsg', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_formatMsg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_formatMsg(...)' code ##################

        
        # Call to assertEqual(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to _formatMessage(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'None' (line 136)
        None_193889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'None', False)
        str_193890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 65), 'str', 'foo')
        # Processing the call keyword arguments (line 136)
        kwargs_193891 = {}
        # Getting the type of 'self' (line 136)
        self_193886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 'self', False)
        # Obtaining the member 'testableFalse' of a type (line 136)
        testableFalse_193887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 25), self_193886, 'testableFalse')
        # Obtaining the member '_formatMessage' of a type (line 136)
        _formatMessage_193888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 25), testableFalse_193887, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 136)
        _formatMessage_call_result_193892 = invoke(stypy.reporting.localization.Localization(__file__, 136, 25), _formatMessage_193888, *[None_193889, str_193890], **kwargs_193891)
        
        str_193893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 73), 'str', 'foo')
        # Processing the call keyword arguments (line 136)
        kwargs_193894 = {}
        # Getting the type of 'self' (line 136)
        self_193884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 136)
        assertEqual_193885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_193884, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 136)
        assertEqual_call_result_193895 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), assertEqual_193885, *[_formatMessage_call_result_193892, str_193893], **kwargs_193894)
        
        
        # Call to assertEqual(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Call to _formatMessage(...): (line 137)
        # Processing the call arguments (line 137)
        str_193901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 59), 'str', 'foo')
        str_193902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 66), 'str', 'bar')
        # Processing the call keyword arguments (line 137)
        kwargs_193903 = {}
        # Getting the type of 'self' (line 137)
        self_193898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'self', False)
        # Obtaining the member 'testableFalse' of a type (line 137)
        testableFalse_193899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 25), self_193898, 'testableFalse')
        # Obtaining the member '_formatMessage' of a type (line 137)
        _formatMessage_193900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 25), testableFalse_193899, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 137)
        _formatMessage_call_result_193904 = invoke(stypy.reporting.localization.Localization(__file__, 137, 25), _formatMessage_193900, *[str_193901, str_193902], **kwargs_193903)
        
        str_193905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 74), 'str', 'foo')
        # Processing the call keyword arguments (line 137)
        kwargs_193906 = {}
        # Getting the type of 'self' (line 137)
        self_193896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 137)
        assertEqual_193897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_193896, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 137)
        assertEqual_call_result_193907 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assertEqual_193897, *[_formatMessage_call_result_193904, str_193905], **kwargs_193906)
        
        
        # Call to assertEqual(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to _formatMessage(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'None' (line 139)
        None_193913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 58), 'None', False)
        str_193914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 64), 'str', 'foo')
        # Processing the call keyword arguments (line 139)
        kwargs_193915 = {}
        # Getting the type of 'self' (line 139)
        self_193910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'self', False)
        # Obtaining the member 'testableTrue' of a type (line 139)
        testableTrue_193911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), self_193910, 'testableTrue')
        # Obtaining the member '_formatMessage' of a type (line 139)
        _formatMessage_193912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), testableTrue_193911, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 139)
        _formatMessage_call_result_193916 = invoke(stypy.reporting.localization.Localization(__file__, 139, 25), _formatMessage_193912, *[None_193913, str_193914], **kwargs_193915)
        
        str_193917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 72), 'str', 'foo')
        # Processing the call keyword arguments (line 139)
        kwargs_193918 = {}
        # Getting the type of 'self' (line 139)
        self_193908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 139)
        assertEqual_193909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_193908, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 139)
        assertEqual_call_result_193919 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assertEqual_193909, *[_formatMessage_call_result_193916, str_193917], **kwargs_193918)
        
        
        # Call to assertEqual(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to _formatMessage(...): (line 140)
        # Processing the call arguments (line 140)
        str_193925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 58), 'str', 'foo')
        str_193926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 65), 'str', 'bar')
        # Processing the call keyword arguments (line 140)
        kwargs_193927 = {}
        # Getting the type of 'self' (line 140)
        self_193922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'self', False)
        # Obtaining the member 'testableTrue' of a type (line 140)
        testableTrue_193923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 25), self_193922, 'testableTrue')
        # Obtaining the member '_formatMessage' of a type (line 140)
        _formatMessage_193924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 25), testableTrue_193923, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 140)
        _formatMessage_call_result_193928 = invoke(stypy.reporting.localization.Localization(__file__, 140, 25), _formatMessage_193924, *[str_193925, str_193926], **kwargs_193927)
        
        str_193929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 73), 'str', 'bar : foo')
        # Processing the call keyword arguments (line 140)
        kwargs_193930 = {}
        # Getting the type of 'self' (line 140)
        self_193920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 140)
        assertEqual_193921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_193920, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 140)
        assertEqual_call_result_193931 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), assertEqual_193921, *[_formatMessage_call_result_193928, str_193929], **kwargs_193930)
        
        
        # Call to _formatMessage(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Call to object(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_193936 = {}
        # Getting the type of 'object' (line 143)
        object_193935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'object', False)
        # Calling object(args, kwargs) (line 143)
        object_call_result_193937 = invoke(stypy.reporting.localization.Localization(__file__, 143, 41), object_193935, *[], **kwargs_193936)
        
        str_193938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 51), 'str', 'foo')
        # Processing the call keyword arguments (line 143)
        kwargs_193939 = {}
        # Getting the type of 'self' (line 143)
        self_193932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self', False)
        # Obtaining the member 'testableTrue' of a type (line 143)
        testableTrue_193933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_193932, 'testableTrue')
        # Obtaining the member '_formatMessage' of a type (line 143)
        _formatMessage_193934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), testableTrue_193933, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 143)
        _formatMessage_call_result_193940 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), _formatMessage_193934, *[object_call_result_193937, str_193938], **kwargs_193939)
        
        
        # ################# End of 'test_formatMsg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_formatMsg' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_193941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193941)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_formatMsg'
        return stypy_return_type_193941


    @norecursion
    def test_formatMessage_unicode_error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_formatMessage_unicode_error'
        module_type_store = module_type_store.open_function_context('test_formatMessage_unicode_error', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.test_formatMessage_unicode_error')
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.test_formatMessage_unicode_error.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.test_formatMessage_unicode_error', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_formatMessage_unicode_error', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_formatMessage_unicode_error(...)' code ##################

        
        # Assigning a Call to a Name (line 146):
        
        # Call to join(...): (line 146)
        # Processing the call arguments (line 146)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 146, 22, True)
        # Calculating comprehension expression
        
        # Call to range(...): (line 146)
        # Processing the call arguments (line 146)
        int_193949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 44), 'int')
        # Processing the call keyword arguments (line 146)
        kwargs_193950 = {}
        # Getting the type of 'range' (line 146)
        range_193948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'range', False)
        # Calling range(args, kwargs) (line 146)
        range_call_result_193951 = invoke(stypy.reporting.localization.Localization(__file__, 146, 38), range_193948, *[int_193949], **kwargs_193950)
        
        comprehension_193952 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 22), range_call_result_193951)
        # Assigning a type to the variable 'i' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 22), 'i', comprehension_193952)
        
        # Call to chr(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'i' (line 146)
        i_193945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'i', False)
        # Processing the call keyword arguments (line 146)
        kwargs_193946 = {}
        # Getting the type of 'chr' (line 146)
        chr_193944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 22), 'chr', False)
        # Calling chr(args, kwargs) (line 146)
        chr_call_result_193947 = invoke(stypy.reporting.localization.Localization(__file__, 146, 22), chr_193944, *[i_193945], **kwargs_193946)
        
        list_193953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 22), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 22), list_193953, chr_call_result_193947)
        # Processing the call keyword arguments (line 146)
        kwargs_193954 = {}
        str_193942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 14), 'str', '')
        # Obtaining the member 'join' of a type (line 146)
        join_193943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 14), str_193942, 'join')
        # Calling join(args, kwargs) (line 146)
        join_call_result_193955 = invoke(stypy.reporting.localization.Localization(__file__, 146, 14), join_193943, *[list_193953], **kwargs_193954)
        
        # Assigning a type to the variable 'one' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'one', join_call_result_193955)
        
        # Call to _formatMessage(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'one' (line 148)
        one_193959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 41), 'one', False)
        unicode_193960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 46), 'unicode', u'\ufffd')
        # Processing the call keyword arguments (line 148)
        kwargs_193961 = {}
        # Getting the type of 'self' (line 148)
        self_193956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self', False)
        # Obtaining the member 'testableTrue' of a type (line 148)
        testableTrue_193957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_193956, 'testableTrue')
        # Obtaining the member '_formatMessage' of a type (line 148)
        _formatMessage_193958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), testableTrue_193957, '_formatMessage')
        # Calling _formatMessage(args, kwargs) (line 148)
        _formatMessage_call_result_193962 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), _formatMessage_193958, *[one_193959, unicode_193960], **kwargs_193961)
        
        
        # ################# End of 'test_formatMessage_unicode_error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_formatMessage_unicode_error' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_193963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_193963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_formatMessage_unicode_error'
        return stypy_return_type_193963


    @norecursion
    def assertMessages(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'assertMessages'
        module_type_store = module_type_store.open_function_context('assertMessages', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.assertMessages')
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_param_names_list', ['methodName', 'args', 'errors'])
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.assertMessages.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.assertMessages', ['methodName', 'args', 'errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertMessages', localization, ['methodName', 'args', 'errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertMessages(...)' code ##################


        @norecursion
        def getMethod(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'getMethod'
            module_type_store = module_type_store.open_function_context('getMethod', 151, 8, False)
            
            # Passed parameters checking function
            getMethod.stypy_localization = localization
            getMethod.stypy_type_of_self = None
            getMethod.stypy_type_store = module_type_store
            getMethod.stypy_function_name = 'getMethod'
            getMethod.stypy_param_names_list = ['i']
            getMethod.stypy_varargs_param_name = None
            getMethod.stypy_kwargs_param_name = None
            getMethod.stypy_call_defaults = defaults
            getMethod.stypy_call_varargs = varargs
            getMethod.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'getMethod', ['i'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'getMethod', localization, ['i'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'getMethod(...)' code ##################

            
            # Assigning a Compare to a Name (line 152):
            
            # Getting the type of 'i' (line 152)
            i_193964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'i')
            int_193965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 36), 'int')
            # Applying the binary operator '<' (line 152)
            result_lt_193966 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 32), '<', i_193964, int_193965)
            
            # Assigning a type to the variable 'useTestableFalse' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'useTestableFalse', result_lt_193966)
            
            # Getting the type of 'useTestableFalse' (line 153)
            useTestableFalse_193967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'useTestableFalse')
            # Testing the type of an if condition (line 153)
            if_condition_193968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 12), useTestableFalse_193967)
            # Assigning a type to the variable 'if_condition_193968' (line 153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'if_condition_193968', if_condition_193968)
            # SSA begins for if statement (line 153)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 154):
            # Getting the type of 'self' (line 154)
            self_193969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'self')
            # Obtaining the member 'testableFalse' of a type (line 154)
            testableFalse_193970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 23), self_193969, 'testableFalse')
            # Assigning a type to the variable 'test' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'test', testableFalse_193970)
            # SSA branch for the else part of an if statement (line 153)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 156):
            # Getting the type of 'self' (line 156)
            self_193971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'self')
            # Obtaining the member 'testableTrue' of a type (line 156)
            testableTrue_193972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), self_193971, 'testableTrue')
            # Assigning a type to the variable 'test' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'test', testableTrue_193972)
            # SSA join for if statement (line 153)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to getattr(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'test' (line 157)
            test_193974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'test', False)
            # Getting the type of 'methodName' (line 157)
            methodName_193975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 33), 'methodName', False)
            # Processing the call keyword arguments (line 157)
            kwargs_193976 = {}
            # Getting the type of 'getattr' (line 157)
            getattr_193973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), 'getattr', False)
            # Calling getattr(args, kwargs) (line 157)
            getattr_call_result_193977 = invoke(stypy.reporting.localization.Localization(__file__, 157, 19), getattr_193973, *[test_193974, methodName_193975], **kwargs_193976)
            
            # Assigning a type to the variable 'stypy_return_type' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'stypy_return_type', getattr_call_result_193977)
            
            # ################# End of 'getMethod(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'getMethod' in the type store
            # Getting the type of 'stypy_return_type' (line 151)
            stypy_return_type_193978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_193978)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'getMethod'
            return stypy_return_type_193978

        # Assigning a type to the variable 'getMethod' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'getMethod', getMethod)
        
        
        # Call to enumerate(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'errors' (line 159)
        errors_193980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 44), 'errors', False)
        # Processing the call keyword arguments (line 159)
        kwargs_193981 = {}
        # Getting the type of 'enumerate' (line 159)
        enumerate_193979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 159)
        enumerate_call_result_193982 = invoke(stypy.reporting.localization.Localization(__file__, 159, 34), enumerate_193979, *[errors_193980], **kwargs_193981)
        
        # Testing the type of a for loop iterable (line 159)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 8), enumerate_call_result_193982)
        # Getting the type of the for loop variable (line 159)
        for_loop_var_193983 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 8), enumerate_call_result_193982)
        # Assigning a type to the variable 'i' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), for_loop_var_193983))
        # Assigning a type to the variable 'expected_regexp' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'expected_regexp', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), for_loop_var_193983))
        # SSA begins for a for statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 160):
        
        # Call to getMethod(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'i' (line 160)
        i_193985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 35), 'i', False)
        # Processing the call keyword arguments (line 160)
        kwargs_193986 = {}
        # Getting the type of 'getMethod' (line 160)
        getMethod_193984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'getMethod', False)
        # Calling getMethod(args, kwargs) (line 160)
        getMethod_call_result_193987 = invoke(stypy.reporting.localization.Localization(__file__, 160, 25), getMethod_193984, *[i_193985], **kwargs_193986)
        
        # Assigning a type to the variable 'testMethod' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'testMethod', getMethod_call_result_193987)
        
        # Assigning a Dict to a Name (line 161):
        
        # Obtaining an instance of the builtin type 'dict' (line 161)
        dict_193988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 161)
        
        # Assigning a type to the variable 'kwargs' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'kwargs', dict_193988)
        
        # Assigning a BinOp to a Name (line 162):
        # Getting the type of 'i' (line 162)
        i_193989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'i')
        int_193990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 26), 'int')
        # Applying the binary operator '%' (line 162)
        result_mod_193991 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 22), '%', i_193989, int_193990)
        
        # Assigning a type to the variable 'withMsg' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'withMsg', result_mod_193991)
        
        # Getting the type of 'withMsg' (line 163)
        withMsg_193992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'withMsg')
        # Testing the type of an if condition (line 163)
        if_condition_193993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 12), withMsg_193992)
        # Assigning a type to the variable 'if_condition_193993' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'if_condition_193993', if_condition_193993)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Name (line 164):
        
        # Obtaining an instance of the builtin type 'dict' (line 164)
        dict_193994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 164)
        # Adding element type (key, value) (line 164)
        str_193995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 26), 'str', 'msg')
        str_193996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 33), 'str', 'oops')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 25), dict_193994, (str_193995, str_193996))
        
        # Assigning a type to the variable 'kwargs' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'kwargs', dict_193994)
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assertRaisesRegexp(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'self' (line 166)
        self_193999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 41), 'self', False)
        # Obtaining the member 'failureException' of a type (line 166)
        failureException_194000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 41), self_193999, 'failureException')
        # Processing the call keyword arguments (line 166)
        # Getting the type of 'expected_regexp' (line 167)
        expected_regexp_194001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 57), 'expected_regexp', False)
        keyword_194002 = expected_regexp_194001
        kwargs_194003 = {'expected_regexp': keyword_194002}
        # Getting the type of 'self' (line 166)
        self_193997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'self', False)
        # Obtaining the member 'assertRaisesRegexp' of a type (line 166)
        assertRaisesRegexp_193998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), self_193997, 'assertRaisesRegexp')
        # Calling assertRaisesRegexp(args, kwargs) (line 166)
        assertRaisesRegexp_call_result_194004 = invoke(stypy.reporting.localization.Localization(__file__, 166, 17), assertRaisesRegexp_193998, *[failureException_194000], **kwargs_194003)
        
        with_194005 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 166, 17), assertRaisesRegexp_call_result_194004, 'with parameter', '__enter__', '__exit__')

        if with_194005:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 166)
            enter___194006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), assertRaisesRegexp_call_result_194004, '__enter__')
            with_enter_194007 = invoke(stypy.reporting.localization.Localization(__file__, 166, 17), enter___194006)
            
            # Call to testMethod(...): (line 168)
            # Getting the type of 'args' (line 168)
            args_194009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'args', False)
            # Processing the call keyword arguments (line 168)
            # Getting the type of 'kwargs' (line 168)
            kwargs_194010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'kwargs', False)
            kwargs_194011 = {'kwargs_194010': kwargs_194010}
            # Getting the type of 'testMethod' (line 168)
            testMethod_194008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'testMethod', False)
            # Calling testMethod(args, kwargs) (line 168)
            testMethod_call_result_194012 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), testMethod_194008, *[args_194009], **kwargs_194011)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 166)
            exit___194013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 17), assertRaisesRegexp_call_result_194004, '__exit__')
            with_exit_194014 = invoke(stypy.reporting.localization.Localization(__file__, 166, 17), exit___194013, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assertMessages(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertMessages' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_194015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194015)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertMessages'
        return stypy_return_type_194015


    @norecursion
    def testAssertTrue(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertTrue'
        module_type_store = module_type_store.open_function_context('testAssertTrue', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertTrue')
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertTrue.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertTrue', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertTrue', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertTrue(...)' code ##################

        
        # Call to assertMessages(...): (line 171)
        # Processing the call arguments (line 171)
        str_194018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 28), 'str', 'assertTrue')
        
        # Obtaining an instance of the builtin type 'tuple' (line 171)
        tuple_194019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 171)
        # Adding element type (line 171)
        # Getting the type of 'False' (line 171)
        False_194020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 43), 'False', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 43), tuple_194019, False_194020)
        
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_194021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        str_194022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 29), 'str', '^False is not true$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 28), list_194021, str_194022)
        # Adding element type (line 172)
        str_194023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 52), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 28), list_194021, str_194023)
        # Adding element type (line 172)
        str_194024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 62), 'str', '^False is not true$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 28), list_194021, str_194024)
        # Adding element type (line 172)
        str_194025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 29), 'str', '^False is not true : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 28), list_194021, str_194025)
        
        # Processing the call keyword arguments (line 171)
        kwargs_194026 = {}
        # Getting the type of 'self' (line 171)
        self_194016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 171)
        assertMessages_194017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_194016, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 171)
        assertMessages_call_result_194027 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assertMessages_194017, *[str_194018, tuple_194019, list_194021], **kwargs_194026)
        
        
        # ################# End of 'testAssertTrue(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertTrue' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_194028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertTrue'
        return stypy_return_type_194028


    @norecursion
    def testAssertFalse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertFalse'
        module_type_store = module_type_store.open_function_context('testAssertFalse', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertFalse')
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertFalse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertFalse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertFalse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertFalse(...)' code ##################

        
        # Call to assertMessages(...): (line 176)
        # Processing the call arguments (line 176)
        str_194031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', 'assertFalse')
        
        # Obtaining an instance of the builtin type 'tuple' (line 176)
        tuple_194032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 176)
        # Adding element type (line 176)
        # Getting the type of 'True' (line 176)
        True_194033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 44), 'True', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 44), tuple_194032, True_194033)
        
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_194034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        str_194035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 29), 'str', '^True is not false$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 28), list_194034, str_194035)
        # Adding element type (line 177)
        str_194036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 52), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 28), list_194034, str_194036)
        # Adding element type (line 177)
        str_194037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 62), 'str', '^True is not false$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 28), list_194034, str_194037)
        # Adding element type (line 177)
        str_194038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 29), 'str', '^True is not false : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 28), list_194034, str_194038)
        
        # Processing the call keyword arguments (line 176)
        kwargs_194039 = {}
        # Getting the type of 'self' (line 176)
        self_194029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 176)
        assertMessages_194030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_194029, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 176)
        assertMessages_call_result_194040 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assertMessages_194030, *[str_194031, tuple_194032, list_194034], **kwargs_194039)
        
        
        # ################# End of 'testAssertFalse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertFalse' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_194041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertFalse'
        return stypy_return_type_194041


    @norecursion
    def testNotEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testNotEqual'
        module_type_store = module_type_store.open_function_context('testNotEqual', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testNotEqual')
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testNotEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testNotEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testNotEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testNotEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 181)
        # Processing the call arguments (line 181)
        str_194044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 28), 'str', 'assertNotEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_194045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        # Adding element type (line 181)
        int_194046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 47), tuple_194045, int_194046)
        # Adding element type (line 181)
        int_194047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 47), tuple_194045, int_194047)
        
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_194048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        str_194049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 29), 'str', '^1 == 1$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 28), list_194048, str_194049)
        # Adding element type (line 182)
        str_194050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 41), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 28), list_194048, str_194050)
        # Adding element type (line 182)
        str_194051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 51), 'str', '^1 == 1$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 28), list_194048, str_194051)
        # Adding element type (line 182)
        str_194052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 29), 'str', '^1 == 1 : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 28), list_194048, str_194052)
        
        # Processing the call keyword arguments (line 181)
        kwargs_194053 = {}
        # Getting the type of 'self' (line 181)
        self_194042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 181)
        assertMessages_194043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_194042, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 181)
        assertMessages_call_result_194054 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assertMessages_194043, *[str_194044, tuple_194045, list_194048], **kwargs_194053)
        
        
        # ################# End of 'testNotEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testNotEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_194055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194055)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testNotEqual'
        return stypy_return_type_194055


    @norecursion
    def testAlmostEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAlmostEqual'
        module_type_store = module_type_store.open_function_context('testAlmostEqual', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAlmostEqual')
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAlmostEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAlmostEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAlmostEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAlmostEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 186)
        # Processing the call arguments (line 186)
        str_194058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 28), 'str', 'assertAlmostEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_194059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)
        int_194060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 50), tuple_194059, int_194060)
        # Adding element type (line 186)
        int_194061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 50), tuple_194059, int_194061)
        
        
        # Obtaining an instance of the builtin type 'list' (line 187)
        list_194062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 187)
        # Adding element type (line 187)
        str_194063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 29), 'str', '^1 != 2 within 7 places$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), list_194062, str_194063)
        # Adding element type (line 187)
        str_194064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 57), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), list_194062, str_194064)
        # Adding element type (line 187)
        str_194065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 29), 'str', '^1 != 2 within 7 places$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), list_194062, str_194065)
        # Adding element type (line 187)
        str_194066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 57), 'str', '^1 != 2 within 7 places : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 28), list_194062, str_194066)
        
        # Processing the call keyword arguments (line 186)
        kwargs_194067 = {}
        # Getting the type of 'self' (line 186)
        self_194056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 186)
        assertMessages_194057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_194056, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 186)
        assertMessages_call_result_194068 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), assertMessages_194057, *[str_194058, tuple_194059, list_194062], **kwargs_194067)
        
        
        # ################# End of 'testAlmostEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAlmostEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_194069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAlmostEqual'
        return stypy_return_type_194069


    @norecursion
    def testNotAlmostEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testNotAlmostEqual'
        module_type_store = module_type_store.open_function_context('testNotAlmostEqual', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testNotAlmostEqual')
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testNotAlmostEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testNotAlmostEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testNotAlmostEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testNotAlmostEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 191)
        # Processing the call arguments (line 191)
        str_194072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 28), 'str', 'assertNotAlmostEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 191)
        tuple_194073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 191)
        # Adding element type (line 191)
        int_194074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 53), tuple_194073, int_194074)
        # Adding element type (line 191)
        int_194075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 53), tuple_194073, int_194075)
        
        
        # Obtaining an instance of the builtin type 'list' (line 192)
        list_194076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 192)
        # Adding element type (line 192)
        str_194077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 29), 'str', '^1 == 1 within 7 places$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 28), list_194076, str_194077)
        # Adding element type (line 192)
        str_194078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 57), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 28), list_194076, str_194078)
        # Adding element type (line 192)
        str_194079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 29), 'str', '^1 == 1 within 7 places$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 28), list_194076, str_194079)
        # Adding element type (line 192)
        str_194080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 57), 'str', '^1 == 1 within 7 places : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 28), list_194076, str_194080)
        
        # Processing the call keyword arguments (line 191)
        kwargs_194081 = {}
        # Getting the type of 'self' (line 191)
        self_194070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 191)
        assertMessages_194071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_194070, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 191)
        assertMessages_call_result_194082 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), assertMessages_194071, *[str_194072, tuple_194073, list_194076], **kwargs_194081)
        
        
        # ################# End of 'testNotAlmostEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testNotAlmostEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_194083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194083)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testNotAlmostEqual'
        return stypy_return_type_194083


    @norecursion
    def test_baseAssertEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_baseAssertEqual'
        module_type_store = module_type_store.open_function_context('test_baseAssertEqual', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.test_baseAssertEqual')
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.test_baseAssertEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.test_baseAssertEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_baseAssertEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_baseAssertEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 196)
        # Processing the call arguments (line 196)
        str_194086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 28), 'str', '_baseAssertEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_194087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        int_194088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 49), tuple_194087, int_194088)
        # Adding element type (line 196)
        int_194089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 49), tuple_194087, int_194089)
        
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_194090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        str_194091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 29), 'str', '^1 != 2$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 28), list_194090, str_194091)
        # Adding element type (line 197)
        str_194092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 41), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 28), list_194090, str_194092)
        # Adding element type (line 197)
        str_194093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 51), 'str', '^1 != 2$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 28), list_194090, str_194093)
        # Adding element type (line 197)
        str_194094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 63), 'str', '^1 != 2 : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 28), list_194090, str_194094)
        
        # Processing the call keyword arguments (line 196)
        kwargs_194095 = {}
        # Getting the type of 'self' (line 196)
        self_194084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 196)
        assertMessages_194085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_194084, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 196)
        assertMessages_call_result_194096 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), assertMessages_194085, *[str_194086, tuple_194087, list_194090], **kwargs_194095)
        
        
        # ################# End of 'test_baseAssertEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_baseAssertEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_194097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_baseAssertEqual'
        return stypy_return_type_194097


    @norecursion
    def testAssertSequenceEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertSequenceEqual'
        module_type_store = module_type_store.open_function_context('testAssertSequenceEqual', 199, 4, False)
        # Assigning a type to the variable 'self' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertSequenceEqual')
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertSequenceEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertSequenceEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertSequenceEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertSequenceEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 202)
        # Processing the call arguments (line 202)
        str_194100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'str', 'assertSequenceEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 202)
        tuple_194101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 202)
        # Adding element type (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_194102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 52), tuple_194101, list_194102)
        # Adding element type (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_194103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        # Getting the type of 'None' (line 202)
        None_194104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 57), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 56), list_194103, None_194104)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 52), tuple_194101, list_194103)
        
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_194105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        str_194106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 29), 'str', '\\+ \\[None\\]$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 28), list_194105, str_194106)
        # Adding element type (line 203)
        str_194107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 45), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 28), list_194105, str_194107)
        # Adding element type (line 203)
        str_194108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 55), 'str', '\\+ \\[None\\]$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 28), list_194105, str_194108)
        # Adding element type (line 203)
        str_194109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 29), 'str', '\\+ \\[None\\] : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 28), list_194105, str_194109)
        
        # Processing the call keyword arguments (line 202)
        kwargs_194110 = {}
        # Getting the type of 'self' (line 202)
        self_194098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 202)
        assertMessages_194099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_194098, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 202)
        assertMessages_call_result_194111 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assertMessages_194099, *[str_194100, tuple_194101, list_194105], **kwargs_194110)
        
        
        # ################# End of 'testAssertSequenceEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertSequenceEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 199)
        stypy_return_type_194112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertSequenceEqual'
        return stypy_return_type_194112


    @norecursion
    def testAssertSetEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertSetEqual'
        module_type_store = module_type_store.open_function_context('testAssertSetEqual', 206, 4, False)
        # Assigning a type to the variable 'self' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertSetEqual')
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertSetEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertSetEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertSetEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertSetEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 207)
        # Processing the call arguments (line 207)
        str_194115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 28), 'str', 'assertSetEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 207)
        tuple_194116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 207)
        # Adding element type (line 207)
        
        # Call to set(...): (line 207)
        # Processing the call keyword arguments (line 207)
        kwargs_194118 = {}
        # Getting the type of 'set' (line 207)
        set_194117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 47), 'set', False)
        # Calling set(args, kwargs) (line 207)
        set_call_result_194119 = invoke(stypy.reporting.localization.Localization(__file__, 207, 47), set_194117, *[], **kwargs_194118)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 47), tuple_194116, set_call_result_194119)
        # Adding element type (line 207)
        
        # Call to set(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_194121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        # Getting the type of 'None' (line 207)
        None_194122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 59), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 58), list_194121, None_194122)
        
        # Processing the call keyword arguments (line 207)
        kwargs_194123 = {}
        # Getting the type of 'set' (line 207)
        set_194120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 54), 'set', False)
        # Calling set(args, kwargs) (line 207)
        set_call_result_194124 = invoke(stypy.reporting.localization.Localization(__file__, 207, 54), set_194120, *[list_194121], **kwargs_194123)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 47), tuple_194116, set_call_result_194124)
        
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_194125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        str_194126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'str', 'None$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 28), list_194125, str_194126)
        # Adding element type (line 208)
        str_194127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 38), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 28), list_194125, str_194127)
        # Adding element type (line 208)
        str_194128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 48), 'str', 'None$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 28), list_194125, str_194128)
        # Adding element type (line 208)
        str_194129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 29), 'str', 'None : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 28), list_194125, str_194129)
        
        # Processing the call keyword arguments (line 207)
        kwargs_194130 = {}
        # Getting the type of 'self' (line 207)
        self_194113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 207)
        assertMessages_194114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_194113, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 207)
        assertMessages_call_result_194131 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), assertMessages_194114, *[str_194115, tuple_194116, list_194125], **kwargs_194130)
        
        
        # ################# End of 'testAssertSetEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertSetEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 206)
        stypy_return_type_194132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194132)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertSetEqual'
        return stypy_return_type_194132


    @norecursion
    def testAssertIn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertIn'
        module_type_store = module_type_store.open_function_context('testAssertIn', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertIn')
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertIn.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertIn', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertIn', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertIn(...)' code ##################

        
        # Call to assertMessages(...): (line 212)
        # Processing the call arguments (line 212)
        str_194135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 28), 'str', 'assertIn')
        
        # Obtaining an instance of the builtin type 'tuple' (line 212)
        tuple_194136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 212)
        # Adding element type (line 212)
        # Getting the type of 'None' (line 212)
        None_194137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 41), tuple_194136, None_194137)
        # Adding element type (line 212)
        
        # Obtaining an instance of the builtin type 'list' (line 212)
        list_194138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 212)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 41), tuple_194136, list_194138)
        
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_194139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        str_194140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 29), 'str', '^None not found in \\[\\]$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 28), list_194139, str_194140)
        # Adding element type (line 213)
        str_194141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 57), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 28), list_194139, str_194141)
        # Adding element type (line 213)
        str_194142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 29), 'str', '^None not found in \\[\\]$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 28), list_194139, str_194142)
        # Adding element type (line 213)
        str_194143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 29), 'str', '^None not found in \\[\\] : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 28), list_194139, str_194143)
        
        # Processing the call keyword arguments (line 212)
        kwargs_194144 = {}
        # Getting the type of 'self' (line 212)
        self_194133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 212)
        assertMessages_194134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_194133, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 212)
        assertMessages_call_result_194145 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), assertMessages_194134, *[str_194135, tuple_194136, list_194139], **kwargs_194144)
        
        
        # ################# End of 'testAssertIn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertIn' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_194146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194146)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertIn'
        return stypy_return_type_194146


    @norecursion
    def testAssertNotIn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertNotIn'
        module_type_store = module_type_store.open_function_context('testAssertNotIn', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertNotIn')
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertNotIn.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertNotIn', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertNotIn', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertNotIn(...)' code ##################

        
        # Call to assertMessages(...): (line 218)
        # Processing the call arguments (line 218)
        str_194149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 28), 'str', 'assertNotIn')
        
        # Obtaining an instance of the builtin type 'tuple' (line 218)
        tuple_194150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 218)
        # Adding element type (line 218)
        # Getting the type of 'None' (line 218)
        None_194151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 44), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 44), tuple_194150, None_194151)
        # Adding element type (line 218)
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_194152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        # Getting the type of 'None' (line 218)
        None_194153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 51), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 50), list_194152, None_194153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 44), tuple_194150, list_194152)
        
        
        # Obtaining an instance of the builtin type 'list' (line 219)
        list_194154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 219)
        # Adding element type (line 219)
        str_194155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 29), 'str', '^None unexpectedly found in \\[None\\]$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 28), list_194154, str_194155)
        # Adding element type (line 219)
        str_194156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 70), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 28), list_194154, str_194156)
        # Adding element type (line 219)
        str_194157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 29), 'str', '^None unexpectedly found in \\[None\\]$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 28), list_194154, str_194157)
        # Adding element type (line 219)
        str_194158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 29), 'str', '^None unexpectedly found in \\[None\\] : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 28), list_194154, str_194158)
        
        # Processing the call keyword arguments (line 218)
        kwargs_194159 = {}
        # Getting the type of 'self' (line 218)
        self_194147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 218)
        assertMessages_194148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_194147, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 218)
        assertMessages_call_result_194160 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), assertMessages_194148, *[str_194149, tuple_194150, list_194154], **kwargs_194159)
        
        
        # ################# End of 'testAssertNotIn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertNotIn' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_194161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194161)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertNotIn'
        return stypy_return_type_194161


    @norecursion
    def testAssertDictEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertDictEqual'
        module_type_store = module_type_store.open_function_context('testAssertDictEqual', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertDictEqual')
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertDictEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertDictEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertDictEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertDictEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 224)
        # Processing the call arguments (line 224)
        str_194164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 28), 'str', 'assertDictEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 224)
        tuple_194165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 224)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'dict' (line 224)
        dict_194166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 48), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 224)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 48), tuple_194165, dict_194166)
        # Adding element type (line 224)
        
        # Obtaining an instance of the builtin type 'dict' (line 224)
        dict_194167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 52), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 224)
        # Adding element type (key, value) (line 224)
        str_194168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 53), 'str', 'key')
        str_194169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 60), 'str', 'value')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 52), dict_194167, (str_194168, str_194169))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 48), tuple_194165, dict_194167)
        
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_194170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        # Adding element type (line 225)
        str_194171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 29), 'str', "\\+ \\{'key': 'value'\\}$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 28), list_194170, str_194171)
        # Adding element type (line 225)
        str_194172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 56), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 28), list_194170, str_194172)
        # Adding element type (line 225)
        str_194173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 29), 'str', "\\+ \\{'key': 'value'\\}$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 28), list_194170, str_194173)
        # Adding element type (line 225)
        str_194174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 29), 'str', "\\+ \\{'key': 'value'\\} : oops$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 28), list_194170, str_194174)
        
        # Processing the call keyword arguments (line 224)
        kwargs_194175 = {}
        # Getting the type of 'self' (line 224)
        self_194162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 224)
        assertMessages_194163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_194162, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 224)
        assertMessages_call_result_194176 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), assertMessages_194163, *[str_194164, tuple_194165, list_194170], **kwargs_194175)
        
        
        # ################# End of 'testAssertDictEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertDictEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_194177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertDictEqual'
        return stypy_return_type_194177


    @norecursion
    def testAssertDictContainsSubset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertDictContainsSubset'
        module_type_store = module_type_store.open_function_context('testAssertDictContainsSubset', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertDictContainsSubset')
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertDictContainsSubset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertDictContainsSubset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertDictContainsSubset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertDictContainsSubset(...)' code ##################

        
        # Call to assertMessages(...): (line 230)
        # Processing the call arguments (line 230)
        str_194180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 28), 'str', 'assertDictContainsSubset')
        
        # Obtaining an instance of the builtin type 'tuple' (line 230)
        tuple_194181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 230)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'dict' (line 230)
        dict_194182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 57), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 230)
        # Adding element type (key, value) (line 230)
        str_194183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 58), 'str', 'key')
        str_194184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 65), 'str', 'value')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 57), dict_194182, (str_194183, str_194184))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 57), tuple_194181, dict_194182)
        # Adding element type (line 230)
        
        # Obtaining an instance of the builtin type 'dict' (line 230)
        dict_194185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 75), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 57), tuple_194181, dict_194185)
        
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_194186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        str_194187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 29), 'str', "^Missing: 'key'$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 28), list_194186, str_194187)
        # Adding element type (line 231)
        str_194188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 49), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 28), list_194186, str_194188)
        # Adding element type (line 231)
        str_194189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 29), 'str', "^Missing: 'key'$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 28), list_194186, str_194189)
        # Adding element type (line 231)
        str_194190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'str', "^Missing: 'key' : oops$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 28), list_194186, str_194190)
        
        # Processing the call keyword arguments (line 230)
        kwargs_194191 = {}
        # Getting the type of 'self' (line 230)
        self_194178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 230)
        assertMessages_194179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), self_194178, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 230)
        assertMessages_call_result_194192 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assertMessages_194179, *[str_194180, tuple_194181, list_194186], **kwargs_194191)
        
        
        # ################# End of 'testAssertDictContainsSubset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertDictContainsSubset' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_194193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194193)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertDictContainsSubset'
        return stypy_return_type_194193


    @norecursion
    def testAssertMultiLineEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertMultiLineEqual'
        module_type_store = module_type_store.open_function_context('testAssertMultiLineEqual', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertMultiLineEqual')
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertMultiLineEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertMultiLineEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertMultiLineEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertMultiLineEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 236)
        # Processing the call arguments (line 236)
        str_194196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 28), 'str', 'assertMultiLineEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 236)
        tuple_194197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 236)
        # Adding element type (line 236)
        str_194198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 53), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 53), tuple_194197, str_194198)
        # Adding element type (line 236)
        str_194199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 57), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 53), tuple_194197, str_194199)
        
        
        # Obtaining an instance of the builtin type 'list' (line 237)
        list_194200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 237)
        # Adding element type (line 237)
        str_194201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 29), 'str', '\\+ foo$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 28), list_194200, str_194201)
        # Adding element type (line 237)
        str_194202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 41), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 28), list_194200, str_194202)
        # Adding element type (line 237)
        str_194203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 29), 'str', '\\+ foo$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 28), list_194200, str_194203)
        # Adding element type (line 237)
        str_194204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 29), 'str', '\\+ foo : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 28), list_194200, str_194204)
        
        # Processing the call keyword arguments (line 236)
        kwargs_194205 = {}
        # Getting the type of 'self' (line 236)
        self_194194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 236)
        assertMessages_194195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_194194, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 236)
        assertMessages_call_result_194206 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), assertMessages_194195, *[str_194196, tuple_194197, list_194200], **kwargs_194205)
        
        
        # ################# End of 'testAssertMultiLineEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertMultiLineEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_194207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194207)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertMultiLineEqual'
        return stypy_return_type_194207


    @norecursion
    def testAssertLess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertLess'
        module_type_store = module_type_store.open_function_context('testAssertLess', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertLess')
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertLess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertLess', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertLess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertLess(...)' code ##################

        
        # Call to assertMessages(...): (line 242)
        # Processing the call arguments (line 242)
        str_194210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 28), 'str', 'assertLess')
        
        # Obtaining an instance of the builtin type 'tuple' (line 242)
        tuple_194211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 242)
        # Adding element type (line 242)
        int_194212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 43), tuple_194211, int_194212)
        # Adding element type (line 242)
        int_194213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 43), tuple_194211, int_194213)
        
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_194214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        str_194215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'str', '^2 not less than 1$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 28), list_194214, str_194215)
        # Adding element type (line 243)
        str_194216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 52), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 28), list_194214, str_194216)
        # Adding element type (line 243)
        str_194217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 29), 'str', '^2 not less than 1$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 28), list_194214, str_194217)
        # Adding element type (line 243)
        str_194218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 52), 'str', '^2 not less than 1 : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 28), list_194214, str_194218)
        
        # Processing the call keyword arguments (line 242)
        kwargs_194219 = {}
        # Getting the type of 'self' (line 242)
        self_194208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 242)
        assertMessages_194209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), self_194208, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 242)
        assertMessages_call_result_194220 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assertMessages_194209, *[str_194210, tuple_194211, list_194214], **kwargs_194219)
        
        
        # ################# End of 'testAssertLess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertLess' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_194221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194221)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertLess'
        return stypy_return_type_194221


    @norecursion
    def testAssertLessEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertLessEqual'
        module_type_store = module_type_store.open_function_context('testAssertLessEqual', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertLessEqual')
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertLessEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertLessEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertLessEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertLessEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 247)
        # Processing the call arguments (line 247)
        str_194224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 28), 'str', 'assertLessEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 247)
        tuple_194225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 247)
        # Adding element type (line 247)
        int_194226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 48), tuple_194225, int_194226)
        # Adding element type (line 247)
        int_194227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 48), tuple_194225, int_194227)
        
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_194228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        str_194229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 29), 'str', '^2 not less than or equal to 1$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 28), list_194228, str_194229)
        # Adding element type (line 248)
        str_194230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 64), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 28), list_194228, str_194230)
        # Adding element type (line 248)
        str_194231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'str', '^2 not less than or equal to 1$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 28), list_194228, str_194231)
        # Adding element type (line 248)
        str_194232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 29), 'str', '^2 not less than or equal to 1 : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 28), list_194228, str_194232)
        
        # Processing the call keyword arguments (line 247)
        kwargs_194233 = {}
        # Getting the type of 'self' (line 247)
        self_194222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 247)
        assertMessages_194223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_194222, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 247)
        assertMessages_call_result_194234 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), assertMessages_194223, *[str_194224, tuple_194225, list_194228], **kwargs_194233)
        
        
        # ################# End of 'testAssertLessEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertLessEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_194235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertLessEqual'
        return stypy_return_type_194235


    @norecursion
    def testAssertGreater(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertGreater'
        module_type_store = module_type_store.open_function_context('testAssertGreater', 252, 4, False)
        # Assigning a type to the variable 'self' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertGreater')
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertGreater.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertGreater', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertGreater', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertGreater(...)' code ##################

        
        # Call to assertMessages(...): (line 253)
        # Processing the call arguments (line 253)
        str_194238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 28), 'str', 'assertGreater')
        
        # Obtaining an instance of the builtin type 'tuple' (line 253)
        tuple_194239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 253)
        # Adding element type (line 253)
        int_194240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 46), tuple_194239, int_194240)
        # Adding element type (line 253)
        int_194241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 46), tuple_194239, int_194241)
        
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_194242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        str_194243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'str', '^1 not greater than 2$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 28), list_194242, str_194243)
        # Adding element type (line 254)
        str_194244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 55), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 28), list_194242, str_194244)
        # Adding element type (line 254)
        str_194245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 29), 'str', '^1 not greater than 2$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 28), list_194242, str_194245)
        # Adding element type (line 254)
        str_194246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 29), 'str', '^1 not greater than 2 : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 28), list_194242, str_194246)
        
        # Processing the call keyword arguments (line 253)
        kwargs_194247 = {}
        # Getting the type of 'self' (line 253)
        self_194236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 253)
        assertMessages_194237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), self_194236, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 253)
        assertMessages_call_result_194248 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), assertMessages_194237, *[str_194238, tuple_194239, list_194242], **kwargs_194247)
        
        
        # ################# End of 'testAssertGreater(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertGreater' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_194249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194249)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertGreater'
        return stypy_return_type_194249


    @norecursion
    def testAssertGreaterEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertGreaterEqual'
        module_type_store = module_type_store.open_function_context('testAssertGreaterEqual', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertGreaterEqual')
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertGreaterEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertGreaterEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertGreaterEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertGreaterEqual(...)' code ##################

        
        # Call to assertMessages(...): (line 259)
        # Processing the call arguments (line 259)
        str_194252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 28), 'str', 'assertGreaterEqual')
        
        # Obtaining an instance of the builtin type 'tuple' (line 259)
        tuple_194253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 259)
        # Adding element type (line 259)
        int_194254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 51), tuple_194253, int_194254)
        # Adding element type (line 259)
        int_194255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 51), tuple_194253, int_194255)
        
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_194256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        str_194257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 29), 'str', '^1 not greater than or equal to 2$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 28), list_194256, str_194257)
        # Adding element type (line 260)
        str_194258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 67), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 28), list_194256, str_194258)
        # Adding element type (line 260)
        str_194259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 29), 'str', '^1 not greater than or equal to 2$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 28), list_194256, str_194259)
        # Adding element type (line 260)
        str_194260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 29), 'str', '^1 not greater than or equal to 2 : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 28), list_194256, str_194260)
        
        # Processing the call keyword arguments (line 259)
        kwargs_194261 = {}
        # Getting the type of 'self' (line 259)
        self_194250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 259)
        assertMessages_194251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_194250, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 259)
        assertMessages_call_result_194262 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), assertMessages_194251, *[str_194252, tuple_194253, list_194256], **kwargs_194261)
        
        
        # ################# End of 'testAssertGreaterEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertGreaterEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_194263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertGreaterEqual'
        return stypy_return_type_194263


    @norecursion
    def testAssertIsNone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertIsNone'
        module_type_store = module_type_store.open_function_context('testAssertIsNone', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertIsNone')
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertIsNone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertIsNone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertIsNone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertIsNone(...)' code ##################

        
        # Call to assertMessages(...): (line 265)
        # Processing the call arguments (line 265)
        str_194266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 28), 'str', 'assertIsNone')
        
        # Obtaining an instance of the builtin type 'tuple' (line 265)
        tuple_194267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 265)
        # Adding element type (line 265)
        str_194268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 45), 'str', 'not None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 45), tuple_194267, str_194268)
        
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_194269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        # Adding element type (line 266)
        str_194270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 29), 'str', "^'not None' is not None$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 28), list_194269, str_194270)
        # Adding element type (line 266)
        str_194271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 57), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 28), list_194269, str_194271)
        # Adding element type (line 266)
        str_194272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 29), 'str', "^'not None' is not None$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 28), list_194269, str_194272)
        # Adding element type (line 266)
        str_194273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 29), 'str', "^'not None' is not None : oops$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 28), list_194269, str_194273)
        
        # Processing the call keyword arguments (line 265)
        kwargs_194274 = {}
        # Getting the type of 'self' (line 265)
        self_194264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 265)
        assertMessages_194265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_194264, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 265)
        assertMessages_call_result_194275 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), assertMessages_194265, *[str_194266, tuple_194267, list_194269], **kwargs_194274)
        
        
        # ################# End of 'testAssertIsNone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertIsNone' in the type store
        # Getting the type of 'stypy_return_type' (line 264)
        stypy_return_type_194276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertIsNone'
        return stypy_return_type_194276


    @norecursion
    def testAssertIsNotNone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertIsNotNone'
        module_type_store = module_type_store.open_function_context('testAssertIsNotNone', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertIsNotNone')
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertIsNotNone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertIsNotNone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertIsNotNone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertIsNotNone(...)' code ##################

        
        # Call to assertMessages(...): (line 271)
        # Processing the call arguments (line 271)
        str_194279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 28), 'str', 'assertIsNotNone')
        
        # Obtaining an instance of the builtin type 'tuple' (line 271)
        tuple_194280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 271)
        # Adding element type (line 271)
        # Getting the type of 'None' (line 271)
        None_194281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 48), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 48), tuple_194280, None_194281)
        
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_194282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        str_194283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 29), 'str', '^unexpectedly None$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 28), list_194282, str_194283)
        # Adding element type (line 272)
        str_194284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 52), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 28), list_194282, str_194284)
        # Adding element type (line 272)
        str_194285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 29), 'str', '^unexpectedly None$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 28), list_194282, str_194285)
        # Adding element type (line 272)
        str_194286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 29), 'str', '^unexpectedly None : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 28), list_194282, str_194286)
        
        # Processing the call keyword arguments (line 271)
        kwargs_194287 = {}
        # Getting the type of 'self' (line 271)
        self_194277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 271)
        assertMessages_194278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), self_194277, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 271)
        assertMessages_call_result_194288 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), assertMessages_194278, *[str_194279, tuple_194280, list_194282], **kwargs_194287)
        
        
        # ################# End of 'testAssertIsNotNone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertIsNotNone' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_194289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertIsNotNone'
        return stypy_return_type_194289


    @norecursion
    def testAssertIs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertIs'
        module_type_store = module_type_store.open_function_context('testAssertIs', 276, 4, False)
        # Assigning a type to the variable 'self' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertIs')
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertIs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertIs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertIs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertIs(...)' code ##################

        
        # Call to assertMessages(...): (line 277)
        # Processing the call arguments (line 277)
        str_194292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 28), 'str', 'assertIs')
        
        # Obtaining an instance of the builtin type 'tuple' (line 277)
        tuple_194293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 277)
        # Adding element type (line 277)
        # Getting the type of 'None' (line 277)
        None_194294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 41), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 41), tuple_194293, None_194294)
        # Adding element type (line 277)
        str_194295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 47), 'str', 'foo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 41), tuple_194293, str_194295)
        
        
        # Obtaining an instance of the builtin type 'list' (line 278)
        list_194296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 278)
        # Adding element type (line 278)
        str_194297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 29), 'str', "^None is not 'foo'$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 28), list_194296, str_194297)
        # Adding element type (line 278)
        str_194298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 52), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 28), list_194296, str_194298)
        # Adding element type (line 278)
        str_194299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 29), 'str', "^None is not 'foo'$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 28), list_194296, str_194299)
        # Adding element type (line 278)
        str_194300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 29), 'str', "^None is not 'foo' : oops$")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 28), list_194296, str_194300)
        
        # Processing the call keyword arguments (line 277)
        kwargs_194301 = {}
        # Getting the type of 'self' (line 277)
        self_194290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 277)
        assertMessages_194291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 8), self_194290, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 277)
        assertMessages_call_result_194302 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), assertMessages_194291, *[str_194292, tuple_194293, list_194296], **kwargs_194301)
        
        
        # ################# End of 'testAssertIs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertIs' in the type store
        # Getting the type of 'stypy_return_type' (line 276)
        stypy_return_type_194303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194303)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertIs'
        return stypy_return_type_194303


    @norecursion
    def testAssertIsNot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'testAssertIsNot'
        module_type_store = module_type_store.open_function_context('testAssertIsNot', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_localization', localization)
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_function_name', 'TestLongMessage.testAssertIsNot')
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_param_names_list', [])
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLongMessage.testAssertIsNot.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.testAssertIsNot', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'testAssertIsNot', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'testAssertIsNot(...)' code ##################

        
        # Call to assertMessages(...): (line 283)
        # Processing the call arguments (line 283)
        str_194306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'str', 'assertIsNot')
        
        # Obtaining an instance of the builtin type 'tuple' (line 283)
        tuple_194307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 283)
        # Adding element type (line 283)
        # Getting the type of 'None' (line 283)
        None_194308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 44), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 44), tuple_194307, None_194308)
        # Adding element type (line 283)
        # Getting the type of 'None' (line 283)
        None_194309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 50), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 44), tuple_194307, None_194309)
        
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_194310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        # Adding element type (line 284)
        str_194311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 29), 'str', '^unexpectedly identical: None$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 28), list_194310, str_194311)
        # Adding element type (line 284)
        str_194312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 63), 'str', '^oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 28), list_194310, str_194312)
        # Adding element type (line 284)
        str_194313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 29), 'str', '^unexpectedly identical: None$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 28), list_194310, str_194313)
        # Adding element type (line 284)
        str_194314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 29), 'str', '^unexpectedly identical: None : oops$')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 28), list_194310, str_194314)
        
        # Processing the call keyword arguments (line 283)
        kwargs_194315 = {}
        # Getting the type of 'self' (line 283)
        self_194304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self', False)
        # Obtaining the member 'assertMessages' of a type (line 283)
        assertMessages_194305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_194304, 'assertMessages')
        # Calling assertMessages(args, kwargs) (line 283)
        assertMessages_call_result_194316 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assertMessages_194305, *[str_194306, tuple_194307, list_194310], **kwargs_194315)
        
        
        # ################# End of 'testAssertIsNot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'testAssertIsNot' in the type store
        # Getting the type of 'stypy_return_type' (line 282)
        stypy_return_type_194317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_194317)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'testAssertIsNot'
        return stypy_return_type_194317


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 109, 0, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLongMessage.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLongMessage' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'TestLongMessage', TestLongMessage)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 290)
    # Processing the call keyword arguments (line 290)
    kwargs_194320 = {}
    # Getting the type of 'unittest' (line 290)
    unittest_194318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 290)
    main_194319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 4), unittest_194318, 'main')
    # Calling main(args, kwargs) (line 290)
    main_call_result_194321 = invoke(stypy.reporting.localization.Localization(__file__, 290, 4), main_194319, *[], **kwargs_194320)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

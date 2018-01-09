
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Utilities for writing code that runs on Python 2 and 3'''
2: 
3: # Copyright (c) 2010-2012 Benjamin Peterson
4: #
5: # Permission is hereby granted, free of charge, to any person obtaining a copy of
6: # this software and associated documentation files (the "Software"), to deal in
7: # the Software without restriction, including without limitation the rights to
8: # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
9: # the Software, and to permit persons to whom the Software is furnished to do so,
10: # subject to the following conditions:
11: #
12: # The above copyright notice and this permission notice shall be included in all
13: # copies or substantial portions of the Software.
14: #
15: # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
16: # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
17: # FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
18: # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
19: # IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
20: # CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
21: 
22: import operator
23: import sys
24: import types
25: 
26: __author__ = "Benjamin Peterson <benjamin@python.org>"
27: __version__ = "1.2.0"
28: 
29: 
30: # True if we are running on Python 3.
31: PY3 = sys.version_info[0] == 3
32: 
33: if PY3:
34:     string_types = str,
35:     integer_types = int,
36:     class_types = type,
37:     text_type = str
38:     binary_type = bytes
39: 
40:     MAXSIZE = sys.maxsize
41: else:
42:     string_types = basestring,
43:     integer_types = (int, long)
44:     class_types = (type, types.ClassType)
45:     text_type = unicode
46:     binary_type = str
47: 
48:     if sys.platform.startswith("java"):
49:         # Jython always uses 32 bits.
50:         MAXSIZE = int((1 << 31) - 1)
51:     else:
52:         # It's possible to have sizeof(long) != sizeof(Py_ssize_t).
53:         class X(object):
54:             def __len__(self):
55:                 return 1 << 31
56:         try:
57:             len(X())
58:         except OverflowError:
59:             # 32-bit
60:             MAXSIZE = int((1 << 31) - 1)
61:         else:
62:             # 64-bit
63:             MAXSIZE = int((1 << 63) - 1)
64:             del X
65: 
66: 
67: def _add_doc(func, doc):
68:     '''Add documentation to a function.'''
69:     func.__doc__ = doc
70: 
71: 
72: def _import_module(name):
73:     '''Import module, returning the module after the last dot.'''
74:     __import__(name)
75:     return sys.modules[name]
76: 
77: 
78: # Replacement for lazy loading stuff in upstream six.  See gh-2764
79: if PY3:
80:     import builtins
81:     import functools
82:     reduce = functools.reduce
83:     zip = builtins.zip
84:     xrange = builtins.range
85: else:
86:     import __builtin__
87:     import itertools
88:     builtins = __builtin__
89:     reduce = __builtin__.reduce
90:     zip = itertools.izip
91:     xrange = __builtin__.xrange
92: 
93: 
94: if PY3:
95:     _meth_func = "__func__"
96:     _meth_self = "__self__"
97: 
98:     _func_code = "__code__"
99:     _func_defaults = "__defaults__"
100: 
101:     _iterkeys = "keys"
102:     _itervalues = "values"
103:     _iteritems = "items"
104: else:
105:     _meth_func = "im_func"
106:     _meth_self = "im_self"
107: 
108:     _func_code = "func_code"
109:     _func_defaults = "func_defaults"
110: 
111:     _iterkeys = "iterkeys"
112:     _itervalues = "itervalues"
113:     _iteritems = "iteritems"
114: 
115: 
116: try:
117:     advance_iterator = next
118: except NameError:
119:     def advance_iterator(it):
120:         return it.next()
121: next = advance_iterator
122: 
123: 
124: if PY3:
125:     def get_unbound_function(unbound):
126:         return unbound
127: 
128:     Iterator = object
129: 
130:     def callable(obj):
131:         return any("__call__" in klass.__dict__ for klass in type(obj).__mro__)
132: else:
133:     def get_unbound_function(unbound):
134:         return unbound.im_func
135: 
136:     class Iterator(object):
137: 
138:         def next(self):
139:             return type(self).__next__(self)
140: 
141:     callable = callable
142: _add_doc(get_unbound_function,
143:          '''Get the function out of a possibly unbound function''')
144: 
145: 
146: get_method_function = operator.attrgetter(_meth_func)
147: get_method_self = operator.attrgetter(_meth_self)
148: get_function_code = operator.attrgetter(_func_code)
149: get_function_defaults = operator.attrgetter(_func_defaults)
150: 
151: 
152: def iterkeys(d):
153:     '''Return an iterator over the keys of a dictionary.'''
154:     return iter(getattr(d, _iterkeys)())
155: 
156: 
157: def itervalues(d):
158:     '''Return an iterator over the values of a dictionary.'''
159:     return iter(getattr(d, _itervalues)())
160: 
161: 
162: def iteritems(d):
163:     '''Return an iterator over the (key, value) pairs of a dictionary.'''
164:     return iter(getattr(d, _iteritems)())
165: 
166: 
167: if PY3:
168:     def b(s):
169:         return s.encode("latin-1")
170: 
171:     def u(s):
172:         return s
173: 
174:     if sys.version_info[1] <= 1:
175:         def int2byte(i):
176:             return bytes((i,))
177:     else:
178:         # This is about 2x faster than the implementation above on 3.2+
179:         int2byte = operator.methodcaller("to_bytes", 1, "big")
180:     import io
181:     StringIO = io.StringIO
182:     BytesIO = io.BytesIO
183: else:
184:     def b(s):
185:         return s
186: 
187:     def u(s):
188:         return unicode(s, "unicode_escape")
189:     int2byte = chr
190:     import StringIO
191:     StringIO = BytesIO = StringIO.StringIO
192: _add_doc(b, '''Byte literal''')
193: _add_doc(u, '''Text literal''')
194: 
195: 
196: if PY3:
197:     import builtins
198:     exec_ = getattr(builtins, "exec")
199: 
200:     def reraise(tp, value, tb=None):
201:         if value.__traceback__ is not tb:
202:             raise value.with_traceback(tb)
203:         raise value
204: 
205:     print_ = getattr(builtins, "print")
206:     del builtins
207: 
208: else:
209:     def exec_(code, globs=None, locs=None):
210:         '''Execute code in a namespace.'''
211:         if globs is None:
212:             frame = sys._getframe(1)
213:             globs = frame.f_globals
214:             if locs is None:
215:                 locs = frame.f_locals
216:             del frame
217:         elif locs is None:
218:             locs = globs
219:         exec('''exec code in globs, locs''')
220: 
221:     exec_('''def reraise(tp, value, tb=None):
222:     raise tp, value, tb
223: ''')
224: 
225:     def print_(*args, **kwargs):
226:         '''The new-style print function.'''
227:         fp = kwargs.pop("file", sys.stdout)
228:         if fp is None:
229:             return
230: 
231:         def write(data):
232:             if not isinstance(data, basestring):
233:                 data = str(data)
234:             fp.write(data)
235:         want_unicode = False
236:         sep = kwargs.pop("sep", None)
237:         if sep is not None:
238:             if isinstance(sep, unicode):
239:                 want_unicode = True
240:             elif not isinstance(sep, str):
241:                 raise TypeError("sep must be None or a string")
242:         end = kwargs.pop("end", None)
243:         if end is not None:
244:             if isinstance(end, unicode):
245:                 want_unicode = True
246:             elif not isinstance(end, str):
247:                 raise TypeError("end must be None or a string")
248:         if kwargs:
249:             raise TypeError("invalid keyword arguments to print()")
250:         if not want_unicode:
251:             for arg in args:
252:                 if isinstance(arg, unicode):
253:                     want_unicode = True
254:                     break
255:         if want_unicode:
256:             newline = unicode("\n")
257:             space = unicode(" ")
258:         else:
259:             newline = "\n"
260:             space = " "
261:         if sep is None:
262:             sep = space
263:         if end is None:
264:             end = newline
265:         for i, arg in enumerate(args):
266:             if i:
267:                 write(sep)
268:             write(arg)
269:         write(end)
270: 
271: _add_doc(reraise, '''Reraise an exception.''')
272: 
273: 
274: def with_metaclass(meta, base=object):
275:     '''Create a base class with a metaclass.'''
276:     return meta("NewBase", (base,), {})
277: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_707692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Utilities for writing code that runs on Python 2 and 3')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import operator' statement (line 22)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import sys' statement (line 23)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import types' statement (line 24)
import types

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'types', types, module_type_store)


# Assigning a Str to a Name (line 26):
str_707693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'str', 'Benjamin Peterson <benjamin@python.org>')
# Assigning a type to the variable '__author__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__author__', str_707693)

# Assigning a Str to a Name (line 27):
str_707694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'str', '1.2.0')
# Assigning a type to the variable '__version__' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '__version__', str_707694)

# Assigning a Compare to a Name (line 31):


# Obtaining the type of the subscript
int_707695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'int')
# Getting the type of 'sys' (line 31)
sys_707696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 6), 'sys')
# Obtaining the member 'version_info' of a type (line 31)
version_info_707697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 6), sys_707696, 'version_info')
# Obtaining the member '__getitem__' of a type (line 31)
getitem___707698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 6), version_info_707697, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 31)
subscript_call_result_707699 = invoke(stypy.reporting.localization.Localization(__file__, 31, 6), getitem___707698, int_707695)

int_707700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'int')
# Applying the binary operator '==' (line 31)
result_eq_707701 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 6), '==', subscript_call_result_707699, int_707700)

# Assigning a type to the variable 'PY3' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'PY3', result_eq_707701)

# Getting the type of 'PY3' (line 33)
PY3_707702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 3), 'PY3')
# Testing the type of an if condition (line 33)
if_condition_707703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 0), PY3_707702)
# Assigning a type to the variable 'if_condition_707703' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'if_condition_707703', if_condition_707703)
# SSA begins for if statement (line 33)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Tuple to a Name (line 34):

# Obtaining an instance of the builtin type 'tuple' (line 34)
tuple_707704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 34)
# Adding element type (line 34)
# Getting the type of 'str' (line 34)
str_707705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), tuple_707704, str_707705)

# Assigning a type to the variable 'string_types' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'string_types', tuple_707704)

# Assigning a Tuple to a Name (line 35):

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_707706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
# Getting the type of 'int' (line 35)
int_707707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 20), tuple_707706, int_707707)

# Assigning a type to the variable 'integer_types' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'integer_types', tuple_707706)

# Assigning a Tuple to a Name (line 36):

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_707708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
# Getting the type of 'type' (line 36)
type_707709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 18), tuple_707708, type_707709)

# Assigning a type to the variable 'class_types' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'class_types', tuple_707708)

# Assigning a Name to a Name (line 37):
# Getting the type of 'str' (line 37)
str_707710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'str')
# Assigning a type to the variable 'text_type' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'text_type', str_707710)

# Assigning a Name to a Name (line 38):
# Getting the type of 'bytes' (line 38)
bytes_707711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'bytes')
# Assigning a type to the variable 'binary_type' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'binary_type', bytes_707711)

# Assigning a Attribute to a Name (line 40):
# Getting the type of 'sys' (line 40)
sys_707712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'sys')
# Obtaining the member 'maxsize' of a type (line 40)
maxsize_707713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 14), sys_707712, 'maxsize')
# Assigning a type to the variable 'MAXSIZE' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'MAXSIZE', maxsize_707713)
# SSA branch for the else part of an if statement (line 33)
module_type_store.open_ssa_branch('else')

# Assigning a Tuple to a Name (line 42):

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_707714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
# Getting the type of 'basestring' (line 42)
basestring_707715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'basestring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), tuple_707714, basestring_707715)

# Assigning a type to the variable 'string_types' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'string_types', tuple_707714)

# Assigning a Tuple to a Name (line 43):

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_707716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
# Getting the type of 'int' (line 43)
int_707717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_707716, int_707717)
# Adding element type (line 43)
# Getting the type of 'long' (line 43)
long_707718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_707716, long_707718)

# Assigning a type to the variable 'integer_types' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'integer_types', tuple_707716)

# Assigning a Tuple to a Name (line 44):

# Obtaining an instance of the builtin type 'tuple' (line 44)
tuple_707719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 44)
# Adding element type (line 44)
# Getting the type of 'type' (line 44)
type_707720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), tuple_707719, type_707720)
# Adding element type (line 44)
# Getting the type of 'types' (line 44)
types_707721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'types')
# Obtaining the member 'ClassType' of a type (line 44)
ClassType_707722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 25), types_707721, 'ClassType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), tuple_707719, ClassType_707722)

# Assigning a type to the variable 'class_types' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'class_types', tuple_707719)

# Assigning a Name to a Name (line 45):
# Getting the type of 'unicode' (line 45)
unicode_707723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'unicode')
# Assigning a type to the variable 'text_type' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'text_type', unicode_707723)

# Assigning a Name to a Name (line 46):
# Getting the type of 'str' (line 46)
str_707724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'str')
# Assigning a type to the variable 'binary_type' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'binary_type', str_707724)


# Call to startswith(...): (line 48)
# Processing the call arguments (line 48)
str_707728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 31), 'str', 'java')
# Processing the call keyword arguments (line 48)
kwargs_707729 = {}
# Getting the type of 'sys' (line 48)
sys_707725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 'sys', False)
# Obtaining the member 'platform' of a type (line 48)
platform_707726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 7), sys_707725, 'platform')
# Obtaining the member 'startswith' of a type (line 48)
startswith_707727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 7), platform_707726, 'startswith')
# Calling startswith(args, kwargs) (line 48)
startswith_call_result_707730 = invoke(stypy.reporting.localization.Localization(__file__, 48, 7), startswith_707727, *[str_707728], **kwargs_707729)

# Testing the type of an if condition (line 48)
if_condition_707731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), startswith_call_result_707730)
# Assigning a type to the variable 'if_condition_707731' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'if_condition_707731', if_condition_707731)
# SSA begins for if statement (line 48)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 50):

# Call to int(...): (line 50)
# Processing the call arguments (line 50)
int_707733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'int')
int_707734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 28), 'int')
# Applying the binary operator '<<' (line 50)
result_lshift_707735 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 23), '<<', int_707733, int_707734)

int_707736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 34), 'int')
# Applying the binary operator '-' (line 50)
result_sub_707737 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 22), '-', result_lshift_707735, int_707736)

# Processing the call keyword arguments (line 50)
kwargs_707738 = {}
# Getting the type of 'int' (line 50)
int_707732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'int', False)
# Calling int(args, kwargs) (line 50)
int_call_result_707739 = invoke(stypy.reporting.localization.Localization(__file__, 50, 18), int_707732, *[result_sub_707737], **kwargs_707738)

# Assigning a type to the variable 'MAXSIZE' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'MAXSIZE', int_call_result_707739)
# SSA branch for the else part of an if statement (line 48)
module_type_store.open_ssa_branch('else')
# Declaration of the 'X' class

class X(object, ):

    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 54, 12, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'self', type_of_self)
        
        # Passed parameters checking function
        X.__len__.__dict__.__setitem__('stypy_localization', localization)
        X.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        X.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        X.__len__.__dict__.__setitem__('stypy_function_name', 'X.__len__')
        X.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        X.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        X.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        X.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        X.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        X.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        X.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'X.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        int_707740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'int')
        int_707741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 28), 'int')
        # Applying the binary operator '<<' (line 55)
        result_lshift_707742 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 23), '<<', int_707740, int_707741)
        
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'stypy_return_type', result_lshift_707742)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_707743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_707743)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_707743


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 53, 8, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'X.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'X' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'X', X)


# SSA begins for try-except statement (line 56)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Call to len(...): (line 57)
# Processing the call arguments (line 57)

# Call to X(...): (line 57)
# Processing the call keyword arguments (line 57)
kwargs_707746 = {}
# Getting the type of 'X' (line 57)
X_707745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'X', False)
# Calling X(args, kwargs) (line 57)
X_call_result_707747 = invoke(stypy.reporting.localization.Localization(__file__, 57, 16), X_707745, *[], **kwargs_707746)

# Processing the call keyword arguments (line 57)
kwargs_707748 = {}
# Getting the type of 'len' (line 57)
len_707744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'len', False)
# Calling len(args, kwargs) (line 57)
len_call_result_707749 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), len_707744, *[X_call_result_707747], **kwargs_707748)

# SSA branch for the except part of a try statement (line 56)
# SSA branch for the except 'OverflowError' branch of a try statement (line 56)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 60):

# Call to int(...): (line 60)
# Processing the call arguments (line 60)
int_707751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 27), 'int')
int_707752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
# Applying the binary operator '<<' (line 60)
result_lshift_707753 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 27), '<<', int_707751, int_707752)

int_707754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 38), 'int')
# Applying the binary operator '-' (line 60)
result_sub_707755 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 26), '-', result_lshift_707753, int_707754)

# Processing the call keyword arguments (line 60)
kwargs_707756 = {}
# Getting the type of 'int' (line 60)
int_707750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'int', False)
# Calling int(args, kwargs) (line 60)
int_call_result_707757 = invoke(stypy.reporting.localization.Localization(__file__, 60, 22), int_707750, *[result_sub_707755], **kwargs_707756)

# Assigning a type to the variable 'MAXSIZE' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'MAXSIZE', int_call_result_707757)
# SSA branch for the else branch of a try statement (line 56)
module_type_store.open_ssa_branch('except else')

# Assigning a Call to a Name (line 63):

# Call to int(...): (line 63)
# Processing the call arguments (line 63)
int_707759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'int')
int_707760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'int')
# Applying the binary operator '<<' (line 63)
result_lshift_707761 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 27), '<<', int_707759, int_707760)

int_707762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 38), 'int')
# Applying the binary operator '-' (line 63)
result_sub_707763 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 26), '-', result_lshift_707761, int_707762)

# Processing the call keyword arguments (line 63)
kwargs_707764 = {}
# Getting the type of 'int' (line 63)
int_707758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'int', False)
# Calling int(args, kwargs) (line 63)
int_call_result_707765 = invoke(stypy.reporting.localization.Localization(__file__, 63, 22), int_707758, *[result_sub_707763], **kwargs_707764)

# Assigning a type to the variable 'MAXSIZE' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'MAXSIZE', int_call_result_707765)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 64, 12), module_type_store, 'X')
# SSA join for try-except statement (line 56)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 48)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 33)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _add_doc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_add_doc'
    module_type_store = module_type_store.open_function_context('_add_doc', 67, 0, False)
    
    # Passed parameters checking function
    _add_doc.stypy_localization = localization
    _add_doc.stypy_type_of_self = None
    _add_doc.stypy_type_store = module_type_store
    _add_doc.stypy_function_name = '_add_doc'
    _add_doc.stypy_param_names_list = ['func', 'doc']
    _add_doc.stypy_varargs_param_name = None
    _add_doc.stypy_kwargs_param_name = None
    _add_doc.stypy_call_defaults = defaults
    _add_doc.stypy_call_varargs = varargs
    _add_doc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_add_doc', ['func', 'doc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_add_doc', localization, ['func', 'doc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_add_doc(...)' code ##################

    str_707766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'str', 'Add documentation to a function.')
    
    # Assigning a Name to a Attribute (line 69):
    # Getting the type of 'doc' (line 69)
    doc_707767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'doc')
    # Getting the type of 'func' (line 69)
    func_707768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'func')
    # Setting the type of the member '__doc__' of a type (line 69)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), func_707768, '__doc__', doc_707767)
    
    # ################# End of '_add_doc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_add_doc' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_707769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707769)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_add_doc'
    return stypy_return_type_707769

# Assigning a type to the variable '_add_doc' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), '_add_doc', _add_doc)

@norecursion
def _import_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_import_module'
    module_type_store = module_type_store.open_function_context('_import_module', 72, 0, False)
    
    # Passed parameters checking function
    _import_module.stypy_localization = localization
    _import_module.stypy_type_of_self = None
    _import_module.stypy_type_store = module_type_store
    _import_module.stypy_function_name = '_import_module'
    _import_module.stypy_param_names_list = ['name']
    _import_module.stypy_varargs_param_name = None
    _import_module.stypy_kwargs_param_name = None
    _import_module.stypy_call_defaults = defaults
    _import_module.stypy_call_varargs = varargs
    _import_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_import_module', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_import_module', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_import_module(...)' code ##################

    str_707770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'Import module, returning the module after the last dot.')
    
    # Call to __import__(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'name' (line 74)
    name_707772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'name', False)
    # Processing the call keyword arguments (line 74)
    kwargs_707773 = {}
    # Getting the type of '__import__' (line 74)
    import___707771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), '__import__', False)
    # Calling __import__(args, kwargs) (line 74)
    import___call_result_707774 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), import___707771, *[name_707772], **kwargs_707773)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 75)
    name_707775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'name')
    # Getting the type of 'sys' (line 75)
    sys_707776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'sys')
    # Obtaining the member 'modules' of a type (line 75)
    modules_707777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), sys_707776, 'modules')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___707778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), modules_707777, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_707779 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), getitem___707778, name_707775)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', subscript_call_result_707779)
    
    # ################# End of '_import_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_import_module' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_707780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_import_module'
    return stypy_return_type_707780

# Assigning a type to the variable '_import_module' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), '_import_module', _import_module)

# Getting the type of 'PY3' (line 79)
PY3_707781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 3), 'PY3')
# Testing the type of an if condition (line 79)
if_condition_707782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 0), PY3_707781)
# Assigning a type to the variable 'if_condition_707782' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'if_condition_707782', if_condition_707782)
# SSA begins for if statement (line 79)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 80, 4))

# 'import builtins' statement (line 80)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_707783 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 80, 4), 'builtins')

if (type(import_707783) is not StypyTypeError):

    if (import_707783 != 'pyd_module'):
        __import__(import_707783)
        sys_modules_707784 = sys.modules[import_707783]
        import_module(stypy.reporting.localization.Localization(__file__, 80, 4), 'builtins', sys_modules_707784.module_type_store, module_type_store)
    else:
        import builtins

        import_module(stypy.reporting.localization.Localization(__file__, 80, 4), 'builtins', builtins, module_type_store)

else:
    # Assigning a type to the variable 'builtins' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'builtins', import_707783)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 81, 4))

# 'import functools' statement (line 81)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 81, 4), 'functools', functools, module_type_store)


# Assigning a Attribute to a Name (line 82):
# Getting the type of 'functools' (line 82)
functools_707785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'functools')
# Obtaining the member 'reduce' of a type (line 82)
reduce_707786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), functools_707785, 'reduce')
# Assigning a type to the variable 'reduce' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'reduce', reduce_707786)

# Assigning a Attribute to a Name (line 83):
# Getting the type of 'builtins' (line 83)
builtins_707787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 10), 'builtins')
# Obtaining the member 'zip' of a type (line 83)
zip_707788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 10), builtins_707787, 'zip')
# Assigning a type to the variable 'zip' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'zip', zip_707788)

# Assigning a Attribute to a Name (line 84):
# Getting the type of 'builtins' (line 84)
builtins_707789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'builtins')
# Obtaining the member 'range' of a type (line 84)
range_707790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), builtins_707789, 'range')
# Assigning a type to the variable 'xrange' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'xrange', range_707790)
# SSA branch for the else part of an if statement (line 79)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 86, 4))

# 'import __builtin__' statement (line 86)
import __builtin__

import_module(stypy.reporting.localization.Localization(__file__, 86, 4), '__builtin__', __builtin__, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 87, 4))

# 'import itertools' statement (line 87)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 87, 4), 'itertools', itertools, module_type_store)


# Assigning a Name to a Name (line 88):
# Getting the type of '__builtin__' (line 88)
builtin___707791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), '__builtin__')
# Assigning a type to the variable 'builtins' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'builtins', builtin___707791)

# Assigning a Attribute to a Name (line 89):
# Getting the type of '__builtin__' (line 89)
builtin___707792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), '__builtin__')
# Obtaining the member 'reduce' of a type (line 89)
reduce_707793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), builtin___707792, 'reduce')
# Assigning a type to the variable 'reduce' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'reduce', reduce_707793)

# Assigning a Attribute to a Name (line 90):
# Getting the type of 'itertools' (line 90)
itertools_707794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 10), 'itertools')
# Obtaining the member 'izip' of a type (line 90)
izip_707795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 10), itertools_707794, 'izip')
# Assigning a type to the variable 'zip' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'zip', izip_707795)

# Assigning a Attribute to a Name (line 91):
# Getting the type of '__builtin__' (line 91)
builtin___707796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), '__builtin__')
# Obtaining the member 'xrange' of a type (line 91)
xrange_707797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), builtin___707796, 'xrange')
# Assigning a type to the variable 'xrange' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'xrange', xrange_707797)
# SSA join for if statement (line 79)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'PY3' (line 94)
PY3_707798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 3), 'PY3')
# Testing the type of an if condition (line 94)
if_condition_707799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 0), PY3_707798)
# Assigning a type to the variable 'if_condition_707799' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'if_condition_707799', if_condition_707799)
# SSA begins for if statement (line 94)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 95):
str_707800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 17), 'str', '__func__')
# Assigning a type to the variable '_meth_func' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), '_meth_func', str_707800)

# Assigning a Str to a Name (line 96):
str_707801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'str', '__self__')
# Assigning a type to the variable '_meth_self' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), '_meth_self', str_707801)

# Assigning a Str to a Name (line 98):
str_707802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 17), 'str', '__code__')
# Assigning a type to the variable '_func_code' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), '_func_code', str_707802)

# Assigning a Str to a Name (line 99):
str_707803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 21), 'str', '__defaults__')
# Assigning a type to the variable '_func_defaults' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), '_func_defaults', str_707803)

# Assigning a Str to a Name (line 101):
str_707804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'str', 'keys')
# Assigning a type to the variable '_iterkeys' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), '_iterkeys', str_707804)

# Assigning a Str to a Name (line 102):
str_707805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 18), 'str', 'values')
# Assigning a type to the variable '_itervalues' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), '_itervalues', str_707805)

# Assigning a Str to a Name (line 103):
str_707806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 17), 'str', 'items')
# Assigning a type to the variable '_iteritems' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), '_iteritems', str_707806)
# SSA branch for the else part of an if statement (line 94)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 105):
str_707807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 17), 'str', 'im_func')
# Assigning a type to the variable '_meth_func' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), '_meth_func', str_707807)

# Assigning a Str to a Name (line 106):
str_707808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 17), 'str', 'im_self')
# Assigning a type to the variable '_meth_self' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), '_meth_self', str_707808)

# Assigning a Str to a Name (line 108):
str_707809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 17), 'str', 'func_code')
# Assigning a type to the variable '_func_code' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), '_func_code', str_707809)

# Assigning a Str to a Name (line 109):
str_707810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'str', 'func_defaults')
# Assigning a type to the variable '_func_defaults' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), '_func_defaults', str_707810)

# Assigning a Str to a Name (line 111):
str_707811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 16), 'str', 'iterkeys')
# Assigning a type to the variable '_iterkeys' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), '_iterkeys', str_707811)

# Assigning a Str to a Name (line 112):
str_707812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 18), 'str', 'itervalues')
# Assigning a type to the variable '_itervalues' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), '_itervalues', str_707812)

# Assigning a Str to a Name (line 113):
str_707813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 17), 'str', 'iteritems')
# Assigning a type to the variable '_iteritems' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), '_iteritems', str_707813)
# SSA join for if statement (line 94)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 116)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Name to a Name (line 117):
# Getting the type of 'next' (line 117)
next_707814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'next')
# Assigning a type to the variable 'advance_iterator' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'advance_iterator', next_707814)
# SSA branch for the except part of a try statement (line 116)
# SSA branch for the except 'NameError' branch of a try statement (line 116)
module_type_store.open_ssa_branch('except')

@norecursion
def advance_iterator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'advance_iterator'
    module_type_store = module_type_store.open_function_context('advance_iterator', 119, 4, False)
    
    # Passed parameters checking function
    advance_iterator.stypy_localization = localization
    advance_iterator.stypy_type_of_self = None
    advance_iterator.stypy_type_store = module_type_store
    advance_iterator.stypy_function_name = 'advance_iterator'
    advance_iterator.stypy_param_names_list = ['it']
    advance_iterator.stypy_varargs_param_name = None
    advance_iterator.stypy_kwargs_param_name = None
    advance_iterator.stypy_call_defaults = defaults
    advance_iterator.stypy_call_varargs = varargs
    advance_iterator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'advance_iterator', ['it'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'advance_iterator', localization, ['it'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'advance_iterator(...)' code ##################

    
    # Call to next(...): (line 120)
    # Processing the call keyword arguments (line 120)
    kwargs_707817 = {}
    # Getting the type of 'it' (line 120)
    it_707815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'it', False)
    # Obtaining the member 'next' of a type (line 120)
    next_707816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), it_707815, 'next')
    # Calling next(args, kwargs) (line 120)
    next_call_result_707818 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), next_707816, *[], **kwargs_707817)
    
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', next_call_result_707818)
    
    # ################# End of 'advance_iterator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'advance_iterator' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_707819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707819)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'advance_iterator'
    return stypy_return_type_707819

# Assigning a type to the variable 'advance_iterator' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'advance_iterator', advance_iterator)
# SSA join for try-except statement (line 116)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 121):
# Getting the type of 'advance_iterator' (line 121)
advance_iterator_707820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'advance_iterator')
# Assigning a type to the variable 'next' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'next', advance_iterator_707820)

# Getting the type of 'PY3' (line 124)
PY3_707821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 3), 'PY3')
# Testing the type of an if condition (line 124)
if_condition_707822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 0), PY3_707821)
# Assigning a type to the variable 'if_condition_707822' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'if_condition_707822', if_condition_707822)
# SSA begins for if statement (line 124)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def get_unbound_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_unbound_function'
    module_type_store = module_type_store.open_function_context('get_unbound_function', 125, 4, False)
    
    # Passed parameters checking function
    get_unbound_function.stypy_localization = localization
    get_unbound_function.stypy_type_of_self = None
    get_unbound_function.stypy_type_store = module_type_store
    get_unbound_function.stypy_function_name = 'get_unbound_function'
    get_unbound_function.stypy_param_names_list = ['unbound']
    get_unbound_function.stypy_varargs_param_name = None
    get_unbound_function.stypy_kwargs_param_name = None
    get_unbound_function.stypy_call_defaults = defaults
    get_unbound_function.stypy_call_varargs = varargs
    get_unbound_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_unbound_function', ['unbound'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_unbound_function', localization, ['unbound'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_unbound_function(...)' code ##################

    # Getting the type of 'unbound' (line 126)
    unbound_707823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'unbound')
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', unbound_707823)
    
    # ################# End of 'get_unbound_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_unbound_function' in the type store
    # Getting the type of 'stypy_return_type' (line 125)
    stypy_return_type_707824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707824)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_unbound_function'
    return stypy_return_type_707824

# Assigning a type to the variable 'get_unbound_function' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'get_unbound_function', get_unbound_function)

# Assigning a Name to a Name (line 128):
# Getting the type of 'object' (line 128)
object_707825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'object')
# Assigning a type to the variable 'Iterator' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'Iterator', object_707825)

@norecursion
def callable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'callable'
    module_type_store = module_type_store.open_function_context('callable', 130, 4, False)
    
    # Passed parameters checking function
    callable.stypy_localization = localization
    callable.stypy_type_of_self = None
    callable.stypy_type_store = module_type_store
    callable.stypy_function_name = 'callable'
    callable.stypy_param_names_list = ['obj']
    callable.stypy_varargs_param_name = None
    callable.stypy_kwargs_param_name = None
    callable.stypy_call_defaults = defaults
    callable.stypy_call_varargs = varargs
    callable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'callable', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'callable', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'callable(...)' code ##################

    
    # Call to any(...): (line 131)
    # Processing the call arguments (line 131)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 131, 19, True)
    # Calculating comprehension expression
    
    # Call to type(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'obj' (line 131)
    obj_707832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 66), 'obj', False)
    # Processing the call keyword arguments (line 131)
    kwargs_707833 = {}
    # Getting the type of 'type' (line 131)
    type_707831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 61), 'type', False)
    # Calling type(args, kwargs) (line 131)
    type_call_result_707834 = invoke(stypy.reporting.localization.Localization(__file__, 131, 61), type_707831, *[obj_707832], **kwargs_707833)
    
    # Obtaining the member '__mro__' of a type (line 131)
    mro___707835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 61), type_call_result_707834, '__mro__')
    comprehension_707836 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 19), mro___707835)
    # Assigning a type to the variable 'klass' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'klass', comprehension_707836)
    
    str_707827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 19), 'str', '__call__')
    # Getting the type of 'klass' (line 131)
    klass_707828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 33), 'klass', False)
    # Obtaining the member '__dict__' of a type (line 131)
    dict___707829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 33), klass_707828, '__dict__')
    # Applying the binary operator 'in' (line 131)
    result_contains_707830 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 19), 'in', str_707827, dict___707829)
    
    list_707837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 19), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 19), list_707837, result_contains_707830)
    # Processing the call keyword arguments (line 131)
    kwargs_707838 = {}
    # Getting the type of 'any' (line 131)
    any_707826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'any', False)
    # Calling any(args, kwargs) (line 131)
    any_call_result_707839 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), any_707826, *[list_707837], **kwargs_707838)
    
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', any_call_result_707839)
    
    # ################# End of 'callable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'callable' in the type store
    # Getting the type of 'stypy_return_type' (line 130)
    stypy_return_type_707840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707840)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'callable'
    return stypy_return_type_707840

# Assigning a type to the variable 'callable' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'callable', callable)
# SSA branch for the else part of an if statement (line 124)
module_type_store.open_ssa_branch('else')

@norecursion
def get_unbound_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_unbound_function'
    module_type_store = module_type_store.open_function_context('get_unbound_function', 133, 4, False)
    
    # Passed parameters checking function
    get_unbound_function.stypy_localization = localization
    get_unbound_function.stypy_type_of_self = None
    get_unbound_function.stypy_type_store = module_type_store
    get_unbound_function.stypy_function_name = 'get_unbound_function'
    get_unbound_function.stypy_param_names_list = ['unbound']
    get_unbound_function.stypy_varargs_param_name = None
    get_unbound_function.stypy_kwargs_param_name = None
    get_unbound_function.stypy_call_defaults = defaults
    get_unbound_function.stypy_call_varargs = varargs
    get_unbound_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_unbound_function', ['unbound'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_unbound_function', localization, ['unbound'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_unbound_function(...)' code ##################

    # Getting the type of 'unbound' (line 134)
    unbound_707841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'unbound')
    # Obtaining the member 'im_func' of a type (line 134)
    im_func_707842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), unbound_707841, 'im_func')
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', im_func_707842)
    
    # ################# End of 'get_unbound_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_unbound_function' in the type store
    # Getting the type of 'stypy_return_type' (line 133)
    stypy_return_type_707843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707843)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_unbound_function'
    return stypy_return_type_707843

# Assigning a type to the variable 'get_unbound_function' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'get_unbound_function', get_unbound_function)
# Declaration of the 'Iterator' class

class Iterator(object, ):

    @norecursion
    def next(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'next'
        module_type_store = module_type_store.open_function_context('next', 138, 8, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        Iterator.next.__dict__.__setitem__('stypy_localization', localization)
        Iterator.next.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Iterator.next.__dict__.__setitem__('stypy_type_store', module_type_store)
        Iterator.next.__dict__.__setitem__('stypy_function_name', 'Iterator.next')
        Iterator.next.__dict__.__setitem__('stypy_param_names_list', [])
        Iterator.next.__dict__.__setitem__('stypy_varargs_param_name', None)
        Iterator.next.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Iterator.next.__dict__.__setitem__('stypy_call_defaults', defaults)
        Iterator.next.__dict__.__setitem__('stypy_call_varargs', varargs)
        Iterator.next.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Iterator.next.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Iterator.next', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'next', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'next(...)' code ##################

        
        # Call to __next__(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'self' (line 139)
        self_707849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 39), 'self', False)
        # Processing the call keyword arguments (line 139)
        kwargs_707850 = {}
        
        # Call to type(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'self' (line 139)
        self_707845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'self', False)
        # Processing the call keyword arguments (line 139)
        kwargs_707846 = {}
        # Getting the type of 'type' (line 139)
        type_707844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'type', False)
        # Calling type(args, kwargs) (line 139)
        type_call_result_707847 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), type_707844, *[self_707845], **kwargs_707846)
        
        # Obtaining the member '__next__' of a type (line 139)
        next___707848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 19), type_call_result_707847, '__next__')
        # Calling __next__(args, kwargs) (line 139)
        next___call_result_707851 = invoke(stypy.reporting.localization.Localization(__file__, 139, 19), next___707848, *[self_707849], **kwargs_707850)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'stypy_return_type', next___call_result_707851)
        
        # ################# End of 'next(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'next' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_707852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_707852)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'next'
        return stypy_return_type_707852


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Iterator.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Iterator' (line 136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'Iterator', Iterator)

# Assigning a Name to a Name (line 141):
# Getting the type of 'callable' (line 141)
callable_707853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'callable')
# Assigning a type to the variable 'callable' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'callable', callable_707853)
# SSA join for if statement (line 124)
module_type_store = module_type_store.join_ssa_context()


# Call to _add_doc(...): (line 142)
# Processing the call arguments (line 142)
# Getting the type of 'get_unbound_function' (line 142)
get_unbound_function_707855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 9), 'get_unbound_function', False)
str_707856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 9), 'str', 'Get the function out of a possibly unbound function')
# Processing the call keyword arguments (line 142)
kwargs_707857 = {}
# Getting the type of '_add_doc' (line 142)
_add_doc_707854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 142)
_add_doc_call_result_707858 = invoke(stypy.reporting.localization.Localization(__file__, 142, 0), _add_doc_707854, *[get_unbound_function_707855, str_707856], **kwargs_707857)


# Assigning a Call to a Name (line 146):

# Call to attrgetter(...): (line 146)
# Processing the call arguments (line 146)
# Getting the type of '_meth_func' (line 146)
_meth_func_707861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), '_meth_func', False)
# Processing the call keyword arguments (line 146)
kwargs_707862 = {}
# Getting the type of 'operator' (line 146)
operator_707859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 22), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 146)
attrgetter_707860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 22), operator_707859, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 146)
attrgetter_call_result_707863 = invoke(stypy.reporting.localization.Localization(__file__, 146, 22), attrgetter_707860, *[_meth_func_707861], **kwargs_707862)

# Assigning a type to the variable 'get_method_function' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'get_method_function', attrgetter_call_result_707863)

# Assigning a Call to a Name (line 147):

# Call to attrgetter(...): (line 147)
# Processing the call arguments (line 147)
# Getting the type of '_meth_self' (line 147)
_meth_self_707866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), '_meth_self', False)
# Processing the call keyword arguments (line 147)
kwargs_707867 = {}
# Getting the type of 'operator' (line 147)
operator_707864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 147)
attrgetter_707865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 18), operator_707864, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 147)
attrgetter_call_result_707868 = invoke(stypy.reporting.localization.Localization(__file__, 147, 18), attrgetter_707865, *[_meth_self_707866], **kwargs_707867)

# Assigning a type to the variable 'get_method_self' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'get_method_self', attrgetter_call_result_707868)

# Assigning a Call to a Name (line 148):

# Call to attrgetter(...): (line 148)
# Processing the call arguments (line 148)
# Getting the type of '_func_code' (line 148)
_func_code_707871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), '_func_code', False)
# Processing the call keyword arguments (line 148)
kwargs_707872 = {}
# Getting the type of 'operator' (line 148)
operator_707869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 148)
attrgetter_707870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), operator_707869, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 148)
attrgetter_call_result_707873 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), attrgetter_707870, *[_func_code_707871], **kwargs_707872)

# Assigning a type to the variable 'get_function_code' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'get_function_code', attrgetter_call_result_707873)

# Assigning a Call to a Name (line 149):

# Call to attrgetter(...): (line 149)
# Processing the call arguments (line 149)
# Getting the type of '_func_defaults' (line 149)
_func_defaults_707876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 44), '_func_defaults', False)
# Processing the call keyword arguments (line 149)
kwargs_707877 = {}
# Getting the type of 'operator' (line 149)
operator_707874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 149)
attrgetter_707875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 24), operator_707874, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 149)
attrgetter_call_result_707878 = invoke(stypy.reporting.localization.Localization(__file__, 149, 24), attrgetter_707875, *[_func_defaults_707876], **kwargs_707877)

# Assigning a type to the variable 'get_function_defaults' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'get_function_defaults', attrgetter_call_result_707878)

@norecursion
def iterkeys(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iterkeys'
    module_type_store = module_type_store.open_function_context('iterkeys', 152, 0, False)
    
    # Passed parameters checking function
    iterkeys.stypy_localization = localization
    iterkeys.stypy_type_of_self = None
    iterkeys.stypy_type_store = module_type_store
    iterkeys.stypy_function_name = 'iterkeys'
    iterkeys.stypy_param_names_list = ['d']
    iterkeys.stypy_varargs_param_name = None
    iterkeys.stypy_kwargs_param_name = None
    iterkeys.stypy_call_defaults = defaults
    iterkeys.stypy_call_varargs = varargs
    iterkeys.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iterkeys', ['d'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iterkeys', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iterkeys(...)' code ##################

    str_707879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'str', 'Return an iterator over the keys of a dictionary.')
    
    # Call to iter(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Call to (...): (line 154)
    # Processing the call keyword arguments (line 154)
    kwargs_707886 = {}
    
    # Call to getattr(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'd' (line 154)
    d_707882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'd', False)
    # Getting the type of '_iterkeys' (line 154)
    _iterkeys_707883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), '_iterkeys', False)
    # Processing the call keyword arguments (line 154)
    kwargs_707884 = {}
    # Getting the type of 'getattr' (line 154)
    getattr_707881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'getattr', False)
    # Calling getattr(args, kwargs) (line 154)
    getattr_call_result_707885 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), getattr_707881, *[d_707882, _iterkeys_707883], **kwargs_707884)
    
    # Calling (args, kwargs) (line 154)
    _call_result_707887 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), getattr_call_result_707885, *[], **kwargs_707886)
    
    # Processing the call keyword arguments (line 154)
    kwargs_707888 = {}
    # Getting the type of 'iter' (line 154)
    iter_707880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'iter', False)
    # Calling iter(args, kwargs) (line 154)
    iter_call_result_707889 = invoke(stypy.reporting.localization.Localization(__file__, 154, 11), iter_707880, *[_call_result_707887], **kwargs_707888)
    
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type', iter_call_result_707889)
    
    # ################# End of 'iterkeys(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iterkeys' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_707890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707890)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iterkeys'
    return stypy_return_type_707890

# Assigning a type to the variable 'iterkeys' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'iterkeys', iterkeys)

@norecursion
def itervalues(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'itervalues'
    module_type_store = module_type_store.open_function_context('itervalues', 157, 0, False)
    
    # Passed parameters checking function
    itervalues.stypy_localization = localization
    itervalues.stypy_type_of_self = None
    itervalues.stypy_type_store = module_type_store
    itervalues.stypy_function_name = 'itervalues'
    itervalues.stypy_param_names_list = ['d']
    itervalues.stypy_varargs_param_name = None
    itervalues.stypy_kwargs_param_name = None
    itervalues.stypy_call_defaults = defaults
    itervalues.stypy_call_varargs = varargs
    itervalues.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'itervalues', ['d'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'itervalues', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'itervalues(...)' code ##################

    str_707891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 4), 'str', 'Return an iterator over the values of a dictionary.')
    
    # Call to iter(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Call to (...): (line 159)
    # Processing the call keyword arguments (line 159)
    kwargs_707898 = {}
    
    # Call to getattr(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'd' (line 159)
    d_707894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'd', False)
    # Getting the type of '_itervalues' (line 159)
    _itervalues_707895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), '_itervalues', False)
    # Processing the call keyword arguments (line 159)
    kwargs_707896 = {}
    # Getting the type of 'getattr' (line 159)
    getattr_707893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'getattr', False)
    # Calling getattr(args, kwargs) (line 159)
    getattr_call_result_707897 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), getattr_707893, *[d_707894, _itervalues_707895], **kwargs_707896)
    
    # Calling (args, kwargs) (line 159)
    _call_result_707899 = invoke(stypy.reporting.localization.Localization(__file__, 159, 16), getattr_call_result_707897, *[], **kwargs_707898)
    
    # Processing the call keyword arguments (line 159)
    kwargs_707900 = {}
    # Getting the type of 'iter' (line 159)
    iter_707892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'iter', False)
    # Calling iter(args, kwargs) (line 159)
    iter_call_result_707901 = invoke(stypy.reporting.localization.Localization(__file__, 159, 11), iter_707892, *[_call_result_707899], **kwargs_707900)
    
    # Assigning a type to the variable 'stypy_return_type' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type', iter_call_result_707901)
    
    # ################# End of 'itervalues(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'itervalues' in the type store
    # Getting the type of 'stypy_return_type' (line 157)
    stypy_return_type_707902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707902)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'itervalues'
    return stypy_return_type_707902

# Assigning a type to the variable 'itervalues' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'itervalues', itervalues)

@norecursion
def iteritems(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iteritems'
    module_type_store = module_type_store.open_function_context('iteritems', 162, 0, False)
    
    # Passed parameters checking function
    iteritems.stypy_localization = localization
    iteritems.stypy_type_of_self = None
    iteritems.stypy_type_store = module_type_store
    iteritems.stypy_function_name = 'iteritems'
    iteritems.stypy_param_names_list = ['d']
    iteritems.stypy_varargs_param_name = None
    iteritems.stypy_kwargs_param_name = None
    iteritems.stypy_call_defaults = defaults
    iteritems.stypy_call_varargs = varargs
    iteritems.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iteritems', ['d'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iteritems', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iteritems(...)' code ##################

    str_707903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 4), 'str', 'Return an iterator over the (key, value) pairs of a dictionary.')
    
    # Call to iter(...): (line 164)
    # Processing the call arguments (line 164)
    
    # Call to (...): (line 164)
    # Processing the call keyword arguments (line 164)
    kwargs_707910 = {}
    
    # Call to getattr(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'd' (line 164)
    d_707906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'd', False)
    # Getting the type of '_iteritems' (line 164)
    _iteritems_707907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), '_iteritems', False)
    # Processing the call keyword arguments (line 164)
    kwargs_707908 = {}
    # Getting the type of 'getattr' (line 164)
    getattr_707905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'getattr', False)
    # Calling getattr(args, kwargs) (line 164)
    getattr_call_result_707909 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), getattr_707905, *[d_707906, _iteritems_707907], **kwargs_707908)
    
    # Calling (args, kwargs) (line 164)
    _call_result_707911 = invoke(stypy.reporting.localization.Localization(__file__, 164, 16), getattr_call_result_707909, *[], **kwargs_707910)
    
    # Processing the call keyword arguments (line 164)
    kwargs_707912 = {}
    # Getting the type of 'iter' (line 164)
    iter_707904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'iter', False)
    # Calling iter(args, kwargs) (line 164)
    iter_call_result_707913 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), iter_707904, *[_call_result_707911], **kwargs_707912)
    
    # Assigning a type to the variable 'stypy_return_type' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type', iter_call_result_707913)
    
    # ################# End of 'iteritems(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iteritems' in the type store
    # Getting the type of 'stypy_return_type' (line 162)
    stypy_return_type_707914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707914)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iteritems'
    return stypy_return_type_707914

# Assigning a type to the variable 'iteritems' (line 162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'iteritems', iteritems)

# Getting the type of 'PY3' (line 167)
PY3_707915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 3), 'PY3')
# Testing the type of an if condition (line 167)
if_condition_707916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 0), PY3_707915)
# Assigning a type to the variable 'if_condition_707916' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'if_condition_707916', if_condition_707916)
# SSA begins for if statement (line 167)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'b'
    module_type_store = module_type_store.open_function_context('b', 168, 4, False)
    
    # Passed parameters checking function
    b.stypy_localization = localization
    b.stypy_type_of_self = None
    b.stypy_type_store = module_type_store
    b.stypy_function_name = 'b'
    b.stypy_param_names_list = ['s']
    b.stypy_varargs_param_name = None
    b.stypy_kwargs_param_name = None
    b.stypy_call_defaults = defaults
    b.stypy_call_varargs = varargs
    b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'b', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'b', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'b(...)' code ##################

    
    # Call to encode(...): (line 169)
    # Processing the call arguments (line 169)
    str_707919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'str', 'latin-1')
    # Processing the call keyword arguments (line 169)
    kwargs_707920 = {}
    # Getting the type of 's' (line 169)
    s_707917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 's', False)
    # Obtaining the member 'encode' of a type (line 169)
    encode_707918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 15), s_707917, 'encode')
    # Calling encode(args, kwargs) (line 169)
    encode_call_result_707921 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), encode_707918, *[str_707919], **kwargs_707920)
    
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', encode_call_result_707921)
    
    # ################# End of 'b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'b' in the type store
    # Getting the type of 'stypy_return_type' (line 168)
    stypy_return_type_707922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707922)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'b'
    return stypy_return_type_707922

# Assigning a type to the variable 'b' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'b', b)

@norecursion
def u(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'u'
    module_type_store = module_type_store.open_function_context('u', 171, 4, False)
    
    # Passed parameters checking function
    u.stypy_localization = localization
    u.stypy_type_of_self = None
    u.stypy_type_store = module_type_store
    u.stypy_function_name = 'u'
    u.stypy_param_names_list = ['s']
    u.stypy_varargs_param_name = None
    u.stypy_kwargs_param_name = None
    u.stypy_call_defaults = defaults
    u.stypy_call_varargs = varargs
    u.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'u', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'u', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'u(...)' code ##################

    # Getting the type of 's' (line 172)
    s_707923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', s_707923)
    
    # ################# End of 'u(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'u' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_707924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707924)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'u'
    return stypy_return_type_707924

# Assigning a type to the variable 'u' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'u', u)



# Obtaining the type of the subscript
int_707925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 24), 'int')
# Getting the type of 'sys' (line 174)
sys_707926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), 'sys')
# Obtaining the member 'version_info' of a type (line 174)
version_info_707927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 7), sys_707926, 'version_info')
# Obtaining the member '__getitem__' of a type (line 174)
getitem___707928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 7), version_info_707927, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 174)
subscript_call_result_707929 = invoke(stypy.reporting.localization.Localization(__file__, 174, 7), getitem___707928, int_707925)

int_707930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 30), 'int')
# Applying the binary operator '<=' (line 174)
result_le_707931 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), '<=', subscript_call_result_707929, int_707930)

# Testing the type of an if condition (line 174)
if_condition_707932 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_le_707931)
# Assigning a type to the variable 'if_condition_707932' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_707932', if_condition_707932)
# SSA begins for if statement (line 174)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def int2byte(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'int2byte'
    module_type_store = module_type_store.open_function_context('int2byte', 175, 8, False)
    
    # Passed parameters checking function
    int2byte.stypy_localization = localization
    int2byte.stypy_type_of_self = None
    int2byte.stypy_type_store = module_type_store
    int2byte.stypy_function_name = 'int2byte'
    int2byte.stypy_param_names_list = ['i']
    int2byte.stypy_varargs_param_name = None
    int2byte.stypy_kwargs_param_name = None
    int2byte.stypy_call_defaults = defaults
    int2byte.stypy_call_varargs = varargs
    int2byte.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'int2byte', ['i'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'int2byte', localization, ['i'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'int2byte(...)' code ##################

    
    # Call to bytes(...): (line 176)
    # Processing the call arguments (line 176)
    
    # Obtaining an instance of the builtin type 'tuple' (line 176)
    tuple_707934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 176)
    # Adding element type (line 176)
    # Getting the type of 'i' (line 176)
    i_707935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'i', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 26), tuple_707934, i_707935)
    
    # Processing the call keyword arguments (line 176)
    kwargs_707936 = {}
    # Getting the type of 'bytes' (line 176)
    bytes_707933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'bytes', False)
    # Calling bytes(args, kwargs) (line 176)
    bytes_call_result_707937 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), bytes_707933, *[tuple_707934], **kwargs_707936)
    
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'stypy_return_type', bytes_call_result_707937)
    
    # ################# End of 'int2byte(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'int2byte' in the type store
    # Getting the type of 'stypy_return_type' (line 175)
    stypy_return_type_707938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707938)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'int2byte'
    return stypy_return_type_707938

# Assigning a type to the variable 'int2byte' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'int2byte', int2byte)
# SSA branch for the else part of an if statement (line 174)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 179):

# Call to methodcaller(...): (line 179)
# Processing the call arguments (line 179)
str_707941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 41), 'str', 'to_bytes')
int_707942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 53), 'int')
str_707943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 56), 'str', 'big')
# Processing the call keyword arguments (line 179)
kwargs_707944 = {}
# Getting the type of 'operator' (line 179)
operator_707939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 179)
methodcaller_707940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 19), operator_707939, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 179)
methodcaller_call_result_707945 = invoke(stypy.reporting.localization.Localization(__file__, 179, 19), methodcaller_707940, *[str_707941, int_707942, str_707943], **kwargs_707944)

# Assigning a type to the variable 'int2byte' (line 179)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'int2byte', methodcaller_call_result_707945)
# SSA join for if statement (line 174)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 180, 4))

# 'import io' statement (line 180)
import io

import_module(stypy.reporting.localization.Localization(__file__, 180, 4), 'io', io, module_type_store)


# Assigning a Attribute to a Name (line 181):
# Getting the type of 'io' (line 181)
io_707946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'io')
# Obtaining the member 'StringIO' of a type (line 181)
StringIO_707947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 15), io_707946, 'StringIO')
# Assigning a type to the variable 'StringIO' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'StringIO', StringIO_707947)

# Assigning a Attribute to a Name (line 182):
# Getting the type of 'io' (line 182)
io_707948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'io')
# Obtaining the member 'BytesIO' of a type (line 182)
BytesIO_707949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 14), io_707948, 'BytesIO')
# Assigning a type to the variable 'BytesIO' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'BytesIO', BytesIO_707949)
# SSA branch for the else part of an if statement (line 167)
module_type_store.open_ssa_branch('else')

@norecursion
def b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'b'
    module_type_store = module_type_store.open_function_context('b', 184, 4, False)
    
    # Passed parameters checking function
    b.stypy_localization = localization
    b.stypy_type_of_self = None
    b.stypy_type_store = module_type_store
    b.stypy_function_name = 'b'
    b.stypy_param_names_list = ['s']
    b.stypy_varargs_param_name = None
    b.stypy_kwargs_param_name = None
    b.stypy_call_defaults = defaults
    b.stypy_call_varargs = varargs
    b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'b', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'b', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'b(...)' code ##################

    # Getting the type of 's' (line 185)
    s_707950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', s_707950)
    
    # ################# End of 'b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'b' in the type store
    # Getting the type of 'stypy_return_type' (line 184)
    stypy_return_type_707951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707951)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'b'
    return stypy_return_type_707951

# Assigning a type to the variable 'b' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'b', b)

@norecursion
def u(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'u'
    module_type_store = module_type_store.open_function_context('u', 187, 4, False)
    
    # Passed parameters checking function
    u.stypy_localization = localization
    u.stypy_type_of_self = None
    u.stypy_type_store = module_type_store
    u.stypy_function_name = 'u'
    u.stypy_param_names_list = ['s']
    u.stypy_varargs_param_name = None
    u.stypy_kwargs_param_name = None
    u.stypy_call_defaults = defaults
    u.stypy_call_varargs = varargs
    u.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'u', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'u', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'u(...)' code ##################

    
    # Call to unicode(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 's' (line 188)
    s_707953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 's', False)
    str_707954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 26), 'str', 'unicode_escape')
    # Processing the call keyword arguments (line 188)
    kwargs_707955 = {}
    # Getting the type of 'unicode' (line 188)
    unicode_707952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'unicode', False)
    # Calling unicode(args, kwargs) (line 188)
    unicode_call_result_707956 = invoke(stypy.reporting.localization.Localization(__file__, 188, 15), unicode_707952, *[s_707953, str_707954], **kwargs_707955)
    
    # Assigning a type to the variable 'stypy_return_type' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', unicode_call_result_707956)
    
    # ################# End of 'u(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'u' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_707957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707957)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'u'
    return stypy_return_type_707957

# Assigning a type to the variable 'u' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'u', u)

# Assigning a Name to a Name (line 189):
# Getting the type of 'chr' (line 189)
chr_707958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'chr')
# Assigning a type to the variable 'int2byte' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'int2byte', chr_707958)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 190, 4))

# 'import StringIO' statement (line 190)
import StringIO

import_module(stypy.reporting.localization.Localization(__file__, 190, 4), 'StringIO', StringIO, module_type_store)


# Multiple assignment of 2 elements.
# Getting the type of 'StringIO' (line 191)
StringIO_707959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'StringIO')
# Obtaining the member 'StringIO' of a type (line 191)
StringIO_707960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 25), StringIO_707959, 'StringIO')
# Assigning a type to the variable 'BytesIO' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'BytesIO', StringIO_707960)
# Getting the type of 'BytesIO' (line 191)
BytesIO_707961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'BytesIO')
# Assigning a type to the variable 'StringIO' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'StringIO', BytesIO_707961)
# SSA join for if statement (line 167)
module_type_store = module_type_store.join_ssa_context()


# Call to _add_doc(...): (line 192)
# Processing the call arguments (line 192)
# Getting the type of 'b' (line 192)
b_707963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 9), 'b', False)
str_707964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 12), 'str', 'Byte literal')
# Processing the call keyword arguments (line 192)
kwargs_707965 = {}
# Getting the type of '_add_doc' (line 192)
_add_doc_707962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 192)
_add_doc_call_result_707966 = invoke(stypy.reporting.localization.Localization(__file__, 192, 0), _add_doc_707962, *[b_707963, str_707964], **kwargs_707965)


# Call to _add_doc(...): (line 193)
# Processing the call arguments (line 193)
# Getting the type of 'u' (line 193)
u_707968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 9), 'u', False)
str_707969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 12), 'str', 'Text literal')
# Processing the call keyword arguments (line 193)
kwargs_707970 = {}
# Getting the type of '_add_doc' (line 193)
_add_doc_707967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 193)
_add_doc_call_result_707971 = invoke(stypy.reporting.localization.Localization(__file__, 193, 0), _add_doc_707967, *[u_707968, str_707969], **kwargs_707970)


# Getting the type of 'PY3' (line 196)
PY3_707972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 3), 'PY3')
# Testing the type of an if condition (line 196)
if_condition_707973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 0), PY3_707972)
# Assigning a type to the variable 'if_condition_707973' (line 196)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'if_condition_707973', if_condition_707973)
# SSA begins for if statement (line 196)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 197, 4))

# 'import builtins' statement (line 197)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_707974 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'builtins')

if (type(import_707974) is not StypyTypeError):

    if (import_707974 != 'pyd_module'):
        __import__(import_707974)
        sys_modules_707975 = sys.modules[import_707974]
        import_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'builtins', sys_modules_707975.module_type_store, module_type_store)
    else:
        import builtins

        import_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'builtins', builtins, module_type_store)

else:
    # Assigning a type to the variable 'builtins' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'builtins', import_707974)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')


# Assigning a Call to a Name (line 198):

# Call to getattr(...): (line 198)
# Processing the call arguments (line 198)
# Getting the type of 'builtins' (line 198)
builtins_707977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'builtins', False)
str_707978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 30), 'str', 'exec')
# Processing the call keyword arguments (line 198)
kwargs_707979 = {}
# Getting the type of 'getattr' (line 198)
getattr_707976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'getattr', False)
# Calling getattr(args, kwargs) (line 198)
getattr_call_result_707980 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), getattr_707976, *[builtins_707977, str_707978], **kwargs_707979)

# Assigning a type to the variable 'exec_' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'exec_', getattr_call_result_707980)

@norecursion
def reraise(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 200)
    None_707981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 30), 'None')
    defaults = [None_707981]
    # Create a new context for function 'reraise'
    module_type_store = module_type_store.open_function_context('reraise', 200, 4, False)
    
    # Passed parameters checking function
    reraise.stypy_localization = localization
    reraise.stypy_type_of_self = None
    reraise.stypy_type_store = module_type_store
    reraise.stypy_function_name = 'reraise'
    reraise.stypy_param_names_list = ['tp', 'value', 'tb']
    reraise.stypy_varargs_param_name = None
    reraise.stypy_kwargs_param_name = None
    reraise.stypy_call_defaults = defaults
    reraise.stypy_call_varargs = varargs
    reraise.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reraise', ['tp', 'value', 'tb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reraise', localization, ['tp', 'value', 'tb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reraise(...)' code ##################

    
    
    # Getting the type of 'value' (line 201)
    value_707982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'value')
    # Obtaining the member '__traceback__' of a type (line 201)
    traceback___707983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), value_707982, '__traceback__')
    # Getting the type of 'tb' (line 201)
    tb_707984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 38), 'tb')
    # Applying the binary operator 'isnot' (line 201)
    result_is_not_707985 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), 'isnot', traceback___707983, tb_707984)
    
    # Testing the type of an if condition (line 201)
    if_condition_707986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), result_is_not_707985)
    # Assigning a type to the variable 'if_condition_707986' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_707986', if_condition_707986)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to with_traceback(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'tb' (line 202)
    tb_707989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 39), 'tb', False)
    # Processing the call keyword arguments (line 202)
    kwargs_707990 = {}
    # Getting the type of 'value' (line 202)
    value_707987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'value', False)
    # Obtaining the member 'with_traceback' of a type (line 202)
    with_traceback_707988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 18), value_707987, 'with_traceback')
    # Calling with_traceback(args, kwargs) (line 202)
    with_traceback_call_result_707991 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), with_traceback_707988, *[tb_707989], **kwargs_707990)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 202, 12), with_traceback_call_result_707991, 'raise parameter', BaseException)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'value' (line 203)
    value_707992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'value')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 203, 8), value_707992, 'raise parameter', BaseException)
    
    # ################# End of 'reraise(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reraise' in the type store
    # Getting the type of 'stypy_return_type' (line 200)
    stypy_return_type_707993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_707993)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reraise'
    return stypy_return_type_707993

# Assigning a type to the variable 'reraise' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'reraise', reraise)

# Assigning a Call to a Name (line 205):

# Call to getattr(...): (line 205)
# Processing the call arguments (line 205)
# Getting the type of 'builtins' (line 205)
builtins_707995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'builtins', False)
str_707996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 31), 'str', 'print')
# Processing the call keyword arguments (line 205)
kwargs_707997 = {}
# Getting the type of 'getattr' (line 205)
getattr_707994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'getattr', False)
# Calling getattr(args, kwargs) (line 205)
getattr_call_result_707998 = invoke(stypy.reporting.localization.Localization(__file__, 205, 13), getattr_707994, *[builtins_707995, str_707996], **kwargs_707997)

# Assigning a type to the variable 'print_' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'print_', getattr_call_result_707998)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 206, 4), module_type_store, 'builtins')
# SSA branch for the else part of an if statement (line 196)
module_type_store.open_ssa_branch('else')

@norecursion
def exec_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 209)
    None_707999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 26), 'None')
    # Getting the type of 'None' (line 209)
    None_708000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 37), 'None')
    defaults = [None_707999, None_708000]
    # Create a new context for function 'exec_'
    module_type_store = module_type_store.open_function_context('exec_', 209, 4, False)
    
    # Passed parameters checking function
    exec_.stypy_localization = localization
    exec_.stypy_type_of_self = None
    exec_.stypy_type_store = module_type_store
    exec_.stypy_function_name = 'exec_'
    exec_.stypy_param_names_list = ['code', 'globs', 'locs']
    exec_.stypy_varargs_param_name = None
    exec_.stypy_kwargs_param_name = None
    exec_.stypy_call_defaults = defaults
    exec_.stypy_call_varargs = varargs
    exec_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exec_', ['code', 'globs', 'locs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exec_', localization, ['code', 'globs', 'locs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exec_(...)' code ##################

    str_708001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 8), 'str', 'Execute code in a namespace.')
    
    # Type idiom detected: calculating its left and rigth part (line 211)
    # Getting the type of 'globs' (line 211)
    globs_708002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'globs')
    # Getting the type of 'None' (line 211)
    None_708003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'None')
    
    (may_be_708004, more_types_in_union_708005) = may_be_none(globs_708002, None_708003)

    if may_be_708004:

        if more_types_in_union_708005:
            # Runtime conditional SSA (line 211)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 212):
        
        # Call to _getframe(...): (line 212)
        # Processing the call arguments (line 212)
        int_708008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 34), 'int')
        # Processing the call keyword arguments (line 212)
        kwargs_708009 = {}
        # Getting the type of 'sys' (line 212)
        sys_708006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'sys', False)
        # Obtaining the member '_getframe' of a type (line 212)
        _getframe_708007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), sys_708006, '_getframe')
        # Calling _getframe(args, kwargs) (line 212)
        _getframe_call_result_708010 = invoke(stypy.reporting.localization.Localization(__file__, 212, 20), _getframe_708007, *[int_708008], **kwargs_708009)
        
        # Assigning a type to the variable 'frame' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'frame', _getframe_call_result_708010)
        
        # Assigning a Attribute to a Name (line 213):
        # Getting the type of 'frame' (line 213)
        frame_708011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'frame')
        # Obtaining the member 'f_globals' of a type (line 213)
        f_globals_708012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), frame_708011, 'f_globals')
        # Assigning a type to the variable 'globs' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'globs', f_globals_708012)
        
        # Type idiom detected: calculating its left and rigth part (line 214)
        # Getting the type of 'locs' (line 214)
        locs_708013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'locs')
        # Getting the type of 'None' (line 214)
        None_708014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'None')
        
        (may_be_708015, more_types_in_union_708016) = may_be_none(locs_708013, None_708014)

        if may_be_708015:

            if more_types_in_union_708016:
                # Runtime conditional SSA (line 214)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 215):
            # Getting the type of 'frame' (line 215)
            frame_708017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'frame')
            # Obtaining the member 'f_locals' of a type (line 215)
            f_locals_708018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 23), frame_708017, 'f_locals')
            # Assigning a type to the variable 'locs' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'locs', f_locals_708018)

            if more_types_in_union_708016:
                # SSA join for if statement (line 214)
                module_type_store = module_type_store.join_ssa_context()


        
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 216, 12), module_type_store, 'frame')

        if more_types_in_union_708005:
            # Runtime conditional SSA for else branch (line 211)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_708004) or more_types_in_union_708005):
        
        # Type idiom detected: calculating its left and rigth part (line 217)
        # Getting the type of 'locs' (line 217)
        locs_708019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'locs')
        # Getting the type of 'None' (line 217)
        None_708020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'None')
        
        (may_be_708021, more_types_in_union_708022) = may_be_none(locs_708019, None_708020)

        if may_be_708021:

            if more_types_in_union_708022:
                # Runtime conditional SSA (line 217)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'globs' (line 218)
            globs_708023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'globs')
            # Assigning a type to the variable 'locs' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'locs', globs_708023)

            if more_types_in_union_708022:
                # SSA join for if statement (line 217)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_708004 and more_types_in_union_708005):
            # SSA join for if statement (line 211)
            module_type_store = module_type_store.join_ssa_context()


    
    # Dynamic code evaluation using an exec statement
    str_708024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 13), 'str', 'exec code in globs, locs')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 219, 8), str_708024, 'exec parameter', 'StringType', 'FileType', 'CodeType')
    enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 219, 8))
    
    # ################# End of 'exec_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exec_' in the type store
    # Getting the type of 'stypy_return_type' (line 209)
    stypy_return_type_708025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708025)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exec_'
    return stypy_return_type_708025

# Assigning a type to the variable 'exec_' (line 209)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'exec_', exec_)

# Call to exec_(...): (line 221)
# Processing the call arguments (line 221)
str_708027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, (-1)), 'str', 'def reraise(tp, value, tb=None):\n    raise tp, value, tb\n')
# Processing the call keyword arguments (line 221)
kwargs_708028 = {}
# Getting the type of 'exec_' (line 221)
exec__708026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'exec_', False)
# Calling exec_(args, kwargs) (line 221)
exec__call_result_708029 = invoke(stypy.reporting.localization.Localization(__file__, 221, 4), exec__708026, *[str_708027], **kwargs_708028)


@norecursion
def print_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_'
    module_type_store = module_type_store.open_function_context('print_', 225, 4, False)
    
    # Passed parameters checking function
    print_.stypy_localization = localization
    print_.stypy_type_of_self = None
    print_.stypy_type_store = module_type_store
    print_.stypy_function_name = 'print_'
    print_.stypy_param_names_list = []
    print_.stypy_varargs_param_name = 'args'
    print_.stypy_kwargs_param_name = 'kwargs'
    print_.stypy_call_defaults = defaults
    print_.stypy_call_varargs = varargs
    print_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_(...)' code ##################

    str_708030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'str', 'The new-style print function.')
    
    # Assigning a Call to a Name (line 227):
    
    # Call to pop(...): (line 227)
    # Processing the call arguments (line 227)
    str_708033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 24), 'str', 'file')
    # Getting the type of 'sys' (line 227)
    sys_708034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 32), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 227)
    stdout_708035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 32), sys_708034, 'stdout')
    # Processing the call keyword arguments (line 227)
    kwargs_708036 = {}
    # Getting the type of 'kwargs' (line 227)
    kwargs_708031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 13), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 227)
    pop_708032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 13), kwargs_708031, 'pop')
    # Calling pop(args, kwargs) (line 227)
    pop_call_result_708037 = invoke(stypy.reporting.localization.Localization(__file__, 227, 13), pop_708032, *[str_708033, stdout_708035], **kwargs_708036)
    
    # Assigning a type to the variable 'fp' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'fp', pop_call_result_708037)
    
    # Type idiom detected: calculating its left and rigth part (line 228)
    # Getting the type of 'fp' (line 228)
    fp_708038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'fp')
    # Getting the type of 'None' (line 228)
    None_708039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'None')
    
    (may_be_708040, more_types_in_union_708041) = may_be_none(fp_708038, None_708039)

    if may_be_708040:

        if more_types_in_union_708041:
            # Runtime conditional SSA (line 228)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'stypy_return_type', types.NoneType)

        if more_types_in_union_708041:
            # SSA join for if statement (line 228)
            module_type_store = module_type_store.join_ssa_context()


    

    @norecursion
    def write(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write'
        module_type_store = module_type_store.open_function_context('write', 231, 8, False)
        
        # Passed parameters checking function
        write.stypy_localization = localization
        write.stypy_type_of_self = None
        write.stypy_type_store = module_type_store
        write.stypy_function_name = 'write'
        write.stypy_param_names_list = ['data']
        write.stypy_varargs_param_name = None
        write.stypy_kwargs_param_name = None
        write.stypy_call_defaults = defaults
        write.stypy_call_varargs = varargs
        write.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'write', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 232)
        # Getting the type of 'basestring' (line 232)
        basestring_708042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 36), 'basestring')
        # Getting the type of 'data' (line 232)
        data_708043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'data')
        
        (may_be_708044, more_types_in_union_708045) = may_not_be_subtype(basestring_708042, data_708043)

        if may_be_708044:

            if more_types_in_union_708045:
                # Runtime conditional SSA (line 232)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'data' (line 232)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'data', remove_subtype_from_union(data_708043, basestring))
            
            # Assigning a Call to a Name (line 233):
            
            # Call to str(...): (line 233)
            # Processing the call arguments (line 233)
            # Getting the type of 'data' (line 233)
            data_708047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'data', False)
            # Processing the call keyword arguments (line 233)
            kwargs_708048 = {}
            # Getting the type of 'str' (line 233)
            str_708046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 23), 'str', False)
            # Calling str(args, kwargs) (line 233)
            str_call_result_708049 = invoke(stypy.reporting.localization.Localization(__file__, 233, 23), str_708046, *[data_708047], **kwargs_708048)
            
            # Assigning a type to the variable 'data' (line 233)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'data', str_call_result_708049)

            if more_types_in_union_708045:
                # SSA join for if statement (line 232)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to write(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'data' (line 234)
        data_708052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'data', False)
        # Processing the call keyword arguments (line 234)
        kwargs_708053 = {}
        # Getting the type of 'fp' (line 234)
        fp_708050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'fp', False)
        # Obtaining the member 'write' of a type (line 234)
        write_708051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), fp_708050, 'write')
        # Calling write(args, kwargs) (line 234)
        write_call_result_708054 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), write_708051, *[data_708052], **kwargs_708053)
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_708055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708055)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_708055

    # Assigning a type to the variable 'write' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'write', write)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'False' (line 235)
    False_708056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'False')
    # Assigning a type to the variable 'want_unicode' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'want_unicode', False_708056)
    
    # Assigning a Call to a Name (line 236):
    
    # Call to pop(...): (line 236)
    # Processing the call arguments (line 236)
    str_708059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 25), 'str', 'sep')
    # Getting the type of 'None' (line 236)
    None_708060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 32), 'None', False)
    # Processing the call keyword arguments (line 236)
    kwargs_708061 = {}
    # Getting the type of 'kwargs' (line 236)
    kwargs_708057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 14), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 236)
    pop_708058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 14), kwargs_708057, 'pop')
    # Calling pop(args, kwargs) (line 236)
    pop_call_result_708062 = invoke(stypy.reporting.localization.Localization(__file__, 236, 14), pop_708058, *[str_708059, None_708060], **kwargs_708061)
    
    # Assigning a type to the variable 'sep' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'sep', pop_call_result_708062)
    
    # Type idiom detected: calculating its left and rigth part (line 237)
    # Getting the type of 'sep' (line 237)
    sep_708063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'sep')
    # Getting the type of 'None' (line 237)
    None_708064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'None')
    
    (may_be_708065, more_types_in_union_708066) = may_not_be_none(sep_708063, None_708064)

    if may_be_708065:

        if more_types_in_union_708066:
            # Runtime conditional SSA (line 237)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 238)
        # Getting the type of 'unicode' (line 238)
        unicode_708067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 31), 'unicode')
        # Getting the type of 'sep' (line 238)
        sep_708068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 26), 'sep')
        
        (may_be_708069, more_types_in_union_708070) = may_be_subtype(unicode_708067, sep_708068)

        if may_be_708069:

            if more_types_in_union_708070:
                # Runtime conditional SSA (line 238)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'sep' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'sep', remove_not_subtype_from_union(sep_708068, unicode))
            
            # Assigning a Name to a Name (line 239):
            # Getting the type of 'True' (line 239)
            True_708071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'True')
            # Assigning a type to the variable 'want_unicode' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'want_unicode', True_708071)

            if more_types_in_union_708070:
                # Runtime conditional SSA for else branch (line 238)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_708069) or more_types_in_union_708070):
            # Assigning a type to the variable 'sep' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'sep', remove_subtype_from_union(sep_708068, unicode))
            
            # Type idiom detected: calculating its left and rigth part (line 240)
            # Getting the type of 'str' (line 240)
            str_708072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 37), 'str')
            # Getting the type of 'sep' (line 240)
            sep_708073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 32), 'sep')
            
            (may_be_708074, more_types_in_union_708075) = may_not_be_subtype(str_708072, sep_708073)

            if may_be_708074:

                if more_types_in_union_708075:
                    # Runtime conditional SSA (line 240)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'sep' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 17), 'sep', remove_subtype_from_union(sep_708073, str))
                
                # Call to TypeError(...): (line 241)
                # Processing the call arguments (line 241)
                str_708077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 32), 'str', 'sep must be None or a string')
                # Processing the call keyword arguments (line 241)
                kwargs_708078 = {}
                # Getting the type of 'TypeError' (line 241)
                TypeError_708076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 241)
                TypeError_call_result_708079 = invoke(stypy.reporting.localization.Localization(__file__, 241, 22), TypeError_708076, *[str_708077], **kwargs_708078)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 241, 16), TypeError_call_result_708079, 'raise parameter', BaseException)

                if more_types_in_union_708075:
                    # SSA join for if statement (line 240)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_708069 and more_types_in_union_708070):
                # SSA join for if statement (line 238)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_708066:
            # SSA join for if statement (line 237)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 242):
    
    # Call to pop(...): (line 242)
    # Processing the call arguments (line 242)
    str_708082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 25), 'str', 'end')
    # Getting the type of 'None' (line 242)
    None_708083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 'None', False)
    # Processing the call keyword arguments (line 242)
    kwargs_708084 = {}
    # Getting the type of 'kwargs' (line 242)
    kwargs_708080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 242)
    pop_708081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 14), kwargs_708080, 'pop')
    # Calling pop(args, kwargs) (line 242)
    pop_call_result_708085 = invoke(stypy.reporting.localization.Localization(__file__, 242, 14), pop_708081, *[str_708082, None_708083], **kwargs_708084)
    
    # Assigning a type to the variable 'end' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'end', pop_call_result_708085)
    
    # Type idiom detected: calculating its left and rigth part (line 243)
    # Getting the type of 'end' (line 243)
    end_708086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'end')
    # Getting the type of 'None' (line 243)
    None_708087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'None')
    
    (may_be_708088, more_types_in_union_708089) = may_not_be_none(end_708086, None_708087)

    if may_be_708088:

        if more_types_in_union_708089:
            # Runtime conditional SSA (line 243)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 244)
        # Getting the type of 'unicode' (line 244)
        unicode_708090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 31), 'unicode')
        # Getting the type of 'end' (line 244)
        end_708091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 26), 'end')
        
        (may_be_708092, more_types_in_union_708093) = may_be_subtype(unicode_708090, end_708091)

        if may_be_708092:

            if more_types_in_union_708093:
                # Runtime conditional SSA (line 244)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'end' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'end', remove_not_subtype_from_union(end_708091, unicode))
            
            # Assigning a Name to a Name (line 245):
            # Getting the type of 'True' (line 245)
            True_708094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 31), 'True')
            # Assigning a type to the variable 'want_unicode' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 16), 'want_unicode', True_708094)

            if more_types_in_union_708093:
                # Runtime conditional SSA for else branch (line 244)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_708092) or more_types_in_union_708093):
            # Assigning a type to the variable 'end' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'end', remove_subtype_from_union(end_708091, unicode))
            
            # Type idiom detected: calculating its left and rigth part (line 246)
            # Getting the type of 'str' (line 246)
            str_708095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'str')
            # Getting the type of 'end' (line 246)
            end_708096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 32), 'end')
            
            (may_be_708097, more_types_in_union_708098) = may_not_be_subtype(str_708095, end_708096)

            if may_be_708097:

                if more_types_in_union_708098:
                    # Runtime conditional SSA (line 246)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'end' (line 246)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'end', remove_subtype_from_union(end_708096, str))
                
                # Call to TypeError(...): (line 247)
                # Processing the call arguments (line 247)
                str_708100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 32), 'str', 'end must be None or a string')
                # Processing the call keyword arguments (line 247)
                kwargs_708101 = {}
                # Getting the type of 'TypeError' (line 247)
                TypeError_708099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 247)
                TypeError_call_result_708102 = invoke(stypy.reporting.localization.Localization(__file__, 247, 22), TypeError_708099, *[str_708100], **kwargs_708101)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 247, 16), TypeError_call_result_708102, 'raise parameter', BaseException)

                if more_types_in_union_708098:
                    # SSA join for if statement (line 246)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_708092 and more_types_in_union_708093):
                # SSA join for if statement (line 244)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_708089:
            # SSA join for if statement (line 243)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'kwargs' (line 248)
    kwargs_708103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'kwargs')
    # Testing the type of an if condition (line 248)
    if_condition_708104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 8), kwargs_708103)
    # Assigning a type to the variable 'if_condition_708104' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'if_condition_708104', if_condition_708104)
    # SSA begins for if statement (line 248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 249)
    # Processing the call arguments (line 249)
    str_708106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 28), 'str', 'invalid keyword arguments to print()')
    # Processing the call keyword arguments (line 249)
    kwargs_708107 = {}
    # Getting the type of 'TypeError' (line 249)
    TypeError_708105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 249)
    TypeError_call_result_708108 = invoke(stypy.reporting.localization.Localization(__file__, 249, 18), TypeError_708105, *[str_708106], **kwargs_708107)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 249, 12), TypeError_call_result_708108, 'raise parameter', BaseException)
    # SSA join for if statement (line 248)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'want_unicode' (line 250)
    want_unicode_708109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'want_unicode')
    # Applying the 'not' unary operator (line 250)
    result_not__708110 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 11), 'not', want_unicode_708109)
    
    # Testing the type of an if condition (line 250)
    if_condition_708111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 8), result_not__708110)
    # Assigning a type to the variable 'if_condition_708111' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'if_condition_708111', if_condition_708111)
    # SSA begins for if statement (line 250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'args' (line 251)
    args_708112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'args')
    # Testing the type of a for loop iterable (line 251)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 251, 12), args_708112)
    # Getting the type of the for loop variable (line 251)
    for_loop_var_708113 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 251, 12), args_708112)
    # Assigning a type to the variable 'arg' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'arg', for_loop_var_708113)
    # SSA begins for a for statement (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 252)
    # Getting the type of 'unicode' (line 252)
    unicode_708114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 35), 'unicode')
    # Getting the type of 'arg' (line 252)
    arg_708115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 30), 'arg')
    
    (may_be_708116, more_types_in_union_708117) = may_be_subtype(unicode_708114, arg_708115)

    if may_be_708116:

        if more_types_in_union_708117:
            # Runtime conditional SSA (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'arg' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'arg', remove_not_subtype_from_union(arg_708115, unicode))
        
        # Assigning a Name to a Name (line 253):
        # Getting the type of 'True' (line 253)
        True_708118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'True')
        # Assigning a type to the variable 'want_unicode' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'want_unicode', True_708118)

        if more_types_in_union_708117:
            # SSA join for if statement (line 252)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 250)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'want_unicode' (line 255)
    want_unicode_708119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'want_unicode')
    # Testing the type of an if condition (line 255)
    if_condition_708120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), want_unicode_708119)
    # Assigning a type to the variable 'if_condition_708120' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_708120', if_condition_708120)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 256):
    
    # Call to unicode(...): (line 256)
    # Processing the call arguments (line 256)
    str_708122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 30), 'str', '\n')
    # Processing the call keyword arguments (line 256)
    kwargs_708123 = {}
    # Getting the type of 'unicode' (line 256)
    unicode_708121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'unicode', False)
    # Calling unicode(args, kwargs) (line 256)
    unicode_call_result_708124 = invoke(stypy.reporting.localization.Localization(__file__, 256, 22), unicode_708121, *[str_708122], **kwargs_708123)
    
    # Assigning a type to the variable 'newline' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'newline', unicode_call_result_708124)
    
    # Assigning a Call to a Name (line 257):
    
    # Call to unicode(...): (line 257)
    # Processing the call arguments (line 257)
    str_708126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'str', ' ')
    # Processing the call keyword arguments (line 257)
    kwargs_708127 = {}
    # Getting the type of 'unicode' (line 257)
    unicode_708125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'unicode', False)
    # Calling unicode(args, kwargs) (line 257)
    unicode_call_result_708128 = invoke(stypy.reporting.localization.Localization(__file__, 257, 20), unicode_708125, *[str_708126], **kwargs_708127)
    
    # Assigning a type to the variable 'space' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'space', unicode_call_result_708128)
    # SSA branch for the else part of an if statement (line 255)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 259):
    str_708129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 22), 'str', '\n')
    # Assigning a type to the variable 'newline' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'newline', str_708129)
    
    # Assigning a Str to a Name (line 260):
    str_708130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 20), 'str', ' ')
    # Assigning a type to the variable 'space' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'space', str_708130)
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 261)
    # Getting the type of 'sep' (line 261)
    sep_708131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'sep')
    # Getting the type of 'None' (line 261)
    None_708132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 18), 'None')
    
    (may_be_708133, more_types_in_union_708134) = may_be_none(sep_708131, None_708132)

    if may_be_708133:

        if more_types_in_union_708134:
            # Runtime conditional SSA (line 261)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 262):
        # Getting the type of 'space' (line 262)
        space_708135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 18), 'space')
        # Assigning a type to the variable 'sep' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'sep', space_708135)

        if more_types_in_union_708134:
            # SSA join for if statement (line 261)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 263)
    # Getting the type of 'end' (line 263)
    end_708136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'end')
    # Getting the type of 'None' (line 263)
    None_708137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 18), 'None')
    
    (may_be_708138, more_types_in_union_708139) = may_be_none(end_708136, None_708137)

    if may_be_708138:

        if more_types_in_union_708139:
            # Runtime conditional SSA (line 263)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 264):
        # Getting the type of 'newline' (line 264)
        newline_708140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 18), 'newline')
        # Assigning a type to the variable 'end' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'end', newline_708140)

        if more_types_in_union_708139:
            # SSA join for if statement (line 263)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to enumerate(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'args' (line 265)
    args_708142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 32), 'args', False)
    # Processing the call keyword arguments (line 265)
    kwargs_708143 = {}
    # Getting the type of 'enumerate' (line 265)
    enumerate_708141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 265)
    enumerate_call_result_708144 = invoke(stypy.reporting.localization.Localization(__file__, 265, 22), enumerate_708141, *[args_708142], **kwargs_708143)
    
    # Testing the type of a for loop iterable (line 265)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 265, 8), enumerate_call_result_708144)
    # Getting the type of the for loop variable (line 265)
    for_loop_var_708145 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 265, 8), enumerate_call_result_708144)
    # Assigning a type to the variable 'i' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), for_loop_var_708145))
    # Assigning a type to the variable 'arg' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 8), for_loop_var_708145))
    # SSA begins for a for statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'i' (line 266)
    i_708146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'i')
    # Testing the type of an if condition (line 266)
    if_condition_708147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 12), i_708146)
    # Assigning a type to the variable 'if_condition_708147' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'if_condition_708147', if_condition_708147)
    # SSA begins for if statement (line 266)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to write(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'sep' (line 267)
    sep_708149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 22), 'sep', False)
    # Processing the call keyword arguments (line 267)
    kwargs_708150 = {}
    # Getting the type of 'write' (line 267)
    write_708148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'write', False)
    # Calling write(args, kwargs) (line 267)
    write_call_result_708151 = invoke(stypy.reporting.localization.Localization(__file__, 267, 16), write_708148, *[sep_708149], **kwargs_708150)
    
    # SSA join for if statement (line 266)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'arg' (line 268)
    arg_708153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 18), 'arg', False)
    # Processing the call keyword arguments (line 268)
    kwargs_708154 = {}
    # Getting the type of 'write' (line 268)
    write_708152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'write', False)
    # Calling write(args, kwargs) (line 268)
    write_call_result_708155 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), write_708152, *[arg_708153], **kwargs_708154)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'end' (line 269)
    end_708157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 14), 'end', False)
    # Processing the call keyword arguments (line 269)
    kwargs_708158 = {}
    # Getting the type of 'write' (line 269)
    write_708156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'write', False)
    # Calling write(args, kwargs) (line 269)
    write_call_result_708159 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), write_708156, *[end_708157], **kwargs_708158)
    
    
    # ################# End of 'print_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_708160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708160)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_'
    return stypy_return_type_708160

# Assigning a type to the variable 'print_' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'print_', print_)
# SSA join for if statement (line 196)
module_type_store = module_type_store.join_ssa_context()


# Call to _add_doc(...): (line 271)
# Processing the call arguments (line 271)
# Getting the type of 'reraise' (line 271)
reraise_708162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 9), 'reraise', False)
str_708163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 18), 'str', 'Reraise an exception.')
# Processing the call keyword arguments (line 271)
kwargs_708164 = {}
# Getting the type of '_add_doc' (line 271)
_add_doc_708161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), '_add_doc', False)
# Calling _add_doc(args, kwargs) (line 271)
_add_doc_call_result_708165 = invoke(stypy.reporting.localization.Localization(__file__, 271, 0), _add_doc_708161, *[reraise_708162, str_708163], **kwargs_708164)


@norecursion
def with_metaclass(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'object' (line 274)
    object_708166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 30), 'object')
    defaults = [object_708166]
    # Create a new context for function 'with_metaclass'
    module_type_store = module_type_store.open_function_context('with_metaclass', 274, 0, False)
    
    # Passed parameters checking function
    with_metaclass.stypy_localization = localization
    with_metaclass.stypy_type_of_self = None
    with_metaclass.stypy_type_store = module_type_store
    with_metaclass.stypy_function_name = 'with_metaclass'
    with_metaclass.stypy_param_names_list = ['meta', 'base']
    with_metaclass.stypy_varargs_param_name = None
    with_metaclass.stypy_kwargs_param_name = None
    with_metaclass.stypy_call_defaults = defaults
    with_metaclass.stypy_call_varargs = varargs
    with_metaclass.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'with_metaclass', ['meta', 'base'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'with_metaclass', localization, ['meta', 'base'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'with_metaclass(...)' code ##################

    str_708167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 4), 'str', 'Create a base class with a metaclass.')
    
    # Call to meta(...): (line 276)
    # Processing the call arguments (line 276)
    str_708169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'str', 'NewBase')
    
    # Obtaining an instance of the builtin type 'tuple' (line 276)
    tuple_708170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 276)
    # Adding element type (line 276)
    # Getting the type of 'base' (line 276)
    base_708171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'base', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 28), tuple_708170, base_708171)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 276)
    dict_708172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 36), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 276)
    
    # Processing the call keyword arguments (line 276)
    kwargs_708173 = {}
    # Getting the type of 'meta' (line 276)
    meta_708168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), 'meta', False)
    # Calling meta(args, kwargs) (line 276)
    meta_call_result_708174 = invoke(stypy.reporting.localization.Localization(__file__, 276, 11), meta_708168, *[str_708169, tuple_708170, dict_708172], **kwargs_708173)
    
    # Assigning a type to the variable 'stypy_return_type' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type', meta_call_result_708174)
    
    # ################# End of 'with_metaclass(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'with_metaclass' in the type store
    # Getting the type of 'stypy_return_type' (line 274)
    stypy_return_type_708175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708175)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'with_metaclass'
    return stypy_return_type_708175

# Assigning a type to the variable 'with_metaclass' (line 274)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 0), 'with_metaclass', with_metaclass)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

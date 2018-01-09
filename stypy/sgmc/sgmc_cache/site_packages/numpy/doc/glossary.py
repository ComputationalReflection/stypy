
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ========
3: Glossary
4: ========
5: 
6: .. glossary::
7: 
8:    along an axis
9:        Axes are defined for arrays with more than one dimension.  A
10:        2-dimensional array has two corresponding axes: the first running
11:        vertically downwards across rows (axis 0), and the second running
12:        horizontally across columns (axis 1).
13: 
14:        Many operation can take place along one of these axes.  For example,
15:        we can sum each row of an array, in which case we operate along
16:        columns, or axis 1::
17: 
18:          >>> x = np.arange(12).reshape((3,4))
19: 
20:          >>> x
21:          array([[ 0,  1,  2,  3],
22:                 [ 4,  5,  6,  7],
23:                 [ 8,  9, 10, 11]])
24: 
25:          >>> x.sum(axis=1)
26:          array([ 6, 22, 38])
27: 
28:    array
29:        A homogeneous container of numerical elements.  Each element in the
30:        array occupies a fixed amount of memory (hence homogeneous), and
31:        can be a numerical element of a single type (such as float, int
32:        or complex) or a combination (such as ``(float, int, float)``).  Each
33:        array has an associated data-type (or ``dtype``), which describes
34:        the numerical type of its elements::
35: 
36:          >>> x = np.array([1, 2, 3], float)
37: 
38:          >>> x
39:          array([ 1.,  2.,  3.])
40: 
41:          >>> x.dtype # floating point number, 64 bits of memory per element
42:          dtype('float64')
43: 
44: 
45:          # More complicated data type: each array element is a combination of
46:          # and integer and a floating point number
47:          >>> np.array([(1, 2.0), (3, 4.0)], dtype=[('x', int), ('y', float)])
48:          array([(1, 2.0), (3, 4.0)],
49:                dtype=[('x', '<i4'), ('y', '<f8')])
50: 
51:        Fast element-wise operations, called `ufuncs`_, operate on arrays.
52: 
53:    array_like
54:        Any sequence that can be interpreted as an ndarray.  This includes
55:        nested lists, tuples, scalars and existing arrays.
56: 
57:    attribute
58:        A property of an object that can be accessed using ``obj.attribute``,
59:        e.g., ``shape`` is an attribute of an array::
60: 
61:          >>> x = np.array([1, 2, 3])
62:          >>> x.shape
63:          (3,)
64: 
65:    BLAS
66:        `Basic Linear Algebra Subprograms <http://en.wikipedia.org/wiki/BLAS>`_
67: 
68:    broadcast
69:        NumPy can do operations on arrays whose shapes are mismatched::
70: 
71:          >>> x = np.array([1, 2])
72:          >>> y = np.array([[3], [4]])
73: 
74:          >>> x
75:          array([1, 2])
76: 
77:          >>> y
78:          array([[3],
79:                 [4]])
80: 
81:          >>> x + y
82:          array([[4, 5],
83:                 [5, 6]])
84: 
85:        See `doc.broadcasting`_ for more information.
86: 
87:    C order
88:        See `row-major`
89: 
90:    column-major
91:        A way to represent items in a N-dimensional array in the 1-dimensional
92:        computer memory. In column-major order, the leftmost index "varies the
93:        fastest": for example the array::
94: 
95:             [[1, 2, 3],
96:              [4, 5, 6]]
97: 
98:        is represented in the column-major order as::
99: 
100:            [1, 4, 2, 5, 3, 6]
101: 
102:        Column-major order is also known as the Fortran order, as the Fortran
103:        programming language uses it.
104: 
105:    decorator
106:        An operator that transforms a function.  For example, a ``log``
107:        decorator may be defined to print debugging information upon
108:        function execution::
109: 
110:          >>> def log(f):
111:          ...     def new_logging_func(*args, **kwargs):
112:          ...         print("Logging call with parameters:", args, kwargs)
113:          ...         return f(*args, **kwargs)
114:          ...
115:          ...     return new_logging_func
116: 
117:        Now, when we define a function, we can "decorate" it using ``log``::
118: 
119:          >>> @log
120:          ... def add(a, b):
121:          ...     return a + b
122: 
123:        Calling ``add`` then yields:
124: 
125:        >>> add(1, 2)
126:        Logging call with parameters: (1, 2) {}
127:        3
128: 
129:    dictionary
130:        Resembling a language dictionary, which provides a mapping between
131:        words and descriptions thereof, a Python dictionary is a mapping
132:        between two objects::
133: 
134:          >>> x = {1: 'one', 'two': [1, 2]}
135: 
136:        Here, `x` is a dictionary mapping keys to values, in this case
137:        the integer 1 to the string "one", and the string "two" to
138:        the list ``[1, 2]``.  The values may be accessed using their
139:        corresponding keys::
140: 
141:          >>> x[1]
142:          'one'
143: 
144:          >>> x['two']
145:          [1, 2]
146: 
147:        Note that dictionaries are not stored in any specific order.  Also,
148:        most mutable (see *immutable* below) objects, such as lists, may not
149:        be used as keys.
150: 
151:        For more information on dictionaries, read the
152:        `Python tutorial <http://docs.python.org/tut>`_.
153: 
154:    Fortran order
155:        See `column-major`
156: 
157:    flattened
158:        Collapsed to a one-dimensional array. See `ndarray.flatten`_ for details.
159: 
160:    immutable
161:        An object that cannot be modified after execution is called
162:        immutable.  Two common examples are strings and tuples.
163: 
164:    instance
165:        A class definition gives the blueprint for constructing an object::
166: 
167:          >>> class House(object):
168:          ...     wall_colour = 'white'
169: 
170:        Yet, we have to *build* a house before it exists::
171: 
172:          >>> h = House() # build a house
173: 
174:        Now, ``h`` is called a ``House`` instance.  An instance is therefore
175:        a specific realisation of a class.
176: 
177:    iterable
178:        A sequence that allows "walking" (iterating) over items, typically
179:        using a loop such as::
180: 
181:          >>> x = [1, 2, 3]
182:          >>> [item**2 for item in x]
183:          [1, 4, 9]
184: 
185:        It is often used in combintion with ``enumerate``::
186:          >>> keys = ['a','b','c']
187:          >>> for n, k in enumerate(keys):
188:          ...     print("Key %d: %s" % (n, k))
189:          ...
190:          Key 0: a
191:          Key 1: b
192:          Key 2: c
193: 
194:    list
195:        A Python container that can hold any number of objects or items.
196:        The items do not have to be of the same type, and can even be
197:        lists themselves::
198: 
199:          >>> x = [2, 2.0, "two", [2, 2.0]]
200: 
201:        The list `x` contains 4 items, each which can be accessed individually::
202: 
203:          >>> x[2] # the string 'two'
204:          'two'
205: 
206:          >>> x[3] # a list, containing an integer 2 and a float 2.0
207:          [2, 2.0]
208: 
209:        It is also possible to select more than one item at a time,
210:        using *slicing*::
211: 
212:          >>> x[0:2] # or, equivalently, x[:2]
213:          [2, 2.0]
214: 
215:        In code, arrays are often conveniently expressed as nested lists::
216: 
217: 
218:          >>> np.array([[1, 2], [3, 4]])
219:          array([[1, 2],
220:                 [3, 4]])
221: 
222:        For more information, read the section on lists in the `Python
223:        tutorial <http://docs.python.org/tut>`_.  For a mapping
224:        type (key-value), see *dictionary*.
225: 
226:    mask
227:        A boolean array, used to select only certain elements for an operation::
228: 
229:          >>> x = np.arange(5)
230:          >>> x
231:          array([0, 1, 2, 3, 4])
232: 
233:          >>> mask = (x > 2)
234:          >>> mask
235:          array([False, False, False, True,  True], dtype=bool)
236: 
237:          >>> x[mask] = -1
238:          >>> x
239:          array([ 0,  1,  2,  -1, -1])
240: 
241:    masked array
242:        Array that suppressed values indicated by a mask::
243: 
244:          >>> x = np.ma.masked_array([np.nan, 2, np.nan], [True, False, True])
245:          >>> x
246:          masked_array(data = [-- 2.0 --],
247:                       mask = [ True False  True],
248:                 fill_value = 1e+20)
249:          <BLANKLINE>
250: 
251:          >>> x + [1, 2, 3]
252:          masked_array(data = [-- 4.0 --],
253:                       mask = [ True False  True],
254:                 fill_value = 1e+20)
255:          <BLANKLINE>
256: 
257: 
258:        Masked arrays are often used when operating on arrays containing
259:        missing or invalid entries.
260: 
261:    matrix
262:        A 2-dimensional ndarray that preserves its two-dimensional nature
263:        throughout operations.  It has certain special operations, such as ``*``
264:        (matrix multiplication) and ``**`` (matrix power), defined::
265: 
266:          >>> x = np.mat([[1, 2], [3, 4]])
267:          >>> x
268:          matrix([[1, 2],
269:                  [3, 4]])
270: 
271:          >>> x**2
272:          matrix([[ 7, 10],
273:                [15, 22]])
274: 
275:    method
276:        A function associated with an object.  For example, each ndarray has a
277:        method called ``repeat``::
278: 
279:          >>> x = np.array([1, 2, 3])
280:          >>> x.repeat(2)
281:          array([1, 1, 2, 2, 3, 3])
282: 
283:    ndarray
284:        See *array*.
285: 
286:    record array
287:        An `ndarray`_ with `structured data type`_ which has been subclassed as
288:        np.recarray and whose dtype is of type np.record, making the
289:        fields of its data type to be accessible by attribute.
290: 
291:    reference
292:        If ``a`` is a reference to ``b``, then ``(a is b) == True``.  Therefore,
293:        ``a`` and ``b`` are different names for the same Python object.
294: 
295:    row-major
296:        A way to represent items in a N-dimensional array in the 1-dimensional
297:        computer memory. In row-major order, the rightmost index "varies
298:        the fastest": for example the array::
299: 
300:             [[1, 2, 3],
301:              [4, 5, 6]]
302: 
303:        is represented in the row-major order as::
304: 
305:            [1, 2, 3, 4, 5, 6]
306: 
307:        Row-major order is also known as the C order, as the C programming
308:        language uses it. New Numpy arrays are by default in row-major order.
309: 
310:    self
311:        Often seen in method signatures, ``self`` refers to the instance
312:        of the associated class.  For example:
313: 
314:          >>> class Paintbrush(object):
315:          ...     color = 'blue'
316:          ...
317:          ...     def paint(self):
318:          ...         print("Painting the city %s!" % self.color)
319:          ...
320:          >>> p = Paintbrush()
321:          >>> p.color = 'red'
322:          >>> p.paint() # self refers to 'p'
323:          Painting the city red!
324: 
325:    slice
326:        Used to select only certain elements from a sequence::
327: 
328:          >>> x = range(5)
329:          >>> x
330:          [0, 1, 2, 3, 4]
331: 
332:          >>> x[1:3] # slice from 1 to 3 (excluding 3 itself)
333:          [1, 2]
334: 
335:          >>> x[1:5:2] # slice from 1 to 5, but skipping every second element
336:          [1, 3]
337: 
338:          >>> x[::-1] # slice a sequence in reverse
339:          [4, 3, 2, 1, 0]
340: 
341:        Arrays may have more than one dimension, each which can be sliced
342:        individually::
343: 
344:          >>> x = np.array([[1, 2], [3, 4]])
345:          >>> x
346:          array([[1, 2],
347:                 [3, 4]])
348: 
349:          >>> x[:, 1]
350:          array([2, 4])
351:    
352:    structured data type
353:        A data type composed of other datatypes
354:    
355:    tuple
356:        A sequence that may contain a variable number of types of any
357:        kind.  A tuple is immutable, i.e., once constructed it cannot be
358:        changed.  Similar to a list, it can be indexed and sliced::
359: 
360:          >>> x = (1, 'one', [1, 2])
361:          >>> x
362:          (1, 'one', [1, 2])
363: 
364:          >>> x[0]
365:          1
366: 
367:          >>> x[:2]
368:          (1, 'one')
369: 
370:        A useful concept is "tuple unpacking", which allows variables to
371:        be assigned to the contents of a tuple::
372: 
373:          >>> x, y = (1, 2)
374:          >>> x, y = 1, 2
375: 
376:        This is often used when a function returns multiple values:
377: 
378:          >>> def return_many():
379:          ...     return 1, 'alpha', None
380: 
381:          >>> a, b, c = return_many()
382:          >>> a, b, c
383:          (1, 'alpha', None)
384: 
385:          >>> a
386:          1
387:          >>> b
388:          'alpha'
389: 
390:    ufunc
391:        Universal function.  A fast element-wise array operation.  Examples include
392:        ``add``, ``sin`` and ``logical_or``.
393: 
394:    view
395:        An array that does not own its data, but refers to another array's
396:        data instead.  For example, we may create a view that only shows
397:        every second element of another array::
398: 
399:          >>> x = np.arange(5)
400:          >>> x
401:          array([0, 1, 2, 3, 4])
402: 
403:          >>> y = x[::2]
404:          >>> y
405:          array([0, 2, 4])
406: 
407:          >>> x[0] = 3 # changing x changes y as well, since y is a view on x
408:          >>> y
409:          array([3, 2, 4])
410: 
411:    wrapper
412:        Python is a high-level (highly abstracted, or English-like) language.
413:        This abstraction comes at a price in execution speed, and sometimes
414:        it becomes necessary to use lower level languages to do fast
415:        computations.  A wrapper is code that provides a bridge between
416:        high and the low level languages, allowing, e.g., Python to execute
417:        code written in C or Fortran.
418: 
419:        Examples include ctypes, SWIG and Cython (which wraps C and C++)
420:        and f2py (which wraps Fortran).
421: 
422: '''
423: from __future__ import division, absolute_import, print_function
424: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, (-1)), 'str', '\n========\nGlossary\n========\n\n.. glossary::\n\n   along an axis\n       Axes are defined for arrays with more than one dimension.  A\n       2-dimensional array has two corresponding axes: the first running\n       vertically downwards across rows (axis 0), and the second running\n       horizontally across columns (axis 1).\n\n       Many operation can take place along one of these axes.  For example,\n       we can sum each row of an array, in which case we operate along\n       columns, or axis 1::\n\n         >>> x = np.arange(12).reshape((3,4))\n\n         >>> x\n         array([[ 0,  1,  2,  3],\n                [ 4,  5,  6,  7],\n                [ 8,  9, 10, 11]])\n\n         >>> x.sum(axis=1)\n         array([ 6, 22, 38])\n\n   array\n       A homogeneous container of numerical elements.  Each element in the\n       array occupies a fixed amount of memory (hence homogeneous), and\n       can be a numerical element of a single type (such as float, int\n       or complex) or a combination (such as ``(float, int, float)``).  Each\n       array has an associated data-type (or ``dtype``), which describes\n       the numerical type of its elements::\n\n         >>> x = np.array([1, 2, 3], float)\n\n         >>> x\n         array([ 1.,  2.,  3.])\n\n         >>> x.dtype # floating point number, 64 bits of memory per element\n         dtype(\'float64\')\n\n\n         # More complicated data type: each array element is a combination of\n         # and integer and a floating point number\n         >>> np.array([(1, 2.0), (3, 4.0)], dtype=[(\'x\', int), (\'y\', float)])\n         array([(1, 2.0), (3, 4.0)],\n               dtype=[(\'x\', \'<i4\'), (\'y\', \'<f8\')])\n\n       Fast element-wise operations, called `ufuncs`_, operate on arrays.\n\n   array_like\n       Any sequence that can be interpreted as an ndarray.  This includes\n       nested lists, tuples, scalars and existing arrays.\n\n   attribute\n       A property of an object that can be accessed using ``obj.attribute``,\n       e.g., ``shape`` is an attribute of an array::\n\n         >>> x = np.array([1, 2, 3])\n         >>> x.shape\n         (3,)\n\n   BLAS\n       `Basic Linear Algebra Subprograms <http://en.wikipedia.org/wiki/BLAS>`_\n\n   broadcast\n       NumPy can do operations on arrays whose shapes are mismatched::\n\n         >>> x = np.array([1, 2])\n         >>> y = np.array([[3], [4]])\n\n         >>> x\n         array([1, 2])\n\n         >>> y\n         array([[3],\n                [4]])\n\n         >>> x + y\n         array([[4, 5],\n                [5, 6]])\n\n       See `doc.broadcasting`_ for more information.\n\n   C order\n       See `row-major`\n\n   column-major\n       A way to represent items in a N-dimensional array in the 1-dimensional\n       computer memory. In column-major order, the leftmost index "varies the\n       fastest": for example the array::\n\n            [[1, 2, 3],\n             [4, 5, 6]]\n\n       is represented in the column-major order as::\n\n           [1, 4, 2, 5, 3, 6]\n\n       Column-major order is also known as the Fortran order, as the Fortran\n       programming language uses it.\n\n   decorator\n       An operator that transforms a function.  For example, a ``log``\n       decorator may be defined to print debugging information upon\n       function execution::\n\n         >>> def log(f):\n         ...     def new_logging_func(*args, **kwargs):\n         ...         print("Logging call with parameters:", args, kwargs)\n         ...         return f(*args, **kwargs)\n         ...\n         ...     return new_logging_func\n\n       Now, when we define a function, we can "decorate" it using ``log``::\n\n         >>> @log\n         ... def add(a, b):\n         ...     return a + b\n\n       Calling ``add`` then yields:\n\n       >>> add(1, 2)\n       Logging call with parameters: (1, 2) {}\n       3\n\n   dictionary\n       Resembling a language dictionary, which provides a mapping between\n       words and descriptions thereof, a Python dictionary is a mapping\n       between two objects::\n\n         >>> x = {1: \'one\', \'two\': [1, 2]}\n\n       Here, `x` is a dictionary mapping keys to values, in this case\n       the integer 1 to the string "one", and the string "two" to\n       the list ``[1, 2]``.  The values may be accessed using their\n       corresponding keys::\n\n         >>> x[1]\n         \'one\'\n\n         >>> x[\'two\']\n         [1, 2]\n\n       Note that dictionaries are not stored in any specific order.  Also,\n       most mutable (see *immutable* below) objects, such as lists, may not\n       be used as keys.\n\n       For more information on dictionaries, read the\n       `Python tutorial <http://docs.python.org/tut>`_.\n\n   Fortran order\n       See `column-major`\n\n   flattened\n       Collapsed to a one-dimensional array. See `ndarray.flatten`_ for details.\n\n   immutable\n       An object that cannot be modified after execution is called\n       immutable.  Two common examples are strings and tuples.\n\n   instance\n       A class definition gives the blueprint for constructing an object::\n\n         >>> class House(object):\n         ...     wall_colour = \'white\'\n\n       Yet, we have to *build* a house before it exists::\n\n         >>> h = House() # build a house\n\n       Now, ``h`` is called a ``House`` instance.  An instance is therefore\n       a specific realisation of a class.\n\n   iterable\n       A sequence that allows "walking" (iterating) over items, typically\n       using a loop such as::\n\n         >>> x = [1, 2, 3]\n         >>> [item**2 for item in x]\n         [1, 4, 9]\n\n       It is often used in combintion with ``enumerate``::\n         >>> keys = [\'a\',\'b\',\'c\']\n         >>> for n, k in enumerate(keys):\n         ...     print("Key %d: %s" % (n, k))\n         ...\n         Key 0: a\n         Key 1: b\n         Key 2: c\n\n   list\n       A Python container that can hold any number of objects or items.\n       The items do not have to be of the same type, and can even be\n       lists themselves::\n\n         >>> x = [2, 2.0, "two", [2, 2.0]]\n\n       The list `x` contains 4 items, each which can be accessed individually::\n\n         >>> x[2] # the string \'two\'\n         \'two\'\n\n         >>> x[3] # a list, containing an integer 2 and a float 2.0\n         [2, 2.0]\n\n       It is also possible to select more than one item at a time,\n       using *slicing*::\n\n         >>> x[0:2] # or, equivalently, x[:2]\n         [2, 2.0]\n\n       In code, arrays are often conveniently expressed as nested lists::\n\n\n         >>> np.array([[1, 2], [3, 4]])\n         array([[1, 2],\n                [3, 4]])\n\n       For more information, read the section on lists in the `Python\n       tutorial <http://docs.python.org/tut>`_.  For a mapping\n       type (key-value), see *dictionary*.\n\n   mask\n       A boolean array, used to select only certain elements for an operation::\n\n         >>> x = np.arange(5)\n         >>> x\n         array([0, 1, 2, 3, 4])\n\n         >>> mask = (x > 2)\n         >>> mask\n         array([False, False, False, True,  True], dtype=bool)\n\n         >>> x[mask] = -1\n         >>> x\n         array([ 0,  1,  2,  -1, -1])\n\n   masked array\n       Array that suppressed values indicated by a mask::\n\n         >>> x = np.ma.masked_array([np.nan, 2, np.nan], [True, False, True])\n         >>> x\n         masked_array(data = [-- 2.0 --],\n                      mask = [ True False  True],\n                fill_value = 1e+20)\n         <BLANKLINE>\n\n         >>> x + [1, 2, 3]\n         masked_array(data = [-- 4.0 --],\n                      mask = [ True False  True],\n                fill_value = 1e+20)\n         <BLANKLINE>\n\n\n       Masked arrays are often used when operating on arrays containing\n       missing or invalid entries.\n\n   matrix\n       A 2-dimensional ndarray that preserves its two-dimensional nature\n       throughout operations.  It has certain special operations, such as ``*``\n       (matrix multiplication) and ``**`` (matrix power), defined::\n\n         >>> x = np.mat([[1, 2], [3, 4]])\n         >>> x\n         matrix([[1, 2],\n                 [3, 4]])\n\n         >>> x**2\n         matrix([[ 7, 10],\n               [15, 22]])\n\n   method\n       A function associated with an object.  For example, each ndarray has a\n       method called ``repeat``::\n\n         >>> x = np.array([1, 2, 3])\n         >>> x.repeat(2)\n         array([1, 1, 2, 2, 3, 3])\n\n   ndarray\n       See *array*.\n\n   record array\n       An `ndarray`_ with `structured data type`_ which has been subclassed as\n       np.recarray and whose dtype is of type np.record, making the\n       fields of its data type to be accessible by attribute.\n\n   reference\n       If ``a`` is a reference to ``b``, then ``(a is b) == True``.  Therefore,\n       ``a`` and ``b`` are different names for the same Python object.\n\n   row-major\n       A way to represent items in a N-dimensional array in the 1-dimensional\n       computer memory. In row-major order, the rightmost index "varies\n       the fastest": for example the array::\n\n            [[1, 2, 3],\n             [4, 5, 6]]\n\n       is represented in the row-major order as::\n\n           [1, 2, 3, 4, 5, 6]\n\n       Row-major order is also known as the C order, as the C programming\n       language uses it. New Numpy arrays are by default in row-major order.\n\n   self\n       Often seen in method signatures, ``self`` refers to the instance\n       of the associated class.  For example:\n\n         >>> class Paintbrush(object):\n         ...     color = \'blue\'\n         ...\n         ...     def paint(self):\n         ...         print("Painting the city %s!" % self.color)\n         ...\n         >>> p = Paintbrush()\n         >>> p.color = \'red\'\n         >>> p.paint() # self refers to \'p\'\n         Painting the city red!\n\n   slice\n       Used to select only certain elements from a sequence::\n\n         >>> x = range(5)\n         >>> x\n         [0, 1, 2, 3, 4]\n\n         >>> x[1:3] # slice from 1 to 3 (excluding 3 itself)\n         [1, 2]\n\n         >>> x[1:5:2] # slice from 1 to 5, but skipping every second element\n         [1, 3]\n\n         >>> x[::-1] # slice a sequence in reverse\n         [4, 3, 2, 1, 0]\n\n       Arrays may have more than one dimension, each which can be sliced\n       individually::\n\n         >>> x = np.array([[1, 2], [3, 4]])\n         >>> x\n         array([[1, 2],\n                [3, 4]])\n\n         >>> x[:, 1]\n         array([2, 4])\n   \n   structured data type\n       A data type composed of other datatypes\n   \n   tuple\n       A sequence that may contain a variable number of types of any\n       kind.  A tuple is immutable, i.e., once constructed it cannot be\n       changed.  Similar to a list, it can be indexed and sliced::\n\n         >>> x = (1, \'one\', [1, 2])\n         >>> x\n         (1, \'one\', [1, 2])\n\n         >>> x[0]\n         1\n\n         >>> x[:2]\n         (1, \'one\')\n\n       A useful concept is "tuple unpacking", which allows variables to\n       be assigned to the contents of a tuple::\n\n         >>> x, y = (1, 2)\n         >>> x, y = 1, 2\n\n       This is often used when a function returns multiple values:\n\n         >>> def return_many():\n         ...     return 1, \'alpha\', None\n\n         >>> a, b, c = return_many()\n         >>> a, b, c\n         (1, \'alpha\', None)\n\n         >>> a\n         1\n         >>> b\n         \'alpha\'\n\n   ufunc\n       Universal function.  A fast element-wise array operation.  Examples include\n       ``add``, ``sin`` and ``logical_or``.\n\n   view\n       An array that does not own its data, but refers to another array\'s\n       data instead.  For example, we may create a view that only shows\n       every second element of another array::\n\n         >>> x = np.arange(5)\n         >>> x\n         array([0, 1, 2, 3, 4])\n\n         >>> y = x[::2]\n         >>> y\n         array([0, 2, 4])\n\n         >>> x[0] = 3 # changing x changes y as well, since y is a view on x\n         >>> y\n         array([3, 2, 4])\n\n   wrapper\n       Python is a high-level (highly abstracted, or English-like) language.\n       This abstraction comes at a price in execution speed, and sometimes\n       it becomes necessary to use lower level languages to do fast\n       computations.  A wrapper is code that provides a bridge between\n       high and the low level languages, allowing, e.g., Python to execute\n       code written in C or Fortran.\n\n       Examples include ctypes, SWIG and Cython (which wraps C and C++)\n       and f2py (which wraps Fortran).\n\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

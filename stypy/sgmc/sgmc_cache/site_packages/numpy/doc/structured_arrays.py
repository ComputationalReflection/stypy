
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =================
3: Structured Arrays
4: =================
5: 
6: Introduction
7: ============
8: 
9: Numpy provides powerful capabilities to create arrays of structured datatype.
10: These arrays permit one to manipulate the data by named fields. A simple 
11: example will show what is meant.: ::
12: 
13:  >>> x = np.array([(1,2.,'Hello'), (2,3.,"World")],
14:  ...              dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
15:  >>> x
16:  array([(1, 2.0, 'Hello'), (2, 3.0, 'World')],
17:       dtype=[('foo', '>i4'), ('bar', '>f4'), ('baz', '|S10')])
18: 
19: Here we have created a one-dimensional array of length 2. Each element of
20: this array is a structure that contains three items, a 32-bit integer, a 32-bit
21: float, and a string of length 10 or less. If we index this array at the second
22: position we get the second structure: ::
23: 
24:  >>> x[1]
25:  (2,3.,"World")
26: 
27: Conveniently, one can access any field of the array by indexing using the
28: string that names that field. ::
29: 
30:  >>> y = x['bar']
31:  >>> y
32:  array([ 2.,  3.], dtype=float32)
33:  >>> y[:] = 2*y
34:  >>> y
35:  array([ 4.,  6.], dtype=float32)
36:  >>> x
37:  array([(1, 4.0, 'Hello'), (2, 6.0, 'World')],
38:        dtype=[('foo', '>i4'), ('bar', '>f4'), ('baz', '|S10')])
39: 
40: In these examples, y is a simple float array consisting of the 2nd field
41: in the structured type. But, rather than being a copy of the data in the structured
42: array, it is a view, i.e., it shares exactly the same memory locations.
43: Thus, when we updated this array by doubling its values, the structured
44: array shows the corresponding values as doubled as well. Likewise, if one
45: changes the structured array, the field view also changes: ::
46: 
47:  >>> x[1] = (-1,-1.,"Master")
48:  >>> x
49:  array([(1, 4.0, 'Hello'), (-1, -1.0, 'Master')],
50:        dtype=[('foo', '>i4'), ('bar', '>f4'), ('baz', '|S10')])
51:  >>> y
52:  array([ 4., -1.], dtype=float32)
53: 
54: Defining Structured Arrays
55: ==========================
56: 
57: One defines a structured array through the dtype object.  There are
58: **several** alternative ways to define the fields of a record.  Some of
59: these variants provide backward compatibility with Numeric, numarray, or
60: another module, and should not be used except for such purposes. These
61: will be so noted. One specifies record structure in
62: one of four alternative ways, using an argument (as supplied to a dtype
63: function keyword or a dtype object constructor itself).  This
64: argument must be one of the following: 1) string, 2) tuple, 3) list, or
65: 4) dictionary.  Each of these is briefly described below.
66: 
67: 1) String argument.
68: In this case, the constructor expects a comma-separated list of type
69: specifiers, optionally with extra shape information. The fields are 
70: given the default names 'f0', 'f1', 'f2' and so on.
71: The type specifiers can take 4 different forms: ::
72: 
73:   a) b1, i1, i2, i4, i8, u1, u2, u4, u8, f2, f4, f8, c8, c16, a<n>
74:      (representing bytes, ints, unsigned ints, floats, complex and
75:       fixed length strings of specified byte lengths)
76:   b) int8,...,uint8,...,float16, float32, float64, complex64, complex128
77:      (this time with bit sizes)
78:   c) older Numeric/numarray type specifications (e.g. Float32).
79:      Don't use these in new code!
80:   d) Single character type specifiers (e.g H for unsigned short ints).
81:      Avoid using these unless you must. Details can be found in the
82:      Numpy book
83: 
84: These different styles can be mixed within the same string (but why would you
85: want to do that?). Furthermore, each type specifier can be prefixed
86: with a repetition number, or a shape. In these cases an array
87: element is created, i.e., an array within a record. That array
88: is still referred to as a single field. An example: ::
89: 
90:  >>> x = np.zeros(3, dtype='3int8, float32, (2,3)float64')
91:  >>> x
92:  array([([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
93:         ([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
94:         ([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])],
95:        dtype=[('f0', '|i1', 3), ('f1', '>f4'), ('f2', '>f8', (2, 3))])
96: 
97: By using strings to define the record structure, it precludes being
98: able to name the fields in the original definition. The names can
99: be changed as shown later, however.
100: 
101: 2) Tuple argument: The only relevant tuple case that applies to record
102: structures is when a structure is mapped to an existing data type. This
103: is done by pairing in a tuple, the existing data type with a matching
104: dtype definition (using any of the variants being described here). As
105: an example (using a definition using a list, so see 3) for further
106: details): ::
107: 
108:  >>> x = np.zeros(3, dtype=('i4',[('r','u1'), ('g','u1'), ('b','u1'), ('a','u1')]))
109:  >>> x
110:  array([0, 0, 0])
111:  >>> x['r']
112:  array([0, 0, 0], dtype=uint8)
113: 
114: In this case, an array is produced that looks and acts like a simple int32 array,
115: but also has definitions for fields that use only one byte of the int32 (a bit
116: like Fortran equivalencing).
117: 
118: 3) List argument: In this case the record structure is defined with a list of
119: tuples. Each tuple has 2 or 3 elements specifying: 1) The name of the field
120: ('' is permitted), 2) the type of the field, and 3) the shape (optional).
121: For example::
122: 
123:  >>> x = np.zeros(3, dtype=[('x','f4'),('y',np.float32),('value','f4',(2,2))])
124:  >>> x
125:  array([(0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]]),
126:         (0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]]),
127:         (0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]])],
128:        dtype=[('x', '>f4'), ('y', '>f4'), ('value', '>f4', (2, 2))])
129: 
130: 4) Dictionary argument: two different forms are permitted. The first consists
131: of a dictionary with two required keys ('names' and 'formats'), each having an
132: equal sized list of values. The format list contains any type/shape specifier
133: allowed in other contexts. The names must be strings. There are two optional
134: keys: 'offsets' and 'titles'. Each must be a correspondingly matching list to
135: the required two where offsets contain integer offsets for each field, and
136: titles are objects containing metadata for each field (these do not have
137: to be strings), where the value of None is permitted. As an example: ::
138: 
139:  >>> x = np.zeros(3, dtype={'names':['col1', 'col2'], 'formats':['i4','f4']})
140:  >>> x
141:  array([(0, 0.0), (0, 0.0), (0, 0.0)],
142:        dtype=[('col1', '>i4'), ('col2', '>f4')])
143: 
144: The other dictionary form permitted is a dictionary of name keys with tuple
145: values specifying type, offset, and an optional title. ::
146: 
147:  >>> x = np.zeros(3, dtype={'col1':('i1',0,'title 1'), 'col2':('f4',1,'title 2')})
148:  >>> x
149:  array([(0, 0.0), (0, 0.0), (0, 0.0)],
150:        dtype=[(('title 1', 'col1'), '|i1'), (('title 2', 'col2'), '>f4')])
151: 
152: Accessing and modifying field names
153: ===================================
154: 
155: The field names are an attribute of the dtype object defining the structure.
156: For the last example: ::
157: 
158:  >>> x.dtype.names
159:  ('col1', 'col2')
160:  >>> x.dtype.names = ('x', 'y')
161:  >>> x
162:  array([(0, 0.0), (0, 0.0), (0, 0.0)],
163:       dtype=[(('title 1', 'x'), '|i1'), (('title 2', 'y'), '>f4')])
164:  >>> x.dtype.names = ('x', 'y', 'z') # wrong number of names
165:  <type 'exceptions.ValueError'>: must replace all names at once with a sequence of length 2
166: 
167: Accessing field titles
168: ====================================
169: 
170: The field titles provide a standard place to put associated info for fields.
171: They do not have to be strings. ::
172: 
173:  >>> x.dtype.fields['x'][2]
174:  'title 1'
175: 
176: Accessing multiple fields at once
177: ====================================
178: 
179: You can access multiple fields at once using a list of field names: ::
180: 
181:  >>> x = np.array([(1.5,2.5,(1.0,2.0)),(3.,4.,(4.,5.)),(1.,3.,(2.,6.))],
182:          dtype=[('x','f4'),('y',np.float32),('value','f4',(2,2))])
183: 
184: Notice that `x` is created with a list of tuples. ::
185: 
186:  >>> x[['x','y']]
187:  array([(1.5, 2.5), (3.0, 4.0), (1.0, 3.0)],
188:       dtype=[('x', '<f4'), ('y', '<f4')])
189:  >>> x[['x','value']]
190:  array([(1.5, [[1.0, 2.0], [1.0, 2.0]]), (3.0, [[4.0, 5.0], [4.0, 5.0]]),
191:        (1.0, [[2.0, 6.0], [2.0, 6.0]])],
192:       dtype=[('x', '<f4'), ('value', '<f4', (2, 2))])
193: 
194: The fields are returned in the order they are asked for.::
195: 
196:  >>> x[['y','x']]
197:  array([(2.5, 1.5), (4.0, 3.0), (3.0, 1.0)],
198:       dtype=[('y', '<f4'), ('x', '<f4')])
199: 
200: Filling structured arrays
201: =========================
202: 
203: Structured arrays can be filled by field or row by row. ::
204: 
205:  >>> arr = np.zeros((5,), dtype=[('var1','f8'),('var2','f8')])
206:  >>> arr['var1'] = np.arange(5)
207: 
208: If you fill it in row by row, it takes a take a tuple
209: (but not a list or array!)::
210: 
211:  >>> arr[0] = (10,20)
212:  >>> arr
213:  array([(10.0, 20.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
214:       dtype=[('var1', '<f8'), ('var2', '<f8')])
215: 
216: Record Arrays
217: =============
218: 
219: For convenience, numpy provides "record arrays" which allow one to access
220: fields of structured arrays by attribute rather than by index. Record arrays
221: are structured arrays wrapped using a subclass of ndarray,
222: :class:`numpy.recarray`, which allows field access by attribute on the array
223: object, and record arrays also use a special datatype, :class:`numpy.record`,
224: which allows field access by attribute on the individual elements of the array. 
225: 
226: The simplest way to create a record array is with :func:`numpy.rec.array`: ::
227: 
228:  >>> recordarr = np.rec.array([(1,2.,'Hello'),(2,3.,"World")], 
229:  ...                    dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
230:  >>> recordarr.bar
231:  array([ 2.,  3.], dtype=float32)
232:  >>> recordarr[1:2]
233:  rec.array([(2, 3.0, 'World')], 
234:        dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])
235:  >>> recordarr[1:2].foo
236:  array([2], dtype=int32)
237:  >>> recordarr.foo[1:2]
238:  array([2], dtype=int32)
239:  >>> recordarr[1].baz
240:  'World'
241: 
242: numpy.rec.array can convert a wide variety of arguments into record arrays,
243: including normal structured arrays: ::
244: 
245:  >>> arr = array([(1,2.,'Hello'),(2,3.,"World")], 
246:  ...             dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
247:  >>> recordarr = np.rec.array(arr)
248: 
249: The numpy.rec module provides a number of other convenience functions for
250: creating record arrays, see :ref:`record array creation routines
251: <routines.array-creation.rec>`.
252: 
253: A record array representation of a structured array can be obtained using the
254: appropriate :ref:`view`: ::
255: 
256:  >>> arr = np.array([(1,2.,'Hello'),(2,3.,"World")], 
257:  ...                dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'a10')])
258:  >>> recordarr = arr.view(dtype=dtype((np.record, arr.dtype)), 
259:  ...                      type=np.recarray)
260: 
261: For convenience, viewing an ndarray as type `np.recarray` will automatically
262: convert to `np.record` datatype, so the dtype can be left out of the view: ::
263: 
264:  >>> recordarr = arr.view(np.recarray)
265:  >>> recordarr.dtype
266:  dtype((numpy.record, [('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')]))
267: 
268: To get back to a plain ndarray both the dtype and type must be reset. The
269: following view does so, taking into account the unusual case that the
270: recordarr was not a structured type: ::
271: 
272:  >>> arr2 = recordarr.view(recordarr.dtype.fields or recordarr.dtype, np.ndarray)
273: 
274: Record array fields accessed by index or by attribute are returned as a record
275: array if the field has a structured type but as a plain ndarray otherwise. ::
276: 
277:  >>> recordarr = np.rec.array([('Hello', (1,2)),("World", (3,4))], 
278:  ...                 dtype=[('foo', 'S6'),('bar', [('A', int), ('B', int)])])
279:  >>> type(recordarr.foo)
280:  <type 'numpy.ndarray'>
281:  >>> type(recordarr.bar)
282:  <class 'numpy.core.records.recarray'>
283: 
284: Note that if a field has the same name as an ndarray attribute, the ndarray
285: attribute takes precedence. Such fields will be inaccessible by attribute but
286: may still be accessed by index.
287: 
288: 
289: '''
290: from __future__ import division, absolute_import, print_function
291: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', '\n=================\nStructured Arrays\n=================\n\nIntroduction\n============\n\nNumpy provides powerful capabilities to create arrays of structured datatype.\nThese arrays permit one to manipulate the data by named fields. A simple \nexample will show what is meant.: ::\n\n >>> x = np.array([(1,2.,\'Hello\'), (2,3.,"World")],\n ...              dtype=[(\'foo\', \'i4\'),(\'bar\', \'f4\'), (\'baz\', \'S10\')])\n >>> x\n array([(1, 2.0, \'Hello\'), (2, 3.0, \'World\')],\n      dtype=[(\'foo\', \'>i4\'), (\'bar\', \'>f4\'), (\'baz\', \'|S10\')])\n\nHere we have created a one-dimensional array of length 2. Each element of\nthis array is a structure that contains three items, a 32-bit integer, a 32-bit\nfloat, and a string of length 10 or less. If we index this array at the second\nposition we get the second structure: ::\n\n >>> x[1]\n (2,3.,"World")\n\nConveniently, one can access any field of the array by indexing using the\nstring that names that field. ::\n\n >>> y = x[\'bar\']\n >>> y\n array([ 2.,  3.], dtype=float32)\n >>> y[:] = 2*y\n >>> y\n array([ 4.,  6.], dtype=float32)\n >>> x\n array([(1, 4.0, \'Hello\'), (2, 6.0, \'World\')],\n       dtype=[(\'foo\', \'>i4\'), (\'bar\', \'>f4\'), (\'baz\', \'|S10\')])\n\nIn these examples, y is a simple float array consisting of the 2nd field\nin the structured type. But, rather than being a copy of the data in the structured\narray, it is a view, i.e., it shares exactly the same memory locations.\nThus, when we updated this array by doubling its values, the structured\narray shows the corresponding values as doubled as well. Likewise, if one\nchanges the structured array, the field view also changes: ::\n\n >>> x[1] = (-1,-1.,"Master")\n >>> x\n array([(1, 4.0, \'Hello\'), (-1, -1.0, \'Master\')],\n       dtype=[(\'foo\', \'>i4\'), (\'bar\', \'>f4\'), (\'baz\', \'|S10\')])\n >>> y\n array([ 4., -1.], dtype=float32)\n\nDefining Structured Arrays\n==========================\n\nOne defines a structured array through the dtype object.  There are\n**several** alternative ways to define the fields of a record.  Some of\nthese variants provide backward compatibility with Numeric, numarray, or\nanother module, and should not be used except for such purposes. These\nwill be so noted. One specifies record structure in\none of four alternative ways, using an argument (as supplied to a dtype\nfunction keyword or a dtype object constructor itself).  This\nargument must be one of the following: 1) string, 2) tuple, 3) list, or\n4) dictionary.  Each of these is briefly described below.\n\n1) String argument.\nIn this case, the constructor expects a comma-separated list of type\nspecifiers, optionally with extra shape information. The fields are \ngiven the default names \'f0\', \'f1\', \'f2\' and so on.\nThe type specifiers can take 4 different forms: ::\n\n  a) b1, i1, i2, i4, i8, u1, u2, u4, u8, f2, f4, f8, c8, c16, a<n>\n     (representing bytes, ints, unsigned ints, floats, complex and\n      fixed length strings of specified byte lengths)\n  b) int8,...,uint8,...,float16, float32, float64, complex64, complex128\n     (this time with bit sizes)\n  c) older Numeric/numarray type specifications (e.g. Float32).\n     Don\'t use these in new code!\n  d) Single character type specifiers (e.g H for unsigned short ints).\n     Avoid using these unless you must. Details can be found in the\n     Numpy book\n\nThese different styles can be mixed within the same string (but why would you\nwant to do that?). Furthermore, each type specifier can be prefixed\nwith a repetition number, or a shape. In these cases an array\nelement is created, i.e., an array within a record. That array\nis still referred to as a single field. An example: ::\n\n >>> x = np.zeros(3, dtype=\'3int8, float32, (2,3)float64\')\n >>> x\n array([([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),\n        ([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),\n        ([0, 0, 0], 0.0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])],\n       dtype=[(\'f0\', \'|i1\', 3), (\'f1\', \'>f4\'), (\'f2\', \'>f8\', (2, 3))])\n\nBy using strings to define the record structure, it precludes being\nable to name the fields in the original definition. The names can\nbe changed as shown later, however.\n\n2) Tuple argument: The only relevant tuple case that applies to record\nstructures is when a structure is mapped to an existing data type. This\nis done by pairing in a tuple, the existing data type with a matching\ndtype definition (using any of the variants being described here). As\nan example (using a definition using a list, so see 3) for further\ndetails): ::\n\n >>> x = np.zeros(3, dtype=(\'i4\',[(\'r\',\'u1\'), (\'g\',\'u1\'), (\'b\',\'u1\'), (\'a\',\'u1\')]))\n >>> x\n array([0, 0, 0])\n >>> x[\'r\']\n array([0, 0, 0], dtype=uint8)\n\nIn this case, an array is produced that looks and acts like a simple int32 array,\nbut also has definitions for fields that use only one byte of the int32 (a bit\nlike Fortran equivalencing).\n\n3) List argument: In this case the record structure is defined with a list of\ntuples. Each tuple has 2 or 3 elements specifying: 1) The name of the field\n(\'\' is permitted), 2) the type of the field, and 3) the shape (optional).\nFor example::\n\n >>> x = np.zeros(3, dtype=[(\'x\',\'f4\'),(\'y\',np.float32),(\'value\',\'f4\',(2,2))])\n >>> x\n array([(0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]]),\n        (0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]]),\n        (0.0, 0.0, [[0.0, 0.0], [0.0, 0.0]])],\n       dtype=[(\'x\', \'>f4\'), (\'y\', \'>f4\'), (\'value\', \'>f4\', (2, 2))])\n\n4) Dictionary argument: two different forms are permitted. The first consists\nof a dictionary with two required keys (\'names\' and \'formats\'), each having an\nequal sized list of values. The format list contains any type/shape specifier\nallowed in other contexts. The names must be strings. There are two optional\nkeys: \'offsets\' and \'titles\'. Each must be a correspondingly matching list to\nthe required two where offsets contain integer offsets for each field, and\ntitles are objects containing metadata for each field (these do not have\nto be strings), where the value of None is permitted. As an example: ::\n\n >>> x = np.zeros(3, dtype={\'names\':[\'col1\', \'col2\'], \'formats\':[\'i4\',\'f4\']})\n >>> x\n array([(0, 0.0), (0, 0.0), (0, 0.0)],\n       dtype=[(\'col1\', \'>i4\'), (\'col2\', \'>f4\')])\n\nThe other dictionary form permitted is a dictionary of name keys with tuple\nvalues specifying type, offset, and an optional title. ::\n\n >>> x = np.zeros(3, dtype={\'col1\':(\'i1\',0,\'title 1\'), \'col2\':(\'f4\',1,\'title 2\')})\n >>> x\n array([(0, 0.0), (0, 0.0), (0, 0.0)],\n       dtype=[((\'title 1\', \'col1\'), \'|i1\'), ((\'title 2\', \'col2\'), \'>f4\')])\n\nAccessing and modifying field names\n===================================\n\nThe field names are an attribute of the dtype object defining the structure.\nFor the last example: ::\n\n >>> x.dtype.names\n (\'col1\', \'col2\')\n >>> x.dtype.names = (\'x\', \'y\')\n >>> x\n array([(0, 0.0), (0, 0.0), (0, 0.0)],\n      dtype=[((\'title 1\', \'x\'), \'|i1\'), ((\'title 2\', \'y\'), \'>f4\')])\n >>> x.dtype.names = (\'x\', \'y\', \'z\') # wrong number of names\n <type \'exceptions.ValueError\'>: must replace all names at once with a sequence of length 2\n\nAccessing field titles\n====================================\n\nThe field titles provide a standard place to put associated info for fields.\nThey do not have to be strings. ::\n\n >>> x.dtype.fields[\'x\'][2]\n \'title 1\'\n\nAccessing multiple fields at once\n====================================\n\nYou can access multiple fields at once using a list of field names: ::\n\n >>> x = np.array([(1.5,2.5,(1.0,2.0)),(3.,4.,(4.,5.)),(1.,3.,(2.,6.))],\n         dtype=[(\'x\',\'f4\'),(\'y\',np.float32),(\'value\',\'f4\',(2,2))])\n\nNotice that `x` is created with a list of tuples. ::\n\n >>> x[[\'x\',\'y\']]\n array([(1.5, 2.5), (3.0, 4.0), (1.0, 3.0)],\n      dtype=[(\'x\', \'<f4\'), (\'y\', \'<f4\')])\n >>> x[[\'x\',\'value\']]\n array([(1.5, [[1.0, 2.0], [1.0, 2.0]]), (3.0, [[4.0, 5.0], [4.0, 5.0]]),\n       (1.0, [[2.0, 6.0], [2.0, 6.0]])],\n      dtype=[(\'x\', \'<f4\'), (\'value\', \'<f4\', (2, 2))])\n\nThe fields are returned in the order they are asked for.::\n\n >>> x[[\'y\',\'x\']]\n array([(2.5, 1.5), (4.0, 3.0), (3.0, 1.0)],\n      dtype=[(\'y\', \'<f4\'), (\'x\', \'<f4\')])\n\nFilling structured arrays\n=========================\n\nStructured arrays can be filled by field or row by row. ::\n\n >>> arr = np.zeros((5,), dtype=[(\'var1\',\'f8\'),(\'var2\',\'f8\')])\n >>> arr[\'var1\'] = np.arange(5)\n\nIf you fill it in row by row, it takes a take a tuple\n(but not a list or array!)::\n\n >>> arr[0] = (10,20)\n >>> arr\n array([(10.0, 20.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],\n      dtype=[(\'var1\', \'<f8\'), (\'var2\', \'<f8\')])\n\nRecord Arrays\n=============\n\nFor convenience, numpy provides "record arrays" which allow one to access\nfields of structured arrays by attribute rather than by index. Record arrays\nare structured arrays wrapped using a subclass of ndarray,\n:class:`numpy.recarray`, which allows field access by attribute on the array\nobject, and record arrays also use a special datatype, :class:`numpy.record`,\nwhich allows field access by attribute on the individual elements of the array. \n\nThe simplest way to create a record array is with :func:`numpy.rec.array`: ::\n\n >>> recordarr = np.rec.array([(1,2.,\'Hello\'),(2,3.,"World")], \n ...                    dtype=[(\'foo\', \'i4\'),(\'bar\', \'f4\'), (\'baz\', \'S10\')])\n >>> recordarr.bar\n array([ 2.,  3.], dtype=float32)\n >>> recordarr[1:2]\n rec.array([(2, 3.0, \'World\')], \n       dtype=[(\'foo\', \'<i4\'), (\'bar\', \'<f4\'), (\'baz\', \'S10\')])\n >>> recordarr[1:2].foo\n array([2], dtype=int32)\n >>> recordarr.foo[1:2]\n array([2], dtype=int32)\n >>> recordarr[1].baz\n \'World\'\n\nnumpy.rec.array can convert a wide variety of arguments into record arrays,\nincluding normal structured arrays: ::\n\n >>> arr = array([(1,2.,\'Hello\'),(2,3.,"World")], \n ...             dtype=[(\'foo\', \'i4\'), (\'bar\', \'f4\'), (\'baz\', \'S10\')])\n >>> recordarr = np.rec.array(arr)\n\nThe numpy.rec module provides a number of other convenience functions for\ncreating record arrays, see :ref:`record array creation routines\n<routines.array-creation.rec>`.\n\nA record array representation of a structured array can be obtained using the\nappropriate :ref:`view`: ::\n\n >>> arr = np.array([(1,2.,\'Hello\'),(2,3.,"World")], \n ...                dtype=[(\'foo\', \'i4\'),(\'bar\', \'f4\'), (\'baz\', \'a10\')])\n >>> recordarr = arr.view(dtype=dtype((np.record, arr.dtype)), \n ...                      type=np.recarray)\n\nFor convenience, viewing an ndarray as type `np.recarray` will automatically\nconvert to `np.record` datatype, so the dtype can be left out of the view: ::\n\n >>> recordarr = arr.view(np.recarray)\n >>> recordarr.dtype\n dtype((numpy.record, [(\'foo\', \'<i4\'), (\'bar\', \'<f4\'), (\'baz\', \'S10\')]))\n\nTo get back to a plain ndarray both the dtype and type must be reset. The\nfollowing view does so, taking into account the unusual case that the\nrecordarr was not a structured type: ::\n\n >>> arr2 = recordarr.view(recordarr.dtype.fields or recordarr.dtype, np.ndarray)\n\nRecord array fields accessed by index or by attribute are returned as a record\narray if the field has a structured type but as a plain ndarray otherwise. ::\n\n >>> recordarr = np.rec.array([(\'Hello\', (1,2)),("World", (3,4))], \n ...                 dtype=[(\'foo\', \'S6\'),(\'bar\', [(\'A\', int), (\'B\', int)])])\n >>> type(recordarr.foo)\n <type \'numpy.ndarray\'>\n >>> type(recordarr.bar)\n <class \'numpy.core.records.recarray\'>\n\nNote that if a field has the same name as an ndarray attribute, the ndarray\nattribute takes precedence. Such fields will be inaccessible by attribute but\nmay still be accessed by index.\n\n\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

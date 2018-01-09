
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''==============
2: Array indexing
3: ==============
4: 
5: Array indexing refers to any use of the square brackets ([]) to index
6: array values. There are many options to indexing, which give numpy
7: indexing great power, but with power comes some complexity and the
8: potential for confusion. This section is just an overview of the
9: various options and issues related to indexing. Aside from single
10: element indexing, the details on most of these options are to be
11: found in related sections.
12: 
13: Assignment vs referencing
14: =========================
15: 
16: Most of the following examples show the use of indexing when
17: referencing data in an array. The examples work just as well
18: when assigning to an array. See the section at the end for
19: specific examples and explanations on how assignments work.
20: 
21: Single element indexing
22: =======================
23: 
24: Single element indexing for a 1-D array is what one expects. It work
25: exactly like that for other standard Python sequences. It is 0-based,
26: and accepts negative indices for indexing from the end of the array. ::
27: 
28:     >>> x = np.arange(10)
29:     >>> x[2]
30:     2
31:     >>> x[-2]
32:     8
33: 
34: Unlike lists and tuples, numpy arrays support multidimensional indexing
35: for multidimensional arrays. That means that it is not necessary to
36: separate each dimension's index into its own set of square brackets. ::
37: 
38:     >>> x.shape = (2,5) # now x is 2-dimensional
39:     >>> x[1,3]
40:     8
41:     >>> x[1,-1]
42:     9
43: 
44: Note that if one indexes a multidimensional array with fewer indices
45: than dimensions, one gets a subdimensional array. For example: ::
46: 
47:     >>> x[0]
48:     array([0, 1, 2, 3, 4])
49: 
50: That is, each index specified selects the array corresponding to the
51: rest of the dimensions selected. In the above example, choosing 0
52: means that the remaining dimension of length 5 is being left unspecified,
53: and that what is returned is an array of that dimensionality and size.
54: It must be noted that the returned array is not a copy of the original,
55: but points to the same values in memory as does the original array.
56: In  this case, the 1-D array at the first position (0) is returned.
57: So using a single index on the returned array, results in a single
58: element being returned. That is: ::
59: 
60:     >>> x[0][2]
61:     2
62: 
63: So note that ``x[0,2] = x[0][2]`` though the second case is more
64: inefficient as a new temporary array is created after the first index
65: that is subsequently indexed by 2.
66: 
67: Note to those used to IDL or Fortran memory order as it relates to
68: indexing.  Numpy uses C-order indexing. That means that the last
69: index usually represents the most rapidly changing memory location,
70: unlike Fortran or IDL, where the first index represents the most
71: rapidly changing location in memory. This difference represents a
72: great potential for confusion.
73: 
74: Other indexing options
75: ======================
76: 
77: It is possible to slice and stride arrays to extract arrays of the
78: same number of dimensions, but of different sizes than the original.
79: The slicing and striding works exactly the same way it does for lists
80: and tuples except that they can be applied to multiple dimensions as
81: well. A few examples illustrates best: ::
82: 
83:  >>> x = np.arange(10)
84:  >>> x[2:5]
85:  array([2, 3, 4])
86:  >>> x[:-7]
87:  array([0, 1, 2])
88:  >>> x[1:7:2]
89:  array([1, 3, 5])
90:  >>> y = np.arange(35).reshape(5,7)
91:  >>> y[1:5:2,::3]
92:  array([[ 7, 10, 13],
93:         [21, 24, 27]])
94: 
95: Note that slices of arrays do not copy the internal array data but
96: also produce new views of the original data.
97: 
98: It is possible to index arrays with other arrays for the purposes of
99: selecting lists of values out of arrays into new arrays. There are
100: two different ways of accomplishing this. One uses one or more arrays
101: of index values. The other involves giving a boolean array of the proper
102: shape to indicate the values to be selected. Index arrays are a very
103: powerful tool that allow one to avoid looping over individual elements in
104: arrays and thus greatly improve performance.
105: 
106: It is possible to use special features to effectively increase the
107: number of dimensions in an array through indexing so the resulting
108: array aquires the shape needed for use in an expression or with a
109: specific function.
110: 
111: Index arrays
112: ============
113: 
114: Numpy arrays may be indexed with other arrays (or any other sequence-
115: like object that can be converted to an array, such as lists, with the
116: exception of tuples; see the end of this document for why this is). The
117: use of index arrays ranges from simple, straightforward cases to
118: complex, hard-to-understand cases. For all cases of index arrays, what
119: is returned is a copy of the original data, not a view as one gets for
120: slices.
121: 
122: Index arrays must be of integer type. Each value in the array indicates
123: which value in the array to use in place of the index. To illustrate: ::
124: 
125:  >>> x = np.arange(10,1,-1)
126:  >>> x
127:  array([10,  9,  8,  7,  6,  5,  4,  3,  2])
128:  >>> x[np.array([3, 3, 1, 8])]
129:  array([7, 7, 9, 2])
130: 
131: 
132: The index array consisting of the values 3, 3, 1 and 8 correspondingly
133: create an array of length 4 (same as the index array) where each index
134: is replaced by the value the index array has in the array being indexed.
135: 
136: Negative values are permitted and work as they do with single indices
137: or slices: ::
138: 
139:  >>> x[np.array([3,3,-3,8])]
140:  array([7, 7, 4, 2])
141: 
142: It is an error to have index values out of bounds: ::
143: 
144:  >>> x[np.array([3, 3, 20, 8])]
145:  <type 'exceptions.IndexError'>: index 20 out of bounds 0<=index<9
146: 
147: Generally speaking, what is returned when index arrays are used is
148: an array with the same shape as the index array, but with the type
149: and values of the array being indexed. As an example, we can use a
150: multidimensional index array instead: ::
151: 
152:  >>> x[np.array([[1,1],[2,3]])]
153:  array([[9, 9],
154:         [8, 7]])
155: 
156: Indexing Multi-dimensional arrays
157: =================================
158: 
159: Things become more complex when multidimensional arrays are indexed,
160: particularly with multidimensional index arrays. These tend to be
161: more unusal uses, but theyare permitted, and they are useful for some
162: problems. We'll  start with thesimplest multidimensional case (using
163: the array y from the previous examples): ::
164: 
165:  >>> y[np.array([0,2,4]), np.array([0,1,2])]
166:  array([ 0, 15, 30])
167: 
168: In this case, if the index arrays have a matching shape, and there is
169: an index array for each dimension of the array being indexed, the
170: resultant array has the same shape as the index arrays, and the values
171: correspond to the index set for each position in the index arrays. In
172: this example, the first index value is 0 for both index arrays, and
173: thus the first value of the resultant array is y[0,0]. The next value
174: is y[2,1], and the last is y[4,2].
175: 
176: If the index arrays do not have the same shape, there is an attempt to
177: broadcast them to the same shape.  If they cannot be broadcast to the
178: same shape, an exception is raised: ::
179: 
180:  >>> y[np.array([0,2,4]), np.array([0,1])]
181:  <type 'exceptions.ValueError'>: shape mismatch: objects cannot be
182:  broadcast to a single shape
183: 
184: The broadcasting mechanism permits index arrays to be combined with
185: scalars for other indices. The effect is that the scalar value is used
186: for all the corresponding values of the index arrays: ::
187: 
188:  >>> y[np.array([0,2,4]), 1]
189:  array([ 1, 15, 29])
190: 
191: Jumping to the next level of complexity, it is possible to only
192: partially index an array with index arrays. It takes a bit of thought
193: to understand what happens in such cases. For example if we just use
194: one index array with y: ::
195: 
196:  >>> y[np.array([0,2,4])]
197:  array([[ 0,  1,  2,  3,  4,  5,  6],
198:         [14, 15, 16, 17, 18, 19, 20],
199:         [28, 29, 30, 31, 32, 33, 34]])
200: 
201: What results is the construction of a new array where each value of
202: the index array selects one row from the array being indexed and the
203: resultant array has the resulting shape (size of row, number index
204: elements).
205: 
206: An example of where this may be useful is for a color lookup table
207: where we want to map the values of an image into RGB triples for
208: display. The lookup table could have a shape (nlookup, 3). Indexing
209: such an array with an image with shape (ny, nx) with dtype=np.uint8
210: (or any integer type so long as values are with the bounds of the
211: lookup table) will result in an array of shape (ny, nx, 3) where a
212: triple of RGB values is associated with each pixel location.
213: 
214: In general, the shape of the resulant array will be the concatenation
215: of the shape of the index array (or the shape that all the index arrays
216: were broadcast to) with the shape of any unused dimensions (those not
217: indexed) in the array being indexed.
218: 
219: Boolean or "mask" index arrays
220: ==============================
221: 
222: Boolean arrays used as indices are treated in a different manner
223: entirely than index arrays. Boolean arrays must be of the same shape
224: as the initial dimensions of the array being indexed. In the
225: most straightforward case, the boolean array has the same shape: ::
226: 
227:  >>> b = y>20
228:  >>> y[b]
229:  array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
230: 
231: Unlike in the case of integer index arrays, in the boolean case, the
232: result is a 1-D array containing all the elements in the indexed array
233: corresponding to all the true elements in the boolean array. The
234: elements in the indexed array are always iterated and returned in
235: :term:`row-major` (C-style) order. The result is also identical to
236: ``y[np.nonzero(b)]``. As with index arrays, what is returned is a copy
237: of the data, not a view as one gets with slices.
238: 
239: The result will be multidimensional if y has more dimensions than b.
240: For example: ::
241: 
242:  >>> b[:,5] # use a 1-D boolean whose first dim agrees with the first dim of y
243:  array([False, False, False,  True,  True], dtype=bool)
244:  >>> y[b[:,5]]
245:  array([[21, 22, 23, 24, 25, 26, 27],
246:         [28, 29, 30, 31, 32, 33, 34]])
247: 
248: Here the 4th and 5th rows are selected from the indexed array and
249: combined to make a 2-D array.
250: 
251: In general, when the boolean array has fewer dimensions than the array
252: being indexed, this is equivalent to y[b, ...], which means
253: y is indexed by b followed by as many : as are needed to fill
254: out the rank of y.
255: Thus the shape of the result is one dimension containing the number
256: of True elements of the boolean array, followed by the remaining
257: dimensions of the array being indexed.
258: 
259: For example, using a 2-D boolean array of shape (2,3)
260: with four True elements to select rows from a 3-D array of shape
261: (2,3,5) results in a 2-D result of shape (4,5): ::
262: 
263:  >>> x = np.arange(30).reshape(2,3,5)
264:  >>> x
265:  array([[[ 0,  1,  2,  3,  4],
266:          [ 5,  6,  7,  8,  9],
267:          [10, 11, 12, 13, 14]],
268:         [[15, 16, 17, 18, 19],
269:          [20, 21, 22, 23, 24],
270:          [25, 26, 27, 28, 29]]])
271:  >>> b = np.array([[True, True, False], [False, True, True]])
272:  >>> x[b]
273:  array([[ 0,  1,  2,  3,  4],
274:         [ 5,  6,  7,  8,  9],
275:         [20, 21, 22, 23, 24],
276:         [25, 26, 27, 28, 29]])
277: 
278: For further details, consult the numpy reference documentation on array indexing.
279: 
280: Combining index arrays with slices
281: ==================================
282: 
283: Index arrays may be combined with slices. For example: ::
284: 
285:  >>> y[np.array([0,2,4]),1:3]
286:  array([[ 1,  2],
287:         [15, 16],
288:         [29, 30]])
289: 
290: In effect, the slice is converted to an index array
291: np.array([[1,2]]) (shape (1,2)) that is broadcast with the index array
292: to produce a resultant array of shape (3,2).
293: 
294: Likewise, slicing can be combined with broadcasted boolean indices: ::
295: 
296:  >>> y[b[:,5],1:3]
297:  array([[22, 23],
298:         [29, 30]])
299: 
300: Structural indexing tools
301: =========================
302: 
303: To facilitate easy matching of array shapes with expressions and in
304: assignments, the np.newaxis object can be used within array indices
305: to add new dimensions with a size of 1. For example: ::
306: 
307:  >>> y.shape
308:  (5, 7)
309:  >>> y[:,np.newaxis,:].shape
310:  (5, 1, 7)
311: 
312: Note that there are no new elements in the array, just that the
313: dimensionality is increased. This can be handy to combine two
314: arrays in a way that otherwise would require explicitly reshaping
315: operations. For example: ::
316: 
317:  >>> x = np.arange(5)
318:  >>> x[:,np.newaxis] + x[np.newaxis,:]
319:  array([[0, 1, 2, 3, 4],
320:         [1, 2, 3, 4, 5],
321:         [2, 3, 4, 5, 6],
322:         [3, 4, 5, 6, 7],
323:         [4, 5, 6, 7, 8]])
324: 
325: The ellipsis syntax maybe used to indicate selecting in full any
326: remaining unspecified dimensions. For example: ::
327: 
328:  >>> z = np.arange(81).reshape(3,3,3,3)
329:  >>> z[1,...,2]
330:  array([[29, 32, 35],
331:         [38, 41, 44],
332:         [47, 50, 53]])
333: 
334: This is equivalent to: ::
335: 
336:  >>> z[1,:,:,2]
337:  array([[29, 32, 35],
338:         [38, 41, 44],
339:         [47, 50, 53]])
340: 
341: Assigning values to indexed arrays
342: ==================================
343: 
344: As mentioned, one can select a subset of an array to assign to using
345: a single index, slices, and index and mask arrays. The value being
346: assigned to the indexed array must be shape consistent (the same shape
347: or broadcastable to the shape the index produces). For example, it is
348: permitted to assign a constant to a slice: ::
349: 
350:  >>> x = np.arange(10)
351:  >>> x[2:7] = 1
352: 
353: or an array of the right size: ::
354: 
355:  >>> x[2:7] = np.arange(5)
356: 
357: Note that assignments may result in changes if assigning
358: higher types to lower types (like floats to ints) or even
359: exceptions (assigning complex to floats or ints): ::
360: 
361:  >>> x[1] = 1.2
362:  >>> x[1]
363:  1
364:  >>> x[1] = 1.2j
365:  <type 'exceptions.TypeError'>: can't convert complex to long; use
366:  long(abs(z))
367: 
368: 
369: Unlike some of the references (such as array and mask indices)
370: assignments are always made to the original data in the array
371: (indeed, nothing else would make sense!). Note though, that some
372: actions may not work as one may naively expect. This particular
373: example is often surprising to people: ::
374: 
375:  >>> x = np.arange(0, 50, 10)
376:  >>> x
377:  array([ 0, 10, 20, 30, 40])
378:  >>> x[np.array([1, 1, 3, 1])] += 1
379:  >>> x
380:  array([ 0, 11, 20, 31, 40])
381: 
382: Where people expect that the 1st location will be incremented by 3.
383: In fact, it will only be incremented by 1. The reason is because
384: a new array is extracted from the original (as a temporary) containing
385: the values at 1, 1, 3, 1, then the value 1 is added to the temporary,
386: and then the temporary is assigned back to the original array. Thus
387: the value of the array at x[1]+1 is assigned to x[1] three times,
388: rather than being incremented 3 times.
389: 
390: Dealing with variable numbers of indices within programs
391: ========================================================
392: 
393: The index syntax is very powerful but limiting when dealing with
394: a variable number of indices. For example, if you want to write
395: a function that can handle arguments with various numbers of
396: dimensions without having to write special case code for each
397: number of possible dimensions, how can that be done? If one
398: supplies to the index a tuple, the tuple will be interpreted
399: as a list of indices. For example (using the previous definition
400: for the array z): ::
401: 
402:  >>> indices = (1,1,1,1)
403:  >>> z[indices]
404:  40
405: 
406: So one can use code to construct tuples of any number of indices
407: and then use these within an index.
408: 
409: Slices can be specified within programs by using the slice() function
410: in Python. For example: ::
411: 
412:  >>> indices = (1,1,1,slice(0,2)) # same as [1,1,1,0:2]
413:  >>> z[indices]
414:  array([39, 40])
415: 
416: Likewise, ellipsis can be specified by code by using the Ellipsis
417: object: ::
418: 
419:  >>> indices = (1, Ellipsis, 1) # same as [1,...,1]
420:  >>> z[indices]
421:  array([[28, 31, 34],
422:         [37, 40, 43],
423:         [46, 49, 52]])
424: 
425: For this reason it is possible to use the output from the np.where()
426: function directly as an index since it always returns a tuple of index
427: arrays.
428: 
429: Because the special treatment of tuples, they are not automatically
430: converted to an array as a list would be. As an example: ::
431: 
432:  >>> z[[1,1,1,1]] # produces a large array
433:  array([[[[27, 28, 29],
434:           [30, 31, 32], ...
435:  >>> z[(1,1,1,1)] # returns a single value
436:  40
437: 
438: '''
439: from __future__ import division, absolute_import, print_function
440: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, (-1)), 'str', '==============\nArray indexing\n==============\n\nArray indexing refers to any use of the square brackets ([]) to index\narray values. There are many options to indexing, which give numpy\nindexing great power, but with power comes some complexity and the\npotential for confusion. This section is just an overview of the\nvarious options and issues related to indexing. Aside from single\nelement indexing, the details on most of these options are to be\nfound in related sections.\n\nAssignment vs referencing\n=========================\n\nMost of the following examples show the use of indexing when\nreferencing data in an array. The examples work just as well\nwhen assigning to an array. See the section at the end for\nspecific examples and explanations on how assignments work.\n\nSingle element indexing\n=======================\n\nSingle element indexing for a 1-D array is what one expects. It work\nexactly like that for other standard Python sequences. It is 0-based,\nand accepts negative indices for indexing from the end of the array. ::\n\n    >>> x = np.arange(10)\n    >>> x[2]\n    2\n    >>> x[-2]\n    8\n\nUnlike lists and tuples, numpy arrays support multidimensional indexing\nfor multidimensional arrays. That means that it is not necessary to\nseparate each dimension\'s index into its own set of square brackets. ::\n\n    >>> x.shape = (2,5) # now x is 2-dimensional\n    >>> x[1,3]\n    8\n    >>> x[1,-1]\n    9\n\nNote that if one indexes a multidimensional array with fewer indices\nthan dimensions, one gets a subdimensional array. For example: ::\n\n    >>> x[0]\n    array([0, 1, 2, 3, 4])\n\nThat is, each index specified selects the array corresponding to the\nrest of the dimensions selected. In the above example, choosing 0\nmeans that the remaining dimension of length 5 is being left unspecified,\nand that what is returned is an array of that dimensionality and size.\nIt must be noted that the returned array is not a copy of the original,\nbut points to the same values in memory as does the original array.\nIn  this case, the 1-D array at the first position (0) is returned.\nSo using a single index on the returned array, results in a single\nelement being returned. That is: ::\n\n    >>> x[0][2]\n    2\n\nSo note that ``x[0,2] = x[0][2]`` though the second case is more\ninefficient as a new temporary array is created after the first index\nthat is subsequently indexed by 2.\n\nNote to those used to IDL or Fortran memory order as it relates to\nindexing.  Numpy uses C-order indexing. That means that the last\nindex usually represents the most rapidly changing memory location,\nunlike Fortran or IDL, where the first index represents the most\nrapidly changing location in memory. This difference represents a\ngreat potential for confusion.\n\nOther indexing options\n======================\n\nIt is possible to slice and stride arrays to extract arrays of the\nsame number of dimensions, but of different sizes than the original.\nThe slicing and striding works exactly the same way it does for lists\nand tuples except that they can be applied to multiple dimensions as\nwell. A few examples illustrates best: ::\n\n >>> x = np.arange(10)\n >>> x[2:5]\n array([2, 3, 4])\n >>> x[:-7]\n array([0, 1, 2])\n >>> x[1:7:2]\n array([1, 3, 5])\n >>> y = np.arange(35).reshape(5,7)\n >>> y[1:5:2,::3]\n array([[ 7, 10, 13],\n        [21, 24, 27]])\n\nNote that slices of arrays do not copy the internal array data but\nalso produce new views of the original data.\n\nIt is possible to index arrays with other arrays for the purposes of\nselecting lists of values out of arrays into new arrays. There are\ntwo different ways of accomplishing this. One uses one or more arrays\nof index values. The other involves giving a boolean array of the proper\nshape to indicate the values to be selected. Index arrays are a very\npowerful tool that allow one to avoid looping over individual elements in\narrays and thus greatly improve performance.\n\nIt is possible to use special features to effectively increase the\nnumber of dimensions in an array through indexing so the resulting\narray aquires the shape needed for use in an expression or with a\nspecific function.\n\nIndex arrays\n============\n\nNumpy arrays may be indexed with other arrays (or any other sequence-\nlike object that can be converted to an array, such as lists, with the\nexception of tuples; see the end of this document for why this is). The\nuse of index arrays ranges from simple, straightforward cases to\ncomplex, hard-to-understand cases. For all cases of index arrays, what\nis returned is a copy of the original data, not a view as one gets for\nslices.\n\nIndex arrays must be of integer type. Each value in the array indicates\nwhich value in the array to use in place of the index. To illustrate: ::\n\n >>> x = np.arange(10,1,-1)\n >>> x\n array([10,  9,  8,  7,  6,  5,  4,  3,  2])\n >>> x[np.array([3, 3, 1, 8])]\n array([7, 7, 9, 2])\n\n\nThe index array consisting of the values 3, 3, 1 and 8 correspondingly\ncreate an array of length 4 (same as the index array) where each index\nis replaced by the value the index array has in the array being indexed.\n\nNegative values are permitted and work as they do with single indices\nor slices: ::\n\n >>> x[np.array([3,3,-3,8])]\n array([7, 7, 4, 2])\n\nIt is an error to have index values out of bounds: ::\n\n >>> x[np.array([3, 3, 20, 8])]\n <type \'exceptions.IndexError\'>: index 20 out of bounds 0<=index<9\n\nGenerally speaking, what is returned when index arrays are used is\nan array with the same shape as the index array, but with the type\nand values of the array being indexed. As an example, we can use a\nmultidimensional index array instead: ::\n\n >>> x[np.array([[1,1],[2,3]])]\n array([[9, 9],\n        [8, 7]])\n\nIndexing Multi-dimensional arrays\n=================================\n\nThings become more complex when multidimensional arrays are indexed,\nparticularly with multidimensional index arrays. These tend to be\nmore unusal uses, but theyare permitted, and they are useful for some\nproblems. We\'ll  start with thesimplest multidimensional case (using\nthe array y from the previous examples): ::\n\n >>> y[np.array([0,2,4]), np.array([0,1,2])]\n array([ 0, 15, 30])\n\nIn this case, if the index arrays have a matching shape, and there is\nan index array for each dimension of the array being indexed, the\nresultant array has the same shape as the index arrays, and the values\ncorrespond to the index set for each position in the index arrays. In\nthis example, the first index value is 0 for both index arrays, and\nthus the first value of the resultant array is y[0,0]. The next value\nis y[2,1], and the last is y[4,2].\n\nIf the index arrays do not have the same shape, there is an attempt to\nbroadcast them to the same shape.  If they cannot be broadcast to the\nsame shape, an exception is raised: ::\n\n >>> y[np.array([0,2,4]), np.array([0,1])]\n <type \'exceptions.ValueError\'>: shape mismatch: objects cannot be\n broadcast to a single shape\n\nThe broadcasting mechanism permits index arrays to be combined with\nscalars for other indices. The effect is that the scalar value is used\nfor all the corresponding values of the index arrays: ::\n\n >>> y[np.array([0,2,4]), 1]\n array([ 1, 15, 29])\n\nJumping to the next level of complexity, it is possible to only\npartially index an array with index arrays. It takes a bit of thought\nto understand what happens in such cases. For example if we just use\none index array with y: ::\n\n >>> y[np.array([0,2,4])]\n array([[ 0,  1,  2,  3,  4,  5,  6],\n        [14, 15, 16, 17, 18, 19, 20],\n        [28, 29, 30, 31, 32, 33, 34]])\n\nWhat results is the construction of a new array where each value of\nthe index array selects one row from the array being indexed and the\nresultant array has the resulting shape (size of row, number index\nelements).\n\nAn example of where this may be useful is for a color lookup table\nwhere we want to map the values of an image into RGB triples for\ndisplay. The lookup table could have a shape (nlookup, 3). Indexing\nsuch an array with an image with shape (ny, nx) with dtype=np.uint8\n(or any integer type so long as values are with the bounds of the\nlookup table) will result in an array of shape (ny, nx, 3) where a\ntriple of RGB values is associated with each pixel location.\n\nIn general, the shape of the resulant array will be the concatenation\nof the shape of the index array (or the shape that all the index arrays\nwere broadcast to) with the shape of any unused dimensions (those not\nindexed) in the array being indexed.\n\nBoolean or "mask" index arrays\n==============================\n\nBoolean arrays used as indices are treated in a different manner\nentirely than index arrays. Boolean arrays must be of the same shape\nas the initial dimensions of the array being indexed. In the\nmost straightforward case, the boolean array has the same shape: ::\n\n >>> b = y>20\n >>> y[b]\n array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])\n\nUnlike in the case of integer index arrays, in the boolean case, the\nresult is a 1-D array containing all the elements in the indexed array\ncorresponding to all the true elements in the boolean array. The\nelements in the indexed array are always iterated and returned in\n:term:`row-major` (C-style) order. The result is also identical to\n``y[np.nonzero(b)]``. As with index arrays, what is returned is a copy\nof the data, not a view as one gets with slices.\n\nThe result will be multidimensional if y has more dimensions than b.\nFor example: ::\n\n >>> b[:,5] # use a 1-D boolean whose first dim agrees with the first dim of y\n array([False, False, False,  True,  True], dtype=bool)\n >>> y[b[:,5]]\n array([[21, 22, 23, 24, 25, 26, 27],\n        [28, 29, 30, 31, 32, 33, 34]])\n\nHere the 4th and 5th rows are selected from the indexed array and\ncombined to make a 2-D array.\n\nIn general, when the boolean array has fewer dimensions than the array\nbeing indexed, this is equivalent to y[b, ...], which means\ny is indexed by b followed by as many : as are needed to fill\nout the rank of y.\nThus the shape of the result is one dimension containing the number\nof True elements of the boolean array, followed by the remaining\ndimensions of the array being indexed.\n\nFor example, using a 2-D boolean array of shape (2,3)\nwith four True elements to select rows from a 3-D array of shape\n(2,3,5) results in a 2-D result of shape (4,5): ::\n\n >>> x = np.arange(30).reshape(2,3,5)\n >>> x\n array([[[ 0,  1,  2,  3,  4],\n         [ 5,  6,  7,  8,  9],\n         [10, 11, 12, 13, 14]],\n        [[15, 16, 17, 18, 19],\n         [20, 21, 22, 23, 24],\n         [25, 26, 27, 28, 29]]])\n >>> b = np.array([[True, True, False], [False, True, True]])\n >>> x[b]\n array([[ 0,  1,  2,  3,  4],\n        [ 5,  6,  7,  8,  9],\n        [20, 21, 22, 23, 24],\n        [25, 26, 27, 28, 29]])\n\nFor further details, consult the numpy reference documentation on array indexing.\n\nCombining index arrays with slices\n==================================\n\nIndex arrays may be combined with slices. For example: ::\n\n >>> y[np.array([0,2,4]),1:3]\n array([[ 1,  2],\n        [15, 16],\n        [29, 30]])\n\nIn effect, the slice is converted to an index array\nnp.array([[1,2]]) (shape (1,2)) that is broadcast with the index array\nto produce a resultant array of shape (3,2).\n\nLikewise, slicing can be combined with broadcasted boolean indices: ::\n\n >>> y[b[:,5],1:3]\n array([[22, 23],\n        [29, 30]])\n\nStructural indexing tools\n=========================\n\nTo facilitate easy matching of array shapes with expressions and in\nassignments, the np.newaxis object can be used within array indices\nto add new dimensions with a size of 1. For example: ::\n\n >>> y.shape\n (5, 7)\n >>> y[:,np.newaxis,:].shape\n (5, 1, 7)\n\nNote that there are no new elements in the array, just that the\ndimensionality is increased. This can be handy to combine two\narrays in a way that otherwise would require explicitly reshaping\noperations. For example: ::\n\n >>> x = np.arange(5)\n >>> x[:,np.newaxis] + x[np.newaxis,:]\n array([[0, 1, 2, 3, 4],\n        [1, 2, 3, 4, 5],\n        [2, 3, 4, 5, 6],\n        [3, 4, 5, 6, 7],\n        [4, 5, 6, 7, 8]])\n\nThe ellipsis syntax maybe used to indicate selecting in full any\nremaining unspecified dimensions. For example: ::\n\n >>> z = np.arange(81).reshape(3,3,3,3)\n >>> z[1,...,2]\n array([[29, 32, 35],\n        [38, 41, 44],\n        [47, 50, 53]])\n\nThis is equivalent to: ::\n\n >>> z[1,:,:,2]\n array([[29, 32, 35],\n        [38, 41, 44],\n        [47, 50, 53]])\n\nAssigning values to indexed arrays\n==================================\n\nAs mentioned, one can select a subset of an array to assign to using\na single index, slices, and index and mask arrays. The value being\nassigned to the indexed array must be shape consistent (the same shape\nor broadcastable to the shape the index produces). For example, it is\npermitted to assign a constant to a slice: ::\n\n >>> x = np.arange(10)\n >>> x[2:7] = 1\n\nor an array of the right size: ::\n\n >>> x[2:7] = np.arange(5)\n\nNote that assignments may result in changes if assigning\nhigher types to lower types (like floats to ints) or even\nexceptions (assigning complex to floats or ints): ::\n\n >>> x[1] = 1.2\n >>> x[1]\n 1\n >>> x[1] = 1.2j\n <type \'exceptions.TypeError\'>: can\'t convert complex to long; use\n long(abs(z))\n\n\nUnlike some of the references (such as array and mask indices)\nassignments are always made to the original data in the array\n(indeed, nothing else would make sense!). Note though, that some\nactions may not work as one may naively expect. This particular\nexample is often surprising to people: ::\n\n >>> x = np.arange(0, 50, 10)\n >>> x\n array([ 0, 10, 20, 30, 40])\n >>> x[np.array([1, 1, 3, 1])] += 1\n >>> x\n array([ 0, 11, 20, 31, 40])\n\nWhere people expect that the 1st location will be incremented by 3.\nIn fact, it will only be incremented by 1. The reason is because\na new array is extracted from the original (as a temporary) containing\nthe values at 1, 1, 3, 1, then the value 1 is added to the temporary,\nand then the temporary is assigned back to the original array. Thus\nthe value of the array at x[1]+1 is assigned to x[1] three times,\nrather than being incremented 3 times.\n\nDealing with variable numbers of indices within programs\n========================================================\n\nThe index syntax is very powerful but limiting when dealing with\na variable number of indices. For example, if you want to write\na function that can handle arguments with various numbers of\ndimensions without having to write special case code for each\nnumber of possible dimensions, how can that be done? If one\nsupplies to the index a tuple, the tuple will be interpreted\nas a list of indices. For example (using the previous definition\nfor the array z): ::\n\n >>> indices = (1,1,1,1)\n >>> z[indices]\n 40\n\nSo one can use code to construct tuples of any number of indices\nand then use these within an index.\n\nSlices can be specified within programs by using the slice() function\nin Python. For example: ::\n\n >>> indices = (1,1,1,slice(0,2)) # same as [1,1,1,0:2]\n >>> z[indices]\n array([39, 40])\n\nLikewise, ellipsis can be specified by code by using the Ellipsis\nobject: ::\n\n >>> indices = (1, Ellipsis, 1) # same as [1,...,1]\n >>> z[indices]\n array([[28, 31, 34],\n        [37, 40, 43],\n        [46, 49, 52]])\n\nFor this reason it is possible to use the output from the np.where()\nfunction directly as an index since it always returns a tuple of index\narrays.\n\nBecause the special treatment of tuples, they are not automatically\nconverted to an array as a list would be. As an example: ::\n\n >>> z[[1,1,1,1]] # produces a large array\n array([[[[27, 28, 29],\n          [30, 31, 32], ...\n >>> z[(1,1,1,1)] # returns a single value\n 40\n\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

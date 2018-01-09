
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ===============
3: Array Internals
4: ===============
5: 
6: Internal organization of numpy arrays
7: =====================================
8: 
9: It helps to understand a bit about how numpy arrays are handled under the covers to help understand numpy better. This section will not go into great detail. Those wishing to understand the full details are referred to Travis Oliphant's book "Guide to Numpy".
10: 
11: Numpy arrays consist of two major components, the raw array data (from now on,
12: referred to as the data buffer), and the information about the raw array data.
13: The data buffer is typically what people think of as arrays in C or Fortran,
14: a contiguous (and fixed) block of memory containing fixed sized data items.
15: Numpy also contains a significant set of data that describes how to interpret
16: the data in the data buffer. This extra information contains (among other things):
17: 
18:  1) The basic data element's size in bytes
19:  2) The start of the data within the data buffer (an offset relative to the
20:     beginning of the data buffer).
21:  3) The number of dimensions and the size of each dimension
22:  4) The separation between elements for each dimension (the 'stride'). This
23:     does not have to be a multiple of the element size
24:  5) The byte order of the data (which may not be the native byte order)
25:  6) Whether the buffer is read-only
26:  7) Information (via the dtype object) about the interpretation of the basic
27:     data element. The basic data element may be as simple as a int or a float,
28:     or it may be a compound object (e.g., struct-like), a fixed character field,
29:     or Python object pointers.
30:  8) Whether the array is to interpreted as C-order or Fortran-order.
31: 
32: This arrangement allow for very flexible use of arrays. One thing that it allows
33: is simple changes of the metadata to change the interpretation of the array buffer.
34: Changing the byteorder of the array is a simple change involving no rearrangement
35: of the data. The shape of the array can be changed very easily without changing
36: anything in the data buffer or any data copying at all
37: 
38: Among other things that are made possible is one can create a new array metadata
39: object that uses the same data buffer
40: to create a new view of that data buffer that has a different interpretation
41: of the buffer (e.g., different shape, offset, byte order, strides, etc) but
42: shares the same data bytes. Many operations in numpy do just this such as
43: slices. Other operations, such as transpose, don't move data elements
44: around in the array, but rather change the information about the shape and strides so that the indexing of the array changes, but the data in the doesn't move.
45: 
46: Typically these new versions of the array metadata but the same data buffer are
47: new 'views' into the data buffer. There is a different ndarray object, but it
48: uses the same data buffer. This is why it is necessary to force copies through
49: use of the .copy() method if one really wants to make a new and independent
50: copy of the data buffer.
51: 
52: New views into arrays mean the the object reference counts for the data buffer
53: increase. Simply doing away with the original array object will not remove the
54: data buffer if other views of it still exist.
55: 
56: Multidimensional Array Indexing Order Issues
57: ============================================
58: 
59: What is the right way to index
60: multi-dimensional arrays? Before you jump to conclusions about the one and
61: true way to index multi-dimensional arrays, it pays to understand why this is
62: a confusing issue. This section will try to explain in detail how numpy
63: indexing works and why we adopt the convention we do for images, and when it
64: may be appropriate to adopt other conventions.
65: 
66: The first thing to understand is
67: that there are two conflicting conventions for indexing 2-dimensional arrays.
68: Matrix notation uses the first index to indicate which row is being selected and
69: the second index to indicate which column is selected. This is opposite the
70: geometrically oriented-convention for images where people generally think the
71: first index represents x position (i.e., column) and the second represents y
72: position (i.e., row). This alone is the source of much confusion;
73: matrix-oriented users and image-oriented users expect two different things with
74: regard to indexing.
75: 
76: The second issue to understand is how indices correspond
77: to the order the array is stored in memory. In Fortran the first index is the
78: most rapidly varying index when moving through the elements of a two
79: dimensional array as it is stored in memory. If you adopt the matrix
80: convention for indexing, then this means the matrix is stored one column at a
81: time (since the first index moves to the next row as it changes). Thus Fortran
82: is considered a Column-major language. C has just the opposite convention. In
83: C, the last index changes most rapidly as one moves through the array as
84: stored in memory. Thus C is a Row-major language. The matrix is stored by
85: rows. Note that in both cases it presumes that the matrix convention for
86: indexing is being used, i.e., for both Fortran and C, the first index is the
87: row. Note this convention implies that the indexing convention is invariant
88: and that the data order changes to keep that so.
89: 
90: But that's not the only way
91: to look at it. Suppose one has large two-dimensional arrays (images or
92: matrices) stored in data files. Suppose the data are stored by rows rather than
93: by columns. If we are to preserve our index convention (whether matrix or
94: image) that means that depending on the language we use, we may be forced to
95: reorder the data if it is read into memory to preserve our indexing
96: convention. For example if we read row-ordered data into memory without
97: reordering, it will match the matrix indexing convention for C, but not for
98: Fortran. Conversely, it will match the image indexing convention for Fortran,
99: but not for C. For C, if one is using data stored in row order, and one wants
100: to preserve the image index convention, the data must be reordered when
101: reading into memory.
102: 
103: In the end, which you do for Fortran or C depends on
104: which is more important, not reordering data or preserving the indexing
105: convention. For large images, reordering data is potentially expensive, and
106: often the indexing convention is inverted to avoid that.
107: 
108: The situation with
109: numpy makes this issue yet more complicated. The internal machinery of numpy
110: arrays is flexible enough to accept any ordering of indices. One can simply
111: reorder indices by manipulating the internal stride information for arrays
112: without reordering the data at all. Numpy will know how to map the new index
113: order to the data without moving the data.
114: 
115: So if this is true, why not choose
116: the index order that matches what you most expect? In particular, why not define
117: row-ordered images to use the image convention? (This is sometimes referred
118: to as the Fortran convention vs the C convention, thus the 'C' and 'FORTRAN'
119: order options for array ordering in numpy.) The drawback of doing this is
120: potential performance penalties. It's common to access the data sequentially,
121: either implicitly in array operations or explicitly by looping over rows of an
122: image. When that is done, then the data will be accessed in non-optimal order.
123: As the first index is incremented, what is actually happening is that elements
124: spaced far apart in memory are being sequentially accessed, with usually poor
125: memory access speeds. For example, for a two dimensional image 'im' defined so
126: that im[0, 10] represents the value at x=0, y=10. To be consistent with usual
127: Python behavior then im[0] would represent a column at x=0. Yet that data
128: would be spread over the whole array since the data are stored in row order.
129: Despite the flexibility of numpy's indexing, it can't really paper over the fact
130: basic operations are rendered inefficient because of data order or that getting
131: contiguous subarrays is still awkward (e.g., im[:,0] for the first row, vs
132: im[0]), thus one can't use an idiom such as for row in im; for col in im does
133: work, but doesn't yield contiguous column data.
134: 
135: As it turns out, numpy is
136: smart enough when dealing with ufuncs to determine which index is the most
137: rapidly varying one in memory and uses that for the innermost loop. Thus for
138: ufuncs there is no large intrinsic advantage to either approach in most cases.
139: On the other hand, use of .flat with an FORTRAN ordered array will lead to
140: non-optimal memory access as adjacent elements in the flattened array (iterator,
141: actually) are not contiguous in memory.
142: 
143: Indeed, the fact is that Python
144: indexing on lists and other sequences naturally leads to an outside-to inside
145: ordering (the first index gets the largest grouping, the next the next largest,
146: and the last gets the smallest element). Since image data are normally stored
147: by rows, this corresponds to position within rows being the last item indexed.
148: 
149: If you do want to use Fortran ordering realize that
150: there are two approaches to consider: 1) accept that the first index is just not
151: the most rapidly changing in memory and have all your I/O routines reorder
152: your data when going from memory to disk or visa versa, or use numpy's
153: mechanism for mapping the first index to the most rapidly varying data. We
154: recommend the former if possible. The disadvantage of the latter is that many
155: of numpy's functions will yield arrays without Fortran ordering unless you are
156: careful to use the 'order' keyword. Doing this would be highly inconvenient.
157: 
158: Otherwise we recommend simply learning to reverse the usual order of indices
159: when accessing elements of an array. Granted, it goes against the grain, but
160: it is more in line with Python semantics and the natural order of the data.
161: 
162: '''
163: from __future__ import division, absolute_import, print_function
164: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', '\n===============\nArray Internals\n===============\n\nInternal organization of numpy arrays\n=====================================\n\nIt helps to understand a bit about how numpy arrays are handled under the covers to help understand numpy better. This section will not go into great detail. Those wishing to understand the full details are referred to Travis Oliphant\'s book "Guide to Numpy".\n\nNumpy arrays consist of two major components, the raw array data (from now on,\nreferred to as the data buffer), and the information about the raw array data.\nThe data buffer is typically what people think of as arrays in C or Fortran,\na contiguous (and fixed) block of memory containing fixed sized data items.\nNumpy also contains a significant set of data that describes how to interpret\nthe data in the data buffer. This extra information contains (among other things):\n\n 1) The basic data element\'s size in bytes\n 2) The start of the data within the data buffer (an offset relative to the\n    beginning of the data buffer).\n 3) The number of dimensions and the size of each dimension\n 4) The separation between elements for each dimension (the \'stride\'). This\n    does not have to be a multiple of the element size\n 5) The byte order of the data (which may not be the native byte order)\n 6) Whether the buffer is read-only\n 7) Information (via the dtype object) about the interpretation of the basic\n    data element. The basic data element may be as simple as a int or a float,\n    or it may be a compound object (e.g., struct-like), a fixed character field,\n    or Python object pointers.\n 8) Whether the array is to interpreted as C-order or Fortran-order.\n\nThis arrangement allow for very flexible use of arrays. One thing that it allows\nis simple changes of the metadata to change the interpretation of the array buffer.\nChanging the byteorder of the array is a simple change involving no rearrangement\nof the data. The shape of the array can be changed very easily without changing\nanything in the data buffer or any data copying at all\n\nAmong other things that are made possible is one can create a new array metadata\nobject that uses the same data buffer\nto create a new view of that data buffer that has a different interpretation\nof the buffer (e.g., different shape, offset, byte order, strides, etc) but\nshares the same data bytes. Many operations in numpy do just this such as\nslices. Other operations, such as transpose, don\'t move data elements\naround in the array, but rather change the information about the shape and strides so that the indexing of the array changes, but the data in the doesn\'t move.\n\nTypically these new versions of the array metadata but the same data buffer are\nnew \'views\' into the data buffer. There is a different ndarray object, but it\nuses the same data buffer. This is why it is necessary to force copies through\nuse of the .copy() method if one really wants to make a new and independent\ncopy of the data buffer.\n\nNew views into arrays mean the the object reference counts for the data buffer\nincrease. Simply doing away with the original array object will not remove the\ndata buffer if other views of it still exist.\n\nMultidimensional Array Indexing Order Issues\n============================================\n\nWhat is the right way to index\nmulti-dimensional arrays? Before you jump to conclusions about the one and\ntrue way to index multi-dimensional arrays, it pays to understand why this is\na confusing issue. This section will try to explain in detail how numpy\nindexing works and why we adopt the convention we do for images, and when it\nmay be appropriate to adopt other conventions.\n\nThe first thing to understand is\nthat there are two conflicting conventions for indexing 2-dimensional arrays.\nMatrix notation uses the first index to indicate which row is being selected and\nthe second index to indicate which column is selected. This is opposite the\ngeometrically oriented-convention for images where people generally think the\nfirst index represents x position (i.e., column) and the second represents y\nposition (i.e., row). This alone is the source of much confusion;\nmatrix-oriented users and image-oriented users expect two different things with\nregard to indexing.\n\nThe second issue to understand is how indices correspond\nto the order the array is stored in memory. In Fortran the first index is the\nmost rapidly varying index when moving through the elements of a two\ndimensional array as it is stored in memory. If you adopt the matrix\nconvention for indexing, then this means the matrix is stored one column at a\ntime (since the first index moves to the next row as it changes). Thus Fortran\nis considered a Column-major language. C has just the opposite convention. In\nC, the last index changes most rapidly as one moves through the array as\nstored in memory. Thus C is a Row-major language. The matrix is stored by\nrows. Note that in both cases it presumes that the matrix convention for\nindexing is being used, i.e., for both Fortran and C, the first index is the\nrow. Note this convention implies that the indexing convention is invariant\nand that the data order changes to keep that so.\n\nBut that\'s not the only way\nto look at it. Suppose one has large two-dimensional arrays (images or\nmatrices) stored in data files. Suppose the data are stored by rows rather than\nby columns. If we are to preserve our index convention (whether matrix or\nimage) that means that depending on the language we use, we may be forced to\nreorder the data if it is read into memory to preserve our indexing\nconvention. For example if we read row-ordered data into memory without\nreordering, it will match the matrix indexing convention for C, but not for\nFortran. Conversely, it will match the image indexing convention for Fortran,\nbut not for C. For C, if one is using data stored in row order, and one wants\nto preserve the image index convention, the data must be reordered when\nreading into memory.\n\nIn the end, which you do for Fortran or C depends on\nwhich is more important, not reordering data or preserving the indexing\nconvention. For large images, reordering data is potentially expensive, and\noften the indexing convention is inverted to avoid that.\n\nThe situation with\nnumpy makes this issue yet more complicated. The internal machinery of numpy\narrays is flexible enough to accept any ordering of indices. One can simply\nreorder indices by manipulating the internal stride information for arrays\nwithout reordering the data at all. Numpy will know how to map the new index\norder to the data without moving the data.\n\nSo if this is true, why not choose\nthe index order that matches what you most expect? In particular, why not define\nrow-ordered images to use the image convention? (This is sometimes referred\nto as the Fortran convention vs the C convention, thus the \'C\' and \'FORTRAN\'\norder options for array ordering in numpy.) The drawback of doing this is\npotential performance penalties. It\'s common to access the data sequentially,\neither implicitly in array operations or explicitly by looping over rows of an\nimage. When that is done, then the data will be accessed in non-optimal order.\nAs the first index is incremented, what is actually happening is that elements\nspaced far apart in memory are being sequentially accessed, with usually poor\nmemory access speeds. For example, for a two dimensional image \'im\' defined so\nthat im[0, 10] represents the value at x=0, y=10. To be consistent with usual\nPython behavior then im[0] would represent a column at x=0. Yet that data\nwould be spread over the whole array since the data are stored in row order.\nDespite the flexibility of numpy\'s indexing, it can\'t really paper over the fact\nbasic operations are rendered inefficient because of data order or that getting\ncontiguous subarrays is still awkward (e.g., im[:,0] for the first row, vs\nim[0]), thus one can\'t use an idiom such as for row in im; for col in im does\nwork, but doesn\'t yield contiguous column data.\n\nAs it turns out, numpy is\nsmart enough when dealing with ufuncs to determine which index is the most\nrapidly varying one in memory and uses that for the innermost loop. Thus for\nufuncs there is no large intrinsic advantage to either approach in most cases.\nOn the other hand, use of .flat with an FORTRAN ordered array will lead to\nnon-optimal memory access as adjacent elements in the flattened array (iterator,\nactually) are not contiguous in memory.\n\nIndeed, the fact is that Python\nindexing on lists and other sequences naturally leads to an outside-to inside\nordering (the first index gets the largest grouping, the next the next largest,\nand the last gets the smallest element). Since image data are normally stored\nby rows, this corresponds to position within rows being the last item indexed.\n\nIf you do want to use Fortran ordering realize that\nthere are two approaches to consider: 1) accept that the first index is just not\nthe most rapidly changing in memory and have all your I/O routines reorder\nyour data when going from memory to disk or visa versa, or use numpy\'s\nmechanism for mapping the first index to the most rapidly varying data. We\nrecommend the former if possible. The disadvantage of the latter is that many\nof numpy\'s functions will yield arrays without Fortran ordering unless you are\ncareful to use the \'order\' keyword. Doing this would be highly inconvenient.\n\nOtherwise we recommend simply learning to reverse the usual order of indices\nwhen accessing elements of an array. Granted, it goes against the grain, but\nit is more in line with Python semantics and the natural order of the data.\n\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

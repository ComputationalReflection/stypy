
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ========================
3: Broadcasting over arrays
4: ========================
5: 
6: The term broadcasting describes how numpy treats arrays with different
7: shapes during arithmetic operations. Subject to certain constraints,
8: the smaller array is "broadcast" across the larger array so that they
9: have compatible shapes. Broadcasting provides a means of vectorizing
10: array operations so that looping occurs in C instead of Python. It does
11: this without making needless copies of data and usually leads to
12: efficient algorithm implementations. There are, however, cases where
13: broadcasting is a bad idea because it leads to inefficient use of memory
14: that slows computation.
15: 
16: NumPy operations are usually done on pairs of arrays on an
17: element-by-element basis.  In the simplest case, the two arrays must
18: have exactly the same shape, as in the following example:
19: 
20:   >>> a = np.array([1.0, 2.0, 3.0])
21:   >>> b = np.array([2.0, 2.0, 2.0])
22:   >>> a * b
23:   array([ 2.,  4.,  6.])
24: 
25: NumPy's broadcasting rule relaxes this constraint when the arrays'
26: shapes meet certain constraints. The simplest broadcasting example occurs
27: when an array and a scalar value are combined in an operation:
28: 
29: >>> a = np.array([1.0, 2.0, 3.0])
30: >>> b = 2.0
31: >>> a * b
32: array([ 2.,  4.,  6.])
33: 
34: The result is equivalent to the previous example where ``b`` was an array.
35: We can think of the scalar ``b`` being *stretched* during the arithmetic
36: operation into an array with the same shape as ``a``. The new elements in
37: ``b`` are simply copies of the original scalar. The stretching analogy is
38: only conceptual.  NumPy is smart enough to use the original scalar value
39: without actually making copies, so that broadcasting operations are as
40: memory and computationally efficient as possible.
41: 
42: The code in the second example is more efficient than that in the first
43: because broadcasting moves less memory around during the multiplication
44: (``b`` is a scalar rather than an array).
45: 
46: General Broadcasting Rules
47: ==========================
48: When operating on two arrays, NumPy compares their shapes element-wise.
49: It starts with the trailing dimensions, and works its way forward.  Two
50: dimensions are compatible when
51: 
52: 1) they are equal, or
53: 2) one of them is 1
54: 
55: If these conditions are not met, a
56: ``ValueError: frames are not aligned`` exception is thrown, indicating that
57: the arrays have incompatible shapes. The size of the resulting array
58: is the maximum size along each dimension of the input arrays.
59: 
60: Arrays do not need to have the same *number* of dimensions.  For example,
61: if you have a ``256x256x3`` array of RGB values, and you want to scale
62: each color in the image by a different value, you can multiply the image
63: by a one-dimensional array with 3 values. Lining up the sizes of the
64: trailing axes of these arrays according to the broadcast rules, shows that
65: they are compatible::
66: 
67:   Image  (3d array): 256 x 256 x 3
68:   Scale  (1d array):             3
69:   Result (3d array): 256 x 256 x 3
70: 
71: When either of the dimensions compared is one, the other is
72: used.  In other words, dimensions with size 1 are stretched or "copied"
73: to match the other.
74: 
75: In the following example, both the ``A`` and ``B`` arrays have axes with
76: length one that are expanded to a larger size during the broadcast
77: operation::
78: 
79:   A      (4d array):  8 x 1 x 6 x 1
80:   B      (3d array):      7 x 1 x 5
81:   Result (4d array):  8 x 7 x 6 x 5
82: 
83: Here are some more examples::
84: 
85:   A      (2d array):  5 x 4
86:   B      (1d array):      1
87:   Result (2d array):  5 x 4
88: 
89:   A      (2d array):  5 x 4
90:   B      (1d array):      4
91:   Result (2d array):  5 x 4
92: 
93:   A      (3d array):  15 x 3 x 5
94:   B      (3d array):  15 x 1 x 5
95:   Result (3d array):  15 x 3 x 5
96: 
97:   A      (3d array):  15 x 3 x 5
98:   B      (2d array):       3 x 5
99:   Result (3d array):  15 x 3 x 5
100: 
101:   A      (3d array):  15 x 3 x 5
102:   B      (2d array):       3 x 1
103:   Result (3d array):  15 x 3 x 5
104: 
105: Here are examples of shapes that do not broadcast::
106: 
107:   A      (1d array):  3
108:   B      (1d array):  4 # trailing dimensions do not match
109: 
110:   A      (2d array):      2 x 1
111:   B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched
112: 
113: An example of broadcasting in practice::
114: 
115:  >>> x = np.arange(4)
116:  >>> xx = x.reshape(4,1)
117:  >>> y = np.ones(5)
118:  >>> z = np.ones((3,4))
119: 
120:  >>> x.shape
121:  (4,)
122: 
123:  >>> y.shape
124:  (5,)
125: 
126:  >>> x + y
127:  <type 'exceptions.ValueError'>: shape mismatch: objects cannot be broadcast to a single shape
128: 
129:  >>> xx.shape
130:  (4, 1)
131: 
132:  >>> y.shape
133:  (5,)
134: 
135:  >>> (xx + y).shape
136:  (4, 5)
137: 
138:  >>> xx + y
139:  array([[ 1.,  1.,  1.,  1.,  1.],
140:         [ 2.,  2.,  2.,  2.,  2.],
141:         [ 3.,  3.,  3.,  3.,  3.],
142:         [ 4.,  4.,  4.,  4.,  4.]])
143: 
144:  >>> x.shape
145:  (4,)
146: 
147:  >>> z.shape
148:  (3, 4)
149: 
150:  >>> (x + z).shape
151:  (3, 4)
152: 
153:  >>> x + z
154:  array([[ 1.,  2.,  3.,  4.],
155:         [ 1.,  2.,  3.,  4.],
156:         [ 1.,  2.,  3.,  4.]])
157: 
158: Broadcasting provides a convenient way of taking the outer product (or
159: any other outer operation) of two arrays. The following example shows an
160: outer addition operation of two 1-d arrays::
161: 
162:   >>> a = np.array([0.0, 10.0, 20.0, 30.0])
163:   >>> b = np.array([1.0, 2.0, 3.0])
164:   >>> a[:, np.newaxis] + b
165:   array([[  1.,   2.,   3.],
166:          [ 11.,  12.,  13.],
167:          [ 21.,  22.,  23.],
168:          [ 31.,  32.,  33.]])
169: 
170: Here the ``newaxis`` index operator inserts a new axis into ``a``,
171: making it a two-dimensional ``4x1`` array.  Combining the ``4x1`` array
172: with ``b``, which has shape ``(3,)``, yields a ``4x3`` array.
173: 
174: See `this article <http://wiki.scipy.org/EricsBroadcastingDoc>`_
175: for illustrations of broadcasting concepts.
176: 
177: '''
178: from __future__ import division, absolute_import, print_function
179: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, (-1)), 'str', '\n========================\nBroadcasting over arrays\n========================\n\nThe term broadcasting describes how numpy treats arrays with different\nshapes during arithmetic operations. Subject to certain constraints,\nthe smaller array is "broadcast" across the larger array so that they\nhave compatible shapes. Broadcasting provides a means of vectorizing\narray operations so that looping occurs in C instead of Python. It does\nthis without making needless copies of data and usually leads to\nefficient algorithm implementations. There are, however, cases where\nbroadcasting is a bad idea because it leads to inefficient use of memory\nthat slows computation.\n\nNumPy operations are usually done on pairs of arrays on an\nelement-by-element basis.  In the simplest case, the two arrays must\nhave exactly the same shape, as in the following example:\n\n  >>> a = np.array([1.0, 2.0, 3.0])\n  >>> b = np.array([2.0, 2.0, 2.0])\n  >>> a * b\n  array([ 2.,  4.,  6.])\n\nNumPy\'s broadcasting rule relaxes this constraint when the arrays\'\nshapes meet certain constraints. The simplest broadcasting example occurs\nwhen an array and a scalar value are combined in an operation:\n\n>>> a = np.array([1.0, 2.0, 3.0])\n>>> b = 2.0\n>>> a * b\narray([ 2.,  4.,  6.])\n\nThe result is equivalent to the previous example where ``b`` was an array.\nWe can think of the scalar ``b`` being *stretched* during the arithmetic\noperation into an array with the same shape as ``a``. The new elements in\n``b`` are simply copies of the original scalar. The stretching analogy is\nonly conceptual.  NumPy is smart enough to use the original scalar value\nwithout actually making copies, so that broadcasting operations are as\nmemory and computationally efficient as possible.\n\nThe code in the second example is more efficient than that in the first\nbecause broadcasting moves less memory around during the multiplication\n(``b`` is a scalar rather than an array).\n\nGeneral Broadcasting Rules\n==========================\nWhen operating on two arrays, NumPy compares their shapes element-wise.\nIt starts with the trailing dimensions, and works its way forward.  Two\ndimensions are compatible when\n\n1) they are equal, or\n2) one of them is 1\n\nIf these conditions are not met, a\n``ValueError: frames are not aligned`` exception is thrown, indicating that\nthe arrays have incompatible shapes. The size of the resulting array\nis the maximum size along each dimension of the input arrays.\n\nArrays do not need to have the same *number* of dimensions.  For example,\nif you have a ``256x256x3`` array of RGB values, and you want to scale\neach color in the image by a different value, you can multiply the image\nby a one-dimensional array with 3 values. Lining up the sizes of the\ntrailing axes of these arrays according to the broadcast rules, shows that\nthey are compatible::\n\n  Image  (3d array): 256 x 256 x 3\n  Scale  (1d array):             3\n  Result (3d array): 256 x 256 x 3\n\nWhen either of the dimensions compared is one, the other is\nused.  In other words, dimensions with size 1 are stretched or "copied"\nto match the other.\n\nIn the following example, both the ``A`` and ``B`` arrays have axes with\nlength one that are expanded to a larger size during the broadcast\noperation::\n\n  A      (4d array):  8 x 1 x 6 x 1\n  B      (3d array):      7 x 1 x 5\n  Result (4d array):  8 x 7 x 6 x 5\n\nHere are some more examples::\n\n  A      (2d array):  5 x 4\n  B      (1d array):      1\n  Result (2d array):  5 x 4\n\n  A      (2d array):  5 x 4\n  B      (1d array):      4\n  Result (2d array):  5 x 4\n\n  A      (3d array):  15 x 3 x 5\n  B      (3d array):  15 x 1 x 5\n  Result (3d array):  15 x 3 x 5\n\n  A      (3d array):  15 x 3 x 5\n  B      (2d array):       3 x 5\n  Result (3d array):  15 x 3 x 5\n\n  A      (3d array):  15 x 3 x 5\n  B      (2d array):       3 x 1\n  Result (3d array):  15 x 3 x 5\n\nHere are examples of shapes that do not broadcast::\n\n  A      (1d array):  3\n  B      (1d array):  4 # trailing dimensions do not match\n\n  A      (2d array):      2 x 1\n  B      (3d array):  8 x 4 x 3 # second from last dimensions mismatched\n\nAn example of broadcasting in practice::\n\n >>> x = np.arange(4)\n >>> xx = x.reshape(4,1)\n >>> y = np.ones(5)\n >>> z = np.ones((3,4))\n\n >>> x.shape\n (4,)\n\n >>> y.shape\n (5,)\n\n >>> x + y\n <type \'exceptions.ValueError\'>: shape mismatch: objects cannot be broadcast to a single shape\n\n >>> xx.shape\n (4, 1)\n\n >>> y.shape\n (5,)\n\n >>> (xx + y).shape\n (4, 5)\n\n >>> xx + y\n array([[ 1.,  1.,  1.,  1.,  1.],\n        [ 2.,  2.,  2.,  2.,  2.],\n        [ 3.,  3.,  3.,  3.,  3.],\n        [ 4.,  4.,  4.,  4.,  4.]])\n\n >>> x.shape\n (4,)\n\n >>> z.shape\n (3, 4)\n\n >>> (x + z).shape\n (3, 4)\n\n >>> x + z\n array([[ 1.,  2.,  3.,  4.],\n        [ 1.,  2.,  3.,  4.],\n        [ 1.,  2.,  3.,  4.]])\n\nBroadcasting provides a convenient way of taking the outer product (or\nany other outer operation) of two arrays. The following example shows an\nouter addition operation of two 1-d arrays::\n\n  >>> a = np.array([0.0, 10.0, 20.0, 30.0])\n  >>> b = np.array([1.0, 2.0, 3.0])\n  >>> a[:, np.newaxis] + b\n  array([[  1.,   2.,   3.],\n         [ 11.,  12.,  13.],\n         [ 21.,  22.,  23.],\n         [ 31.,  32.,  33.]])\n\nHere the ``newaxis`` index operator inserts a new axis into ``a``,\nmaking it a two-dimensional ``4x1`` array.  Combining the ``4x1`` array\nwith ``b``, which has shape ``(3,)``, yields a ``4x3`` array.\n\nSee `this article <http://wiki.scipy.org/EricsBroadcastingDoc>`_\nfor illustrations of broadcasting concepts.\n\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

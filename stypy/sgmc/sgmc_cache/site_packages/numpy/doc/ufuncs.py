
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ===================
3: Universal Functions
4: ===================
5: 
6: Ufuncs are, generally speaking, mathematical functions or operations that are
7: applied element-by-element to the contents of an array. That is, the result
8: in each output array element only depends on the value in the corresponding
9: input array (or arrays) and on no other array elements. Numpy comes with a
10: large suite of ufuncs, and scipy extends that suite substantially. The simplest
11: example is the addition operator: ::
12: 
13:  >>> np.array([0,2,3,4]) + np.array([1,1,-1,2])
14:  array([1, 3, 2, 6])
15: 
16: The unfunc module lists all the available ufuncs in numpy. Documentation on
17: the specific ufuncs may be found in those modules. This documentation is
18: intended to address the more general aspects of unfuncs common to most of
19: them. All of the ufuncs that make use of Python operators (e.g., +, -, etc.)
20: have equivalent functions defined (e.g. add() for +)
21: 
22: Type coercion
23: =============
24: 
25: What happens when a binary operator (e.g., +,-,\\*,/, etc) deals with arrays of
26: two different types? What is the type of the result? Typically, the result is
27: the higher of the two types. For example: ::
28: 
29:  float32 + float64 -> float64
30:  int8 + int32 -> int32
31:  int16 + float32 -> float32
32:  float32 + complex64 -> complex64
33: 
34: There are some less obvious cases generally involving mixes of types
35: (e.g. uints, ints and floats) where equal bit sizes for each are not
36: capable of saving all the information in a different type of equivalent
37: bit size. Some examples are int32 vs float32 or uint32 vs int32.
38: Generally, the result is the higher type of larger size than both
39: (if available). So: ::
40: 
41:  int32 + float32 -> float64
42:  uint32 + int32 -> int64
43: 
44: Finally, the type coercion behavior when expressions involve Python
45: scalars is different than that seen for arrays. Since Python has a
46: limited number of types, combining a Python int with a dtype=np.int8
47: array does not coerce to the higher type but instead, the type of the
48: array prevails. So the rules for Python scalars combined with arrays is
49: that the result will be that of the array equivalent the Python scalar
50: if the Python scalar is of a higher 'kind' than the array (e.g., float
51: vs. int), otherwise the resultant type will be that of the array.
52: For example: ::
53: 
54:   Python int + int8 -> int8
55:   Python float + int8 -> float64
56: 
57: ufunc methods
58: =============
59: 
60: Binary ufuncs support 4 methods.
61: 
62: **.reduce(arr)** applies the binary operator to elements of the array in
63:   sequence. For example: ::
64: 
65:  >>> np.add.reduce(np.arange(10))  # adds all elements of array
66:  45
67: 
68: For multidimensional arrays, the first dimension is reduced by default: ::
69: 
70:  >>> np.add.reduce(np.arange(10).reshape(2,5))
71:      array([ 5,  7,  9, 11, 13])
72: 
73: The axis keyword can be used to specify different axes to reduce: ::
74: 
75:  >>> np.add.reduce(np.arange(10).reshape(2,5),axis=1)
76:  array([10, 35])
77: 
78: **.accumulate(arr)** applies the binary operator and generates an an
79: equivalently shaped array that includes the accumulated amount for each
80: element of the array. A couple examples: ::
81: 
82:  >>> np.add.accumulate(np.arange(10))
83:  array([ 0,  1,  3,  6, 10, 15, 21, 28, 36, 45])
84:  >>> np.multiply.accumulate(np.arange(1,9))
85:  array([    1,     2,     6,    24,   120,   720,  5040, 40320])
86: 
87: The behavior for multidimensional arrays is the same as for .reduce(),
88: as is the use of the axis keyword).
89: 
90: **.reduceat(arr,indices)** allows one to apply reduce to selected parts
91:   of an array. It is a difficult method to understand. See the documentation
92:   at:
93: 
94: **.outer(arr1,arr2)** generates an outer operation on the two arrays arr1 and
95:   arr2. It will work on multidimensional arrays (the shape of the result is
96:   the concatenation of the two input shapes.: ::
97: 
98:  >>> np.multiply.outer(np.arange(3),np.arange(4))
99:  array([[0, 0, 0, 0],
100:         [0, 1, 2, 3],
101:         [0, 2, 4, 6]])
102: 
103: Output arguments
104: ================
105: 
106: All ufuncs accept an optional output array. The array must be of the expected
107: output shape. Beware that if the type of the output array is of a different
108: (and lower) type than the output result, the results may be silently truncated
109: or otherwise corrupted in the downcast to the lower type. This usage is useful
110: when one wants to avoid creating large temporary arrays and instead allows one
111: to reuse the same array memory repeatedly (at the expense of not being able to
112: use more convenient operator notation in expressions). Note that when the
113: output argument is used, the ufunc still returns a reference to the result.
114: 
115:  >>> x = np.arange(2)
116:  >>> np.add(np.arange(2),np.arange(2.),x)
117:  array([0, 2])
118:  >>> x
119:  array([0, 2])
120: 
121: and & or as ufuncs
122: ==================
123: 
124: Invariably people try to use the python 'and' and 'or' as logical operators
125: (and quite understandably). But these operators do not behave as normal
126: operators since Python treats these quite differently. They cannot be
127: overloaded with array equivalents. Thus using 'and' or 'or' with an array
128: results in an error. There are two alternatives:
129: 
130:  1) use the ufunc functions logical_and() and logical_or().
131:  2) use the bitwise operators & and \\|. The drawback of these is that if
132:     the arguments to these operators are not boolean arrays, the result is
133:     likely incorrect. On the other hand, most usages of logical_and and
134:     logical_or are with boolean arrays. As long as one is careful, this is
135:     a convenient way to apply these operators.
136: 
137: '''
138: from __future__ import division, absolute_import, print_function
139: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, (-1)), 'str', "\n===================\nUniversal Functions\n===================\n\nUfuncs are, generally speaking, mathematical functions or operations that are\napplied element-by-element to the contents of an array. That is, the result\nin each output array element only depends on the value in the corresponding\ninput array (or arrays) and on no other array elements. Numpy comes with a\nlarge suite of ufuncs, and scipy extends that suite substantially. The simplest\nexample is the addition operator: ::\n\n >>> np.array([0,2,3,4]) + np.array([1,1,-1,2])\n array([1, 3, 2, 6])\n\nThe unfunc module lists all the available ufuncs in numpy. Documentation on\nthe specific ufuncs may be found in those modules. This documentation is\nintended to address the more general aspects of unfuncs common to most of\nthem. All of the ufuncs that make use of Python operators (e.g., +, -, etc.)\nhave equivalent functions defined (e.g. add() for +)\n\nType coercion\n=============\n\nWhat happens when a binary operator (e.g., +,-,\\*,/, etc) deals with arrays of\ntwo different types? What is the type of the result? Typically, the result is\nthe higher of the two types. For example: ::\n\n float32 + float64 -> float64\n int8 + int32 -> int32\n int16 + float32 -> float32\n float32 + complex64 -> complex64\n\nThere are some less obvious cases generally involving mixes of types\n(e.g. uints, ints and floats) where equal bit sizes for each are not\ncapable of saving all the information in a different type of equivalent\nbit size. Some examples are int32 vs float32 or uint32 vs int32.\nGenerally, the result is the higher type of larger size than both\n(if available). So: ::\n\n int32 + float32 -> float64\n uint32 + int32 -> int64\n\nFinally, the type coercion behavior when expressions involve Python\nscalars is different than that seen for arrays. Since Python has a\nlimited number of types, combining a Python int with a dtype=np.int8\narray does not coerce to the higher type but instead, the type of the\narray prevails. So the rules for Python scalars combined with arrays is\nthat the result will be that of the array equivalent the Python scalar\nif the Python scalar is of a higher 'kind' than the array (e.g., float\nvs. int), otherwise the resultant type will be that of the array.\nFor example: ::\n\n  Python int + int8 -> int8\n  Python float + int8 -> float64\n\nufunc methods\n=============\n\nBinary ufuncs support 4 methods.\n\n**.reduce(arr)** applies the binary operator to elements of the array in\n  sequence. For example: ::\n\n >>> np.add.reduce(np.arange(10))  # adds all elements of array\n 45\n\nFor multidimensional arrays, the first dimension is reduced by default: ::\n\n >>> np.add.reduce(np.arange(10).reshape(2,5))\n     array([ 5,  7,  9, 11, 13])\n\nThe axis keyword can be used to specify different axes to reduce: ::\n\n >>> np.add.reduce(np.arange(10).reshape(2,5),axis=1)\n array([10, 35])\n\n**.accumulate(arr)** applies the binary operator and generates an an\nequivalently shaped array that includes the accumulated amount for each\nelement of the array. A couple examples: ::\n\n >>> np.add.accumulate(np.arange(10))\n array([ 0,  1,  3,  6, 10, 15, 21, 28, 36, 45])\n >>> np.multiply.accumulate(np.arange(1,9))\n array([    1,     2,     6,    24,   120,   720,  5040, 40320])\n\nThe behavior for multidimensional arrays is the same as for .reduce(),\nas is the use of the axis keyword).\n\n**.reduceat(arr,indices)** allows one to apply reduce to selected parts\n  of an array. It is a difficult method to understand. See the documentation\n  at:\n\n**.outer(arr1,arr2)** generates an outer operation on the two arrays arr1 and\n  arr2. It will work on multidimensional arrays (the shape of the result is\n  the concatenation of the two input shapes.: ::\n\n >>> np.multiply.outer(np.arange(3),np.arange(4))\n array([[0, 0, 0, 0],\n        [0, 1, 2, 3],\n        [0, 2, 4, 6]])\n\nOutput arguments\n================\n\nAll ufuncs accept an optional output array. The array must be of the expected\noutput shape. Beware that if the type of the output array is of a different\n(and lower) type than the output result, the results may be silently truncated\nor otherwise corrupted in the downcast to the lower type. This usage is useful\nwhen one wants to avoid creating large temporary arrays and instead allows one\nto reuse the same array memory repeatedly (at the expense of not being able to\nuse more convenient operator notation in expressions). Note that when the\noutput argument is used, the ufunc still returns a reference to the result.\n\n >>> x = np.arange(2)\n >>> np.add(np.arange(2),np.arange(2.),x)\n array([0, 2])\n >>> x\n array([0, 2])\n\nand & or as ufuncs\n==================\n\nInvariably people try to use the python 'and' and 'or' as logical operators\n(and quite understandably). But these operators do not behave as normal\noperators since Python treats these quite differently. They cannot be\noverloaded with array equivalents. Thus using 'and' or 'or' with an array\nresults in an error. There are two alternatives:\n\n 1) use the ufunc functions logical_and() and logical_or().\n 2) use the bitwise operators & and \\|. The drawback of these is that if\n    the arguments to these operators are not boolean arrays, the result is\n    likely incorrect. On the other hand, most usages of logical_and and\n    logical_or are with boolean arrays. As long as one is careful, this is\n    a convenient way to apply these operators.\n\n")

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

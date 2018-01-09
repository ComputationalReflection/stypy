
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ==============
3: Array Creation
4: ==============
5: 
6: Introduction
7: ============
8: 
9: There are 5 general mechanisms for creating arrays:
10: 
11: 1) Conversion from other Python structures (e.g., lists, tuples)
12: 2) Intrinsic numpy array array creation objects (e.g., arange, ones, zeros,
13:    etc.)
14: 3) Reading arrays from disk, either from standard or custom formats
15: 4) Creating arrays from raw bytes through the use of strings or buffers
16: 5) Use of special library functions (e.g., random)
17: 
18: This section will not cover means of replicating, joining, or otherwise
19: expanding or mutating existing arrays. Nor will it cover creating object
20: arrays or structured arrays. Both of those are covered in their own sections.
21: 
22: Converting Python array_like Objects to Numpy Arrays
23: ====================================================
24: 
25: In general, numerical data arranged in an array-like structure in Python can
26: be converted to arrays through the use of the array() function. The most
27: obvious examples are lists and tuples. See the documentation for array() for
28: details for its use. Some objects may support the array-protocol and allow
29: conversion to arrays this way. A simple way to find out if the object can be
30: converted to a numpy array using array() is simply to try it interactively and
31: see if it works! (The Python Way).
32: 
33: Examples: ::
34: 
35:  >>> x = np.array([2,3,1,0])
36:  >>> x = np.array([2, 3, 1, 0])
37:  >>> x = np.array([[1,2.0],[0,0],(1+1j,3.)]) # note mix of tuple and lists,
38:      and types
39:  >>> x = np.array([[ 1.+0.j, 2.+0.j], [ 0.+0.j, 0.+0.j], [ 1.+1.j, 3.+0.j]])
40: 
41: Intrinsic Numpy Array Creation
42: ==============================
43: 
44: Numpy has built-in functions for creating arrays from scratch:
45: 
46: zeros(shape) will create an array filled with 0 values with the specified
47: shape. The default dtype is float64.
48: 
49: ``>>> np.zeros((2, 3))
50: array([[ 0., 0., 0.], [ 0., 0., 0.]])``
51: 
52: ones(shape) will create an array filled with 1 values. It is identical to
53: zeros in all other respects.
54: 
55: arange() will create arrays with regularly incrementing values. Check the
56: docstring for complete information on the various ways it can be used. A few
57: examples will be given here: ::
58: 
59:  >>> np.arange(10)
60:  array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
61:  >>> np.arange(2, 10, dtype=np.float)
62:  array([ 2., 3., 4., 5., 6., 7., 8., 9.])
63:  >>> np.arange(2, 3, 0.1)
64:  array([ 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])
65: 
66: Note that there are some subtleties regarding the last usage that the user
67: should be aware of that are described in the arange docstring.
68: 
69: linspace() will create arrays with a specified number of elements, and
70: spaced equally between the specified beginning and end values. For
71: example: ::
72: 
73:  >>> np.linspace(1., 4., 6)
74:  array([ 1. ,  1.6,  2.2,  2.8,  3.4,  4. ])
75: 
76: The advantage of this creation function is that one can guarantee the
77: number of elements and the starting and end point, which arange()
78: generally will not do for arbitrary start, stop, and step values.
79: 
80: indices() will create a set of arrays (stacked as a one-higher dimensioned
81: array), one per dimension with each representing variation in that dimension.
82: An example illustrates much better than a verbal description: ::
83: 
84:  >>> np.indices((3,3))
85:  array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
86: 
87: This is particularly useful for evaluating functions of multiple dimensions on
88: a regular grid.
89: 
90: Reading Arrays From Disk
91: ========================
92: 
93: This is presumably the most common case of large array creation. The details,
94: of course, depend greatly on the format of data on disk and so this section
95: can only give general pointers on how to handle various formats.
96: 
97: Standard Binary Formats
98: -----------------------
99: 
100: Various fields have standard formats for array data. The following lists the
101: ones with known python libraries to read them and return numpy arrays (there
102: may be others for which it is possible to read and convert to numpy arrays so
103: check the last section as well)
104: ::
105: 
106:  HDF5: PyTables
107:  FITS: PyFITS
108: 
109: Examples of formats that cannot be read directly but for which it is not hard to
110: convert are those formats supported by libraries like PIL (able to read and
111: write many image formats such as jpg, png, etc).
112: 
113: Common ASCII Formats
114: ------------------------
115: 
116: Comma Separated Value files (CSV) are widely used (and an export and import
117: option for programs like Excel). There are a number of ways of reading these
118: files in Python. There are CSV functions in Python and functions in pylab
119: (part of matplotlib).
120: 
121: More generic ascii files can be read using the io package in scipy.
122: 
123: Custom Binary Formats
124: ---------------------
125: 
126: There are a variety of approaches one can use. If the file has a relatively
127: simple format then one can write a simple I/O library and use the numpy
128: fromfile() function and .tofile() method to read and write numpy arrays
129: directly (mind your byteorder though!) If a good C or C++ library exists that
130: read the data, one can wrap that library with a variety of techniques though
131: that certainly is much more work and requires significantly more advanced
132: knowledge to interface with C or C++.
133: 
134: Use of Special Libraries
135: ------------------------
136: 
137: There are libraries that can be used to generate arrays for special purposes
138: and it isn't possible to enumerate all of them. The most common uses are use
139: of the many array generation functions in random that can generate arrays of
140: random values, and some utility functions to generate special matrices (e.g.
141: diagonal).
142: 
143: '''
144: from __future__ import division, absolute_import, print_function
145: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', "\n==============\nArray Creation\n==============\n\nIntroduction\n============\n\nThere are 5 general mechanisms for creating arrays:\n\n1) Conversion from other Python structures (e.g., lists, tuples)\n2) Intrinsic numpy array array creation objects (e.g., arange, ones, zeros,\n   etc.)\n3) Reading arrays from disk, either from standard or custom formats\n4) Creating arrays from raw bytes through the use of strings or buffers\n5) Use of special library functions (e.g., random)\n\nThis section will not cover means of replicating, joining, or otherwise\nexpanding or mutating existing arrays. Nor will it cover creating object\narrays or structured arrays. Both of those are covered in their own sections.\n\nConverting Python array_like Objects to Numpy Arrays\n====================================================\n\nIn general, numerical data arranged in an array-like structure in Python can\nbe converted to arrays through the use of the array() function. The most\nobvious examples are lists and tuples. See the documentation for array() for\ndetails for its use. Some objects may support the array-protocol and allow\nconversion to arrays this way. A simple way to find out if the object can be\nconverted to a numpy array using array() is simply to try it interactively and\nsee if it works! (The Python Way).\n\nExamples: ::\n\n >>> x = np.array([2,3,1,0])\n >>> x = np.array([2, 3, 1, 0])\n >>> x = np.array([[1,2.0],[0,0],(1+1j,3.)]) # note mix of tuple and lists,\n     and types\n >>> x = np.array([[ 1.+0.j, 2.+0.j], [ 0.+0.j, 0.+0.j], [ 1.+1.j, 3.+0.j]])\n\nIntrinsic Numpy Array Creation\n==============================\n\nNumpy has built-in functions for creating arrays from scratch:\n\nzeros(shape) will create an array filled with 0 values with the specified\nshape. The default dtype is float64.\n\n``>>> np.zeros((2, 3))\narray([[ 0., 0., 0.], [ 0., 0., 0.]])``\n\nones(shape) will create an array filled with 1 values. It is identical to\nzeros in all other respects.\n\narange() will create arrays with regularly incrementing values. Check the\ndocstring for complete information on the various ways it can be used. A few\nexamples will be given here: ::\n\n >>> np.arange(10)\n array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n >>> np.arange(2, 10, dtype=np.float)\n array([ 2., 3., 4., 5., 6., 7., 8., 9.])\n >>> np.arange(2, 3, 0.1)\n array([ 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9])\n\nNote that there are some subtleties regarding the last usage that the user\nshould be aware of that are described in the arange docstring.\n\nlinspace() will create arrays with a specified number of elements, and\nspaced equally between the specified beginning and end values. For\nexample: ::\n\n >>> np.linspace(1., 4., 6)\n array([ 1. ,  1.6,  2.2,  2.8,  3.4,  4. ])\n\nThe advantage of this creation function is that one can guarantee the\nnumber of elements and the starting and end point, which arange()\ngenerally will not do for arbitrary start, stop, and step values.\n\nindices() will create a set of arrays (stacked as a one-higher dimensioned\narray), one per dimension with each representing variation in that dimension.\nAn example illustrates much better than a verbal description: ::\n\n >>> np.indices((3,3))\n array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]])\n\nThis is particularly useful for evaluating functions of multiple dimensions on\na regular grid.\n\nReading Arrays From Disk\n========================\n\nThis is presumably the most common case of large array creation. The details,\nof course, depend greatly on the format of data on disk and so this section\ncan only give general pointers on how to handle various formats.\n\nStandard Binary Formats\n-----------------------\n\nVarious fields have standard formats for array data. The following lists the\nones with known python libraries to read them and return numpy arrays (there\nmay be others for which it is possible to read and convert to numpy arrays so\ncheck the last section as well)\n::\n\n HDF5: PyTables\n FITS: PyFITS\n\nExamples of formats that cannot be read directly but for which it is not hard to\nconvert are those formats supported by libraries like PIL (able to read and\nwrite many image formats such as jpg, png, etc).\n\nCommon ASCII Formats\n------------------------\n\nComma Separated Value files (CSV) are widely used (and an export and import\noption for programs like Excel). There are a number of ways of reading these\nfiles in Python. There are CSV functions in Python and functions in pylab\n(part of matplotlib).\n\nMore generic ascii files can be read using the io package in scipy.\n\nCustom Binary Formats\n---------------------\n\nThere are a variety of approaches one can use. If the file has a relatively\nsimple format then one can write a simple I/O library and use the numpy\nfromfile() function and .tofile() method to read and write numpy arrays\ndirectly (mind your byteorder though!) If a good C or C++ library exists that\nread the data, one can wrap that library with a variety of techniques though\nthat certainly is much more work and requires significantly more advanced\nknowledge to interface with C or C++.\n\nUse of Special Libraries\n------------------------\n\nThere are libraries that can be used to generate arrays for special purposes\nand it isn't possible to enumerate all of them. The most common uses are use\nof the many array generation functions in random that can generate arrays of\nrandom values, and some utility functions to generate special matrices (e.g.\ndiagonal).\n\n")

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

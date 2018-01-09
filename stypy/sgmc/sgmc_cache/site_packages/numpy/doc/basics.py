
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ============
3: Array basics
4: ============
5: 
6: Array types and conversions between types
7: =========================================
8: 
9: Numpy supports a much greater variety of numerical types than Python does.
10: This section shows which are available, and how to modify an array's data-type.
11: 
12: ==========  ==========================================================
13: Data type   Description
14: ==========  ==========================================================
15: bool_       Boolean (True or False) stored as a byte
16: int_        Default integer type (same as C ``long``; normally either
17:             ``int64`` or ``int32``)
18: intc        Identical to C ``int`` (normally ``int32`` or ``int64``)
19: intp        Integer used for indexing (same as C ``ssize_t``; normally
20:             either ``int32`` or ``int64``)
21: int8        Byte (-128 to 127)
22: int16       Integer (-32768 to 32767)
23: int32       Integer (-2147483648 to 2147483647)
24: int64       Integer (-9223372036854775808 to 9223372036854775807)
25: uint8       Unsigned integer (0 to 255)
26: uint16      Unsigned integer (0 to 65535)
27: uint32      Unsigned integer (0 to 4294967295)
28: uint64      Unsigned integer (0 to 18446744073709551615)
29: float_      Shorthand for ``float64``.
30: float16     Half precision float: sign bit, 5 bits exponent,
31:             10 bits mantissa
32: float32     Single precision float: sign bit, 8 bits exponent,
33:             23 bits mantissa
34: float64     Double precision float: sign bit, 11 bits exponent,
35:             52 bits mantissa
36: complex_    Shorthand for ``complex128``.
37: complex64   Complex number, represented by two 32-bit floats (real
38:             and imaginary components)
39: complex128  Complex number, represented by two 64-bit floats (real
40:             and imaginary components)
41: ==========  ==========================================================
42: 
43: Additionally to ``intc`` the platform dependent C integer types ``short``,
44: ``long``, ``longlong`` and their unsigned versions are defined.
45: 
46: Numpy numerical types are instances of ``dtype`` (data-type) objects, each
47: having unique characteristics.  Once you have imported NumPy using
48: 
49:   ::
50: 
51:     >>> import numpy as np
52: 
53: the dtypes are available as ``np.bool_``, ``np.float32``, etc.
54: 
55: Advanced types, not listed in the table above, are explored in
56: section :ref:`structured_arrays`.
57: 
58: There are 5 basic numerical types representing booleans (bool), integers (int),
59: unsigned integers (uint) floating point (float) and complex. Those with numbers
60: in their name indicate the bitsize of the type (i.e. how many bits are needed
61: to represent a single value in memory).  Some types, such as ``int`` and
62: ``intp``, have differing bitsizes, dependent on the platforms (e.g. 32-bit
63: vs. 64-bit machines).  This should be taken into account when interfacing
64: with low-level code (such as C or Fortran) where the raw memory is addressed.
65: 
66: Data-types can be used as functions to convert python numbers to array scalars
67: (see the array scalar section for an explanation), python sequences of numbers
68: to arrays of that type, or as arguments to the dtype keyword that many numpy
69: functions or methods accept. Some examples::
70: 
71:     >>> import numpy as np
72:     >>> x = np.float32(1.0)
73:     >>> x
74:     1.0
75:     >>> y = np.int_([1,2,4])
76:     >>> y
77:     array([1, 2, 4])
78:     >>> z = np.arange(3, dtype=np.uint8)
79:     >>> z
80:     array([0, 1, 2], dtype=uint8)
81: 
82: Array types can also be referred to by character codes, mostly to retain
83: backward compatibility with older packages such as Numeric.  Some
84: documentation may still refer to these, for example::
85: 
86:   >>> np.array([1, 2, 3], dtype='f')
87:   array([ 1.,  2.,  3.], dtype=float32)
88: 
89: We recommend using dtype objects instead.
90: 
91: To convert the type of an array, use the .astype() method (preferred) or
92: the type itself as a function. For example: ::
93: 
94:     >>> z.astype(float)                 #doctest: +NORMALIZE_WHITESPACE
95:     array([  0.,  1.,  2.])
96:     >>> np.int8(z)
97:     array([0, 1, 2], dtype=int8)
98: 
99: Note that, above, we use the *Python* float object as a dtype.  NumPy knows
100: that ``int`` refers to ``np.int_``, ``bool`` means ``np.bool_``,
101: that ``float`` is ``np.float_`` and ``complex`` is ``np.complex_``.
102: The other data-types do not have Python equivalents.
103: 
104: To determine the type of an array, look at the dtype attribute::
105: 
106:     >>> z.dtype
107:     dtype('uint8')
108: 
109: dtype objects also contain information about the type, such as its bit-width
110: and its byte-order.  The data type can also be used indirectly to query
111: properties of the type, such as whether it is an integer::
112: 
113:     >>> d = np.dtype(int)
114:     >>> d
115:     dtype('int32')
116: 
117:     >>> np.issubdtype(d, int)
118:     True
119: 
120:     >>> np.issubdtype(d, float)
121:     False
122: 
123: 
124: Array Scalars
125: =============
126: 
127: Numpy generally returns elements of arrays as array scalars (a scalar
128: with an associated dtype).  Array scalars differ from Python scalars, but
129: for the most part they can be used interchangeably (the primary
130: exception is for versions of Python older than v2.x, where integer array
131: scalars cannot act as indices for lists and tuples).  There are some
132: exceptions, such as when code requires very specific attributes of a scalar
133: or when it checks specifically whether a value is a Python scalar. Generally,
134: problems are easily fixed by explicitly converting array scalars
135: to Python scalars, using the corresponding Python type function
136: (e.g., ``int``, ``float``, ``complex``, ``str``, ``unicode``).
137: 
138: The primary advantage of using array scalars is that
139: they preserve the array type (Python may not have a matching scalar type
140: available, e.g. ``int16``).  Therefore, the use of array scalars ensures
141: identical behaviour between arrays and scalars, irrespective of whether the
142: value is inside an array or not.  NumPy scalars also have many of the same
143: methods arrays do.
144: 
145: Extended Precision
146: ==================
147: 
148: Python's floating-point numbers are usually 64-bit floating-point numbers,
149: nearly equivalent to ``np.float64``. In some unusual situations it may be
150: useful to use floating-point numbers with more precision. Whether this
151: is possible in numpy depends on the hardware and on the development
152: environment: specifically, x86 machines provide hardware floating-point
153: with 80-bit precision, and while most C compilers provide this as their
154: ``long double`` type, MSVC (standard for Windows builds) makes
155: ``long double`` identical to ``double`` (64 bits). Numpy makes the
156: compiler's ``long double`` available as ``np.longdouble`` (and
157: ``np.clongdouble`` for the complex numbers). You can find out what your
158: numpy provides with``np.finfo(np.longdouble)``.
159: 
160: Numpy does not provide a dtype with more precision than C
161: ``long double``s; in particular, the 128-bit IEEE quad precision
162: data type (FORTRAN's ``REAL*16``) is not available.
163: 
164: For efficient memory alignment, ``np.longdouble`` is usually stored
165: padded with zero bits, either to 96 or 128 bits. Which is more efficient
166: depends on hardware and development environment; typically on 32-bit
167: systems they are padded to 96 bits, while on 64-bit systems they are
168: typically padded to 128 bits. ``np.longdouble`` is padded to the system
169: default; ``np.float96`` and ``np.float128`` are provided for users who
170: want specific padding. In spite of the names, ``np.float96`` and
171: ``np.float128`` provide only as much precision as ``np.longdouble``,
172: that is, 80 bits on most x86 machines and 64 bits in standard
173: Windows builds.
174: 
175: Be warned that even if ``np.longdouble`` offers more precision than
176: python ``float``, it is easy to lose that extra precision, since
177: python often forces values to pass through ``float``. For example,
178: the ``%`` formatting operator requires its arguments to be converted
179: to standard python types, and it is therefore impossible to preserve
180: extended precision even if many decimal places are requested. It can
181: be useful to test your code with the value
182: ``1 + np.finfo(np.longdouble).eps``.
183: 
184: '''
185: from __future__ import division, absolute_import, print_function
186: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', "\n============\nArray basics\n============\n\nArray types and conversions between types\n=========================================\n\nNumpy supports a much greater variety of numerical types than Python does.\nThis section shows which are available, and how to modify an array's data-type.\n\n==========  ==========================================================\nData type   Description\n==========  ==========================================================\nbool_       Boolean (True or False) stored as a byte\nint_        Default integer type (same as C ``long``; normally either\n            ``int64`` or ``int32``)\nintc        Identical to C ``int`` (normally ``int32`` or ``int64``)\nintp        Integer used for indexing (same as C ``ssize_t``; normally\n            either ``int32`` or ``int64``)\nint8        Byte (-128 to 127)\nint16       Integer (-32768 to 32767)\nint32       Integer (-2147483648 to 2147483647)\nint64       Integer (-9223372036854775808 to 9223372036854775807)\nuint8       Unsigned integer (0 to 255)\nuint16      Unsigned integer (0 to 65535)\nuint32      Unsigned integer (0 to 4294967295)\nuint64      Unsigned integer (0 to 18446744073709551615)\nfloat_      Shorthand for ``float64``.\nfloat16     Half precision float: sign bit, 5 bits exponent,\n            10 bits mantissa\nfloat32     Single precision float: sign bit, 8 bits exponent,\n            23 bits mantissa\nfloat64     Double precision float: sign bit, 11 bits exponent,\n            52 bits mantissa\ncomplex_    Shorthand for ``complex128``.\ncomplex64   Complex number, represented by two 32-bit floats (real\n            and imaginary components)\ncomplex128  Complex number, represented by two 64-bit floats (real\n            and imaginary components)\n==========  ==========================================================\n\nAdditionally to ``intc`` the platform dependent C integer types ``short``,\n``long``, ``longlong`` and their unsigned versions are defined.\n\nNumpy numerical types are instances of ``dtype`` (data-type) objects, each\nhaving unique characteristics.  Once you have imported NumPy using\n\n  ::\n\n    >>> import numpy as np\n\nthe dtypes are available as ``np.bool_``, ``np.float32``, etc.\n\nAdvanced types, not listed in the table above, are explored in\nsection :ref:`structured_arrays`.\n\nThere are 5 basic numerical types representing booleans (bool), integers (int),\nunsigned integers (uint) floating point (float) and complex. Those with numbers\nin their name indicate the bitsize of the type (i.e. how many bits are needed\nto represent a single value in memory).  Some types, such as ``int`` and\n``intp``, have differing bitsizes, dependent on the platforms (e.g. 32-bit\nvs. 64-bit machines).  This should be taken into account when interfacing\nwith low-level code (such as C or Fortran) where the raw memory is addressed.\n\nData-types can be used as functions to convert python numbers to array scalars\n(see the array scalar section for an explanation), python sequences of numbers\nto arrays of that type, or as arguments to the dtype keyword that many numpy\nfunctions or methods accept. Some examples::\n\n    >>> import numpy as np\n    >>> x = np.float32(1.0)\n    >>> x\n    1.0\n    >>> y = np.int_([1,2,4])\n    >>> y\n    array([1, 2, 4])\n    >>> z = np.arange(3, dtype=np.uint8)\n    >>> z\n    array([0, 1, 2], dtype=uint8)\n\nArray types can also be referred to by character codes, mostly to retain\nbackward compatibility with older packages such as Numeric.  Some\ndocumentation may still refer to these, for example::\n\n  >>> np.array([1, 2, 3], dtype='f')\n  array([ 1.,  2.,  3.], dtype=float32)\n\nWe recommend using dtype objects instead.\n\nTo convert the type of an array, use the .astype() method (preferred) or\nthe type itself as a function. For example: ::\n\n    >>> z.astype(float)                 #doctest: +NORMALIZE_WHITESPACE\n    array([  0.,  1.,  2.])\n    >>> np.int8(z)\n    array([0, 1, 2], dtype=int8)\n\nNote that, above, we use the *Python* float object as a dtype.  NumPy knows\nthat ``int`` refers to ``np.int_``, ``bool`` means ``np.bool_``,\nthat ``float`` is ``np.float_`` and ``complex`` is ``np.complex_``.\nThe other data-types do not have Python equivalents.\n\nTo determine the type of an array, look at the dtype attribute::\n\n    >>> z.dtype\n    dtype('uint8')\n\ndtype objects also contain information about the type, such as its bit-width\nand its byte-order.  The data type can also be used indirectly to query\nproperties of the type, such as whether it is an integer::\n\n    >>> d = np.dtype(int)\n    >>> d\n    dtype('int32')\n\n    >>> np.issubdtype(d, int)\n    True\n\n    >>> np.issubdtype(d, float)\n    False\n\n\nArray Scalars\n=============\n\nNumpy generally returns elements of arrays as array scalars (a scalar\nwith an associated dtype).  Array scalars differ from Python scalars, but\nfor the most part they can be used interchangeably (the primary\nexception is for versions of Python older than v2.x, where integer array\nscalars cannot act as indices for lists and tuples).  There are some\nexceptions, such as when code requires very specific attributes of a scalar\nor when it checks specifically whether a value is a Python scalar. Generally,\nproblems are easily fixed by explicitly converting array scalars\nto Python scalars, using the corresponding Python type function\n(e.g., ``int``, ``float``, ``complex``, ``str``, ``unicode``).\n\nThe primary advantage of using array scalars is that\nthey preserve the array type (Python may not have a matching scalar type\navailable, e.g. ``int16``).  Therefore, the use of array scalars ensures\nidentical behaviour between arrays and scalars, irrespective of whether the\nvalue is inside an array or not.  NumPy scalars also have many of the same\nmethods arrays do.\n\nExtended Precision\n==================\n\nPython's floating-point numbers are usually 64-bit floating-point numbers,\nnearly equivalent to ``np.float64``. In some unusual situations it may be\nuseful to use floating-point numbers with more precision. Whether this\nis possible in numpy depends on the hardware and on the development\nenvironment: specifically, x86 machines provide hardware floating-point\nwith 80-bit precision, and while most C compilers provide this as their\n``long double`` type, MSVC (standard for Windows builds) makes\n``long double`` identical to ``double`` (64 bits). Numpy makes the\ncompiler's ``long double`` available as ``np.longdouble`` (and\n``np.clongdouble`` for the complex numbers). You can find out what your\nnumpy provides with``np.finfo(np.longdouble)``.\n\nNumpy does not provide a dtype with more precision than C\n``long double``s; in particular, the 128-bit IEEE quad precision\ndata type (FORTRAN's ``REAL*16``) is not available.\n\nFor efficient memory alignment, ``np.longdouble`` is usually stored\npadded with zero bits, either to 96 or 128 bits. Which is more efficient\ndepends on hardware and development environment; typically on 32-bit\nsystems they are padded to 96 bits, while on 64-bit systems they are\ntypically padded to 128 bits. ``np.longdouble`` is padded to the system\ndefault; ``np.float96`` and ``np.float128`` are provided for users who\nwant specific padding. In spite of the names, ``np.float96`` and\n``np.float128`` provide only as much precision as ``np.longdouble``,\nthat is, 80 bits on most x86 machines and 64 bits in standard\nWindows builds.\n\nBe warned that even if ``np.longdouble`` offers more precision than\npython ``float``, it is easy to lose that extra precision, since\npython often forces values to pass through ``float``. For example,\nthe ``%`` formatting operator requires its arguments to be converted\nto standard python types, and it is therefore impossible to preserve\nextended precision even if many decimal places are requested. It can\nbe useful to test your code with the value\n``1 + np.finfo(np.longdouble).eps``.\n\n")

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

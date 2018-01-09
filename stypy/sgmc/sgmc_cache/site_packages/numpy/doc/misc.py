
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =============
3: Miscellaneous
4: =============
5: 
6: IEEE 754 Floating Point Special Values
7: --------------------------------------
8: 
9: Special values defined in numpy: nan, inf,
10: 
11: NaNs can be used as a poor-man's mask (if you don't care what the
12: original value was)
13: 
14: Note: cannot use equality to test NaNs. E.g.: ::
15: 
16:  >>> myarr = np.array([1., 0., np.nan, 3.])
17:  >>> np.where(myarr == np.nan)
18:  >>> np.nan == np.nan  # is always False! Use special numpy functions instead.
19:  False
20:  >>> myarr[myarr == np.nan] = 0. # doesn't work
21:  >>> myarr
22:  array([  1.,   0.,  NaN,   3.])
23:  >>> myarr[np.isnan(myarr)] = 0. # use this instead find
24:  >>> myarr
25:  array([ 1.,  0.,  0.,  3.])
26: 
27: Other related special value functions: ::
28: 
29:  isinf():    True if value is inf
30:  isfinite(): True if not nan or inf
31:  nan_to_num(): Map nan to 0, inf to max float, -inf to min float
32: 
33: The following corresponds to the usual functions except that nans are excluded
34: from the results: ::
35: 
36:  nansum()
37:  nanmax()
38:  nanmin()
39:  nanargmax()
40:  nanargmin()
41: 
42:  >>> x = np.arange(10.)
43:  >>> x[3] = np.nan
44:  >>> x.sum()
45:  nan
46:  >>> np.nansum(x)
47:  42.0
48: 
49: How numpy handles numerical exceptions
50: --------------------------------------
51: 
52: The default is to ``'warn'`` for ``invalid``, ``divide``, and ``overflow``
53: and ``'ignore'`` for ``underflow``.  But this can be changed, and it can be
54: set individually for different kinds of exceptions. The different behaviors
55: are:
56: 
57:  - 'ignore' : Take no action when the exception occurs.
58:  - 'warn'   : Print a `RuntimeWarning` (via the Python `warnings` module).
59:  - 'raise'  : Raise a `FloatingPointError`.
60:  - 'call'   : Call a function specified using the `seterrcall` function.
61:  - 'print'  : Print a warning directly to ``stdout``.
62:  - 'log'    : Record error in a Log object specified by `seterrcall`.
63: 
64: These behaviors can be set for all kinds of errors or specific ones:
65: 
66:  - all       : apply to all numeric exceptions
67:  - invalid   : when NaNs are generated
68:  - divide    : divide by zero (for integers as well!)
69:  - overflow  : floating point overflows
70:  - underflow : floating point underflows
71: 
72: Note that integer divide-by-zero is handled by the same machinery.
73: These behaviors are set on a per-thread basis.
74: 
75: Examples
76: --------
77: 
78: ::
79: 
80:  >>> oldsettings = np.seterr(all='warn')
81:  >>> np.zeros(5,dtype=np.float32)/0.
82:  invalid value encountered in divide
83:  >>> j = np.seterr(under='ignore')
84:  >>> np.array([1.e-100])**10
85:  >>> j = np.seterr(invalid='raise')
86:  >>> np.sqrt(np.array([-1.]))
87:  FloatingPointError: invalid value encountered in sqrt
88:  >>> def errorhandler(errstr, errflag):
89:  ...      print("saw stupid error!")
90:  >>> np.seterrcall(errorhandler)
91:  <function err_handler at 0x...>
92:  >>> j = np.seterr(all='call')
93:  >>> np.zeros(5, dtype=np.int32)/0
94:  FloatingPointError: invalid value encountered in divide
95:  saw stupid error!
96:  >>> j = np.seterr(**oldsettings) # restore previous
97:  ...                              # error-handling settings
98: 
99: Interfacing to C
100: ----------------
101: Only a survey of the choices. Little detail on how each works.
102: 
103: 1) Bare metal, wrap your own C-code manually.
104: 
105:  - Plusses:
106: 
107:    - Efficient
108:    - No dependencies on other tools
109: 
110:  - Minuses:
111: 
112:    - Lots of learning overhead:
113: 
114:      - need to learn basics of Python C API
115:      - need to learn basics of numpy C API
116:      - need to learn how to handle reference counting and love it.
117: 
118:    - Reference counting often difficult to get right.
119: 
120:      - getting it wrong leads to memory leaks, and worse, segfaults
121: 
122:    - API will change for Python 3.0!
123: 
124: 2) Cython
125: 
126:  - Plusses:
127: 
128:    - avoid learning C API's
129:    - no dealing with reference counting
130:    - can code in pseudo python and generate C code
131:    - can also interface to existing C code
132:    - should shield you from changes to Python C api
133:    - has become the de-facto standard within the scientific Python community
134:    - fast indexing support for arrays
135: 
136:  - Minuses:
137: 
138:    - Can write code in non-standard form which may become obsolete
139:    - Not as flexible as manual wrapping
140: 
141: 3) ctypes
142: 
143:  - Plusses:
144: 
145:    - part of Python standard library
146:    - good for interfacing to existing sharable libraries, particularly
147:      Windows DLLs
148:    - avoids API/reference counting issues
149:    - good numpy support: arrays have all these in their ctypes
150:      attribute: ::
151: 
152:        a.ctypes.data              a.ctypes.get_strides
153:        a.ctypes.data_as           a.ctypes.shape
154:        a.ctypes.get_as_parameter  a.ctypes.shape_as
155:        a.ctypes.get_data          a.ctypes.strides
156:        a.ctypes.get_shape         a.ctypes.strides_as
157: 
158:  - Minuses:
159: 
160:    - can't use for writing code to be turned into C extensions, only a wrapper
161:      tool.
162: 
163: 4) SWIG (automatic wrapper generator)
164: 
165:  - Plusses:
166: 
167:    - around a long time
168:    - multiple scripting language support
169:    - C++ support
170:    - Good for wrapping large (many functions) existing C libraries
171: 
172:  - Minuses:
173: 
174:    - generates lots of code between Python and the C code
175:    - can cause performance problems that are nearly impossible to optimize
176:      out
177:    - interface files can be hard to write
178:    - doesn't necessarily avoid reference counting issues or needing to know
179:      API's
180: 
181: 5) scipy.weave
182: 
183:  - Plusses:
184: 
185:    - can turn many numpy expressions into C code
186:    - dynamic compiling and loading of generated C code
187:    - can embed pure C code in Python module and have weave extract, generate
188:      interfaces and compile, etc.
189: 
190:  - Minuses:
191: 
192:    - Future very uncertain: it's the only part of Scipy not ported to Python 3
193:      and is effectively deprecated in favor of Cython.
194: 
195: 6) Psyco
196: 
197:  - Plusses:
198: 
199:    - Turns pure python into efficient machine code through jit-like
200:      optimizations
201:    - very fast when it optimizes well
202: 
203:  - Minuses:
204: 
205:    - Only on intel (windows?)
206:    - Doesn't do much for numpy?
207: 
208: Interfacing to Fortran:
209: -----------------------
210: The clear choice to wrap Fortran code is
211: `f2py <http://docs.scipy.org/doc/numpy-dev/f2py/>`_.
212: 
213: Pyfort is an older alternative, but not supported any longer.
214: Fwrap is a newer project that looked promising but isn't being developed any
215: longer.
216: 
217: Interfacing to C++:
218: -------------------
219:  1) Cython
220:  2) CXX
221:  3) Boost.python
222:  4) SWIG
223:  5) SIP (used mainly in PyQT)
224: 
225: '''
226: from __future__ import division, absolute_import, print_function
227: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, (-1)), 'str', '\n=============\nMiscellaneous\n=============\n\nIEEE 754 Floating Point Special Values\n--------------------------------------\n\nSpecial values defined in numpy: nan, inf,\n\nNaNs can be used as a poor-man\'s mask (if you don\'t care what the\noriginal value was)\n\nNote: cannot use equality to test NaNs. E.g.: ::\n\n >>> myarr = np.array([1., 0., np.nan, 3.])\n >>> np.where(myarr == np.nan)\n >>> np.nan == np.nan  # is always False! Use special numpy functions instead.\n False\n >>> myarr[myarr == np.nan] = 0. # doesn\'t work\n >>> myarr\n array([  1.,   0.,  NaN,   3.])\n >>> myarr[np.isnan(myarr)] = 0. # use this instead find\n >>> myarr\n array([ 1.,  0.,  0.,  3.])\n\nOther related special value functions: ::\n\n isinf():    True if value is inf\n isfinite(): True if not nan or inf\n nan_to_num(): Map nan to 0, inf to max float, -inf to min float\n\nThe following corresponds to the usual functions except that nans are excluded\nfrom the results: ::\n\n nansum()\n nanmax()\n nanmin()\n nanargmax()\n nanargmin()\n\n >>> x = np.arange(10.)\n >>> x[3] = np.nan\n >>> x.sum()\n nan\n >>> np.nansum(x)\n 42.0\n\nHow numpy handles numerical exceptions\n--------------------------------------\n\nThe default is to ``\'warn\'`` for ``invalid``, ``divide``, and ``overflow``\nand ``\'ignore\'`` for ``underflow``.  But this can be changed, and it can be\nset individually for different kinds of exceptions. The different behaviors\nare:\n\n - \'ignore\' : Take no action when the exception occurs.\n - \'warn\'   : Print a `RuntimeWarning` (via the Python `warnings` module).\n - \'raise\'  : Raise a `FloatingPointError`.\n - \'call\'   : Call a function specified using the `seterrcall` function.\n - \'print\'  : Print a warning directly to ``stdout``.\n - \'log\'    : Record error in a Log object specified by `seterrcall`.\n\nThese behaviors can be set for all kinds of errors or specific ones:\n\n - all       : apply to all numeric exceptions\n - invalid   : when NaNs are generated\n - divide    : divide by zero (for integers as well!)\n - overflow  : floating point overflows\n - underflow : floating point underflows\n\nNote that integer divide-by-zero is handled by the same machinery.\nThese behaviors are set on a per-thread basis.\n\nExamples\n--------\n\n::\n\n >>> oldsettings = np.seterr(all=\'warn\')\n >>> np.zeros(5,dtype=np.float32)/0.\n invalid value encountered in divide\n >>> j = np.seterr(under=\'ignore\')\n >>> np.array([1.e-100])**10\n >>> j = np.seterr(invalid=\'raise\')\n >>> np.sqrt(np.array([-1.]))\n FloatingPointError: invalid value encountered in sqrt\n >>> def errorhandler(errstr, errflag):\n ...      print("saw stupid error!")\n >>> np.seterrcall(errorhandler)\n <function err_handler at 0x...>\n >>> j = np.seterr(all=\'call\')\n >>> np.zeros(5, dtype=np.int32)/0\n FloatingPointError: invalid value encountered in divide\n saw stupid error!\n >>> j = np.seterr(**oldsettings) # restore previous\n ...                              # error-handling settings\n\nInterfacing to C\n----------------\nOnly a survey of the choices. Little detail on how each works.\n\n1) Bare metal, wrap your own C-code manually.\n\n - Plusses:\n\n   - Efficient\n   - No dependencies on other tools\n\n - Minuses:\n\n   - Lots of learning overhead:\n\n     - need to learn basics of Python C API\n     - need to learn basics of numpy C API\n     - need to learn how to handle reference counting and love it.\n\n   - Reference counting often difficult to get right.\n\n     - getting it wrong leads to memory leaks, and worse, segfaults\n\n   - API will change for Python 3.0!\n\n2) Cython\n\n - Plusses:\n\n   - avoid learning C API\'s\n   - no dealing with reference counting\n   - can code in pseudo python and generate C code\n   - can also interface to existing C code\n   - should shield you from changes to Python C api\n   - has become the de-facto standard within the scientific Python community\n   - fast indexing support for arrays\n\n - Minuses:\n\n   - Can write code in non-standard form which may become obsolete\n   - Not as flexible as manual wrapping\n\n3) ctypes\n\n - Plusses:\n\n   - part of Python standard library\n   - good for interfacing to existing sharable libraries, particularly\n     Windows DLLs\n   - avoids API/reference counting issues\n   - good numpy support: arrays have all these in their ctypes\n     attribute: ::\n\n       a.ctypes.data              a.ctypes.get_strides\n       a.ctypes.data_as           a.ctypes.shape\n       a.ctypes.get_as_parameter  a.ctypes.shape_as\n       a.ctypes.get_data          a.ctypes.strides\n       a.ctypes.get_shape         a.ctypes.strides_as\n\n - Minuses:\n\n   - can\'t use for writing code to be turned into C extensions, only a wrapper\n     tool.\n\n4) SWIG (automatic wrapper generator)\n\n - Plusses:\n\n   - around a long time\n   - multiple scripting language support\n   - C++ support\n   - Good for wrapping large (many functions) existing C libraries\n\n - Minuses:\n\n   - generates lots of code between Python and the C code\n   - can cause performance problems that are nearly impossible to optimize\n     out\n   - interface files can be hard to write\n   - doesn\'t necessarily avoid reference counting issues or needing to know\n     API\'s\n\n5) scipy.weave\n\n - Plusses:\n\n   - can turn many numpy expressions into C code\n   - dynamic compiling and loading of generated C code\n   - can embed pure C code in Python module and have weave extract, generate\n     interfaces and compile, etc.\n\n - Minuses:\n\n   - Future very uncertain: it\'s the only part of Scipy not ported to Python 3\n     and is effectively deprecated in favor of Cython.\n\n6) Psyco\n\n - Plusses:\n\n   - Turns pure python into efficient machine code through jit-like\n     optimizations\n   - very fast when it optimizes well\n\n - Minuses:\n\n   - Only on intel (windows?)\n   - Doesn\'t do much for numpy?\n\nInterfacing to Fortran:\n-----------------------\nThe clear choice to wrap Fortran code is\n`f2py <http://docs.scipy.org/doc/numpy-dev/f2py/>`_.\n\nPyfort is an older alternative, but not supported any longer.\nFwrap is a newer project that looked promising but isn\'t being developed any\nlonger.\n\nInterfacing to C++:\n-------------------\n 1) Cython\n 2) CXX\n 3) Boost.python\n 4) SWIG\n 5) SIP (used mainly in PyQT)\n\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

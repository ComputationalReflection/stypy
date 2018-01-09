
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Basic functions used by several sub-packages and
3: useful to have in the main name-space.
4: 
5: Type Handling
6: -------------
7: ================ ===================
8: iscomplexobj     Test for complex object, scalar result
9: isrealobj        Test for real object, scalar result
10: iscomplex        Test for complex elements, array result
11: isreal           Test for real elements, array result
12: imag             Imaginary part
13: real             Real part
14: real_if_close    Turns complex number with tiny imaginary part to real
15: isneginf         Tests for negative infinity, array result
16: isposinf         Tests for positive infinity, array result
17: isnan            Tests for nans, array result
18: isinf            Tests for infinity, array result
19: isfinite         Tests for finite numbers, array result
20: isscalar         True if argument is a scalar
21: nan_to_num       Replaces NaN's with 0 and infinities with large numbers
22: cast             Dictionary of functions to force cast to each type
23: common_type      Determine the minimum common type code for a group
24:                  of arrays
25: mintypecode      Return minimal allowed common typecode.
26: ================ ===================
27: 
28: Index Tricks
29: ------------
30: ================ ===================
31: mgrid            Method which allows easy construction of N-d
32:                  'mesh-grids'
33: ``r_``           Append and construct arrays: turns slice objects into
34:                  ranges and concatenates them, for 2d arrays appends rows.
35: index_exp        Konrad Hinsen's index_expression class instance which
36:                  can be useful for building complicated slicing syntax.
37: ================ ===================
38: 
39: Useful Functions
40: ----------------
41: ================ ===================
42: select           Extension of where to multiple conditions and choices
43: extract          Extract 1d array from flattened array according to mask
44: insert           Insert 1d array of values into Nd array according to mask
45: linspace         Evenly spaced samples in linear space
46: logspace         Evenly spaced samples in logarithmic space
47: fix              Round x to nearest integer towards zero
48: mod              Modulo mod(x,y) = x % y except keeps sign of y
49: amax             Array maximum along axis
50: amin             Array minimum along axis
51: ptp              Array max-min along axis
52: cumsum           Cumulative sum along axis
53: prod             Product of elements along axis
54: cumprod          Cumluative product along axis
55: diff             Discrete differences along axis
56: angle            Returns angle of complex argument
57: unwrap           Unwrap phase along given axis (1-d algorithm)
58: sort_complex     Sort a complex-array (based on real, then imaginary)
59: trim_zeros       Trim the leading and trailing zeros from 1D array.
60: vectorize        A class that wraps a Python function taking scalar
61:                  arguments into a generalized function which can handle
62:                  arrays of arguments using the broadcast rules of
63:                  numerix Python.
64: ================ ===================
65: 
66: Shape Manipulation
67: ------------------
68: ================ ===================
69: squeeze          Return a with length-one dimensions removed.
70: atleast_1d       Force arrays to be > 1D
71: atleast_2d       Force arrays to be > 2D
72: atleast_3d       Force arrays to be > 3D
73: vstack           Stack arrays vertically (row on row)
74: hstack           Stack arrays horizontally (column on column)
75: column_stack     Stack 1D arrays as columns into 2D array
76: dstack           Stack arrays depthwise (along third dimension)
77: stack            Stack arrays along a new axis
78: split            Divide array into a list of sub-arrays
79: hsplit           Split into columns
80: vsplit           Split into rows
81: dsplit           Split along third dimension
82: ================ ===================
83: 
84: Matrix (2D Array) Manipulations
85: -------------------------------
86: ================ ===================
87: fliplr           2D array with columns flipped
88: flipud           2D array with rows flipped
89: rot90            Rotate a 2D array a multiple of 90 degrees
90: eye              Return a 2D array with ones down a given diagonal
91: diag             Construct a 2D array from a vector, or return a given
92:                  diagonal from a 2D array.
93: mat              Construct a Matrix
94: bmat             Build a Matrix from blocks
95: ================ ===================
96: 
97: Polynomials
98: -----------
99: ================ ===================
100: poly1d           A one-dimensional polynomial class
101: poly             Return polynomial coefficients from roots
102: roots            Find roots of polynomial given coefficients
103: polyint          Integrate polynomial
104: polyder          Differentiate polynomial
105: polyadd          Add polynomials
106: polysub          Substract polynomials
107: polymul          Multiply polynomials
108: polydiv          Divide polynomials
109: polyval          Evaluate polynomial at given argument
110: ================ ===================
111: 
112: Iterators
113: ---------
114: ================ ===================
115: Arrayterator     A buffered iterator for big arrays.
116: ================ ===================
117: 
118: Import Tricks
119: -------------
120: ================ ===================
121: ppimport         Postpone module import until trying to use it
122: ppimport_attr    Postpone module import until trying to use its attribute
123: ppresolve        Import postponed module and return it.
124: ================ ===================
125: 
126: Machine Arithmetics
127: -------------------
128: ================ ===================
129: machar_single    Single precision floating point arithmetic parameters
130: machar_double    Double precision floating point arithmetic parameters
131: ================ ===================
132: 
133: Threading Tricks
134: ----------------
135: ================ ===================
136: ParallelExec     Execute commands in parallel thread.
137: ================ ===================
138: 
139: 1D Array Set Operations
140: -----------------------
141: Set operations for 1D numeric arrays based on sort() function.
142: 
143: ================ ===================
144: ediff1d          Array difference (auxiliary function).
145: unique           Unique elements of an array.
146: intersect1d      Intersection of 1D arrays with unique elements.
147: setxor1d         Set exclusive-or of 1D arrays with unique elements.
148: in1d             Test whether elements in a 1D array are also present in
149:                  another array.
150: union1d          Union of 1D arrays with unique elements.
151: setdiff1d        Set difference of 1D arrays with unique elements.
152: ================ ===================
153: 
154: '''
155: from __future__ import division, absolute_import, print_function
156: 
157: depends = ['core', 'testing']
158: global_symbols = ['*']
159: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_115591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, (-1)), 'str', "\nBasic functions used by several sub-packages and\nuseful to have in the main name-space.\n\nType Handling\n-------------\n================ ===================\niscomplexobj     Test for complex object, scalar result\nisrealobj        Test for real object, scalar result\niscomplex        Test for complex elements, array result\nisreal           Test for real elements, array result\nimag             Imaginary part\nreal             Real part\nreal_if_close    Turns complex number with tiny imaginary part to real\nisneginf         Tests for negative infinity, array result\nisposinf         Tests for positive infinity, array result\nisnan            Tests for nans, array result\nisinf            Tests for infinity, array result\nisfinite         Tests for finite numbers, array result\nisscalar         True if argument is a scalar\nnan_to_num       Replaces NaN's with 0 and infinities with large numbers\ncast             Dictionary of functions to force cast to each type\ncommon_type      Determine the minimum common type code for a group\n                 of arrays\nmintypecode      Return minimal allowed common typecode.\n================ ===================\n\nIndex Tricks\n------------\n================ ===================\nmgrid            Method which allows easy construction of N-d\n                 'mesh-grids'\n``r_``           Append and construct arrays: turns slice objects into\n                 ranges and concatenates them, for 2d arrays appends rows.\nindex_exp        Konrad Hinsen's index_expression class instance which\n                 can be useful for building complicated slicing syntax.\n================ ===================\n\nUseful Functions\n----------------\n================ ===================\nselect           Extension of where to multiple conditions and choices\nextract          Extract 1d array from flattened array according to mask\ninsert           Insert 1d array of values into Nd array according to mask\nlinspace         Evenly spaced samples in linear space\nlogspace         Evenly spaced samples in logarithmic space\nfix              Round x to nearest integer towards zero\nmod              Modulo mod(x,y) = x % y except keeps sign of y\namax             Array maximum along axis\namin             Array minimum along axis\nptp              Array max-min along axis\ncumsum           Cumulative sum along axis\nprod             Product of elements along axis\ncumprod          Cumluative product along axis\ndiff             Discrete differences along axis\nangle            Returns angle of complex argument\nunwrap           Unwrap phase along given axis (1-d algorithm)\nsort_complex     Sort a complex-array (based on real, then imaginary)\ntrim_zeros       Trim the leading and trailing zeros from 1D array.\nvectorize        A class that wraps a Python function taking scalar\n                 arguments into a generalized function which can handle\n                 arrays of arguments using the broadcast rules of\n                 numerix Python.\n================ ===================\n\nShape Manipulation\n------------------\n================ ===================\nsqueeze          Return a with length-one dimensions removed.\natleast_1d       Force arrays to be > 1D\natleast_2d       Force arrays to be > 2D\natleast_3d       Force arrays to be > 3D\nvstack           Stack arrays vertically (row on row)\nhstack           Stack arrays horizontally (column on column)\ncolumn_stack     Stack 1D arrays as columns into 2D array\ndstack           Stack arrays depthwise (along third dimension)\nstack            Stack arrays along a new axis\nsplit            Divide array into a list of sub-arrays\nhsplit           Split into columns\nvsplit           Split into rows\ndsplit           Split along third dimension\n================ ===================\n\nMatrix (2D Array) Manipulations\n-------------------------------\n================ ===================\nfliplr           2D array with columns flipped\nflipud           2D array with rows flipped\nrot90            Rotate a 2D array a multiple of 90 degrees\neye              Return a 2D array with ones down a given diagonal\ndiag             Construct a 2D array from a vector, or return a given\n                 diagonal from a 2D array.\nmat              Construct a Matrix\nbmat             Build a Matrix from blocks\n================ ===================\n\nPolynomials\n-----------\n================ ===================\npoly1d           A one-dimensional polynomial class\npoly             Return polynomial coefficients from roots\nroots            Find roots of polynomial given coefficients\npolyint          Integrate polynomial\npolyder          Differentiate polynomial\npolyadd          Add polynomials\npolysub          Substract polynomials\npolymul          Multiply polynomials\npolydiv          Divide polynomials\npolyval          Evaluate polynomial at given argument\n================ ===================\n\nIterators\n---------\n================ ===================\nArrayterator     A buffered iterator for big arrays.\n================ ===================\n\nImport Tricks\n-------------\n================ ===================\nppimport         Postpone module import until trying to use it\nppimport_attr    Postpone module import until trying to use its attribute\nppresolve        Import postponed module and return it.\n================ ===================\n\nMachine Arithmetics\n-------------------\n================ ===================\nmachar_single    Single precision floating point arithmetic parameters\nmachar_double    Double precision floating point arithmetic parameters\n================ ===================\n\nThreading Tricks\n----------------\n================ ===================\nParallelExec     Execute commands in parallel thread.\n================ ===================\n\n1D Array Set Operations\n-----------------------\nSet operations for 1D numeric arrays based on sort() function.\n\n================ ===================\nediff1d          Array difference (auxiliary function).\nunique           Unique elements of an array.\nintersect1d      Intersection of 1D arrays with unique elements.\nsetxor1d         Set exclusive-or of 1D arrays with unique elements.\nin1d             Test whether elements in a 1D array are also present in\n                 another array.\nunion1d          Union of 1D arrays with unique elements.\nsetdiff1d        Set difference of 1D arrays with unique elements.\n================ ===================\n\n")

# Assigning a List to a Name (line 157):

# Obtaining an instance of the builtin type 'list' (line 157)
list_115592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 157)
# Adding element type (line 157)
str_115593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 11), 'str', 'core')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 10), list_115592, str_115593)
# Adding element type (line 157)
str_115594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'str', 'testing')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 10), list_115592, str_115594)

# Assigning a type to the variable 'depends' (line 157)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 0), 'depends', list_115592)

# Assigning a List to a Name (line 158):

# Obtaining an instance of the builtin type 'list' (line 158)
list_115595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 158)
# Adding element type (line 158)
str_115596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 18), 'str', '*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 17), list_115595, str_115596)

# Assigning a type to the variable 'global_symbols' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'global_symbols', list_115595)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

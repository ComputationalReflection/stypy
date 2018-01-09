
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Defines a multi-dimensional array and useful procedures for Numerical computation.
2: 
3: Functions
4: 
5: -   array                      - NumPy Array construction
6: -   zeros                      - Return an array of all zeros
7: -   empty                      - Return an unitialized array
8: -   shape                      - Return shape of sequence or array
9: -   rank                       - Return number of dimensions
10: -   size                       - Return number of elements in entire array or a
11:                                  certain dimension
12: -   fromstring                 - Construct array from (byte) string
13: -   take                       - Select sub-arrays using sequence of indices
14: -   put                        - Set sub-arrays using sequence of 1-D indices
15: -   putmask                    - Set portion of arrays using a mask
16: -   reshape                    - Return array with new shape
17: -   repeat                     - Repeat elements of array
18: -   choose                     - Construct new array from indexed array tuple
19: -   correlate                  - Correlate two 1-d arrays
20: -   searchsorted               - Search for element in 1-d array
21: -   sum                        - Total sum over a specified dimension
22: -   average                    - Average, possibly weighted, over axis or array.
23: -   cumsum                     - Cumulative sum over a specified dimension
24: -   product                    - Total product over a specified dimension
25: -   cumproduct                 - Cumulative product over a specified dimension
26: -   alltrue                    - Logical and over an entire axis
27: -   sometrue                   - Logical or over an entire axis
28: -   allclose                   - Tests if sequences are essentially equal
29: 
30: More Functions:
31: 
32: -   arange                     - Return regularly spaced array
33: -   asarray                    - Guarantee NumPy array
34: -   convolve                   - Convolve two 1-d arrays
35: -   swapaxes                   - Exchange axes
36: -   concatenate                - Join arrays together
37: -   transpose                  - Permute axes
38: -   sort                       - Sort elements of array
39: -   argsort                    - Indices of sorted array
40: -   argmax                     - Index of largest value
41: -   argmin                     - Index of smallest value
42: -   inner                      - Innerproduct of two arrays
43: -   dot                        - Dot product (matrix multiplication)
44: -   outer                      - Outerproduct of two arrays
45: -   resize                     - Return array with arbitrary new shape
46: -   indices                    - Tuple of indices
47: -   fromfunction               - Construct array from universal function
48: -   diagonal                   - Return diagonal array
49: -   trace                      - Trace of array
50: -   dump                       - Dump array to file object (pickle)
51: -   dumps                      - Return pickled string representing data
52: -   load                       - Return array stored in file object
53: -   loads                      - Return array from pickled string
54: -   ravel                      - Return array as 1-D
55: -   nonzero                    - Indices of nonzero elements for 1-D array
56: -   shape                      - Shape of array
57: -   where                      - Construct array from binary result
58: -   compress                   - Elements of array where condition is true
59: -   clip                       - Clip array between two values
60: -   ones                       - Array of all ones
61: -   identity                   - 2-D identity array (matrix)
62: 
63: (Universal) Math Functions
64: 
65:        add                    logical_or             exp
66:        subtract               logical_xor            log
67:        multiply               logical_not            log10
68:        divide                 maximum                sin
69:        divide_safe            minimum                sinh
70:        conjugate              bitwise_and            sqrt
71:        power                  bitwise_or             tan
72:        absolute               bitwise_xor            tanh
73:        negative               invert                 ceil
74:        greater                left_shift             fabs
75:        greater_equal          right_shift            floor
76:        less                   arccos                 arctan2
77:        less_equal             arcsin                 fmod
78:        equal                  arctan                 hypot
79:        not_equal              cos                    around
80:        logical_and            cosh                   sign
81:        arccosh                arcsinh                arctanh
82: 
83: '''
84: from __future__ import division, absolute_import, print_function
85: 
86: depends = ['testing']
87: global_symbols = ['*']
88: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_6348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'str', 'Defines a multi-dimensional array and useful procedures for Numerical computation.\n\nFunctions\n\n-   array                      - NumPy Array construction\n-   zeros                      - Return an array of all zeros\n-   empty                      - Return an unitialized array\n-   shape                      - Return shape of sequence or array\n-   rank                       - Return number of dimensions\n-   size                       - Return number of elements in entire array or a\n                                 certain dimension\n-   fromstring                 - Construct array from (byte) string\n-   take                       - Select sub-arrays using sequence of indices\n-   put                        - Set sub-arrays using sequence of 1-D indices\n-   putmask                    - Set portion of arrays using a mask\n-   reshape                    - Return array with new shape\n-   repeat                     - Repeat elements of array\n-   choose                     - Construct new array from indexed array tuple\n-   correlate                  - Correlate two 1-d arrays\n-   searchsorted               - Search for element in 1-d array\n-   sum                        - Total sum over a specified dimension\n-   average                    - Average, possibly weighted, over axis or array.\n-   cumsum                     - Cumulative sum over a specified dimension\n-   product                    - Total product over a specified dimension\n-   cumproduct                 - Cumulative product over a specified dimension\n-   alltrue                    - Logical and over an entire axis\n-   sometrue                   - Logical or over an entire axis\n-   allclose                   - Tests if sequences are essentially equal\n\nMore Functions:\n\n-   arange                     - Return regularly spaced array\n-   asarray                    - Guarantee NumPy array\n-   convolve                   - Convolve two 1-d arrays\n-   swapaxes                   - Exchange axes\n-   concatenate                - Join arrays together\n-   transpose                  - Permute axes\n-   sort                       - Sort elements of array\n-   argsort                    - Indices of sorted array\n-   argmax                     - Index of largest value\n-   argmin                     - Index of smallest value\n-   inner                      - Innerproduct of two arrays\n-   dot                        - Dot product (matrix multiplication)\n-   outer                      - Outerproduct of two arrays\n-   resize                     - Return array with arbitrary new shape\n-   indices                    - Tuple of indices\n-   fromfunction               - Construct array from universal function\n-   diagonal                   - Return diagonal array\n-   trace                      - Trace of array\n-   dump                       - Dump array to file object (pickle)\n-   dumps                      - Return pickled string representing data\n-   load                       - Return array stored in file object\n-   loads                      - Return array from pickled string\n-   ravel                      - Return array as 1-D\n-   nonzero                    - Indices of nonzero elements for 1-D array\n-   shape                      - Shape of array\n-   where                      - Construct array from binary result\n-   compress                   - Elements of array where condition is true\n-   clip                       - Clip array between two values\n-   ones                       - Array of all ones\n-   identity                   - 2-D identity array (matrix)\n\n(Universal) Math Functions\n\n       add                    logical_or             exp\n       subtract               logical_xor            log\n       multiply               logical_not            log10\n       divide                 maximum                sin\n       divide_safe            minimum                sinh\n       conjugate              bitwise_and            sqrt\n       power                  bitwise_or             tan\n       absolute               bitwise_xor            tanh\n       negative               invert                 ceil\n       greater                left_shift             fabs\n       greater_equal          right_shift            floor\n       less                   arccos                 arctan2\n       less_equal             arcsin                 fmod\n       equal                  arctan                 hypot\n       not_equal              cos                    around\n       logical_and            cosh                   sign\n       arccosh                arcsinh                arctanh\n\n')

# Assigning a List to a Name (line 86):

# Obtaining an instance of the builtin type 'list' (line 86)
list_6349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 86)
# Adding element type (line 86)
str_6350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'str', 'testing')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 10), list_6349, str_6350)

# Assigning a type to the variable 'depends' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'depends', list_6349)

# Assigning a List to a Name (line 87):

# Obtaining an instance of the builtin type 'list' (line 87)
list_6351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 87)
# Adding element type (line 87)
str_6352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'str', '*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 17), list_6351, str_6352)

# Assigning a type to the variable 'global_symbols' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'global_symbols', list_6351)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

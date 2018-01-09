
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =========================================================
3: Multi-dimensional image processing (:mod:`scipy.ndimage`)
4: =========================================================
5: 
6: .. currentmodule:: scipy.ndimage
7: 
8: This package contains various functions for multi-dimensional image
9: processing.
10: 
11: 
12: Filters
13: =======
14: 
15: .. autosummary::
16:    :toctree: generated/
17: 
18:    convolve - Multi-dimensional convolution
19:    convolve1d - 1-D convolution along the given axis
20:    correlate - Multi-dimensional correlation
21:    correlate1d - 1-D correlation along the given axis
22:    gaussian_filter
23:    gaussian_filter1d
24:    gaussian_gradient_magnitude
25:    gaussian_laplace
26:    generic_filter - Multi-dimensional filter using a given function
27:    generic_filter1d - 1-D generic filter along the given axis
28:    generic_gradient_magnitude
29:    generic_laplace
30:    laplace - n-D Laplace filter based on approximate second derivatives
31:    maximum_filter
32:    maximum_filter1d
33:    median_filter - Calculates a multi-dimensional median filter
34:    minimum_filter
35:    minimum_filter1d
36:    percentile_filter - Calculates a multi-dimensional percentile filter
37:    prewitt
38:    rank_filter - Calculates a multi-dimensional rank filter
39:    sobel
40:    uniform_filter - Multi-dimensional uniform filter
41:    uniform_filter1d - 1-D uniform filter along the given axis
42: 
43: Fourier filters
44: ===============
45: 
46: .. autosummary::
47:    :toctree: generated/
48: 
49:    fourier_ellipsoid
50:    fourier_gaussian
51:    fourier_shift
52:    fourier_uniform
53: 
54: Interpolation
55: =============
56: 
57: .. autosummary::
58:    :toctree: generated/
59: 
60:    affine_transform - Apply an affine transformation
61:    geometric_transform - Apply an arbritrary geometric transform
62:    map_coordinates - Map input array to new coordinates by interpolation
63:    rotate - Rotate an array
64:    shift - Shift an array
65:    spline_filter
66:    spline_filter1d
67:    zoom - Zoom an array
68: 
69: Measurements
70: ============
71: 
72: .. autosummary::
73:    :toctree: generated/
74: 
75:    center_of_mass - The center of mass of the values of an array at labels
76:    extrema - Min's and max's of an array at labels, with their positions
77:    find_objects - Find objects in a labeled array
78:    histogram - Histogram of the values of an array, optionally at labels
79:    label - Label features in an array
80:    labeled_comprehension
81:    maximum
82:    maximum_position
83:    mean - Mean of the values of an array at labels
84:    median
85:    minimum
86:    minimum_position
87:    standard_deviation - Standard deviation of an n-D image array
88:    sum - Sum of the values of the array
89:    variance - Variance of the values of an n-D image array
90:    watershed_ift
91: 
92: Morphology
93: ==========
94: 
95: .. autosummary::
96:    :toctree: generated/
97: 
98:    binary_closing
99:    binary_dilation
100:    binary_erosion
101:    binary_fill_holes
102:    binary_hit_or_miss
103:    binary_opening
104:    binary_propagation
105:    black_tophat
106:    distance_transform_bf
107:    distance_transform_cdt
108:    distance_transform_edt
109:    generate_binary_structure
110:    grey_closing
111:    grey_dilation
112:    grey_erosion
113:    grey_opening
114:    iterate_structure
115:    morphological_gradient
116:    morphological_laplace
117:    white_tophat
118: 
119: Utility
120: =======
121: 
122: .. autosummary::
123:    :toctree: generated/
124: 
125:    imread - Load an image from a file
126: 
127: '''
128: 
129: # Copyright (C) 2003-2005 Peter J. Verveer
130: #
131: # Redistribution and use in source and binary forms, with or without
132: # modification, are permitted provided that the following conditions
133: # are met:
134: #
135: # 1. Redistributions of source code must retain the above copyright
136: #    notice, this list of conditions and the following disclaimer.
137: #
138: # 2. Redistributions in binary form must reproduce the above
139: #    copyright notice, this list of conditions and the following
140: #    disclaimer in the documentation and/or other materials provided
141: #    with the distribution.
142: #
143: # 3. The name of the author may not be used to endorse or promote
144: #    products derived from this software without specific prior
145: #    written permission.
146: #
147: # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
148: # OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
149: # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
150: # ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
151: # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
152: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
153: # GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
154: # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
155: # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
156: # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
157: # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
158: 
159: from __future__ import division, print_function, absolute_import
160: 
161: from .filters import *
162: from .fourier import *
163: from .interpolation import *
164: from .measurements import *
165: from .morphology import *
166: from .io import *
167: 
168: __version__ = '2.0'
169: 
170: __all__ = [s for s in dir() if not s.startswith('_')]
171: 
172: from scipy._lib._testutils import PytestTester
173: test = PytestTester(__name__)
174: del PytestTester
175: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_126814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, (-1)), 'str', "\n=========================================================\nMulti-dimensional image processing (:mod:`scipy.ndimage`)\n=========================================================\n\n.. currentmodule:: scipy.ndimage\n\nThis package contains various functions for multi-dimensional image\nprocessing.\n\n\nFilters\n=======\n\n.. autosummary::\n   :toctree: generated/\n\n   convolve - Multi-dimensional convolution\n   convolve1d - 1-D convolution along the given axis\n   correlate - Multi-dimensional correlation\n   correlate1d - 1-D correlation along the given axis\n   gaussian_filter\n   gaussian_filter1d\n   gaussian_gradient_magnitude\n   gaussian_laplace\n   generic_filter - Multi-dimensional filter using a given function\n   generic_filter1d - 1-D generic filter along the given axis\n   generic_gradient_magnitude\n   generic_laplace\n   laplace - n-D Laplace filter based on approximate second derivatives\n   maximum_filter\n   maximum_filter1d\n   median_filter - Calculates a multi-dimensional median filter\n   minimum_filter\n   minimum_filter1d\n   percentile_filter - Calculates a multi-dimensional percentile filter\n   prewitt\n   rank_filter - Calculates a multi-dimensional rank filter\n   sobel\n   uniform_filter - Multi-dimensional uniform filter\n   uniform_filter1d - 1-D uniform filter along the given axis\n\nFourier filters\n===============\n\n.. autosummary::\n   :toctree: generated/\n\n   fourier_ellipsoid\n   fourier_gaussian\n   fourier_shift\n   fourier_uniform\n\nInterpolation\n=============\n\n.. autosummary::\n   :toctree: generated/\n\n   affine_transform - Apply an affine transformation\n   geometric_transform - Apply an arbritrary geometric transform\n   map_coordinates - Map input array to new coordinates by interpolation\n   rotate - Rotate an array\n   shift - Shift an array\n   spline_filter\n   spline_filter1d\n   zoom - Zoom an array\n\nMeasurements\n============\n\n.. autosummary::\n   :toctree: generated/\n\n   center_of_mass - The center of mass of the values of an array at labels\n   extrema - Min's and max's of an array at labels, with their positions\n   find_objects - Find objects in a labeled array\n   histogram - Histogram of the values of an array, optionally at labels\n   label - Label features in an array\n   labeled_comprehension\n   maximum\n   maximum_position\n   mean - Mean of the values of an array at labels\n   median\n   minimum\n   minimum_position\n   standard_deviation - Standard deviation of an n-D image array\n   sum - Sum of the values of the array\n   variance - Variance of the values of an n-D image array\n   watershed_ift\n\nMorphology\n==========\n\n.. autosummary::\n   :toctree: generated/\n\n   binary_closing\n   binary_dilation\n   binary_erosion\n   binary_fill_holes\n   binary_hit_or_miss\n   binary_opening\n   binary_propagation\n   black_tophat\n   distance_transform_bf\n   distance_transform_cdt\n   distance_transform_edt\n   generate_binary_structure\n   grey_closing\n   grey_dilation\n   grey_erosion\n   grey_opening\n   iterate_structure\n   morphological_gradient\n   morphological_laplace\n   white_tophat\n\nUtility\n=======\n\n.. autosummary::\n   :toctree: generated/\n\n   imread - Load an image from a file\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 161, 0))

# 'from scipy.ndimage.filters import ' statement (line 161)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126815 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 161, 0), 'scipy.ndimage.filters')

if (type(import_126815) is not StypyTypeError):

    if (import_126815 != 'pyd_module'):
        __import__(import_126815)
        sys_modules_126816 = sys.modules[import_126815]
        import_from_module(stypy.reporting.localization.Localization(__file__, 161, 0), 'scipy.ndimage.filters', sys_modules_126816.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 161, 0), __file__, sys_modules_126816, sys_modules_126816.module_type_store, module_type_store)
    else:
        from scipy.ndimage.filters import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 161, 0), 'scipy.ndimage.filters', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.ndimage.filters' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'scipy.ndimage.filters', import_126815)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 162, 0))

# 'from scipy.ndimage.fourier import ' statement (line 162)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126817 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 162, 0), 'scipy.ndimage.fourier')

if (type(import_126817) is not StypyTypeError):

    if (import_126817 != 'pyd_module'):
        __import__(import_126817)
        sys_modules_126818 = sys.modules[import_126817]
        import_from_module(stypy.reporting.localization.Localization(__file__, 162, 0), 'scipy.ndimage.fourier', sys_modules_126818.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 162, 0), __file__, sys_modules_126818, sys_modules_126818.module_type_store, module_type_store)
    else:
        from scipy.ndimage.fourier import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 162, 0), 'scipy.ndimage.fourier', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.ndimage.fourier' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 0), 'scipy.ndimage.fourier', import_126817)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 163, 0))

# 'from scipy.ndimage.interpolation import ' statement (line 163)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126819 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 163, 0), 'scipy.ndimage.interpolation')

if (type(import_126819) is not StypyTypeError):

    if (import_126819 != 'pyd_module'):
        __import__(import_126819)
        sys_modules_126820 = sys.modules[import_126819]
        import_from_module(stypy.reporting.localization.Localization(__file__, 163, 0), 'scipy.ndimage.interpolation', sys_modules_126820.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 163, 0), __file__, sys_modules_126820, sys_modules_126820.module_type_store, module_type_store)
    else:
        from scipy.ndimage.interpolation import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 163, 0), 'scipy.ndimage.interpolation', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.ndimage.interpolation' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'scipy.ndimage.interpolation', import_126819)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 164, 0))

# 'from scipy.ndimage.measurements import ' statement (line 164)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126821 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 164, 0), 'scipy.ndimage.measurements')

if (type(import_126821) is not StypyTypeError):

    if (import_126821 != 'pyd_module'):
        __import__(import_126821)
        sys_modules_126822 = sys.modules[import_126821]
        import_from_module(stypy.reporting.localization.Localization(__file__, 164, 0), 'scipy.ndimage.measurements', sys_modules_126822.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 164, 0), __file__, sys_modules_126822, sys_modules_126822.module_type_store, module_type_store)
    else:
        from scipy.ndimage.measurements import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 164, 0), 'scipy.ndimage.measurements', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.ndimage.measurements' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'scipy.ndimage.measurements', import_126821)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 165, 0))

# 'from scipy.ndimage.morphology import ' statement (line 165)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126823 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 165, 0), 'scipy.ndimage.morphology')

if (type(import_126823) is not StypyTypeError):

    if (import_126823 != 'pyd_module'):
        __import__(import_126823)
        sys_modules_126824 = sys.modules[import_126823]
        import_from_module(stypy.reporting.localization.Localization(__file__, 165, 0), 'scipy.ndimage.morphology', sys_modules_126824.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 165, 0), __file__, sys_modules_126824, sys_modules_126824.module_type_store, module_type_store)
    else:
        from scipy.ndimage.morphology import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 165, 0), 'scipy.ndimage.morphology', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.ndimage.morphology' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'scipy.ndimage.morphology', import_126823)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 166, 0))

# 'from scipy.ndimage.io import ' statement (line 166)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126825 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 166, 0), 'scipy.ndimage.io')

if (type(import_126825) is not StypyTypeError):

    if (import_126825 != 'pyd_module'):
        __import__(import_126825)
        sys_modules_126826 = sys.modules[import_126825]
        import_from_module(stypy.reporting.localization.Localization(__file__, 166, 0), 'scipy.ndimage.io', sys_modules_126826.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 166, 0), __file__, sys_modules_126826, sys_modules_126826.module_type_store, module_type_store)
    else:
        from scipy.ndimage.io import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 166, 0), 'scipy.ndimage.io', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.ndimage.io' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'scipy.ndimage.io', import_126825)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


# Assigning a Str to a Name (line 168):
str_126827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 14), 'str', '2.0')
# Assigning a type to the variable '__version__' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), '__version__', str_126827)

# Assigning a ListComp to a Name (line 170):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 170)
# Processing the call keyword arguments (line 170)
kwargs_126836 = {}
# Getting the type of 'dir' (line 170)
dir_126835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'dir', False)
# Calling dir(args, kwargs) (line 170)
dir_call_result_126837 = invoke(stypy.reporting.localization.Localization(__file__, 170, 22), dir_126835, *[], **kwargs_126836)

comprehension_126838 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 11), dir_call_result_126837)
# Assigning a type to the variable 's' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 's', comprehension_126838)


# Call to startswith(...): (line 170)
# Processing the call arguments (line 170)
str_126831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 48), 'str', '_')
# Processing the call keyword arguments (line 170)
kwargs_126832 = {}
# Getting the type of 's' (line 170)
s_126829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 170)
startswith_126830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 35), s_126829, 'startswith')
# Calling startswith(args, kwargs) (line 170)
startswith_call_result_126833 = invoke(stypy.reporting.localization.Localization(__file__, 170, 35), startswith_126830, *[str_126831], **kwargs_126832)

# Applying the 'not' unary operator (line 170)
result_not__126834 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 31), 'not', startswith_call_result_126833)

# Getting the type of 's' (line 170)
s_126828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 's')
list_126839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 11), list_126839, s_126828)
# Assigning a type to the variable '__all__' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), '__all__', list_126839)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 172, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 172)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_126840 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 172, 0), 'scipy._lib._testutils')

if (type(import_126840) is not StypyTypeError):

    if (import_126840 != 'pyd_module'):
        __import__(import_126840)
        sys_modules_126841 = sys.modules[import_126840]
        import_from_module(stypy.reporting.localization.Localization(__file__, 172, 0), 'scipy._lib._testutils', sys_modules_126841.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 172, 0), __file__, sys_modules_126841, sys_modules_126841.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 172, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'scipy._lib._testutils', import_126840)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


# Assigning a Call to a Name (line 173):

# Call to PytestTester(...): (line 173)
# Processing the call arguments (line 173)
# Getting the type of '__name__' (line 173)
name___126843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), '__name__', False)
# Processing the call keyword arguments (line 173)
kwargs_126844 = {}
# Getting the type of 'PytestTester' (line 173)
PytestTester_126842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 173)
PytestTester_call_result_126845 = invoke(stypy.reporting.localization.Localization(__file__, 173, 7), PytestTester_126842, *[name___126843], **kwargs_126844)

# Assigning a type to the variable 'test' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'test', PytestTester_call_result_126845)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 174, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

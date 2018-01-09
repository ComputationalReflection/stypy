
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ==================================================
3: Discrete Fourier transforms (:mod:`scipy.fftpack`)
4: ==================================================
5: 
6: Fast Fourier Transforms (FFTs)
7: ==============================
8: 
9: .. autosummary::
10:    :toctree: generated/
11: 
12:    fft - Fast (discrete) Fourier Transform (FFT)
13:    ifft - Inverse FFT
14:    fft2 - Two dimensional FFT
15:    ifft2 - Two dimensional inverse FFT
16:    fftn - n-dimensional FFT
17:    ifftn - n-dimensional inverse FFT
18:    rfft - FFT of strictly real-valued sequence
19:    irfft - Inverse of rfft
20:    dct - Discrete cosine transform
21:    idct - Inverse discrete cosine transform
22:    dctn - n-dimensional Discrete cosine transform
23:    idctn - n-dimensional Inverse discrete cosine transform
24:    dst - Discrete sine transform
25:    idst - Inverse discrete sine transform
26:    dstn - n-dimensional Discrete sine transform
27:    idstn - n-dimensional Inverse discrete sine transform
28: 
29: Differential and pseudo-differential operators
30: ==============================================
31: 
32: .. autosummary::
33:    :toctree: generated/
34: 
35:    diff - Differentiation and integration of periodic sequences
36:    tilbert - Tilbert transform:         cs_diff(x,h,h)
37:    itilbert - Inverse Tilbert transform: sc_diff(x,h,h)
38:    hilbert - Hilbert transform:         cs_diff(x,inf,inf)
39:    ihilbert - Inverse Hilbert transform: sc_diff(x,inf,inf)
40:    cs_diff - cosh/sinh pseudo-derivative of periodic sequences
41:    sc_diff - sinh/cosh pseudo-derivative of periodic sequences
42:    ss_diff - sinh/sinh pseudo-derivative of periodic sequences
43:    cc_diff - cosh/cosh pseudo-derivative of periodic sequences
44:    shift - Shift periodic sequences
45: 
46: Helper functions
47: ================
48: 
49: .. autosummary::
50:    :toctree: generated/
51: 
52:    fftshift - Shift the zero-frequency component to the center of the spectrum
53:    ifftshift - The inverse of `fftshift`
54:    fftfreq - Return the Discrete Fourier Transform sample frequencies
55:    rfftfreq - DFT sample frequencies (for usage with rfft, irfft)
56:    next_fast_len - Find the optimal length to zero-pad an FFT for speed
57: 
58: Note that ``fftshift``, ``ifftshift`` and ``fftfreq`` are numpy functions
59: exposed by ``fftpack``; importing them from ``numpy`` should be preferred.
60: 
61: Convolutions (:mod:`scipy.fftpack.convolve`)
62: ============================================
63: 
64: .. module:: scipy.fftpack.convolve
65: 
66: .. autosummary::
67:    :toctree: generated/
68: 
69:    convolve
70:    convolve_z
71:    init_convolution_kernel
72:    destroy_convolve_cache
73: 
74: '''
75: 
76: # List of possibly useful functions in scipy.fftpack._fftpack:
77: #   drfft
78: #   zfft
79: #   zrfft
80: #   zfftnd
81: #   destroy_drfft_cache
82: #   destroy_zfft_cache
83: #   destroy_zfftnd_cache
84: 
85: from __future__ import division, print_function, absolute_import
86: 
87: 
88: __all__ = ['fft','ifft','fftn','ifftn','rfft','irfft',
89:            'fft2','ifft2',
90:            'diff',
91:            'tilbert','itilbert','hilbert','ihilbert',
92:            'sc_diff','cs_diff','cc_diff','ss_diff',
93:            'shift',
94:            'fftfreq', 'rfftfreq',
95:            'fftshift', 'ifftshift',
96:            'next_fast_len',
97:            ]
98: 
99: from .basic import *
100: from .pseudo_diffs import *
101: from .helper import *
102: 
103: from numpy.dual import register_func
104: for k in ['fft', 'ifft', 'fftn', 'ifftn', 'fft2', 'ifft2']:
105:     register_func(k, eval(k))
106: del k, register_func
107: 
108: from .realtransforms import *
109: __all__.extend(['dct', 'idct', 'dst', 'idst', 'dctn', 'idctn', 'dstn',
110:                 'idstn'])
111: 
112: from scipy._lib._testutils import PytestTester
113: test = PytestTester(__name__)
114: del PytestTester
115: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_18635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', '\n==================================================\nDiscrete Fourier transforms (:mod:`scipy.fftpack`)\n==================================================\n\nFast Fourier Transforms (FFTs)\n==============================\n\n.. autosummary::\n   :toctree: generated/\n\n   fft - Fast (discrete) Fourier Transform (FFT)\n   ifft - Inverse FFT\n   fft2 - Two dimensional FFT\n   ifft2 - Two dimensional inverse FFT\n   fftn - n-dimensional FFT\n   ifftn - n-dimensional inverse FFT\n   rfft - FFT of strictly real-valued sequence\n   irfft - Inverse of rfft\n   dct - Discrete cosine transform\n   idct - Inverse discrete cosine transform\n   dctn - n-dimensional Discrete cosine transform\n   idctn - n-dimensional Inverse discrete cosine transform\n   dst - Discrete sine transform\n   idst - Inverse discrete sine transform\n   dstn - n-dimensional Discrete sine transform\n   idstn - n-dimensional Inverse discrete sine transform\n\nDifferential and pseudo-differential operators\n==============================================\n\n.. autosummary::\n   :toctree: generated/\n\n   diff - Differentiation and integration of periodic sequences\n   tilbert - Tilbert transform:         cs_diff(x,h,h)\n   itilbert - Inverse Tilbert transform: sc_diff(x,h,h)\n   hilbert - Hilbert transform:         cs_diff(x,inf,inf)\n   ihilbert - Inverse Hilbert transform: sc_diff(x,inf,inf)\n   cs_diff - cosh/sinh pseudo-derivative of periodic sequences\n   sc_diff - sinh/cosh pseudo-derivative of periodic sequences\n   ss_diff - sinh/sinh pseudo-derivative of periodic sequences\n   cc_diff - cosh/cosh pseudo-derivative of periodic sequences\n   shift - Shift periodic sequences\n\nHelper functions\n================\n\n.. autosummary::\n   :toctree: generated/\n\n   fftshift - Shift the zero-frequency component to the center of the spectrum\n   ifftshift - The inverse of `fftshift`\n   fftfreq - Return the Discrete Fourier Transform sample frequencies\n   rfftfreq - DFT sample frequencies (for usage with rfft, irfft)\n   next_fast_len - Find the optimal length to zero-pad an FFT for speed\n\nNote that ``fftshift``, ``ifftshift`` and ``fftfreq`` are numpy functions\nexposed by ``fftpack``; importing them from ``numpy`` should be preferred.\n\nConvolutions (:mod:`scipy.fftpack.convolve`)\n============================================\n\n.. module:: scipy.fftpack.convolve\n\n.. autosummary::\n   :toctree: generated/\n\n   convolve\n   convolve_z\n   init_convolution_kernel\n   destroy_convolve_cache\n\n')

# Assigning a List to a Name (line 88):
__all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft', 'fft2', 'ifft2', 'diff', 'tilbert', 'itilbert', 'hilbert', 'ihilbert', 'sc_diff', 'cs_diff', 'cc_diff', 'ss_diff', 'shift', 'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift', 'next_fast_len']
module_type_store.set_exportable_members(['fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft', 'fft2', 'ifft2', 'diff', 'tilbert', 'itilbert', 'hilbert', 'ihilbert', 'sc_diff', 'cs_diff', 'cc_diff', 'ss_diff', 'shift', 'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift', 'next_fast_len'])

# Obtaining an instance of the builtin type 'list' (line 88)
list_18636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 88)
# Adding element type (line 88)
str_18637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 11), 'str', 'fft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18637)
# Adding element type (line 88)
str_18638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 17), 'str', 'ifft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18638)
# Adding element type (line 88)
str_18639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'str', 'fftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18639)
# Adding element type (line 88)
str_18640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'str', 'ifftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18640)
# Adding element type (line 88)
str_18641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 39), 'str', 'rfft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18641)
# Adding element type (line 88)
str_18642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 46), 'str', 'irfft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18642)
# Adding element type (line 88)
str_18643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 11), 'str', 'fft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18643)
# Adding element type (line 88)
str_18644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'str', 'ifft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18644)
# Adding element type (line 88)
str_18645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 11), 'str', 'diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18645)
# Adding element type (line 88)
str_18646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 11), 'str', 'tilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18646)
# Adding element type (line 88)
str_18647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'str', 'itilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18647)
# Adding element type (line 88)
str_18648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'str', 'hilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18648)
# Adding element type (line 88)
str_18649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 42), 'str', 'ihilbert')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18649)
# Adding element type (line 88)
str_18650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 11), 'str', 'sc_diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18650)
# Adding element type (line 88)
str_18651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 21), 'str', 'cs_diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18651)
# Adding element type (line 88)
str_18652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 31), 'str', 'cc_diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18652)
# Adding element type (line 88)
str_18653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 41), 'str', 'ss_diff')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18653)
# Adding element type (line 88)
str_18654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 11), 'str', 'shift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18654)
# Adding element type (line 88)
str_18655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 11), 'str', 'fftfreq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18655)
# Adding element type (line 88)
str_18656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'str', 'rfftfreq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18656)
# Adding element type (line 88)
str_18657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'str', 'fftshift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18657)
# Adding element type (line 88)
str_18658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'str', 'ifftshift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18658)
# Adding element type (line 88)
str_18659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'str', 'next_fast_len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 10), list_18636, str_18659)

# Assigning a type to the variable '__all__' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), '__all__', list_18636)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 99, 0))

# 'from scipy.fftpack.basic import ' statement (line 99)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_18660 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'scipy.fftpack.basic')

if (type(import_18660) is not StypyTypeError):

    if (import_18660 != 'pyd_module'):
        __import__(import_18660)
        sys_modules_18661 = sys.modules[import_18660]
        import_from_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'scipy.fftpack.basic', sys_modules_18661.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 99, 0), __file__, sys_modules_18661, sys_modules_18661.module_type_store, module_type_store)
    else:
        from scipy.fftpack.basic import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'scipy.fftpack.basic', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.fftpack.basic' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'scipy.fftpack.basic', import_18660)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 100, 0))

# 'from scipy.fftpack.pseudo_diffs import ' statement (line 100)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_18662 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.fftpack.pseudo_diffs')

if (type(import_18662) is not StypyTypeError):

    if (import_18662 != 'pyd_module'):
        __import__(import_18662)
        sys_modules_18663 = sys.modules[import_18662]
        import_from_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.fftpack.pseudo_diffs', sys_modules_18663.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 100, 0), __file__, sys_modules_18663, sys_modules_18663.module_type_store, module_type_store)
    else:
        from scipy.fftpack.pseudo_diffs import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.fftpack.pseudo_diffs', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.fftpack.pseudo_diffs' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'scipy.fftpack.pseudo_diffs', import_18662)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 101, 0))

# 'from scipy.fftpack.helper import ' statement (line 101)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_18664 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.fftpack.helper')

if (type(import_18664) is not StypyTypeError):

    if (import_18664 != 'pyd_module'):
        __import__(import_18664)
        sys_modules_18665 = sys.modules[import_18664]
        import_from_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.fftpack.helper', sys_modules_18665.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 101, 0), __file__, sys_modules_18665, sys_modules_18665.module_type_store, module_type_store)
    else:
        from scipy.fftpack.helper import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.fftpack.helper', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.fftpack.helper' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'scipy.fftpack.helper', import_18664)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 103, 0))

# 'from numpy.dual import register_func' statement (line 103)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_18666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'numpy.dual')

if (type(import_18666) is not StypyTypeError):

    if (import_18666 != 'pyd_module'):
        __import__(import_18666)
        sys_modules_18667 = sys.modules[import_18666]
        import_from_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'numpy.dual', sys_modules_18667.module_type_store, module_type_store, ['register_func'])
        nest_module(stypy.reporting.localization.Localization(__file__, 103, 0), __file__, sys_modules_18667, sys_modules_18667.module_type_store, module_type_store)
    else:
        from numpy.dual import register_func

        import_from_module(stypy.reporting.localization.Localization(__file__, 103, 0), 'numpy.dual', None, module_type_store, ['register_func'], [register_func])

else:
    # Assigning a type to the variable 'numpy.dual' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'numpy.dual', import_18666)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')



# Obtaining an instance of the builtin type 'list' (line 104)
list_18668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 104)
# Adding element type (line 104)
str_18669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 10), 'str', 'fft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_18668, str_18669)
# Adding element type (line 104)
str_18670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'str', 'ifft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_18668, str_18670)
# Adding element type (line 104)
str_18671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'str', 'fftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_18668, str_18671)
# Adding element type (line 104)
str_18672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 33), 'str', 'ifftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_18668, str_18672)
# Adding element type (line 104)
str_18673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 42), 'str', 'fft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_18668, str_18673)
# Adding element type (line 104)
str_18674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 50), 'str', 'ifft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_18668, str_18674)

# Testing the type of a for loop iterable (line 104)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 0), list_18668)
# Getting the type of the for loop variable (line 104)
for_loop_var_18675 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 0), list_18668)
# Assigning a type to the variable 'k' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'k', for_loop_var_18675)
# SSA begins for a for statement (line 104)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Call to register_func(...): (line 105)
# Processing the call arguments (line 105)
# Getting the type of 'k' (line 105)
k_18677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'k', False)

# Call to eval(...): (line 105)
# Processing the call arguments (line 105)
# Getting the type of 'k' (line 105)
k_18679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'k', False)
# Processing the call keyword arguments (line 105)
kwargs_18680 = {}
# Getting the type of 'eval' (line 105)
eval_18678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'eval', False)
# Calling eval(args, kwargs) (line 105)
eval_call_result_18681 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), eval_18678, *[k_18679], **kwargs_18680)

# Processing the call keyword arguments (line 105)
kwargs_18682 = {}
# Getting the type of 'register_func' (line 105)
register_func_18676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'register_func', False)
# Calling register_func(args, kwargs) (line 105)
register_func_call_result_18683 = invoke(stypy.reporting.localization.Localization(__file__, 105, 4), register_func_18676, *[k_18677, eval_call_result_18681], **kwargs_18682)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 106, 0), module_type_store, 'k')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 106, 0), module_type_store, 'register_func')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 108, 0))

# 'from scipy.fftpack.realtransforms import ' statement (line 108)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_18684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 108, 0), 'scipy.fftpack.realtransforms')

if (type(import_18684) is not StypyTypeError):

    if (import_18684 != 'pyd_module'):
        __import__(import_18684)
        sys_modules_18685 = sys.modules[import_18684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 108, 0), 'scipy.fftpack.realtransforms', sys_modules_18685.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 108, 0), __file__, sys_modules_18685, sys_modules_18685.module_type_store, module_type_store)
    else:
        from scipy.fftpack.realtransforms import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 108, 0), 'scipy.fftpack.realtransforms', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.fftpack.realtransforms' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'scipy.fftpack.realtransforms', import_18684)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')


# Call to extend(...): (line 109)
# Processing the call arguments (line 109)

# Obtaining an instance of the builtin type 'list' (line 109)
list_18688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 109)
# Adding element type (line 109)
str_18689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 16), 'str', 'dct')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_18688, str_18689)
# Adding element type (line 109)
str_18690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'str', 'idct')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_18688, str_18690)
# Adding element type (line 109)
str_18691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 31), 'str', 'dst')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_18688, str_18691)
# Adding element type (line 109)
str_18692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 38), 'str', 'idst')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_18688, str_18692)
# Adding element type (line 109)
str_18693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 46), 'str', 'dctn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_18688, str_18693)
# Adding element type (line 109)
str_18694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 54), 'str', 'idctn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_18688, str_18694)
# Adding element type (line 109)
str_18695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 63), 'str', 'dstn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_18688, str_18695)
# Adding element type (line 109)
str_18696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 16), 'str', 'idstn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 15), list_18688, str_18696)

# Processing the call keyword arguments (line 109)
kwargs_18697 = {}
# Getting the type of '__all__' (line 109)
all___18686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), '__all__', False)
# Obtaining the member 'extend' of a type (line 109)
extend_18687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 0), all___18686, 'extend')
# Calling extend(args, kwargs) (line 109)
extend_call_result_18698 = invoke(stypy.reporting.localization.Localization(__file__, 109, 0), extend_18687, *[list_18688], **kwargs_18697)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 112, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 112)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_18699 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 112, 0), 'scipy._lib._testutils')

if (type(import_18699) is not StypyTypeError):

    if (import_18699 != 'pyd_module'):
        __import__(import_18699)
        sys_modules_18700 = sys.modules[import_18699]
        import_from_module(stypy.reporting.localization.Localization(__file__, 112, 0), 'scipy._lib._testutils', sys_modules_18700.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 112, 0), __file__, sys_modules_18700, sys_modules_18700.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 112, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'scipy._lib._testutils', import_18699)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')


# Assigning a Call to a Name (line 113):

# Call to PytestTester(...): (line 113)
# Processing the call arguments (line 113)
# Getting the type of '__name__' (line 113)
name___18702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), '__name__', False)
# Processing the call keyword arguments (line 113)
kwargs_18703 = {}
# Getting the type of 'PytestTester' (line 113)
PytestTester_18701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 113)
PytestTester_call_result_18704 = invoke(stypy.reporting.localization.Localization(__file__, 113, 7), PytestTester_18701, *[name___18702], **kwargs_18703)

# Assigning a type to the variable 'test' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'test', PytestTester_call_result_18704)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 114, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

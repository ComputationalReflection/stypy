
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Author: Eric Larson
2: # 2014
3: 
4: '''Tools for MLS generation'''
5: 
6: import numpy as np
7: 
8: from ._max_len_seq_inner import _max_len_seq_inner
9: 
10: __all__ = ['max_len_seq']
11: 
12: 
13: # These are definitions of linear shift register taps for use in max_len_seq()
14: _mls_taps = {2: [1], 3: [2], 4: [3], 5: [3], 6: [5], 7: [6], 8: [7, 6, 1],
15:              9: [5], 10: [7], 11: [9], 12: [11, 10, 4], 13: [12, 11, 8],
16:              14: [13, 12, 2], 15: [14], 16: [15, 13, 4], 17: [14],
17:              18: [11], 19: [18, 17, 14], 20: [17], 21: [19], 22: [21],
18:              23: [18], 24: [23, 22, 17], 25: [22], 26: [25, 24, 20],
19:              27: [26, 25, 22], 28: [25], 29: [27], 30: [29, 28, 7],
20:              31: [28], 32: [31, 30, 10]}
21: 
22: def max_len_seq(nbits, state=None, length=None, taps=None):
23:     '''
24:     Maximum length sequence (MLS) generator.
25: 
26:     Parameters
27:     ----------
28:     nbits : int
29:         Number of bits to use. Length of the resulting sequence will
30:         be ``(2**nbits) - 1``. Note that generating long sequences
31:         (e.g., greater than ``nbits == 16``) can take a long time.
32:     state : array_like, optional
33:         If array, must be of length ``nbits``, and will be cast to binary
34:         (bool) representation. If None, a seed of ones will be used,
35:         producing a repeatable representation. If ``state`` is all
36:         zeros, an error is raised as this is invalid. Default: None.
37:     length : int, optional
38:         Number of samples to compute. If None, the entire length
39:         ``(2**nbits) - 1`` is computed.
40:     taps : array_like, optional
41:         Polynomial taps to use (e.g., ``[7, 6, 1]`` for an 8-bit sequence).
42:         If None, taps will be automatically selected (for up to
43:         ``nbits == 32``).
44: 
45:     Returns
46:     -------
47:     seq : array
48:         Resulting MLS sequence of 0's and 1's.
49:     state : array
50:         The final state of the shift register.
51: 
52:     Notes
53:     -----
54:     The algorithm for MLS generation is generically described in:
55: 
56:         https://en.wikipedia.org/wiki/Maximum_length_sequence
57: 
58:     The default values for taps are specifically taken from the first
59:     option listed for each value of ``nbits`` in:
60: 
61:         http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm
62: 
63:     .. versionadded:: 0.15.0
64: 
65:     Examples
66:     --------
67:     MLS uses binary convention:
68: 
69:     >>> from scipy.signal import max_len_seq
70:     >>> max_len_seq(4)[0]
71:     array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=int8)
72: 
73:     MLS has a white spectrum (except for DC):
74: 
75:     >>> import matplotlib.pyplot as plt
76:     >>> from numpy.fft import fft, ifft, fftshift, fftfreq
77:     >>> seq = max_len_seq(6)[0]*2-1  # +1 and -1
78:     >>> spec = fft(seq)
79:     >>> N = len(seq)
80:     >>> plt.plot(fftshift(fftfreq(N)), fftshift(np.abs(spec)), '.-')
81:     >>> plt.margins(0.1, 0.1)
82:     >>> plt.grid(True)
83:     >>> plt.show()
84: 
85:     Circular autocorrelation of MLS is an impulse:
86: 
87:     >>> acorrcirc = ifft(spec * np.conj(spec)).real
88:     >>> plt.figure()
89:     >>> plt.plot(np.arange(-N/2+1, N/2+1), fftshift(acorrcirc), '.-')
90:     >>> plt.margins(0.1, 0.1)
91:     >>> plt.grid(True)
92:     >>> plt.show()
93: 
94:     Linear autocorrelation of MLS is approximately an impulse:
95: 
96:     >>> acorr = np.correlate(seq, seq, 'full')
97:     >>> plt.figure()
98:     >>> plt.plot(np.arange(-N+1, N), acorr, '.-')
99:     >>> plt.margins(0.1, 0.1)
100:     >>> plt.grid(True)
101:     >>> plt.show()
102: 
103:     '''
104:     if taps is None:
105:         if nbits not in _mls_taps:
106:             known_taps = np.array(list(_mls_taps.keys()))
107:             raise ValueError('nbits must be between %s and %s if taps is None'
108:                              % (known_taps.min(), known_taps.max()))
109:         taps = np.array(_mls_taps[nbits], np.intp)
110:     else:
111:         taps = np.unique(np.array(taps, np.intp))[::-1]
112:         if np.any(taps < 0) or np.any(taps > nbits) or taps.size < 1:
113:             raise ValueError('taps must be non-empty with values between '
114:                              'zero and nbits (inclusive)')
115:         taps = np.ascontiguousarray(taps)  # needed for Cython
116:     n_max = (2**nbits) - 1
117:     if length is None:
118:         length = n_max
119:     else:
120:         length = int(length)
121:         if length < 0:
122:             raise ValueError('length must be greater than or equal to 0')
123:     # We use int8 instead of bool here because numpy arrays of bools
124:     # don't seem to work nicely with Cython
125:     if state is None:
126:         state = np.ones(nbits, dtype=np.int8, order='c')
127:     else:
128:         # makes a copy if need be, ensuring it's 0's and 1's
129:         state = np.array(state, dtype=bool, order='c').astype(np.int8)
130:     if state.ndim != 1 or state.size != nbits:
131:         raise ValueError('state must be a 1-dimensional array of size nbits')
132:     if np.all(state == 0):
133:         raise ValueError('state must not be all zeros')
134: 
135:     seq = np.empty(length, dtype=np.int8, order='c')
136:     state = _max_len_seq_inner(taps, state, nbits, length, seq)
137:     return seq, state
138: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_287110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 0), 'str', 'Tools for MLS generation')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_287111 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_287111) is not StypyTypeError):

    if (import_287111 != 'pyd_module'):
        __import__(import_287111)
        sys_modules_287112 = sys.modules[import_287111]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_287112.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_287111)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.signal._max_len_seq_inner import _max_len_seq_inner' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_287113 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._max_len_seq_inner')

if (type(import_287113) is not StypyTypeError):

    if (import_287113 != 'pyd_module'):
        __import__(import_287113)
        sys_modules_287114 = sys.modules[import_287113]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._max_len_seq_inner', sys_modules_287114.module_type_store, module_type_store, ['_max_len_seq_inner'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_287114, sys_modules_287114.module_type_store, module_type_store)
    else:
        from scipy.signal._max_len_seq_inner import _max_len_seq_inner

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._max_len_seq_inner', None, module_type_store, ['_max_len_seq_inner'], [_max_len_seq_inner])

else:
    # Assigning a type to the variable 'scipy.signal._max_len_seq_inner' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal._max_len_seq_inner', import_287113)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 10):
__all__ = ['max_len_seq']
module_type_store.set_exportable_members(['max_len_seq'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_287115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_287116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'max_len_seq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_287115, str_287116)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_287115)

# Assigning a Dict to a Name (line 14):

# Obtaining an instance of the builtin type 'dict' (line 14)
dict_287117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 14)
# Adding element type (key, value) (line 14)
int_287118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 13), 'int')

# Obtaining an instance of the builtin type 'list' (line 14)
list_287119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_287120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), list_287119, int_287120)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287118, list_287119))
# Adding element type (key, value) (line 14)
int_287121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 21), 'int')

# Obtaining an instance of the builtin type 'list' (line 14)
list_287122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_287123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 24), list_287122, int_287123)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287121, list_287122))
# Adding element type (key, value) (line 14)
int_287124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'int')

# Obtaining an instance of the builtin type 'list' (line 14)
list_287125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_287126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 32), list_287125, int_287126)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287124, list_287125))
# Adding element type (key, value) (line 14)
int_287127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 37), 'int')

# Obtaining an instance of the builtin type 'list' (line 14)
list_287128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_287129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 40), list_287128, int_287129)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287127, list_287128))
# Adding element type (key, value) (line 14)
int_287130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 45), 'int')

# Obtaining an instance of the builtin type 'list' (line 14)
list_287131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 48), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_287132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 48), list_287131, int_287132)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287130, list_287131))
# Adding element type (key, value) (line 14)
int_287133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 53), 'int')

# Obtaining an instance of the builtin type 'list' (line 14)
list_287134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 56), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_287135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 57), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 56), list_287134, int_287135)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287133, list_287134))
# Adding element type (key, value) (line 14)
int_287136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 61), 'int')

# Obtaining an instance of the builtin type 'list' (line 14)
list_287137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 64), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
int_287138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 64), list_287137, int_287138)
# Adding element type (line 14)
int_287139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 64), list_287137, int_287139)
# Adding element type (line 14)
int_287140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 71), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 64), list_287137, int_287140)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287136, list_287137))
# Adding element type (key, value) (line 14)
int_287141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')

# Obtaining an instance of the builtin type 'list' (line 15)
list_287142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_287143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 16), list_287142, int_287143)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287141, list_287142))
# Adding element type (key, value) (line 14)
int_287144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'int')

# Obtaining an instance of the builtin type 'list' (line 15)
list_287145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_287146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_287145, int_287146)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287144, list_287145))
# Adding element type (key, value) (line 14)
int_287147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 30), 'int')

# Obtaining an instance of the builtin type 'list' (line 15)
list_287148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_287149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 34), list_287148, int_287149)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287147, list_287148))
# Adding element type (key, value) (line 14)
int_287150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 39), 'int')

# Obtaining an instance of the builtin type 'list' (line 15)
list_287151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 43), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_287152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 44), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 43), list_287151, int_287152)
# Adding element type (line 15)
int_287153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 43), list_287151, int_287153)
# Adding element type (line 15)
int_287154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 43), list_287151, int_287154)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287150, list_287151))
# Adding element type (key, value) (line 14)
int_287155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 56), 'int')

# Obtaining an instance of the builtin type 'list' (line 15)
list_287156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 60), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
int_287157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 61), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 60), list_287156, int_287157)
# Adding element type (line 15)
int_287158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 65), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 60), list_287156, int_287158)
# Adding element type (line 15)
int_287159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 69), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 60), list_287156, int_287159)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287155, list_287156))
# Adding element type (key, value) (line 14)
int_287160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')

# Obtaining an instance of the builtin type 'list' (line 16)
list_287161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_287162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 17), list_287161, int_287162)
# Adding element type (line 16)
int_287163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 17), list_287161, int_287163)
# Adding element type (line 16)
int_287164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 17), list_287161, int_287164)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287160, list_287161))
# Adding element type (key, value) (line 14)
int_287165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 30), 'int')

# Obtaining an instance of the builtin type 'list' (line 16)
list_287166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_287167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 34), list_287166, int_287167)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287165, list_287166))
# Adding element type (key, value) (line 14)
int_287168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 40), 'int')

# Obtaining an instance of the builtin type 'list' (line 16)
list_287169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 44), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_287170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 44), list_287169, int_287170)
# Adding element type (line 16)
int_287171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 44), list_287169, int_287171)
# Adding element type (line 16)
int_287172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 44), list_287169, int_287172)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287168, list_287169))
# Adding element type (key, value) (line 14)
int_287173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 57), 'int')

# Obtaining an instance of the builtin type 'list' (line 16)
list_287174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 61), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_287175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 62), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 61), list_287174, int_287175)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287173, list_287174))
# Adding element type (key, value) (line 14)
int_287176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'int')

# Obtaining an instance of the builtin type 'list' (line 17)
list_287177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_287178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 17), list_287177, int_287178)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287176, list_287177))
# Adding element type (key, value) (line 14)
int_287179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'int')

# Obtaining an instance of the builtin type 'list' (line 17)
list_287180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_287181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 27), list_287180, int_287181)
# Adding element type (line 17)
int_287182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 27), list_287180, int_287182)
# Adding element type (line 17)
int_287183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 27), list_287180, int_287183)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287179, list_287180))
# Adding element type (key, value) (line 14)
int_287184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'int')

# Obtaining an instance of the builtin type 'list' (line 17)
list_287185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 45), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_287186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 45), list_287185, int_287186)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287184, list_287185))
# Adding element type (key, value) (line 14)
int_287187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 51), 'int')

# Obtaining an instance of the builtin type 'list' (line 17)
list_287188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 55), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_287189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 55), list_287188, int_287189)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287187, list_287188))
# Adding element type (key, value) (line 14)
int_287190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 61), 'int')

# Obtaining an instance of the builtin type 'list' (line 17)
list_287191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 65), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_287192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 65), list_287191, int_287192)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287190, list_287191))
# Adding element type (key, value) (line 14)
int_287193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'int')

# Obtaining an instance of the builtin type 'list' (line 18)
list_287194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
int_287195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 17), list_287194, int_287195)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287193, list_287194))
# Adding element type (key, value) (line 14)
int_287196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 23), 'int')

# Obtaining an instance of the builtin type 'list' (line 18)
list_287197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
int_287198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 27), list_287197, int_287198)
# Adding element type (line 18)
int_287199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 27), list_287197, int_287199)
# Adding element type (line 18)
int_287200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 27), list_287197, int_287200)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287196, list_287197))
# Adding element type (key, value) (line 14)
int_287201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'int')

# Obtaining an instance of the builtin type 'list' (line 18)
list_287202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 45), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
int_287203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 45), list_287202, int_287203)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287201, list_287202))
# Adding element type (key, value) (line 14)
int_287204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 51), 'int')

# Obtaining an instance of the builtin type 'list' (line 18)
list_287205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 55), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
int_287206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 55), list_287205, int_287206)
# Adding element type (line 18)
int_287207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 55), list_287205, int_287207)
# Adding element type (line 18)
int_287208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 55), list_287205, int_287208)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287204, list_287205))
# Adding element type (key, value) (line 14)
int_287209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'int')

# Obtaining an instance of the builtin type 'list' (line 19)
list_287210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_287211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 17), list_287210, int_287211)
# Adding element type (line 19)
int_287212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 17), list_287210, int_287212)
# Adding element type (line 19)
int_287213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 17), list_287210, int_287213)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287209, list_287210))
# Adding element type (key, value) (line 14)
int_287214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')

# Obtaining an instance of the builtin type 'list' (line 19)
list_287215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_287216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 35), list_287215, int_287216)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287214, list_287215))
# Adding element type (key, value) (line 14)
int_287217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'int')

# Obtaining an instance of the builtin type 'list' (line 19)
list_287218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 45), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_287219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 45), list_287218, int_287219)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287217, list_287218))
# Adding element type (key, value) (line 14)
int_287220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 51), 'int')

# Obtaining an instance of the builtin type 'list' (line 19)
list_287221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 55), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
int_287222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 55), list_287221, int_287222)
# Adding element type (line 19)
int_287223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 55), list_287221, int_287223)
# Adding element type (line 19)
int_287224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 55), list_287221, int_287224)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287220, list_287221))
# Adding element type (key, value) (line 14)
int_287225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 13), 'int')

# Obtaining an instance of the builtin type 'list' (line 20)
list_287226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
int_287227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 17), list_287226, int_287227)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287225, list_287226))
# Adding element type (key, value) (line 14)
int_287228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'int')

# Obtaining an instance of the builtin type 'list' (line 20)
list_287229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
int_287230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 27), list_287229, int_287230)
# Adding element type (line 20)
int_287231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 27), list_287229, int_287231)
# Adding element type (line 20)
int_287232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 27), list_287229, int_287232)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), dict_287117, (int_287228, list_287229))

# Assigning a type to the variable '_mls_taps' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '_mls_taps', dict_287117)

@norecursion
def max_len_seq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 22)
    None_287233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 29), 'None')
    # Getting the type of 'None' (line 22)
    None_287234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 42), 'None')
    # Getting the type of 'None' (line 22)
    None_287235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 53), 'None')
    defaults = [None_287233, None_287234, None_287235]
    # Create a new context for function 'max_len_seq'
    module_type_store = module_type_store.open_function_context('max_len_seq', 22, 0, False)
    
    # Passed parameters checking function
    max_len_seq.stypy_localization = localization
    max_len_seq.stypy_type_of_self = None
    max_len_seq.stypy_type_store = module_type_store
    max_len_seq.stypy_function_name = 'max_len_seq'
    max_len_seq.stypy_param_names_list = ['nbits', 'state', 'length', 'taps']
    max_len_seq.stypy_varargs_param_name = None
    max_len_seq.stypy_kwargs_param_name = None
    max_len_seq.stypy_call_defaults = defaults
    max_len_seq.stypy_call_varargs = varargs
    max_len_seq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'max_len_seq', ['nbits', 'state', 'length', 'taps'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'max_len_seq', localization, ['nbits', 'state', 'length', 'taps'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'max_len_seq(...)' code ##################

    str_287236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', "\n    Maximum length sequence (MLS) generator.\n\n    Parameters\n    ----------\n    nbits : int\n        Number of bits to use. Length of the resulting sequence will\n        be ``(2**nbits) - 1``. Note that generating long sequences\n        (e.g., greater than ``nbits == 16``) can take a long time.\n    state : array_like, optional\n        If array, must be of length ``nbits``, and will be cast to binary\n        (bool) representation. If None, a seed of ones will be used,\n        producing a repeatable representation. If ``state`` is all\n        zeros, an error is raised as this is invalid. Default: None.\n    length : int, optional\n        Number of samples to compute. If None, the entire length\n        ``(2**nbits) - 1`` is computed.\n    taps : array_like, optional\n        Polynomial taps to use (e.g., ``[7, 6, 1]`` for an 8-bit sequence).\n        If None, taps will be automatically selected (for up to\n        ``nbits == 32``).\n\n    Returns\n    -------\n    seq : array\n        Resulting MLS sequence of 0's and 1's.\n    state : array\n        The final state of the shift register.\n\n    Notes\n    -----\n    The algorithm for MLS generation is generically described in:\n\n        https://en.wikipedia.org/wiki/Maximum_length_sequence\n\n    The default values for taps are specifically taken from the first\n    option listed for each value of ``nbits`` in:\n\n        http://www.newwaveinstruments.com/resources/articles/m_sequence_linear_feedback_shift_register_lfsr.htm\n\n    .. versionadded:: 0.15.0\n\n    Examples\n    --------\n    MLS uses binary convention:\n\n    >>> from scipy.signal import max_len_seq\n    >>> max_len_seq(4)[0]\n    array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=int8)\n\n    MLS has a white spectrum (except for DC):\n\n    >>> import matplotlib.pyplot as plt\n    >>> from numpy.fft import fft, ifft, fftshift, fftfreq\n    >>> seq = max_len_seq(6)[0]*2-1  # +1 and -1\n    >>> spec = fft(seq)\n    >>> N = len(seq)\n    >>> plt.plot(fftshift(fftfreq(N)), fftshift(np.abs(spec)), '.-')\n    >>> plt.margins(0.1, 0.1)\n    >>> plt.grid(True)\n    >>> plt.show()\n\n    Circular autocorrelation of MLS is an impulse:\n\n    >>> acorrcirc = ifft(spec * np.conj(spec)).real\n    >>> plt.figure()\n    >>> plt.plot(np.arange(-N/2+1, N/2+1), fftshift(acorrcirc), '.-')\n    >>> plt.margins(0.1, 0.1)\n    >>> plt.grid(True)\n    >>> plt.show()\n\n    Linear autocorrelation of MLS is approximately an impulse:\n\n    >>> acorr = np.correlate(seq, seq, 'full')\n    >>> plt.figure()\n    >>> plt.plot(np.arange(-N+1, N), acorr, '.-')\n    >>> plt.margins(0.1, 0.1)\n    >>> plt.grid(True)\n    >>> plt.show()\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 104)
    # Getting the type of 'taps' (line 104)
    taps_287237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 'taps')
    # Getting the type of 'None' (line 104)
    None_287238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'None')
    
    (may_be_287239, more_types_in_union_287240) = may_be_none(taps_287237, None_287238)

    if may_be_287239:

        if more_types_in_union_287240:
            # Runtime conditional SSA (line 104)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'nbits' (line 105)
        nbits_287241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'nbits')
        # Getting the type of '_mls_taps' (line 105)
        _mls_taps_287242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), '_mls_taps')
        # Applying the binary operator 'notin' (line 105)
        result_contains_287243 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), 'notin', nbits_287241, _mls_taps_287242)
        
        # Testing the type of an if condition (line 105)
        if_condition_287244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), result_contains_287243)
        # Assigning a type to the variable 'if_condition_287244' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_287244', if_condition_287244)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 106):
        
        # Call to array(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Call to list(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Call to keys(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_287250 = {}
        # Getting the type of '_mls_taps' (line 106)
        _mls_taps_287248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), '_mls_taps', False)
        # Obtaining the member 'keys' of a type (line 106)
        keys_287249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 39), _mls_taps_287248, 'keys')
        # Calling keys(args, kwargs) (line 106)
        keys_call_result_287251 = invoke(stypy.reporting.localization.Localization(__file__, 106, 39), keys_287249, *[], **kwargs_287250)
        
        # Processing the call keyword arguments (line 106)
        kwargs_287252 = {}
        # Getting the type of 'list' (line 106)
        list_287247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'list', False)
        # Calling list(args, kwargs) (line 106)
        list_call_result_287253 = invoke(stypy.reporting.localization.Localization(__file__, 106, 34), list_287247, *[keys_call_result_287251], **kwargs_287252)
        
        # Processing the call keyword arguments (line 106)
        kwargs_287254 = {}
        # Getting the type of 'np' (line 106)
        np_287245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'np', False)
        # Obtaining the member 'array' of a type (line 106)
        array_287246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 25), np_287245, 'array')
        # Calling array(args, kwargs) (line 106)
        array_call_result_287255 = invoke(stypy.reporting.localization.Localization(__file__, 106, 25), array_287246, *[list_call_result_287253], **kwargs_287254)
        
        # Assigning a type to the variable 'known_taps' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'known_taps', array_call_result_287255)
        
        # Call to ValueError(...): (line 107)
        # Processing the call arguments (line 107)
        str_287257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 29), 'str', 'nbits must be between %s and %s if taps is None')
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_287258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        
        # Call to min(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_287261 = {}
        # Getting the type of 'known_taps' (line 108)
        known_taps_287259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'known_taps', False)
        # Obtaining the member 'min' of a type (line 108)
        min_287260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 32), known_taps_287259, 'min')
        # Calling min(args, kwargs) (line 108)
        min_call_result_287262 = invoke(stypy.reporting.localization.Localization(__file__, 108, 32), min_287260, *[], **kwargs_287261)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 32), tuple_287258, min_call_result_287262)
        # Adding element type (line 108)
        
        # Call to max(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_287265 = {}
        # Getting the type of 'known_taps' (line 108)
        known_taps_287263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 50), 'known_taps', False)
        # Obtaining the member 'max' of a type (line 108)
        max_287264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 50), known_taps_287263, 'max')
        # Calling max(args, kwargs) (line 108)
        max_call_result_287266 = invoke(stypy.reporting.localization.Localization(__file__, 108, 50), max_287264, *[], **kwargs_287265)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 32), tuple_287258, max_call_result_287266)
        
        # Applying the binary operator '%' (line 107)
        result_mod_287267 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 29), '%', str_287257, tuple_287258)
        
        # Processing the call keyword arguments (line 107)
        kwargs_287268 = {}
        # Getting the type of 'ValueError' (line 107)
        ValueError_287256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 107)
        ValueError_call_result_287269 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), ValueError_287256, *[result_mod_287267], **kwargs_287268)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 107, 12), ValueError_call_result_287269, 'raise parameter', BaseException)
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 109):
        
        # Call to array(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining the type of the subscript
        # Getting the type of 'nbits' (line 109)
        nbits_287272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'nbits', False)
        # Getting the type of '_mls_taps' (line 109)
        _mls_taps_287273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), '_mls_taps', False)
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___287274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 24), _mls_taps_287273, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_287275 = invoke(stypy.reporting.localization.Localization(__file__, 109, 24), getitem___287274, nbits_287272)
        
        # Getting the type of 'np' (line 109)
        np_287276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 42), 'np', False)
        # Obtaining the member 'intp' of a type (line 109)
        intp_287277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 42), np_287276, 'intp')
        # Processing the call keyword arguments (line 109)
        kwargs_287278 = {}
        # Getting the type of 'np' (line 109)
        np_287270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 109)
        array_287271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), np_287270, 'array')
        # Calling array(args, kwargs) (line 109)
        array_call_result_287279 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), array_287271, *[subscript_call_result_287275, intp_287277], **kwargs_287278)
        
        # Assigning a type to the variable 'taps' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'taps', array_call_result_287279)

        if more_types_in_union_287240:
            # Runtime conditional SSA for else branch (line 104)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_287239) or more_types_in_union_287240):
        
        # Assigning a Subscript to a Name (line 111):
        
        # Obtaining the type of the subscript
        int_287280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 52), 'int')
        slice_287281 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 111, 15), None, None, int_287280)
        
        # Call to unique(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to array(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'taps' (line 111)
        taps_287286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'taps', False)
        # Getting the type of 'np' (line 111)
        np_287287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 40), 'np', False)
        # Obtaining the member 'intp' of a type (line 111)
        intp_287288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 40), np_287287, 'intp')
        # Processing the call keyword arguments (line 111)
        kwargs_287289 = {}
        # Getting the type of 'np' (line 111)
        np_287284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'np', False)
        # Obtaining the member 'array' of a type (line 111)
        array_287285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), np_287284, 'array')
        # Calling array(args, kwargs) (line 111)
        array_call_result_287290 = invoke(stypy.reporting.localization.Localization(__file__, 111, 25), array_287285, *[taps_287286, intp_287288], **kwargs_287289)
        
        # Processing the call keyword arguments (line 111)
        kwargs_287291 = {}
        # Getting the type of 'np' (line 111)
        np_287282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'np', False)
        # Obtaining the member 'unique' of a type (line 111)
        unique_287283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), np_287282, 'unique')
        # Calling unique(args, kwargs) (line 111)
        unique_call_result_287292 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), unique_287283, *[array_call_result_287290], **kwargs_287291)
        
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___287293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), unique_call_result_287292, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_287294 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), getitem___287293, slice_287281)
        
        # Assigning a type to the variable 'taps' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'taps', subscript_call_result_287294)
        
        
        # Evaluating a boolean operation
        
        # Call to any(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Getting the type of 'taps' (line 112)
        taps_287297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'taps', False)
        int_287298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 25), 'int')
        # Applying the binary operator '<' (line 112)
        result_lt_287299 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 18), '<', taps_287297, int_287298)
        
        # Processing the call keyword arguments (line 112)
        kwargs_287300 = {}
        # Getting the type of 'np' (line 112)
        np_287295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 112)
        any_287296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 11), np_287295, 'any')
        # Calling any(args, kwargs) (line 112)
        any_call_result_287301 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), any_287296, *[result_lt_287299], **kwargs_287300)
        
        
        # Call to any(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Getting the type of 'taps' (line 112)
        taps_287304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'taps', False)
        # Getting the type of 'nbits' (line 112)
        nbits_287305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 45), 'nbits', False)
        # Applying the binary operator '>' (line 112)
        result_gt_287306 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 38), '>', taps_287304, nbits_287305)
        
        # Processing the call keyword arguments (line 112)
        kwargs_287307 = {}
        # Getting the type of 'np' (line 112)
        np_287302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'np', False)
        # Obtaining the member 'any' of a type (line 112)
        any_287303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 31), np_287302, 'any')
        # Calling any(args, kwargs) (line 112)
        any_call_result_287308 = invoke(stypy.reporting.localization.Localization(__file__, 112, 31), any_287303, *[result_gt_287306], **kwargs_287307)
        
        # Applying the binary operator 'or' (line 112)
        result_or_keyword_287309 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), 'or', any_call_result_287301, any_call_result_287308)
        
        # Getting the type of 'taps' (line 112)
        taps_287310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 55), 'taps')
        # Obtaining the member 'size' of a type (line 112)
        size_287311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 55), taps_287310, 'size')
        int_287312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 67), 'int')
        # Applying the binary operator '<' (line 112)
        result_lt_287313 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 55), '<', size_287311, int_287312)
        
        # Applying the binary operator 'or' (line 112)
        result_or_keyword_287314 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), 'or', result_or_keyword_287309, result_lt_287313)
        
        # Testing the type of an if condition (line 112)
        if_condition_287315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_or_keyword_287314)
        # Assigning a type to the variable 'if_condition_287315' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_287315', if_condition_287315)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 113)
        # Processing the call arguments (line 113)
        str_287317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 29), 'str', 'taps must be non-empty with values between zero and nbits (inclusive)')
        # Processing the call keyword arguments (line 113)
        kwargs_287318 = {}
        # Getting the type of 'ValueError' (line 113)
        ValueError_287316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 113)
        ValueError_call_result_287319 = invoke(stypy.reporting.localization.Localization(__file__, 113, 18), ValueError_287316, *[str_287317], **kwargs_287318)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 113, 12), ValueError_call_result_287319, 'raise parameter', BaseException)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 115):
        
        # Call to ascontiguousarray(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'taps' (line 115)
        taps_287322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 36), 'taps', False)
        # Processing the call keyword arguments (line 115)
        kwargs_287323 = {}
        # Getting the type of 'np' (line 115)
        np_287320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 115)
        ascontiguousarray_287321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 15), np_287320, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 115)
        ascontiguousarray_call_result_287324 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), ascontiguousarray_287321, *[taps_287322], **kwargs_287323)
        
        # Assigning a type to the variable 'taps' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'taps', ascontiguousarray_call_result_287324)

        if (may_be_287239 and more_types_in_union_287240):
            # SSA join for if statement (line 104)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 116):
    int_287325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 13), 'int')
    # Getting the type of 'nbits' (line 116)
    nbits_287326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'nbits')
    # Applying the binary operator '**' (line 116)
    result_pow_287327 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 13), '**', int_287325, nbits_287326)
    
    int_287328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 25), 'int')
    # Applying the binary operator '-' (line 116)
    result_sub_287329 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 12), '-', result_pow_287327, int_287328)
    
    # Assigning a type to the variable 'n_max' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'n_max', result_sub_287329)
    
    # Type idiom detected: calculating its left and rigth part (line 117)
    # Getting the type of 'length' (line 117)
    length_287330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 7), 'length')
    # Getting the type of 'None' (line 117)
    None_287331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'None')
    
    (may_be_287332, more_types_in_union_287333) = may_be_none(length_287330, None_287331)

    if may_be_287332:

        if more_types_in_union_287333:
            # Runtime conditional SSA (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'n_max' (line 118)
        n_max_287334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'n_max')
        # Assigning a type to the variable 'length' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'length', n_max_287334)

        if more_types_in_union_287333:
            # Runtime conditional SSA for else branch (line 117)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_287332) or more_types_in_union_287333):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to int(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'length' (line 120)
        length_287336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'length', False)
        # Processing the call keyword arguments (line 120)
        kwargs_287337 = {}
        # Getting the type of 'int' (line 120)
        int_287335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'int', False)
        # Calling int(args, kwargs) (line 120)
        int_call_result_287338 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), int_287335, *[length_287336], **kwargs_287337)
        
        # Assigning a type to the variable 'length' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'length', int_call_result_287338)
        
        
        # Getting the type of 'length' (line 121)
        length_287339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'length')
        int_287340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 20), 'int')
        # Applying the binary operator '<' (line 121)
        result_lt_287341 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 11), '<', length_287339, int_287340)
        
        # Testing the type of an if condition (line 121)
        if_condition_287342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 8), result_lt_287341)
        # Assigning a type to the variable 'if_condition_287342' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'if_condition_287342', if_condition_287342)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 122)
        # Processing the call arguments (line 122)
        str_287344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 29), 'str', 'length must be greater than or equal to 0')
        # Processing the call keyword arguments (line 122)
        kwargs_287345 = {}
        # Getting the type of 'ValueError' (line 122)
        ValueError_287343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 122)
        ValueError_call_result_287346 = invoke(stypy.reporting.localization.Localization(__file__, 122, 18), ValueError_287343, *[str_287344], **kwargs_287345)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 122, 12), ValueError_call_result_287346, 'raise parameter', BaseException)
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_287332 and more_types_in_union_287333):
            # SSA join for if statement (line 117)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 125)
    # Getting the type of 'state' (line 125)
    state_287347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 7), 'state')
    # Getting the type of 'None' (line 125)
    None_287348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'None')
    
    (may_be_287349, more_types_in_union_287350) = may_be_none(state_287347, None_287348)

    if may_be_287349:

        if more_types_in_union_287350:
            # Runtime conditional SSA (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 126):
        
        # Call to ones(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'nbits' (line 126)
        nbits_287353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'nbits', False)
        # Processing the call keyword arguments (line 126)
        # Getting the type of 'np' (line 126)
        np_287354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 37), 'np', False)
        # Obtaining the member 'int8' of a type (line 126)
        int8_287355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 37), np_287354, 'int8')
        keyword_287356 = int8_287355
        str_287357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 52), 'str', 'c')
        keyword_287358 = str_287357
        kwargs_287359 = {'dtype': keyword_287356, 'order': keyword_287358}
        # Getting the type of 'np' (line 126)
        np_287351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'np', False)
        # Obtaining the member 'ones' of a type (line 126)
        ones_287352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 16), np_287351, 'ones')
        # Calling ones(args, kwargs) (line 126)
        ones_call_result_287360 = invoke(stypy.reporting.localization.Localization(__file__, 126, 16), ones_287352, *[nbits_287353], **kwargs_287359)
        
        # Assigning a type to the variable 'state' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'state', ones_call_result_287360)

        if more_types_in_union_287350:
            # Runtime conditional SSA for else branch (line 125)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_287349) or more_types_in_union_287350):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to astype(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'np' (line 129)
        np_287371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 62), 'np', False)
        # Obtaining the member 'int8' of a type (line 129)
        int8_287372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 62), np_287371, 'int8')
        # Processing the call keyword arguments (line 129)
        kwargs_287373 = {}
        
        # Call to array(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'state' (line 129)
        state_287363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'state', False)
        # Processing the call keyword arguments (line 129)
        # Getting the type of 'bool' (line 129)
        bool_287364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), 'bool', False)
        keyword_287365 = bool_287364
        str_287366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 50), 'str', 'c')
        keyword_287367 = str_287366
        kwargs_287368 = {'dtype': keyword_287365, 'order': keyword_287367}
        # Getting the type of 'np' (line 129)
        np_287361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 129)
        array_287362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), np_287361, 'array')
        # Calling array(args, kwargs) (line 129)
        array_call_result_287369 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), array_287362, *[state_287363], **kwargs_287368)
        
        # Obtaining the member 'astype' of a type (line 129)
        astype_287370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), array_call_result_287369, 'astype')
        # Calling astype(args, kwargs) (line 129)
        astype_call_result_287374 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), astype_287370, *[int8_287372], **kwargs_287373)
        
        # Assigning a type to the variable 'state' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'state', astype_call_result_287374)

        if (may_be_287349 and more_types_in_union_287350):
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'state' (line 130)
    state_287375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'state')
    # Obtaining the member 'ndim' of a type (line 130)
    ndim_287376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 7), state_287375, 'ndim')
    int_287377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'int')
    # Applying the binary operator '!=' (line 130)
    result_ne_287378 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), '!=', ndim_287376, int_287377)
    
    
    # Getting the type of 'state' (line 130)
    state_287379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 26), 'state')
    # Obtaining the member 'size' of a type (line 130)
    size_287380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 26), state_287379, 'size')
    # Getting the type of 'nbits' (line 130)
    nbits_287381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'nbits')
    # Applying the binary operator '!=' (line 130)
    result_ne_287382 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 26), '!=', size_287380, nbits_287381)
    
    # Applying the binary operator 'or' (line 130)
    result_or_keyword_287383 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), 'or', result_ne_287378, result_ne_287382)
    
    # Testing the type of an if condition (line 130)
    if_condition_287384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), result_or_keyword_287383)
    # Assigning a type to the variable 'if_condition_287384' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_287384', if_condition_287384)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 131)
    # Processing the call arguments (line 131)
    str_287386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 25), 'str', 'state must be a 1-dimensional array of size nbits')
    # Processing the call keyword arguments (line 131)
    kwargs_287387 = {}
    # Getting the type of 'ValueError' (line 131)
    ValueError_287385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 131)
    ValueError_call_result_287388 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), ValueError_287385, *[str_287386], **kwargs_287387)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 131, 8), ValueError_call_result_287388, 'raise parameter', BaseException)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to all(...): (line 132)
    # Processing the call arguments (line 132)
    
    # Getting the type of 'state' (line 132)
    state_287391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'state', False)
    int_287392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'int')
    # Applying the binary operator '==' (line 132)
    result_eq_287393 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 14), '==', state_287391, int_287392)
    
    # Processing the call keyword arguments (line 132)
    kwargs_287394 = {}
    # Getting the type of 'np' (line 132)
    np_287389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 7), 'np', False)
    # Obtaining the member 'all' of a type (line 132)
    all_287390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 7), np_287389, 'all')
    # Calling all(args, kwargs) (line 132)
    all_call_result_287395 = invoke(stypy.reporting.localization.Localization(__file__, 132, 7), all_287390, *[result_eq_287393], **kwargs_287394)
    
    # Testing the type of an if condition (line 132)
    if_condition_287396 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 4), all_call_result_287395)
    # Assigning a type to the variable 'if_condition_287396' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'if_condition_287396', if_condition_287396)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 133)
    # Processing the call arguments (line 133)
    str_287398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'str', 'state must not be all zeros')
    # Processing the call keyword arguments (line 133)
    kwargs_287399 = {}
    # Getting the type of 'ValueError' (line 133)
    ValueError_287397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 133)
    ValueError_call_result_287400 = invoke(stypy.reporting.localization.Localization(__file__, 133, 14), ValueError_287397, *[str_287398], **kwargs_287399)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 133, 8), ValueError_call_result_287400, 'raise parameter', BaseException)
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 135):
    
    # Call to empty(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'length' (line 135)
    length_287403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'length', False)
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'np' (line 135)
    np_287404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'np', False)
    # Obtaining the member 'int8' of a type (line 135)
    int8_287405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 33), np_287404, 'int8')
    keyword_287406 = int8_287405
    str_287407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 48), 'str', 'c')
    keyword_287408 = str_287407
    kwargs_287409 = {'dtype': keyword_287406, 'order': keyword_287408}
    # Getting the type of 'np' (line 135)
    np_287401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 135)
    empty_287402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 10), np_287401, 'empty')
    # Calling empty(args, kwargs) (line 135)
    empty_call_result_287410 = invoke(stypy.reporting.localization.Localization(__file__, 135, 10), empty_287402, *[length_287403], **kwargs_287409)
    
    # Assigning a type to the variable 'seq' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'seq', empty_call_result_287410)
    
    # Assigning a Call to a Name (line 136):
    
    # Call to _max_len_seq_inner(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'taps' (line 136)
    taps_287412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'taps', False)
    # Getting the type of 'state' (line 136)
    state_287413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 37), 'state', False)
    # Getting the type of 'nbits' (line 136)
    nbits_287414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 44), 'nbits', False)
    # Getting the type of 'length' (line 136)
    length_287415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 51), 'length', False)
    # Getting the type of 'seq' (line 136)
    seq_287416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 59), 'seq', False)
    # Processing the call keyword arguments (line 136)
    kwargs_287417 = {}
    # Getting the type of '_max_len_seq_inner' (line 136)
    _max_len_seq_inner_287411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), '_max_len_seq_inner', False)
    # Calling _max_len_seq_inner(args, kwargs) (line 136)
    _max_len_seq_inner_call_result_287418 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), _max_len_seq_inner_287411, *[taps_287412, state_287413, nbits_287414, length_287415, seq_287416], **kwargs_287417)
    
    # Assigning a type to the variable 'state' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'state', _max_len_seq_inner_call_result_287418)
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_287419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'seq' (line 137)
    seq_287420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'seq')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_287419, seq_287420)
    # Adding element type (line 137)
    # Getting the type of 'state' (line 137)
    state_287421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'state')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_287419, state_287421)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', tuple_287419)
    
    # ################# End of 'max_len_seq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'max_len_seq' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_287422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_287422)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'max_len_seq'
    return stypy_return_type_287422

# Assigning a type to the variable 'max_len_seq' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'max_len_seq', max_len_seq)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Code adapted from "upfirdn" python library with permission:
2: #
3: # Copyright (c) 2009, Motorola, Inc
4: #
5: # All Rights Reserved.
6: #
7: # Redistribution and use in source and binary forms, with or without
8: # modification, are permitted provided that the following conditions are
9: # met:
10: #
11: # * Redistributions of source code must retain the above copyright notice,
12: # this list of conditions and the following disclaimer.
13: #
14: # * Redistributions in binary form must reproduce the above copyright
15: # notice, this list of conditions and the following disclaimer in the
16: # documentation and/or other materials provided with the distribution.
17: #
18: # * Neither the name of Motorola nor the names of its contributors may be
19: # used to endorse or promote products derived from this software without
20: # specific prior written permission.
21: #
22: # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
23: # IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
24: # THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
25: # PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
26: # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
27: # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
28: # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
29: # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
30: # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
31: # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
32: # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
33: 
34: import numpy as np
35: 
36: from ._upfirdn_apply import _output_len, _apply
37: 
38: __all__ = ['upfirdn', '_output_len']
39: 
40: 
41: def _pad_h(h, up):
42:     '''Store coefficients in a transposed, flipped arrangement.
43: 
44:     For example, suppose upRate is 3, and the
45:     input number of coefficients is 10, represented as h[0], ..., h[9].
46: 
47:     Then the internal buffer will look like this::
48: 
49:        h[9], h[6], h[3], h[0],   // flipped phase 0 coefs
50:        0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)
51:        0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)
52: 
53:     '''
54:     h_padlen = len(h) + (-len(h) % up)
55:     h_full = np.zeros(h_padlen, h.dtype)
56:     h_full[:len(h)] = h
57:     h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
58:     return h_full
59: 
60: 
61: class _UpFIRDn(object):
62:     def __init__(self, h, x_dtype, up, down):
63:         '''Helper for resampling'''
64:         h = np.asarray(h)
65:         if h.ndim != 1 or h.size == 0:
66:             raise ValueError('h must be 1D with non-zero length')
67:         self._output_type = np.result_type(h.dtype, x_dtype, np.float32)
68:         h = np.asarray(h, self._output_type)
69:         self._up = int(up)
70:         self._down = int(down)
71:         if self._up < 1 or self._down < 1:
72:             raise ValueError('Both up and down must be >= 1')
73:         # This both transposes, and "flips" each phase for filtering
74:         self._h_trans_flip = _pad_h(h, self._up)
75:         self._h_trans_flip = np.ascontiguousarray(self._h_trans_flip)
76: 
77:     def apply_filter(self, x, axis=-1):
78:         '''Apply the prepared filter to the specified axis of a nD signal x'''
79:         output_len = _output_len(len(self._h_trans_flip), x.shape[axis],
80:                                  self._up, self._down)
81:         output_shape = np.asarray(x.shape)
82:         output_shape[axis] = output_len
83:         out = np.zeros(output_shape, dtype=self._output_type, order='C')
84:         axis = axis % x.ndim
85:         _apply(np.asarray(x, self._output_type),
86:                self._h_trans_flip, out,
87:                self._up, self._down, axis)
88:         return out
89: 
90: 
91: def upfirdn(h, x, up=1, down=1, axis=-1):
92:     '''Upsample, FIR filter, and downsample
93: 
94:     Parameters
95:     ----------
96:     h : array_like
97:         1-dimensional FIR (finite-impulse response) filter coefficients.
98:     x : array_like
99:         Input signal array.
100:     up : int, optional
101:         Upsampling rate. Default is 1.
102:     down : int, optional
103:         Downsampling rate. Default is 1.
104:     axis : int, optional
105:         The axis of the input data array along which to apply the
106:         linear filter. The filter is applied to each subarray along
107:         this axis. Default is -1.
108: 
109:     Returns
110:     -------
111:     y : ndarray
112:         The output signal array. Dimensions will be the same as `x` except
113:         for along `axis`, which will change size according to the `h`,
114:         `up`,  and `down` parameters.
115: 
116:     Notes
117:     -----
118:     The algorithm is an implementation of the block diagram shown on page 129
119:     of the Vaidyanathan text [1]_ (Figure 4.3-8d).
120: 
121:     .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
122:        Prentice Hall, 1993.
123: 
124:     The direct approach of upsampling by factor of P with zero insertion,
125:     FIR filtering of length ``N``, and downsampling by factor of Q is
126:     O(N*Q) per output sample. The polyphase implementation used here is
127:     O(N/P).
128: 
129:     .. versionadded:: 0.18
130: 
131:     Examples
132:     --------
133:     Simple operations:
134: 
135:     >>> from scipy.signal import upfirdn
136:     >>> upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter
137:     array([ 1.,  2.,  3.,  2.,  1.])
138:     >>> upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion
139:     array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.,  0.,  0.])
140:     >>> upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold
141:     array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
142:     >>> upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation
143:     array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5,  0. ])
144:     >>> upfirdn([1], np.arange(10), 1, 3)  # decimation by 3
145:     array([ 0.,  3.,  6.,  9.])
146:     >>> upfirdn([.5, 1, .5], np.arange(10), 2, 3)  # linear interp, rate 2/3
147:     array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5,  0. ])
148: 
149:     Apply a single filter to multiple signals:
150: 
151:     >>> x = np.reshape(np.arange(8), (4, 2))
152:     >>> x
153:     array([[0, 1],
154:            [2, 3],
155:            [4, 5],
156:            [6, 7]])
157: 
158:     Apply along the last dimension of ``x``:
159: 
160:     >>> h = [1, 1]
161:     >>> upfirdn(h, x, 2)
162:     array([[ 0.,  0.,  1.,  1.],
163:            [ 2.,  2.,  3.,  3.],
164:            [ 4.,  4.,  5.,  5.],
165:            [ 6.,  6.,  7.,  7.]])
166: 
167:     Apply along the 0th dimension of ``x``:
168: 
169:     >>> upfirdn(h, x, 2, axis=0)
170:     array([[ 0.,  1.],
171:            [ 0.,  1.],
172:            [ 2.,  3.],
173:            [ 2.,  3.],
174:            [ 4.,  5.],
175:            [ 4.,  5.],
176:            [ 6.,  7.],
177:            [ 6.,  7.]])
178: 
179:     '''
180:     x = np.asarray(x)
181:     ufd = _UpFIRDn(h, x.dtype, up, down)
182:     # This is equivalent to (but faster than) using np.apply_along_axis
183:     return ufd.apply_filter(x, axis)
184: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import numpy' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy')

if (type(import_288630) is not StypyTypeError):

    if (import_288630 != 'pyd_module'):
        __import__(import_288630)
        sys_modules_288631 = sys.modules[import_288630]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'np', sys_modules_288631.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', import_288630)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from scipy.signal._upfirdn_apply import _output_len, _apply' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288632 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.signal._upfirdn_apply')

if (type(import_288632) is not StypyTypeError):

    if (import_288632 != 'pyd_module'):
        __import__(import_288632)
        sys_modules_288633 = sys.modules[import_288632]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.signal._upfirdn_apply', sys_modules_288633.module_type_store, module_type_store, ['_output_len', '_apply'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_288633, sys_modules_288633.module_type_store, module_type_store)
    else:
        from scipy.signal._upfirdn_apply import _output_len, _apply

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.signal._upfirdn_apply', None, module_type_store, ['_output_len', '_apply'], [_output_len, _apply])

else:
    # Assigning a type to the variable 'scipy.signal._upfirdn_apply' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.signal._upfirdn_apply', import_288632)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 38):
__all__ = ['upfirdn', '_output_len']
module_type_store.set_exportable_members(['upfirdn', '_output_len'])

# Obtaining an instance of the builtin type 'list' (line 38)
list_288634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 38)
# Adding element type (line 38)
str_288635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 11), 'str', 'upfirdn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_288634, str_288635)
# Adding element type (line 38)
str_288636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 22), 'str', '_output_len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 10), list_288634, str_288636)

# Assigning a type to the variable '__all__' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), '__all__', list_288634)

@norecursion
def _pad_h(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_pad_h'
    module_type_store = module_type_store.open_function_context('_pad_h', 41, 0, False)
    
    # Passed parameters checking function
    _pad_h.stypy_localization = localization
    _pad_h.stypy_type_of_self = None
    _pad_h.stypy_type_store = module_type_store
    _pad_h.stypy_function_name = '_pad_h'
    _pad_h.stypy_param_names_list = ['h', 'up']
    _pad_h.stypy_varargs_param_name = None
    _pad_h.stypy_kwargs_param_name = None
    _pad_h.stypy_call_defaults = defaults
    _pad_h.stypy_call_varargs = varargs
    _pad_h.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_pad_h', ['h', 'up'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_pad_h', localization, ['h', 'up'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_pad_h(...)' code ##################

    str_288637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, (-1)), 'str', 'Store coefficients in a transposed, flipped arrangement.\n\n    For example, suppose upRate is 3, and the\n    input number of coefficients is 10, represented as h[0], ..., h[9].\n\n    Then the internal buffer will look like this::\n\n       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs\n       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)\n       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)\n\n    ')
    
    # Assigning a BinOp to a Name (line 54):
    
    # Call to len(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'h' (line 54)
    h_288639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'h', False)
    # Processing the call keyword arguments (line 54)
    kwargs_288640 = {}
    # Getting the type of 'len' (line 54)
    len_288638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'len', False)
    # Calling len(args, kwargs) (line 54)
    len_call_result_288641 = invoke(stypy.reporting.localization.Localization(__file__, 54, 15), len_288638, *[h_288639], **kwargs_288640)
    
    
    
    # Call to len(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'h' (line 54)
    h_288643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'h', False)
    # Processing the call keyword arguments (line 54)
    kwargs_288644 = {}
    # Getting the type of 'len' (line 54)
    len_288642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'len', False)
    # Calling len(args, kwargs) (line 54)
    len_call_result_288645 = invoke(stypy.reporting.localization.Localization(__file__, 54, 26), len_288642, *[h_288643], **kwargs_288644)
    
    # Applying the 'usub' unary operator (line 54)
    result___neg___288646 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 25), 'usub', len_call_result_288645)
    
    # Getting the type of 'up' (line 54)
    up_288647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 35), 'up')
    # Applying the binary operator '%' (line 54)
    result_mod_288648 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 25), '%', result___neg___288646, up_288647)
    
    # Applying the binary operator '+' (line 54)
    result_add_288649 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), '+', len_call_result_288641, result_mod_288648)
    
    # Assigning a type to the variable 'h_padlen' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'h_padlen', result_add_288649)
    
    # Assigning a Call to a Name (line 55):
    
    # Call to zeros(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'h_padlen' (line 55)
    h_padlen_288652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'h_padlen', False)
    # Getting the type of 'h' (line 55)
    h_288653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 32), 'h', False)
    # Obtaining the member 'dtype' of a type (line 55)
    dtype_288654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 32), h_288653, 'dtype')
    # Processing the call keyword arguments (line 55)
    kwargs_288655 = {}
    # Getting the type of 'np' (line 55)
    np_288650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'np', False)
    # Obtaining the member 'zeros' of a type (line 55)
    zeros_288651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 13), np_288650, 'zeros')
    # Calling zeros(args, kwargs) (line 55)
    zeros_call_result_288656 = invoke(stypy.reporting.localization.Localization(__file__, 55, 13), zeros_288651, *[h_padlen_288652, dtype_288654], **kwargs_288655)
    
    # Assigning a type to the variable 'h_full' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'h_full', zeros_call_result_288656)
    
    # Assigning a Name to a Subscript (line 56):
    # Getting the type of 'h' (line 56)
    h_288657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'h')
    # Getting the type of 'h_full' (line 56)
    h_full_288658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'h_full')
    
    # Call to len(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'h' (line 56)
    h_288660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'h', False)
    # Processing the call keyword arguments (line 56)
    kwargs_288661 = {}
    # Getting the type of 'len' (line 56)
    len_288659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'len', False)
    # Calling len(args, kwargs) (line 56)
    len_call_result_288662 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), len_288659, *[h_288660], **kwargs_288661)
    
    slice_288663 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 4), None, len_call_result_288662, None)
    # Storing an element on a container (line 56)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 4), h_full_288658, (slice_288663, h_288657))
    
    # Assigning a Call to a Name (line 57):
    
    # Call to ravel(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_288677 = {}
    
    # Obtaining the type of the subscript
    slice_288664 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 57, 13), None, None, None)
    int_288665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 43), 'int')
    slice_288666 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 57, 13), None, None, int_288665)
    
    # Call to reshape(...): (line 57)
    # Processing the call arguments (line 57)
    int_288669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'int')
    # Getting the type of 'up' (line 57)
    up_288670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 32), 'up', False)
    # Processing the call keyword arguments (line 57)
    kwargs_288671 = {}
    # Getting the type of 'h_full' (line 57)
    h_full_288667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'h_full', False)
    # Obtaining the member 'reshape' of a type (line 57)
    reshape_288668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), h_full_288667, 'reshape')
    # Calling reshape(args, kwargs) (line 57)
    reshape_call_result_288672 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), reshape_288668, *[int_288669, up_288670], **kwargs_288671)
    
    # Obtaining the member 'T' of a type (line 57)
    T_288673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), reshape_call_result_288672, 'T')
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___288674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), T_288673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_288675 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), getitem___288674, (slice_288664, slice_288666))
    
    # Obtaining the member 'ravel' of a type (line 57)
    ravel_288676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 13), subscript_call_result_288675, 'ravel')
    # Calling ravel(args, kwargs) (line 57)
    ravel_call_result_288678 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), ravel_288676, *[], **kwargs_288677)
    
    # Assigning a type to the variable 'h_full' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'h_full', ravel_call_result_288678)
    # Getting the type of 'h_full' (line 58)
    h_full_288679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'h_full')
    # Assigning a type to the variable 'stypy_return_type' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type', h_full_288679)
    
    # ################# End of '_pad_h(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_pad_h' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_288680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288680)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_pad_h'
    return stypy_return_type_288680

# Assigning a type to the variable '_pad_h' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_pad_h', _pad_h)
# Declaration of the '_UpFIRDn' class

class _UpFIRDn(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_UpFIRDn.__init__', ['h', 'x_dtype', 'up', 'down'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['h', 'x_dtype', 'up', 'down'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_288681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'str', 'Helper for resampling')
        
        # Assigning a Call to a Name (line 64):
        
        # Call to asarray(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'h' (line 64)
        h_288684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'h', False)
        # Processing the call keyword arguments (line 64)
        kwargs_288685 = {}
        # Getting the type of 'np' (line 64)
        np_288682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 64)
        asarray_288683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), np_288682, 'asarray')
        # Calling asarray(args, kwargs) (line 64)
        asarray_call_result_288686 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), asarray_288683, *[h_288684], **kwargs_288685)
        
        # Assigning a type to the variable 'h' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'h', asarray_call_result_288686)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'h' (line 65)
        h_288687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'h')
        # Obtaining the member 'ndim' of a type (line 65)
        ndim_288688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), h_288687, 'ndim')
        int_288689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'int')
        # Applying the binary operator '!=' (line 65)
        result_ne_288690 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), '!=', ndim_288688, int_288689)
        
        
        # Getting the type of 'h' (line 65)
        h_288691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'h')
        # Obtaining the member 'size' of a type (line 65)
        size_288692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 26), h_288691, 'size')
        int_288693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 36), 'int')
        # Applying the binary operator '==' (line 65)
        result_eq_288694 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 26), '==', size_288692, int_288693)
        
        # Applying the binary operator 'or' (line 65)
        result_or_keyword_288695 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), 'or', result_ne_288690, result_eq_288694)
        
        # Testing the type of an if condition (line 65)
        if_condition_288696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), result_or_keyword_288695)
        # Assigning a type to the variable 'if_condition_288696' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_288696', if_condition_288696)
        # SSA begins for if statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 66)
        # Processing the call arguments (line 66)
        str_288698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'str', 'h must be 1D with non-zero length')
        # Processing the call keyword arguments (line 66)
        kwargs_288699 = {}
        # Getting the type of 'ValueError' (line 66)
        ValueError_288697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 66)
        ValueError_call_result_288700 = invoke(stypy.reporting.localization.Localization(__file__, 66, 18), ValueError_288697, *[str_288698], **kwargs_288699)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 66, 12), ValueError_call_result_288700, 'raise parameter', BaseException)
        # SSA join for if statement (line 65)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 67):
        
        # Call to result_type(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'h' (line 67)
        h_288703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'h', False)
        # Obtaining the member 'dtype' of a type (line 67)
        dtype_288704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 43), h_288703, 'dtype')
        # Getting the type of 'x_dtype' (line 67)
        x_dtype_288705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 52), 'x_dtype', False)
        # Getting the type of 'np' (line 67)
        np_288706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 61), 'np', False)
        # Obtaining the member 'float32' of a type (line 67)
        float32_288707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 61), np_288706, 'float32')
        # Processing the call keyword arguments (line 67)
        kwargs_288708 = {}
        # Getting the type of 'np' (line 67)
        np_288701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'np', False)
        # Obtaining the member 'result_type' of a type (line 67)
        result_type_288702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 28), np_288701, 'result_type')
        # Calling result_type(args, kwargs) (line 67)
        result_type_call_result_288709 = invoke(stypy.reporting.localization.Localization(__file__, 67, 28), result_type_288702, *[dtype_288704, x_dtype_288705, float32_288707], **kwargs_288708)
        
        # Getting the type of 'self' (line 67)
        self_288710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member '_output_type' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_288710, '_output_type', result_type_call_result_288709)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to asarray(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'h' (line 68)
        h_288713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'h', False)
        # Getting the type of 'self' (line 68)
        self_288714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'self', False)
        # Obtaining the member '_output_type' of a type (line 68)
        _output_type_288715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 26), self_288714, '_output_type')
        # Processing the call keyword arguments (line 68)
        kwargs_288716 = {}
        # Getting the type of 'np' (line 68)
        np_288711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 68)
        asarray_288712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), np_288711, 'asarray')
        # Calling asarray(args, kwargs) (line 68)
        asarray_call_result_288717 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), asarray_288712, *[h_288713, _output_type_288715], **kwargs_288716)
        
        # Assigning a type to the variable 'h' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'h', asarray_call_result_288717)
        
        # Assigning a Call to a Attribute (line 69):
        
        # Call to int(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'up' (line 69)
        up_288719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'up', False)
        # Processing the call keyword arguments (line 69)
        kwargs_288720 = {}
        # Getting the type of 'int' (line 69)
        int_288718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'int', False)
        # Calling int(args, kwargs) (line 69)
        int_call_result_288721 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), int_288718, *[up_288719], **kwargs_288720)
        
        # Getting the type of 'self' (line 69)
        self_288722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member '_up' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_288722, '_up', int_call_result_288721)
        
        # Assigning a Call to a Attribute (line 70):
        
        # Call to int(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'down' (line 70)
        down_288724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'down', False)
        # Processing the call keyword arguments (line 70)
        kwargs_288725 = {}
        # Getting the type of 'int' (line 70)
        int_288723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'int', False)
        # Calling int(args, kwargs) (line 70)
        int_call_result_288726 = invoke(stypy.reporting.localization.Localization(__file__, 70, 21), int_288723, *[down_288724], **kwargs_288725)
        
        # Getting the type of 'self' (line 70)
        self_288727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member '_down' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_288727, '_down', int_call_result_288726)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 71)
        self_288728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'self')
        # Obtaining the member '_up' of a type (line 71)
        _up_288729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), self_288728, '_up')
        int_288730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 22), 'int')
        # Applying the binary operator '<' (line 71)
        result_lt_288731 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 11), '<', _up_288729, int_288730)
        
        
        # Getting the type of 'self' (line 71)
        self_288732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'self')
        # Obtaining the member '_down' of a type (line 71)
        _down_288733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 27), self_288732, '_down')
        int_288734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 40), 'int')
        # Applying the binary operator '<' (line 71)
        result_lt_288735 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 27), '<', _down_288733, int_288734)
        
        # Applying the binary operator 'or' (line 71)
        result_or_keyword_288736 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 11), 'or', result_lt_288731, result_lt_288735)
        
        # Testing the type of an if condition (line 71)
        if_condition_288737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 8), result_or_keyword_288736)
        # Assigning a type to the variable 'if_condition_288737' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'if_condition_288737', if_condition_288737)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 72)
        # Processing the call arguments (line 72)
        str_288739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'str', 'Both up and down must be >= 1')
        # Processing the call keyword arguments (line 72)
        kwargs_288740 = {}
        # Getting the type of 'ValueError' (line 72)
        ValueError_288738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 72)
        ValueError_call_result_288741 = invoke(stypy.reporting.localization.Localization(__file__, 72, 18), ValueError_288738, *[str_288739], **kwargs_288740)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 72, 12), ValueError_call_result_288741, 'raise parameter', BaseException)
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 74):
        
        # Call to _pad_h(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'h' (line 74)
        h_288743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 36), 'h', False)
        # Getting the type of 'self' (line 74)
        self_288744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 39), 'self', False)
        # Obtaining the member '_up' of a type (line 74)
        _up_288745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 39), self_288744, '_up')
        # Processing the call keyword arguments (line 74)
        kwargs_288746 = {}
        # Getting the type of '_pad_h' (line 74)
        _pad_h_288742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 29), '_pad_h', False)
        # Calling _pad_h(args, kwargs) (line 74)
        _pad_h_call_result_288747 = invoke(stypy.reporting.localization.Localization(__file__, 74, 29), _pad_h_288742, *[h_288743, _up_288745], **kwargs_288746)
        
        # Getting the type of 'self' (line 74)
        self_288748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member '_h_trans_flip' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_288748, '_h_trans_flip', _pad_h_call_result_288747)
        
        # Assigning a Call to a Attribute (line 75):
        
        # Call to ascontiguousarray(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'self' (line 75)
        self_288751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 50), 'self', False)
        # Obtaining the member '_h_trans_flip' of a type (line 75)
        _h_trans_flip_288752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 50), self_288751, '_h_trans_flip')
        # Processing the call keyword arguments (line 75)
        kwargs_288753 = {}
        # Getting the type of 'np' (line 75)
        np_288749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'np', False)
        # Obtaining the member 'ascontiguousarray' of a type (line 75)
        ascontiguousarray_288750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 29), np_288749, 'ascontiguousarray')
        # Calling ascontiguousarray(args, kwargs) (line 75)
        ascontiguousarray_call_result_288754 = invoke(stypy.reporting.localization.Localization(__file__, 75, 29), ascontiguousarray_288750, *[_h_trans_flip_288752], **kwargs_288753)
        
        # Getting the type of 'self' (line 75)
        self_288755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member '_h_trans_flip' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_288755, '_h_trans_flip', ascontiguousarray_call_result_288754)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def apply_filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_288756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 35), 'int')
        defaults = [int_288756]
        # Create a new context for function 'apply_filter'
        module_type_store = module_type_store.open_function_context('apply_filter', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_localization', localization)
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_function_name', '_UpFIRDn.apply_filter')
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_param_names_list', ['x', 'axis'])
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _UpFIRDn.apply_filter.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_UpFIRDn.apply_filter', ['x', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'apply_filter', localization, ['x', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'apply_filter(...)' code ##################

        str_288757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'str', 'Apply the prepared filter to the specified axis of a nD signal x')
        
        # Assigning a Call to a Name (line 79):
        
        # Call to _output_len(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to len(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_288760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'self', False)
        # Obtaining the member '_h_trans_flip' of a type (line 79)
        _h_trans_flip_288761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 37), self_288760, '_h_trans_flip')
        # Processing the call keyword arguments (line 79)
        kwargs_288762 = {}
        # Getting the type of 'len' (line 79)
        len_288759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'len', False)
        # Calling len(args, kwargs) (line 79)
        len_call_result_288763 = invoke(stypy.reporting.localization.Localization(__file__, 79, 33), len_288759, *[_h_trans_flip_288761], **kwargs_288762)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 79)
        axis_288764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 66), 'axis', False)
        # Getting the type of 'x' (line 79)
        x_288765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 58), 'x', False)
        # Obtaining the member 'shape' of a type (line 79)
        shape_288766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 58), x_288765, 'shape')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___288767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 58), shape_288766, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_288768 = invoke(stypy.reporting.localization.Localization(__file__, 79, 58), getitem___288767, axis_288764)
        
        # Getting the type of 'self' (line 80)
        self_288769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 33), 'self', False)
        # Obtaining the member '_up' of a type (line 80)
        _up_288770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 33), self_288769, '_up')
        # Getting the type of 'self' (line 80)
        self_288771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 43), 'self', False)
        # Obtaining the member '_down' of a type (line 80)
        _down_288772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 43), self_288771, '_down')
        # Processing the call keyword arguments (line 79)
        kwargs_288773 = {}
        # Getting the type of '_output_len' (line 79)
        _output_len_288758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), '_output_len', False)
        # Calling _output_len(args, kwargs) (line 79)
        _output_len_call_result_288774 = invoke(stypy.reporting.localization.Localization(__file__, 79, 21), _output_len_288758, *[len_call_result_288763, subscript_call_result_288768, _up_288770, _down_288772], **kwargs_288773)
        
        # Assigning a type to the variable 'output_len' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'output_len', _output_len_call_result_288774)
        
        # Assigning a Call to a Name (line 81):
        
        # Call to asarray(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'x' (line 81)
        x_288777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 34), 'x', False)
        # Obtaining the member 'shape' of a type (line 81)
        shape_288778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 34), x_288777, 'shape')
        # Processing the call keyword arguments (line 81)
        kwargs_288779 = {}
        # Getting the type of 'np' (line 81)
        np_288775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'np', False)
        # Obtaining the member 'asarray' of a type (line 81)
        asarray_288776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 23), np_288775, 'asarray')
        # Calling asarray(args, kwargs) (line 81)
        asarray_call_result_288780 = invoke(stypy.reporting.localization.Localization(__file__, 81, 23), asarray_288776, *[shape_288778], **kwargs_288779)
        
        # Assigning a type to the variable 'output_shape' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'output_shape', asarray_call_result_288780)
        
        # Assigning a Name to a Subscript (line 82):
        # Getting the type of 'output_len' (line 82)
        output_len_288781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'output_len')
        # Getting the type of 'output_shape' (line 82)
        output_shape_288782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'output_shape')
        # Getting the type of 'axis' (line 82)
        axis_288783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'axis')
        # Storing an element on a container (line 82)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), output_shape_288782, (axis_288783, output_len_288781))
        
        # Assigning a Call to a Name (line 83):
        
        # Call to zeros(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'output_shape' (line 83)
        output_shape_288786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'output_shape', False)
        # Processing the call keyword arguments (line 83)
        # Getting the type of 'self' (line 83)
        self_288787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 43), 'self', False)
        # Obtaining the member '_output_type' of a type (line 83)
        _output_type_288788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 43), self_288787, '_output_type')
        keyword_288789 = _output_type_288788
        str_288790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 68), 'str', 'C')
        keyword_288791 = str_288790
        kwargs_288792 = {'dtype': keyword_288789, 'order': keyword_288791}
        # Getting the type of 'np' (line 83)
        np_288784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'np', False)
        # Obtaining the member 'zeros' of a type (line 83)
        zeros_288785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 14), np_288784, 'zeros')
        # Calling zeros(args, kwargs) (line 83)
        zeros_call_result_288793 = invoke(stypy.reporting.localization.Localization(__file__, 83, 14), zeros_288785, *[output_shape_288786], **kwargs_288792)
        
        # Assigning a type to the variable 'out' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'out', zeros_call_result_288793)
        
        # Assigning a BinOp to a Name (line 84):
        # Getting the type of 'axis' (line 84)
        axis_288794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'axis')
        # Getting the type of 'x' (line 84)
        x_288795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'x')
        # Obtaining the member 'ndim' of a type (line 84)
        ndim_288796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 22), x_288795, 'ndim')
        # Applying the binary operator '%' (line 84)
        result_mod_288797 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), '%', axis_288794, ndim_288796)
        
        # Assigning a type to the variable 'axis' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'axis', result_mod_288797)
        
        # Call to _apply(...): (line 85)
        # Processing the call arguments (line 85)
        
        # Call to asarray(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'x' (line 85)
        x_288801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'x', False)
        # Getting the type of 'self' (line 85)
        self_288802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'self', False)
        # Obtaining the member '_output_type' of a type (line 85)
        _output_type_288803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 29), self_288802, '_output_type')
        # Processing the call keyword arguments (line 85)
        kwargs_288804 = {}
        # Getting the type of 'np' (line 85)
        np_288799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'np', False)
        # Obtaining the member 'asarray' of a type (line 85)
        asarray_288800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), np_288799, 'asarray')
        # Calling asarray(args, kwargs) (line 85)
        asarray_call_result_288805 = invoke(stypy.reporting.localization.Localization(__file__, 85, 15), asarray_288800, *[x_288801, _output_type_288803], **kwargs_288804)
        
        # Getting the type of 'self' (line 86)
        self_288806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self', False)
        # Obtaining the member '_h_trans_flip' of a type (line 86)
        _h_trans_flip_288807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), self_288806, '_h_trans_flip')
        # Getting the type of 'out' (line 86)
        out_288808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 35), 'out', False)
        # Getting the type of 'self' (line 87)
        self_288809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'self', False)
        # Obtaining the member '_up' of a type (line 87)
        _up_288810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), self_288809, '_up')
        # Getting the type of 'self' (line 87)
        self_288811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'self', False)
        # Obtaining the member '_down' of a type (line 87)
        _down_288812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 25), self_288811, '_down')
        # Getting the type of 'axis' (line 87)
        axis_288813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 37), 'axis', False)
        # Processing the call keyword arguments (line 85)
        kwargs_288814 = {}
        # Getting the type of '_apply' (line 85)
        _apply_288798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), '_apply', False)
        # Calling _apply(args, kwargs) (line 85)
        _apply_call_result_288815 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), _apply_288798, *[asarray_call_result_288805, _h_trans_flip_288807, out_288808, _up_288810, _down_288812, axis_288813], **kwargs_288814)
        
        # Getting the type of 'out' (line 88)
        out_288816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', out_288816)
        
        # ################# End of 'apply_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'apply_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_288817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_288817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'apply_filter'
        return stypy_return_type_288817


# Assigning a type to the variable '_UpFIRDn' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), '_UpFIRDn', _UpFIRDn)

@norecursion
def upfirdn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_288818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'int')
    int_288819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 29), 'int')
    int_288820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 37), 'int')
    defaults = [int_288818, int_288819, int_288820]
    # Create a new context for function 'upfirdn'
    module_type_store = module_type_store.open_function_context('upfirdn', 91, 0, False)
    
    # Passed parameters checking function
    upfirdn.stypy_localization = localization
    upfirdn.stypy_type_of_self = None
    upfirdn.stypy_type_store = module_type_store
    upfirdn.stypy_function_name = 'upfirdn'
    upfirdn.stypy_param_names_list = ['h', 'x', 'up', 'down', 'axis']
    upfirdn.stypy_varargs_param_name = None
    upfirdn.stypy_kwargs_param_name = None
    upfirdn.stypy_call_defaults = defaults
    upfirdn.stypy_call_varargs = varargs
    upfirdn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'upfirdn', ['h', 'x', 'up', 'down', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'upfirdn', localization, ['h', 'x', 'up', 'down', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'upfirdn(...)' code ##################

    str_288821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, (-1)), 'str', 'Upsample, FIR filter, and downsample\n\n    Parameters\n    ----------\n    h : array_like\n        1-dimensional FIR (finite-impulse response) filter coefficients.\n    x : array_like\n        Input signal array.\n    up : int, optional\n        Upsampling rate. Default is 1.\n    down : int, optional\n        Downsampling rate. Default is 1.\n    axis : int, optional\n        The axis of the input data array along which to apply the\n        linear filter. The filter is applied to each subarray along\n        this axis. Default is -1.\n\n    Returns\n    -------\n    y : ndarray\n        The output signal array. Dimensions will be the same as `x` except\n        for along `axis`, which will change size according to the `h`,\n        `up`,  and `down` parameters.\n\n    Notes\n    -----\n    The algorithm is an implementation of the block diagram shown on page 129\n    of the Vaidyanathan text [1]_ (Figure 4.3-8d).\n\n    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,\n       Prentice Hall, 1993.\n\n    The direct approach of upsampling by factor of P with zero insertion,\n    FIR filtering of length ``N``, and downsampling by factor of Q is\n    O(N*Q) per output sample. The polyphase implementation used here is\n    O(N/P).\n\n    .. versionadded:: 0.18\n\n    Examples\n    --------\n    Simple operations:\n\n    >>> from scipy.signal import upfirdn\n    >>> upfirdn([1, 1, 1], [1, 1, 1])   # FIR filter\n    array([ 1.,  2.,  3.,  2.,  1.])\n    >>> upfirdn([1], [1, 2, 3], 3)  # upsampling with zeros insertion\n    array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.,  0.,  0.])\n    >>> upfirdn([1, 1, 1], [1, 2, 3], 3)  # upsampling with sample-and-hold\n    array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])\n    >>> upfirdn([.5, 1, .5], [1, 1, 1], 2)  # linear interpolation\n    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5,  0. ])\n    >>> upfirdn([1], np.arange(10), 1, 3)  # decimation by 3\n    array([ 0.,  3.,  6.,  9.])\n    >>> upfirdn([.5, 1, .5], np.arange(10), 2, 3)  # linear interp, rate 2/3\n    array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5,  0. ])\n\n    Apply a single filter to multiple signals:\n\n    >>> x = np.reshape(np.arange(8), (4, 2))\n    >>> x\n    array([[0, 1],\n           [2, 3],\n           [4, 5],\n           [6, 7]])\n\n    Apply along the last dimension of ``x``:\n\n    >>> h = [1, 1]\n    >>> upfirdn(h, x, 2)\n    array([[ 0.,  0.,  1.,  1.],\n           [ 2.,  2.,  3.,  3.],\n           [ 4.,  4.,  5.,  5.],\n           [ 6.,  6.,  7.,  7.]])\n\n    Apply along the 0th dimension of ``x``:\n\n    >>> upfirdn(h, x, 2, axis=0)\n    array([[ 0.,  1.],\n           [ 0.,  1.],\n           [ 2.,  3.],\n           [ 2.,  3.],\n           [ 4.,  5.],\n           [ 4.,  5.],\n           [ 6.,  7.],\n           [ 6.,  7.]])\n\n    ')
    
    # Assigning a Call to a Name (line 180):
    
    # Call to asarray(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'x' (line 180)
    x_288824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'x', False)
    # Processing the call keyword arguments (line 180)
    kwargs_288825 = {}
    # Getting the type of 'np' (line 180)
    np_288822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 180)
    asarray_288823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), np_288822, 'asarray')
    # Calling asarray(args, kwargs) (line 180)
    asarray_call_result_288826 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), asarray_288823, *[x_288824], **kwargs_288825)
    
    # Assigning a type to the variable 'x' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'x', asarray_call_result_288826)
    
    # Assigning a Call to a Name (line 181):
    
    # Call to _UpFIRDn(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'h' (line 181)
    h_288828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'h', False)
    # Getting the type of 'x' (line 181)
    x_288829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'x', False)
    # Obtaining the member 'dtype' of a type (line 181)
    dtype_288830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 22), x_288829, 'dtype')
    # Getting the type of 'up' (line 181)
    up_288831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'up', False)
    # Getting the type of 'down' (line 181)
    down_288832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 35), 'down', False)
    # Processing the call keyword arguments (line 181)
    kwargs_288833 = {}
    # Getting the type of '_UpFIRDn' (line 181)
    _UpFIRDn_288827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 10), '_UpFIRDn', False)
    # Calling _UpFIRDn(args, kwargs) (line 181)
    _UpFIRDn_call_result_288834 = invoke(stypy.reporting.localization.Localization(__file__, 181, 10), _UpFIRDn_288827, *[h_288828, dtype_288830, up_288831, down_288832], **kwargs_288833)
    
    # Assigning a type to the variable 'ufd' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'ufd', _UpFIRDn_call_result_288834)
    
    # Call to apply_filter(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'x' (line 183)
    x_288837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'x', False)
    # Getting the type of 'axis' (line 183)
    axis_288838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 31), 'axis', False)
    # Processing the call keyword arguments (line 183)
    kwargs_288839 = {}
    # Getting the type of 'ufd' (line 183)
    ufd_288835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'ufd', False)
    # Obtaining the member 'apply_filter' of a type (line 183)
    apply_filter_288836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 11), ufd_288835, 'apply_filter')
    # Calling apply_filter(args, kwargs) (line 183)
    apply_filter_call_result_288840 = invoke(stypy.reporting.localization.Localization(__file__, 183, 11), apply_filter_288836, *[x_288837, axis_288838], **kwargs_288839)
    
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type', apply_filter_call_result_288840)
    
    # ################# End of 'upfirdn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'upfirdn' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_288841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288841)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'upfirdn'
    return stypy_return_type_288841

# Assigning a type to the variable 'upfirdn' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'upfirdn', upfirdn)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

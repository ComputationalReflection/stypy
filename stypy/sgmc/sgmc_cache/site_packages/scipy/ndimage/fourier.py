
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2003-2005 Peter J. Verveer
2: #
3: # Redistribution and use in source and binary forms, with or without
4: # modification, are permitted provided that the following conditions
5: # are met:
6: #
7: # 1. Redistributions of source code must retain the above copyright
8: #    notice, this list of conditions and the following disclaimer.
9: #
10: # 2. Redistributions in binary form must reproduce the above
11: #    copyright notice, this list of conditions and the following
12: #    disclaimer in the documentation and/or other materials provided
13: #    with the distribution.
14: #
15: # 3. The name of the author may not be used to endorse or promote
16: #    products derived from this software without specific prior
17: #    written permission.
18: #
19: # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
20: # OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
21: # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
22: # ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
23: # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
24: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
25: # GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
26: # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
27: # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
28: # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
29: # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
30: 
31: from __future__ import division, print_function, absolute_import
32: 
33: import numpy
34: from . import _ni_support
35: from . import _nd_image
36: 
37: __all__ = ['fourier_gaussian', 'fourier_uniform', 'fourier_ellipsoid',
38:            'fourier_shift']
39: 
40: 
41: def _get_output_fourier(output, input):
42:     if output is None:
43:         if input.dtype.type in [numpy.complex64, numpy.complex128,
44:                                 numpy.float32]:
45:             output = numpy.zeros(input.shape, dtype=input.dtype)
46:         else:
47:             output = numpy.zeros(input.shape, dtype=numpy.float64)
48:         return_value = output
49:     elif type(output) is type:
50:         if output not in [numpy.complex64, numpy.complex128,
51:                           numpy.float32, numpy.float64]:
52:             raise RuntimeError("output type not supported")
53:         output = numpy.zeros(input.shape, dtype=output)
54:         return_value = output
55:     else:
56:         if output.shape != input.shape:
57:             raise RuntimeError("output shape not correct")
58:         return_value = None
59:     return output, return_value
60: 
61: 
62: def _get_output_fourier_complex(output, input):
63:     if output is None:
64:         if input.dtype.type in [numpy.complex64, numpy.complex128]:
65:             output = numpy.zeros(input.shape, dtype=input.dtype)
66:         else:
67:             output = numpy.zeros(input.shape, dtype=numpy.complex128)
68:         return_value = output
69:     elif type(output) is type:
70:         if output not in [numpy.complex64, numpy.complex128]:
71:             raise RuntimeError("output type not supported")
72:         output = numpy.zeros(input.shape, dtype=output)
73:         return_value = output
74:     else:
75:         if output.shape != input.shape:
76:             raise RuntimeError("output shape not correct")
77:         return_value = None
78:     return output, return_value
79: 
80: 
81: def fourier_gaussian(input, sigma, n=-1, axis=-1, output=None):
82:     '''
83:     Multi-dimensional Gaussian fourier filter.
84: 
85:     The array is multiplied with the fourier transform of a Gaussian
86:     kernel.
87: 
88:     Parameters
89:     ----------
90:     input : array_like
91:         The input array.
92:     sigma : float or sequence
93:         The sigma of the Gaussian kernel. If a float, `sigma` is the same for
94:         all axes. If a sequence, `sigma` has to contain one value for each
95:         axis.
96:     n : int, optional
97:         If `n` is negative (default), then the input is assumed to be the
98:         result of a complex fft.
99:         If `n` is larger than or equal to zero, the input is assumed to be the
100:         result of a real fft, and `n` gives the length of the array before
101:         transformation along the real transform direction.
102:     axis : int, optional
103:         The axis of the real transform.
104:     output : ndarray, optional
105:         If given, the result of filtering the input is placed in this array.
106:         None is returned in this case.
107: 
108:     Returns
109:     -------
110:     fourier_gaussian : ndarray or None
111:         The filtered input. If `output` is given as a parameter, None is
112:         returned.
113: 
114:     Examples
115:     --------
116:     >>> from scipy import ndimage, misc
117:     >>> import numpy.fft
118:     >>> import matplotlib.pyplot as plt
119:     >>> fig, (ax1, ax2) = plt.subplots(1, 2)
120:     >>> plt.gray()  # show the filtered result in grayscale
121:     >>> ascent = misc.ascent()
122:     >>> input_ = numpy.fft.fft2(ascent)
123:     >>> result = ndimage.fourier_gaussian(input_, sigma=4)
124:     >>> result = numpy.fft.ifft2(result)
125:     >>> ax1.imshow(ascent)
126:     >>> ax2.imshow(result.real)  # the imaginary part is an artifact
127:     >>> plt.show()
128:     '''
129:     input = numpy.asarray(input)
130:     output, return_value = _get_output_fourier(output, input)
131:     axis = _ni_support._check_axis(axis, input.ndim)
132:     sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
133:     sigmas = numpy.asarray(sigmas, dtype=numpy.float64)
134:     if not sigmas.flags.contiguous:
135:         sigmas = sigmas.copy()
136: 
137:     _nd_image.fourier_filter(input, sigmas, n, axis, output, 0)
138:     return return_value
139: 
140: 
141: def fourier_uniform(input, size, n=-1, axis=-1, output=None):
142:     '''
143:     Multi-dimensional uniform fourier filter.
144: 
145:     The array is multiplied with the fourier transform of a box of given
146:     size.
147: 
148:     Parameters
149:     ----------
150:     input : array_like
151:         The input array.
152:     size : float or sequence
153:         The size of the box used for filtering.
154:         If a float, `size` is the same for all axes. If a sequence, `size` has
155:         to contain one value for each axis.
156:     n : int, optional
157:         If `n` is negative (default), then the input is assumed to be the
158:         result of a complex fft.
159:         If `n` is larger than or equal to zero, the input is assumed to be the
160:         result of a real fft, and `n` gives the length of the array before
161:         transformation along the real transform direction.
162:     axis : int, optional
163:         The axis of the real transform.
164:     output : ndarray, optional
165:         If given, the result of filtering the input is placed in this array.
166:         None is returned in this case.
167: 
168:     Returns
169:     -------
170:     fourier_uniform : ndarray or None
171:         The filtered input. If `output` is given as a parameter, None is
172:         returned.
173: 
174:     Examples
175:     --------
176:     >>> from scipy import ndimage, misc
177:     >>> import numpy.fft
178:     >>> import matplotlib.pyplot as plt
179:     >>> fig, (ax1, ax2) = plt.subplots(1, 2)
180:     >>> plt.gray()  # show the filtered result in grayscale
181:     >>> ascent = misc.ascent()
182:     >>> input_ = numpy.fft.fft2(ascent)
183:     >>> result = ndimage.fourier_uniform(input_, size=20)
184:     >>> result = numpy.fft.ifft2(result)
185:     >>> ax1.imshow(ascent)
186:     >>> ax2.imshow(result.real)  # the imaginary part is an artifact
187:     >>> plt.show()
188:     '''
189:     input = numpy.asarray(input)
190:     output, return_value = _get_output_fourier(output, input)
191:     axis = _ni_support._check_axis(axis, input.ndim)
192:     sizes = _ni_support._normalize_sequence(size, input.ndim)
193:     sizes = numpy.asarray(sizes, dtype=numpy.float64)
194:     if not sizes.flags.contiguous:
195:         sizes = sizes.copy()
196:     _nd_image.fourier_filter(input, sizes, n, axis, output, 1)
197:     return return_value
198: 
199: 
200: def fourier_ellipsoid(input, size, n=-1, axis=-1, output=None):
201:     '''
202:     Multi-dimensional ellipsoid fourier filter.
203: 
204:     The array is multiplied with the fourier transform of a ellipsoid of
205:     given sizes.
206: 
207:     Parameters
208:     ----------
209:     input : array_like
210:         The input array.
211:     size : float or sequence
212:         The size of the box used for filtering.
213:         If a float, `size` is the same for all axes. If a sequence, `size` has
214:         to contain one value for each axis.
215:     n : int, optional
216:         If `n` is negative (default), then the input is assumed to be the
217:         result of a complex fft.
218:         If `n` is larger than or equal to zero, the input is assumed to be the
219:         result of a real fft, and `n` gives the length of the array before
220:         transformation along the real transform direction.
221:     axis : int, optional
222:         The axis of the real transform.
223:     output : ndarray, optional
224:         If given, the result of filtering the input is placed in this array.
225:         None is returned in this case.
226: 
227:     Returns
228:     -------
229:     fourier_ellipsoid : ndarray or None
230:         The filtered input. If `output` is given as a parameter, None is
231:         returned.
232: 
233:     Notes
234:     -----
235:     This function is implemented for arrays of rank 1, 2, or 3.
236: 
237:     Examples
238:     --------
239:     >>> from scipy import ndimage, misc
240:     >>> import numpy.fft
241:     >>> import matplotlib.pyplot as plt
242:     >>> fig, (ax1, ax2) = plt.subplots(1, 2)
243:     >>> plt.gray()  # show the filtered result in grayscale
244:     >>> ascent = misc.ascent()
245:     >>> input_ = numpy.fft.fft2(ascent)
246:     >>> result = ndimage.fourier_ellipsoid(input_, size=20)
247:     >>> result = numpy.fft.ifft2(result)
248:     >>> ax1.imshow(ascent)
249:     >>> ax2.imshow(result.real)  # the imaginary part is an artifact
250:     >>> plt.show()
251:     '''
252:     input = numpy.asarray(input)
253:     output, return_value = _get_output_fourier(output, input)
254:     axis = _ni_support._check_axis(axis, input.ndim)
255:     sizes = _ni_support._normalize_sequence(size, input.ndim)
256:     sizes = numpy.asarray(sizes, dtype=numpy.float64)
257:     if not sizes.flags.contiguous:
258:         sizes = sizes.copy()
259:     _nd_image.fourier_filter(input, sizes, n, axis, output, 2)
260:     return return_value
261: 
262: 
263: def fourier_shift(input, shift, n=-1, axis=-1, output=None):
264:     '''
265:     Multi-dimensional fourier shift filter.
266: 
267:     The array is multiplied with the fourier transform of a shift operation.
268: 
269:     Parameters
270:     ----------
271:     input : array_like
272:         The input array.
273:     shift : float or sequence
274:         The size of the box used for filtering.
275:         If a float, `shift` is the same for all axes. If a sequence, `shift`
276:         has to contain one value for each axis.
277:     n : int, optional
278:         If `n` is negative (default), then the input is assumed to be the
279:         result of a complex fft.
280:         If `n` is larger than or equal to zero, the input is assumed to be the
281:         result of a real fft, and `n` gives the length of the array before
282:         transformation along the real transform direction.
283:     axis : int, optional
284:         The axis of the real transform.
285:     output : ndarray, optional
286:         If given, the result of shifting the input is placed in this array.
287:         None is returned in this case.
288: 
289:     Returns
290:     -------
291:     fourier_shift : ndarray or None
292:         The shifted input. If `output` is given as a parameter, None is
293:         returned.
294: 
295:     Examples
296:     --------
297:     >>> from scipy import ndimage, misc
298:     >>> import matplotlib.pyplot as plt
299:     >>> import numpy.fft
300:     >>> fig, (ax1, ax2) = plt.subplots(1, 2)
301:     >>> plt.gray()  # show the filtered result in grayscale
302:     >>> ascent = misc.ascent()
303:     >>> input_ = numpy.fft.fft2(ascent)
304:     >>> result = ndimage.fourier_shift(input_, shift=200)
305:     >>> result = numpy.fft.ifft2(result)
306:     >>> ax1.imshow(ascent)
307:     >>> ax2.imshow(result.real)  # the imaginary part is an artifact
308:     >>> plt.show()
309:     '''
310:     input = numpy.asarray(input)
311:     output, return_value = _get_output_fourier_complex(output, input)
312:     axis = _ni_support._check_axis(axis, input.ndim)
313:     shifts = _ni_support._normalize_sequence(shift, input.ndim)
314:     shifts = numpy.asarray(shifts, dtype=numpy.float64)
315:     if not shifts.flags.contiguous:
316:         shifts = shifts.copy()
317:     _nd_image.fourier_shift(input, shifts, n, axis, output)
318:     return return_value
319: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import numpy' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_119627 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy')

if (type(import_119627) is not StypyTypeError):

    if (import_119627 != 'pyd_module'):
        __import__(import_119627)
        sys_modules_119628 = sys.modules[import_119627]
        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', sys_modules_119628.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', import_119627)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from scipy.ndimage import _ni_support' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_119629 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.ndimage')

if (type(import_119629) is not StypyTypeError):

    if (import_119629 != 'pyd_module'):
        __import__(import_119629)
        sys_modules_119630 = sys.modules[import_119629]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.ndimage', sys_modules_119630.module_type_store, module_type_store, ['_ni_support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_119630, sys_modules_119630.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _ni_support

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.ndimage', None, module_type_store, ['_ni_support'], [_ni_support])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.ndimage', import_119629)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from scipy.ndimage import _nd_image' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_119631 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage')

if (type(import_119631) is not StypyTypeError):

    if (import_119631 != 'pyd_module'):
        __import__(import_119631)
        sys_modules_119632 = sys.modules[import_119631]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', sys_modules_119632.module_type_store, module_type_store, ['_nd_image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_119632, sys_modules_119632.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _nd_image

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', None, module_type_store, ['_nd_image'], [_nd_image])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', import_119631)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


# Assigning a List to a Name (line 37):

# Assigning a List to a Name (line 37):
__all__ = ['fourier_gaussian', 'fourier_uniform', 'fourier_ellipsoid', 'fourier_shift']
module_type_store.set_exportable_members(['fourier_gaussian', 'fourier_uniform', 'fourier_ellipsoid', 'fourier_shift'])

# Obtaining an instance of the builtin type 'list' (line 37)
list_119633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
str_119634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'str', 'fourier_gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 10), list_119633, str_119634)
# Adding element type (line 37)
str_119635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'str', 'fourier_uniform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 10), list_119633, str_119635)
# Adding element type (line 37)
str_119636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 50), 'str', 'fourier_ellipsoid')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 10), list_119633, str_119636)
# Adding element type (line 37)
str_119637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 11), 'str', 'fourier_shift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 10), list_119633, str_119637)

# Assigning a type to the variable '__all__' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '__all__', list_119633)

@norecursion
def _get_output_fourier(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_output_fourier'
    module_type_store = module_type_store.open_function_context('_get_output_fourier', 41, 0, False)
    
    # Passed parameters checking function
    _get_output_fourier.stypy_localization = localization
    _get_output_fourier.stypy_type_of_self = None
    _get_output_fourier.stypy_type_store = module_type_store
    _get_output_fourier.stypy_function_name = '_get_output_fourier'
    _get_output_fourier.stypy_param_names_list = ['output', 'input']
    _get_output_fourier.stypy_varargs_param_name = None
    _get_output_fourier.stypy_kwargs_param_name = None
    _get_output_fourier.stypy_call_defaults = defaults
    _get_output_fourier.stypy_call_varargs = varargs
    _get_output_fourier.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_output_fourier', ['output', 'input'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_output_fourier', localization, ['output', 'input'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_output_fourier(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 42)
    # Getting the type of 'output' (line 42)
    output_119638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'output')
    # Getting the type of 'None' (line 42)
    None_119639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'None')
    
    (may_be_119640, more_types_in_union_119641) = may_be_none(output_119638, None_119639)

    if may_be_119640:

        if more_types_in_union_119641:
            # Runtime conditional SSA (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'input' (line 43)
        input_119642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'input')
        # Obtaining the member 'dtype' of a type (line 43)
        dtype_119643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 11), input_119642, 'dtype')
        # Obtaining the member 'type' of a type (line 43)
        type_119644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 11), dtype_119643, 'type')
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_119645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        # Getting the type of 'numpy' (line 43)
        numpy_119646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'numpy')
        # Obtaining the member 'complex64' of a type (line 43)
        complex64_119647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), numpy_119646, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 31), list_119645, complex64_119647)
        # Adding element type (line 43)
        # Getting the type of 'numpy' (line 43)
        numpy_119648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 49), 'numpy')
        # Obtaining the member 'complex128' of a type (line 43)
        complex128_119649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 49), numpy_119648, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 31), list_119645, complex128_119649)
        # Adding element type (line 43)
        # Getting the type of 'numpy' (line 44)
        numpy_119650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 32), 'numpy')
        # Obtaining the member 'float32' of a type (line 44)
        float32_119651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 32), numpy_119650, 'float32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 31), list_119645, float32_119651)
        
        # Applying the binary operator 'in' (line 43)
        result_contains_119652 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), 'in', type_119644, list_119645)
        
        # Testing the type of an if condition (line 43)
        if_condition_119653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), result_contains_119652)
        # Assigning a type to the variable 'if_condition_119653' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_119653', if_condition_119653)
        # SSA begins for if statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to zeros(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'input' (line 45)
        input_119656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'input', False)
        # Obtaining the member 'shape' of a type (line 45)
        shape_119657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 33), input_119656, 'shape')
        # Processing the call keyword arguments (line 45)
        # Getting the type of 'input' (line 45)
        input_119658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 52), 'input', False)
        # Obtaining the member 'dtype' of a type (line 45)
        dtype_119659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 52), input_119658, 'dtype')
        keyword_119660 = dtype_119659
        kwargs_119661 = {'dtype': keyword_119660}
        # Getting the type of 'numpy' (line 45)
        numpy_119654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 45)
        zeros_119655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 21), numpy_119654, 'zeros')
        # Calling zeros(args, kwargs) (line 45)
        zeros_call_result_119662 = invoke(stypy.reporting.localization.Localization(__file__, 45, 21), zeros_119655, *[shape_119657], **kwargs_119661)
        
        # Assigning a type to the variable 'output' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'output', zeros_call_result_119662)
        # SSA branch for the else part of an if statement (line 43)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to zeros(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'input' (line 47)
        input_119665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'input', False)
        # Obtaining the member 'shape' of a type (line 47)
        shape_119666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 33), input_119665, 'shape')
        # Processing the call keyword arguments (line 47)
        # Getting the type of 'numpy' (line 47)
        numpy_119667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 52), 'numpy', False)
        # Obtaining the member 'float64' of a type (line 47)
        float64_119668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 52), numpy_119667, 'float64')
        keyword_119669 = float64_119668
        kwargs_119670 = {'dtype': keyword_119669}
        # Getting the type of 'numpy' (line 47)
        numpy_119663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 47)
        zeros_119664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 21), numpy_119663, 'zeros')
        # Calling zeros(args, kwargs) (line 47)
        zeros_call_result_119671 = invoke(stypy.reporting.localization.Localization(__file__, 47, 21), zeros_119664, *[shape_119666], **kwargs_119670)
        
        # Assigning a type to the variable 'output' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'output', zeros_call_result_119671)
        # SSA join for if statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 48):
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'output' (line 48)
        output_119672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'output')
        # Assigning a type to the variable 'return_value' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'return_value', output_119672)

        if more_types_in_union_119641:
            # Runtime conditional SSA for else branch (line 42)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_119640) or more_types_in_union_119641):
        
        # Type idiom detected: calculating its left and rigth part (line 49)
        # Getting the type of 'output' (line 49)
        output_119673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'output')
        # Getting the type of 'type' (line 49)
        type_119674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 25), 'type')
        
        (may_be_119675, more_types_in_union_119676) = may_be_type(output_119673, type_119674)

        if may_be_119675:

            if more_types_in_union_119676:
                # Runtime conditional SSA (line 49)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'output' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'output', type_119674())
            
            
            # Getting the type of 'output' (line 50)
            output_119677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'output')
            
            # Obtaining an instance of the builtin type 'list' (line 50)
            list_119678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 50)
            # Adding element type (line 50)
            # Getting the type of 'numpy' (line 50)
            numpy_119679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'numpy')
            # Obtaining the member 'complex64' of a type (line 50)
            complex64_119680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 26), numpy_119679, 'complex64')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 25), list_119678, complex64_119680)
            # Adding element type (line 50)
            # Getting the type of 'numpy' (line 50)
            numpy_119681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 43), 'numpy')
            # Obtaining the member 'complex128' of a type (line 50)
            complex128_119682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 43), numpy_119681, 'complex128')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 25), list_119678, complex128_119682)
            # Adding element type (line 50)
            # Getting the type of 'numpy' (line 51)
            numpy_119683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'numpy')
            # Obtaining the member 'float32' of a type (line 51)
            float32_119684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 26), numpy_119683, 'float32')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 25), list_119678, float32_119684)
            # Adding element type (line 50)
            # Getting the type of 'numpy' (line 51)
            numpy_119685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 41), 'numpy')
            # Obtaining the member 'float64' of a type (line 51)
            float64_119686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 41), numpy_119685, 'float64')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 25), list_119678, float64_119686)
            
            # Applying the binary operator 'notin' (line 50)
            result_contains_119687 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), 'notin', output_119677, list_119678)
            
            # Testing the type of an if condition (line 50)
            if_condition_119688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_contains_119687)
            # Assigning a type to the variable 'if_condition_119688' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_119688', if_condition_119688)
            # SSA begins for if statement (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to RuntimeError(...): (line 52)
            # Processing the call arguments (line 52)
            str_119690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 31), 'str', 'output type not supported')
            # Processing the call keyword arguments (line 52)
            kwargs_119691 = {}
            # Getting the type of 'RuntimeError' (line 52)
            RuntimeError_119689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 52)
            RuntimeError_call_result_119692 = invoke(stypy.reporting.localization.Localization(__file__, 52, 18), RuntimeError_119689, *[str_119690], **kwargs_119691)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 52, 12), RuntimeError_call_result_119692, 'raise parameter', BaseException)
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 53):
            
            # Assigning a Call to a Name (line 53):
            
            # Call to zeros(...): (line 53)
            # Processing the call arguments (line 53)
            # Getting the type of 'input' (line 53)
            input_119695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'input', False)
            # Obtaining the member 'shape' of a type (line 53)
            shape_119696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 29), input_119695, 'shape')
            # Processing the call keyword arguments (line 53)
            # Getting the type of 'output' (line 53)
            output_119697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 48), 'output', False)
            keyword_119698 = output_119697
            kwargs_119699 = {'dtype': keyword_119698}
            # Getting the type of 'numpy' (line 53)
            numpy_119693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'numpy', False)
            # Obtaining the member 'zeros' of a type (line 53)
            zeros_119694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 17), numpy_119693, 'zeros')
            # Calling zeros(args, kwargs) (line 53)
            zeros_call_result_119700 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), zeros_119694, *[shape_119696], **kwargs_119699)
            
            # Assigning a type to the variable 'output' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'output', zeros_call_result_119700)
            
            # Assigning a Name to a Name (line 54):
            
            # Assigning a Name to a Name (line 54):
            # Getting the type of 'output' (line 54)
            output_119701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'output')
            # Assigning a type to the variable 'return_value' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'return_value', output_119701)

            if more_types_in_union_119676:
                # Runtime conditional SSA for else branch (line 49)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_119675) or more_types_in_union_119676):
            # Getting the type of 'output' (line 49)
            output_119702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'output')
            # Assigning a type to the variable 'output' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 9), 'output', remove_type_from_union(output_119702, type_119674))
            
            
            # Getting the type of 'output' (line 56)
            output_119703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'output')
            # Obtaining the member 'shape' of a type (line 56)
            shape_119704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), output_119703, 'shape')
            # Getting the type of 'input' (line 56)
            input_119705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'input')
            # Obtaining the member 'shape' of a type (line 56)
            shape_119706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 27), input_119705, 'shape')
            # Applying the binary operator '!=' (line 56)
            result_ne_119707 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 11), '!=', shape_119704, shape_119706)
            
            # Testing the type of an if condition (line 56)
            if_condition_119708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 8), result_ne_119707)
            # Assigning a type to the variable 'if_condition_119708' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'if_condition_119708', if_condition_119708)
            # SSA begins for if statement (line 56)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to RuntimeError(...): (line 57)
            # Processing the call arguments (line 57)
            str_119710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'str', 'output shape not correct')
            # Processing the call keyword arguments (line 57)
            kwargs_119711 = {}
            # Getting the type of 'RuntimeError' (line 57)
            RuntimeError_119709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 57)
            RuntimeError_call_result_119712 = invoke(stypy.reporting.localization.Localization(__file__, 57, 18), RuntimeError_119709, *[str_119710], **kwargs_119711)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 57, 12), RuntimeError_call_result_119712, 'raise parameter', BaseException)
            # SSA join for if statement (line 56)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 58):
            
            # Assigning a Name to a Name (line 58):
            # Getting the type of 'None' (line 58)
            None_119713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'None')
            # Assigning a type to the variable 'return_value' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'return_value', None_119713)

            if (may_be_119675 and more_types_in_union_119676):
                # SSA join for if statement (line 49)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_119640 and more_types_in_union_119641):
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 59)
    tuple_119714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 59)
    # Adding element type (line 59)
    # Getting the type of 'output' (line 59)
    output_119715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'output')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 11), tuple_119714, output_119715)
    # Adding element type (line 59)
    # Getting the type of 'return_value' (line 59)
    return_value_119716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'return_value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 11), tuple_119714, return_value_119716)
    
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type', tuple_119714)
    
    # ################# End of '_get_output_fourier(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_output_fourier' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_119717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119717)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_output_fourier'
    return stypy_return_type_119717

# Assigning a type to the variable '_get_output_fourier' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_get_output_fourier', _get_output_fourier)

@norecursion
def _get_output_fourier_complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_output_fourier_complex'
    module_type_store = module_type_store.open_function_context('_get_output_fourier_complex', 62, 0, False)
    
    # Passed parameters checking function
    _get_output_fourier_complex.stypy_localization = localization
    _get_output_fourier_complex.stypy_type_of_self = None
    _get_output_fourier_complex.stypy_type_store = module_type_store
    _get_output_fourier_complex.stypy_function_name = '_get_output_fourier_complex'
    _get_output_fourier_complex.stypy_param_names_list = ['output', 'input']
    _get_output_fourier_complex.stypy_varargs_param_name = None
    _get_output_fourier_complex.stypy_kwargs_param_name = None
    _get_output_fourier_complex.stypy_call_defaults = defaults
    _get_output_fourier_complex.stypy_call_varargs = varargs
    _get_output_fourier_complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_output_fourier_complex', ['output', 'input'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_output_fourier_complex', localization, ['output', 'input'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_output_fourier_complex(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 63)
    # Getting the type of 'output' (line 63)
    output_119718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'output')
    # Getting the type of 'None' (line 63)
    None_119719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'None')
    
    (may_be_119720, more_types_in_union_119721) = may_be_none(output_119718, None_119719)

    if may_be_119720:

        if more_types_in_union_119721:
            # Runtime conditional SSA (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'input' (line 64)
        input_119722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'input')
        # Obtaining the member 'dtype' of a type (line 64)
        dtype_119723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), input_119722, 'dtype')
        # Obtaining the member 'type' of a type (line 64)
        type_119724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), dtype_119723, 'type')
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_119725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        # Getting the type of 'numpy' (line 64)
        numpy_119726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'numpy')
        # Obtaining the member 'complex64' of a type (line 64)
        complex64_119727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 32), numpy_119726, 'complex64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 31), list_119725, complex64_119727)
        # Adding element type (line 64)
        # Getting the type of 'numpy' (line 64)
        numpy_119728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 49), 'numpy')
        # Obtaining the member 'complex128' of a type (line 64)
        complex128_119729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 49), numpy_119728, 'complex128')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 31), list_119725, complex128_119729)
        
        # Applying the binary operator 'in' (line 64)
        result_contains_119730 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), 'in', type_119724, list_119725)
        
        # Testing the type of an if condition (line 64)
        if_condition_119731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_contains_119730)
        # Assigning a type to the variable 'if_condition_119731' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_119731', if_condition_119731)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to zeros(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'input' (line 65)
        input_119734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'input', False)
        # Obtaining the member 'shape' of a type (line 65)
        shape_119735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 33), input_119734, 'shape')
        # Processing the call keyword arguments (line 65)
        # Getting the type of 'input' (line 65)
        input_119736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 52), 'input', False)
        # Obtaining the member 'dtype' of a type (line 65)
        dtype_119737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 52), input_119736, 'dtype')
        keyword_119738 = dtype_119737
        kwargs_119739 = {'dtype': keyword_119738}
        # Getting the type of 'numpy' (line 65)
        numpy_119732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 65)
        zeros_119733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), numpy_119732, 'zeros')
        # Calling zeros(args, kwargs) (line 65)
        zeros_call_result_119740 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), zeros_119733, *[shape_119735], **kwargs_119739)
        
        # Assigning a type to the variable 'output' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'output', zeros_call_result_119740)
        # SSA branch for the else part of an if statement (line 64)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to zeros(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'input' (line 67)
        input_119743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'input', False)
        # Obtaining the member 'shape' of a type (line 67)
        shape_119744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 33), input_119743, 'shape')
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'numpy' (line 67)
        numpy_119745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 52), 'numpy', False)
        # Obtaining the member 'complex128' of a type (line 67)
        complex128_119746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 52), numpy_119745, 'complex128')
        keyword_119747 = complex128_119746
        kwargs_119748 = {'dtype': keyword_119747}
        # Getting the type of 'numpy' (line 67)
        numpy_119741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'numpy', False)
        # Obtaining the member 'zeros' of a type (line 67)
        zeros_119742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 21), numpy_119741, 'zeros')
        # Calling zeros(args, kwargs) (line 67)
        zeros_call_result_119749 = invoke(stypy.reporting.localization.Localization(__file__, 67, 21), zeros_119742, *[shape_119744], **kwargs_119748)
        
        # Assigning a type to the variable 'output' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'output', zeros_call_result_119749)
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 68):
        
        # Assigning a Name to a Name (line 68):
        # Getting the type of 'output' (line 68)
        output_119750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'output')
        # Assigning a type to the variable 'return_value' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'return_value', output_119750)

        if more_types_in_union_119721:
            # Runtime conditional SSA for else branch (line 63)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_119720) or more_types_in_union_119721):
        
        # Type idiom detected: calculating its left and rigth part (line 69)
        # Getting the type of 'output' (line 69)
        output_119751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'output')
        # Getting the type of 'type' (line 69)
        type_119752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'type')
        
        (may_be_119753, more_types_in_union_119754) = may_be_type(output_119751, type_119752)

        if may_be_119753:

            if more_types_in_union_119754:
                # Runtime conditional SSA (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'output' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'output', type_119752())
            
            
            # Getting the type of 'output' (line 70)
            output_119755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'output')
            
            # Obtaining an instance of the builtin type 'list' (line 70)
            list_119756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 70)
            # Adding element type (line 70)
            # Getting the type of 'numpy' (line 70)
            numpy_119757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'numpy')
            # Obtaining the member 'complex64' of a type (line 70)
            complex64_119758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 26), numpy_119757, 'complex64')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 25), list_119756, complex64_119758)
            # Adding element type (line 70)
            # Getting the type of 'numpy' (line 70)
            numpy_119759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'numpy')
            # Obtaining the member 'complex128' of a type (line 70)
            complex128_119760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 43), numpy_119759, 'complex128')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 25), list_119756, complex128_119760)
            
            # Applying the binary operator 'notin' (line 70)
            result_contains_119761 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), 'notin', output_119755, list_119756)
            
            # Testing the type of an if condition (line 70)
            if_condition_119762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), result_contains_119761)
            # Assigning a type to the variable 'if_condition_119762' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_119762', if_condition_119762)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to RuntimeError(...): (line 71)
            # Processing the call arguments (line 71)
            str_119764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 31), 'str', 'output type not supported')
            # Processing the call keyword arguments (line 71)
            kwargs_119765 = {}
            # Getting the type of 'RuntimeError' (line 71)
            RuntimeError_119763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 71)
            RuntimeError_call_result_119766 = invoke(stypy.reporting.localization.Localization(__file__, 71, 18), RuntimeError_119763, *[str_119764], **kwargs_119765)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 12), RuntimeError_call_result_119766, 'raise parameter', BaseException)
            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 72):
            
            # Assigning a Call to a Name (line 72):
            
            # Call to zeros(...): (line 72)
            # Processing the call arguments (line 72)
            # Getting the type of 'input' (line 72)
            input_119769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'input', False)
            # Obtaining the member 'shape' of a type (line 72)
            shape_119770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 29), input_119769, 'shape')
            # Processing the call keyword arguments (line 72)
            # Getting the type of 'output' (line 72)
            output_119771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 48), 'output', False)
            keyword_119772 = output_119771
            kwargs_119773 = {'dtype': keyword_119772}
            # Getting the type of 'numpy' (line 72)
            numpy_119767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'numpy', False)
            # Obtaining the member 'zeros' of a type (line 72)
            zeros_119768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), numpy_119767, 'zeros')
            # Calling zeros(args, kwargs) (line 72)
            zeros_call_result_119774 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), zeros_119768, *[shape_119770], **kwargs_119773)
            
            # Assigning a type to the variable 'output' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'output', zeros_call_result_119774)
            
            # Assigning a Name to a Name (line 73):
            
            # Assigning a Name to a Name (line 73):
            # Getting the type of 'output' (line 73)
            output_119775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'output')
            # Assigning a type to the variable 'return_value' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'return_value', output_119775)

            if more_types_in_union_119754:
                # Runtime conditional SSA for else branch (line 69)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_119753) or more_types_in_union_119754):
            # Getting the type of 'output' (line 69)
            output_119776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'output')
            # Assigning a type to the variable 'output' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'output', remove_type_from_union(output_119776, type_119752))
            
            
            # Getting the type of 'output' (line 75)
            output_119777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'output')
            # Obtaining the member 'shape' of a type (line 75)
            shape_119778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), output_119777, 'shape')
            # Getting the type of 'input' (line 75)
            input_119779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'input')
            # Obtaining the member 'shape' of a type (line 75)
            shape_119780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 27), input_119779, 'shape')
            # Applying the binary operator '!=' (line 75)
            result_ne_119781 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), '!=', shape_119778, shape_119780)
            
            # Testing the type of an if condition (line 75)
            if_condition_119782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_ne_119781)
            # Assigning a type to the variable 'if_condition_119782' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_119782', if_condition_119782)
            # SSA begins for if statement (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to RuntimeError(...): (line 76)
            # Processing the call arguments (line 76)
            str_119784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 31), 'str', 'output shape not correct')
            # Processing the call keyword arguments (line 76)
            kwargs_119785 = {}
            # Getting the type of 'RuntimeError' (line 76)
            RuntimeError_119783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 76)
            RuntimeError_call_result_119786 = invoke(stypy.reporting.localization.Localization(__file__, 76, 18), RuntimeError_119783, *[str_119784], **kwargs_119785)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 76, 12), RuntimeError_call_result_119786, 'raise parameter', BaseException)
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 77):
            
            # Assigning a Name to a Name (line 77):
            # Getting the type of 'None' (line 77)
            None_119787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'None')
            # Assigning a type to the variable 'return_value' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'return_value', None_119787)

            if (may_be_119753 and more_types_in_union_119754):
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_119720 and more_types_in_union_119721):
            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_119788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'output' (line 78)
    output_119789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'output')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_119788, output_119789)
    # Adding element type (line 78)
    # Getting the type of 'return_value' (line 78)
    return_value_119790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'return_value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_119788, return_value_119790)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', tuple_119788)
    
    # ################# End of '_get_output_fourier_complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_output_fourier_complex' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_119791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119791)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_output_fourier_complex'
    return stypy_return_type_119791

# Assigning a type to the variable '_get_output_fourier_complex' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), '_get_output_fourier_complex', _get_output_fourier_complex)

@norecursion
def fourier_gaussian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_119792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 37), 'int')
    int_119793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 46), 'int')
    # Getting the type of 'None' (line 81)
    None_119794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 57), 'None')
    defaults = [int_119792, int_119793, None_119794]
    # Create a new context for function 'fourier_gaussian'
    module_type_store = module_type_store.open_function_context('fourier_gaussian', 81, 0, False)
    
    # Passed parameters checking function
    fourier_gaussian.stypy_localization = localization
    fourier_gaussian.stypy_type_of_self = None
    fourier_gaussian.stypy_type_store = module_type_store
    fourier_gaussian.stypy_function_name = 'fourier_gaussian'
    fourier_gaussian.stypy_param_names_list = ['input', 'sigma', 'n', 'axis', 'output']
    fourier_gaussian.stypy_varargs_param_name = None
    fourier_gaussian.stypy_kwargs_param_name = None
    fourier_gaussian.stypy_call_defaults = defaults
    fourier_gaussian.stypy_call_varargs = varargs
    fourier_gaussian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fourier_gaussian', ['input', 'sigma', 'n', 'axis', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fourier_gaussian', localization, ['input', 'sigma', 'n', 'axis', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fourier_gaussian(...)' code ##################

    str_119795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', '\n    Multi-dimensional Gaussian fourier filter.\n\n    The array is multiplied with the fourier transform of a Gaussian\n    kernel.\n\n    Parameters\n    ----------\n    input : array_like\n        The input array.\n    sigma : float or sequence\n        The sigma of the Gaussian kernel. If a float, `sigma` is the same for\n        all axes. If a sequence, `sigma` has to contain one value for each\n        axis.\n    n : int, optional\n        If `n` is negative (default), then the input is assumed to be the\n        result of a complex fft.\n        If `n` is larger than or equal to zero, the input is assumed to be the\n        result of a real fft, and `n` gives the length of the array before\n        transformation along the real transform direction.\n    axis : int, optional\n        The axis of the real transform.\n    output : ndarray, optional\n        If given, the result of filtering the input is placed in this array.\n        None is returned in this case.\n\n    Returns\n    -------\n    fourier_gaussian : ndarray or None\n        The filtered input. If `output` is given as a parameter, None is\n        returned.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import numpy.fft\n    >>> import matplotlib.pyplot as plt\n    >>> fig, (ax1, ax2) = plt.subplots(1, 2)\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ascent = misc.ascent()\n    >>> input_ = numpy.fft.fft2(ascent)\n    >>> result = ndimage.fourier_gaussian(input_, sigma=4)\n    >>> result = numpy.fft.ifft2(result)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result.real)  # the imaginary part is an artifact\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to asarray(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'input' (line 129)
    input_119798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'input', False)
    # Processing the call keyword arguments (line 129)
    kwargs_119799 = {}
    # Getting the type of 'numpy' (line 129)
    numpy_119796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 129)
    asarray_119797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), numpy_119796, 'asarray')
    # Calling asarray(args, kwargs) (line 129)
    asarray_call_result_119800 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), asarray_119797, *[input_119798], **kwargs_119799)
    
    # Assigning a type to the variable 'input' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'input', asarray_call_result_119800)
    
    # Assigning a Call to a Tuple (line 130):
    
    # Assigning a Subscript to a Name (line 130):
    
    # Obtaining the type of the subscript
    int_119801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 4), 'int')
    
    # Call to _get_output_fourier(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'output' (line 130)
    output_119803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'output', False)
    # Getting the type of 'input' (line 130)
    input_119804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 55), 'input', False)
    # Processing the call keyword arguments (line 130)
    kwargs_119805 = {}
    # Getting the type of '_get_output_fourier' (line 130)
    _get_output_fourier_119802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), '_get_output_fourier', False)
    # Calling _get_output_fourier(args, kwargs) (line 130)
    _get_output_fourier_call_result_119806 = invoke(stypy.reporting.localization.Localization(__file__, 130, 27), _get_output_fourier_119802, *[output_119803, input_119804], **kwargs_119805)
    
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___119807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), _get_output_fourier_call_result_119806, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_119808 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), getitem___119807, int_119801)
    
    # Assigning a type to the variable 'tuple_var_assignment_119619' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_var_assignment_119619', subscript_call_result_119808)
    
    # Assigning a Subscript to a Name (line 130):
    
    # Obtaining the type of the subscript
    int_119809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 4), 'int')
    
    # Call to _get_output_fourier(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'output' (line 130)
    output_119811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'output', False)
    # Getting the type of 'input' (line 130)
    input_119812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 55), 'input', False)
    # Processing the call keyword arguments (line 130)
    kwargs_119813 = {}
    # Getting the type of '_get_output_fourier' (line 130)
    _get_output_fourier_119810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), '_get_output_fourier', False)
    # Calling _get_output_fourier(args, kwargs) (line 130)
    _get_output_fourier_call_result_119814 = invoke(stypy.reporting.localization.Localization(__file__, 130, 27), _get_output_fourier_119810, *[output_119811, input_119812], **kwargs_119813)
    
    # Obtaining the member '__getitem__' of a type (line 130)
    getitem___119815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), _get_output_fourier_call_result_119814, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 130)
    subscript_call_result_119816 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), getitem___119815, int_119809)
    
    # Assigning a type to the variable 'tuple_var_assignment_119620' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_var_assignment_119620', subscript_call_result_119816)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'tuple_var_assignment_119619' (line 130)
    tuple_var_assignment_119619_119817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_var_assignment_119619')
    # Assigning a type to the variable 'output' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'output', tuple_var_assignment_119619_119817)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'tuple_var_assignment_119620' (line 130)
    tuple_var_assignment_119620_119818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'tuple_var_assignment_119620')
    # Assigning a type to the variable 'return_value' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'return_value', tuple_var_assignment_119620_119818)
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to _check_axis(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'axis' (line 131)
    axis_119821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'axis', False)
    # Getting the type of 'input' (line 131)
    input_119822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 131)
    ndim_119823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 41), input_119822, 'ndim')
    # Processing the call keyword arguments (line 131)
    kwargs_119824 = {}
    # Getting the type of '_ni_support' (line 131)
    _ni_support_119819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 131)
    _check_axis_119820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 11), _ni_support_119819, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 131)
    _check_axis_call_result_119825 = invoke(stypy.reporting.localization.Localization(__file__, 131, 11), _check_axis_119820, *[axis_119821, ndim_119823], **kwargs_119824)
    
    # Assigning a type to the variable 'axis' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'axis', _check_axis_call_result_119825)
    
    # Assigning a Call to a Name (line 132):
    
    # Assigning a Call to a Name (line 132):
    
    # Call to _normalize_sequence(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'sigma' (line 132)
    sigma_119828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 45), 'sigma', False)
    # Getting the type of 'input' (line 132)
    input_119829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 52), 'input', False)
    # Obtaining the member 'ndim' of a type (line 132)
    ndim_119830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 52), input_119829, 'ndim')
    # Processing the call keyword arguments (line 132)
    kwargs_119831 = {}
    # Getting the type of '_ni_support' (line 132)
    _ni_support_119826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 13), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 132)
    _normalize_sequence_119827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 13), _ni_support_119826, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 132)
    _normalize_sequence_call_result_119832 = invoke(stypy.reporting.localization.Localization(__file__, 132, 13), _normalize_sequence_119827, *[sigma_119828, ndim_119830], **kwargs_119831)
    
    # Assigning a type to the variable 'sigmas' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'sigmas', _normalize_sequence_call_result_119832)
    
    # Assigning a Call to a Name (line 133):
    
    # Assigning a Call to a Name (line 133):
    
    # Call to asarray(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'sigmas' (line 133)
    sigmas_119835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), 'sigmas', False)
    # Processing the call keyword arguments (line 133)
    # Getting the type of 'numpy' (line 133)
    numpy_119836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 133)
    float64_119837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 41), numpy_119836, 'float64')
    keyword_119838 = float64_119837
    kwargs_119839 = {'dtype': keyword_119838}
    # Getting the type of 'numpy' (line 133)
    numpy_119833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 133)
    asarray_119834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 13), numpy_119833, 'asarray')
    # Calling asarray(args, kwargs) (line 133)
    asarray_call_result_119840 = invoke(stypy.reporting.localization.Localization(__file__, 133, 13), asarray_119834, *[sigmas_119835], **kwargs_119839)
    
    # Assigning a type to the variable 'sigmas' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'sigmas', asarray_call_result_119840)
    
    
    # Getting the type of 'sigmas' (line 134)
    sigmas_119841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'sigmas')
    # Obtaining the member 'flags' of a type (line 134)
    flags_119842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), sigmas_119841, 'flags')
    # Obtaining the member 'contiguous' of a type (line 134)
    contiguous_119843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), flags_119842, 'contiguous')
    # Applying the 'not' unary operator (line 134)
    result_not__119844 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 7), 'not', contiguous_119843)
    
    # Testing the type of an if condition (line 134)
    if_condition_119845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 4), result_not__119844)
    # Assigning a type to the variable 'if_condition_119845' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'if_condition_119845', if_condition_119845)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 135):
    
    # Assigning a Call to a Name (line 135):
    
    # Call to copy(...): (line 135)
    # Processing the call keyword arguments (line 135)
    kwargs_119848 = {}
    # Getting the type of 'sigmas' (line 135)
    sigmas_119846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'sigmas', False)
    # Obtaining the member 'copy' of a type (line 135)
    copy_119847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 17), sigmas_119846, 'copy')
    # Calling copy(args, kwargs) (line 135)
    copy_call_result_119849 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), copy_119847, *[], **kwargs_119848)
    
    # Assigning a type to the variable 'sigmas' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'sigmas', copy_call_result_119849)
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fourier_filter(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'input' (line 137)
    input_119852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'input', False)
    # Getting the type of 'sigmas' (line 137)
    sigmas_119853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'sigmas', False)
    # Getting the type of 'n' (line 137)
    n_119854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 44), 'n', False)
    # Getting the type of 'axis' (line 137)
    axis_119855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 47), 'axis', False)
    # Getting the type of 'output' (line 137)
    output_119856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 53), 'output', False)
    int_119857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 61), 'int')
    # Processing the call keyword arguments (line 137)
    kwargs_119858 = {}
    # Getting the type of '_nd_image' (line 137)
    _nd_image_119850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), '_nd_image', False)
    # Obtaining the member 'fourier_filter' of a type (line 137)
    fourier_filter_119851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 4), _nd_image_119850, 'fourier_filter')
    # Calling fourier_filter(args, kwargs) (line 137)
    fourier_filter_call_result_119859 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), fourier_filter_119851, *[input_119852, sigmas_119853, n_119854, axis_119855, output_119856, int_119857], **kwargs_119858)
    
    # Getting the type of 'return_value' (line 138)
    return_value_119860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', return_value_119860)
    
    # ################# End of 'fourier_gaussian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fourier_gaussian' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_119861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119861)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fourier_gaussian'
    return stypy_return_type_119861

# Assigning a type to the variable 'fourier_gaussian' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'fourier_gaussian', fourier_gaussian)

@norecursion
def fourier_uniform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_119862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'int')
    int_119863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 44), 'int')
    # Getting the type of 'None' (line 141)
    None_119864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 55), 'None')
    defaults = [int_119862, int_119863, None_119864]
    # Create a new context for function 'fourier_uniform'
    module_type_store = module_type_store.open_function_context('fourier_uniform', 141, 0, False)
    
    # Passed parameters checking function
    fourier_uniform.stypy_localization = localization
    fourier_uniform.stypy_type_of_self = None
    fourier_uniform.stypy_type_store = module_type_store
    fourier_uniform.stypy_function_name = 'fourier_uniform'
    fourier_uniform.stypy_param_names_list = ['input', 'size', 'n', 'axis', 'output']
    fourier_uniform.stypy_varargs_param_name = None
    fourier_uniform.stypy_kwargs_param_name = None
    fourier_uniform.stypy_call_defaults = defaults
    fourier_uniform.stypy_call_varargs = varargs
    fourier_uniform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fourier_uniform', ['input', 'size', 'n', 'axis', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fourier_uniform', localization, ['input', 'size', 'n', 'axis', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fourier_uniform(...)' code ##################

    str_119865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, (-1)), 'str', '\n    Multi-dimensional uniform fourier filter.\n\n    The array is multiplied with the fourier transform of a box of given\n    size.\n\n    Parameters\n    ----------\n    input : array_like\n        The input array.\n    size : float or sequence\n        The size of the box used for filtering.\n        If a float, `size` is the same for all axes. If a sequence, `size` has\n        to contain one value for each axis.\n    n : int, optional\n        If `n` is negative (default), then the input is assumed to be the\n        result of a complex fft.\n        If `n` is larger than or equal to zero, the input is assumed to be the\n        result of a real fft, and `n` gives the length of the array before\n        transformation along the real transform direction.\n    axis : int, optional\n        The axis of the real transform.\n    output : ndarray, optional\n        If given, the result of filtering the input is placed in this array.\n        None is returned in this case.\n\n    Returns\n    -------\n    fourier_uniform : ndarray or None\n        The filtered input. If `output` is given as a parameter, None is\n        returned.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import numpy.fft\n    >>> import matplotlib.pyplot as plt\n    >>> fig, (ax1, ax2) = plt.subplots(1, 2)\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ascent = misc.ascent()\n    >>> input_ = numpy.fft.fft2(ascent)\n    >>> result = ndimage.fourier_uniform(input_, size=20)\n    >>> result = numpy.fft.ifft2(result)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result.real)  # the imaginary part is an artifact\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 189):
    
    # Assigning a Call to a Name (line 189):
    
    # Call to asarray(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'input' (line 189)
    input_119868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 26), 'input', False)
    # Processing the call keyword arguments (line 189)
    kwargs_119869 = {}
    # Getting the type of 'numpy' (line 189)
    numpy_119866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 189)
    asarray_119867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), numpy_119866, 'asarray')
    # Calling asarray(args, kwargs) (line 189)
    asarray_call_result_119870 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), asarray_119867, *[input_119868], **kwargs_119869)
    
    # Assigning a type to the variable 'input' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'input', asarray_call_result_119870)
    
    # Assigning a Call to a Tuple (line 190):
    
    # Assigning a Subscript to a Name (line 190):
    
    # Obtaining the type of the subscript
    int_119871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 4), 'int')
    
    # Call to _get_output_fourier(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'output' (line 190)
    output_119873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 47), 'output', False)
    # Getting the type of 'input' (line 190)
    input_119874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 55), 'input', False)
    # Processing the call keyword arguments (line 190)
    kwargs_119875 = {}
    # Getting the type of '_get_output_fourier' (line 190)
    _get_output_fourier_119872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), '_get_output_fourier', False)
    # Calling _get_output_fourier(args, kwargs) (line 190)
    _get_output_fourier_call_result_119876 = invoke(stypy.reporting.localization.Localization(__file__, 190, 27), _get_output_fourier_119872, *[output_119873, input_119874], **kwargs_119875)
    
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___119877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 4), _get_output_fourier_call_result_119876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 190)
    subscript_call_result_119878 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), getitem___119877, int_119871)
    
    # Assigning a type to the variable 'tuple_var_assignment_119621' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'tuple_var_assignment_119621', subscript_call_result_119878)
    
    # Assigning a Subscript to a Name (line 190):
    
    # Obtaining the type of the subscript
    int_119879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 4), 'int')
    
    # Call to _get_output_fourier(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'output' (line 190)
    output_119881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 47), 'output', False)
    # Getting the type of 'input' (line 190)
    input_119882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 55), 'input', False)
    # Processing the call keyword arguments (line 190)
    kwargs_119883 = {}
    # Getting the type of '_get_output_fourier' (line 190)
    _get_output_fourier_119880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), '_get_output_fourier', False)
    # Calling _get_output_fourier(args, kwargs) (line 190)
    _get_output_fourier_call_result_119884 = invoke(stypy.reporting.localization.Localization(__file__, 190, 27), _get_output_fourier_119880, *[output_119881, input_119882], **kwargs_119883)
    
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___119885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 4), _get_output_fourier_call_result_119884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 190)
    subscript_call_result_119886 = invoke(stypy.reporting.localization.Localization(__file__, 190, 4), getitem___119885, int_119879)
    
    # Assigning a type to the variable 'tuple_var_assignment_119622' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'tuple_var_assignment_119622', subscript_call_result_119886)
    
    # Assigning a Name to a Name (line 190):
    # Getting the type of 'tuple_var_assignment_119621' (line 190)
    tuple_var_assignment_119621_119887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'tuple_var_assignment_119621')
    # Assigning a type to the variable 'output' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'output', tuple_var_assignment_119621_119887)
    
    # Assigning a Name to a Name (line 190):
    # Getting the type of 'tuple_var_assignment_119622' (line 190)
    tuple_var_assignment_119622_119888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'tuple_var_assignment_119622')
    # Assigning a type to the variable 'return_value' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'return_value', tuple_var_assignment_119622_119888)
    
    # Assigning a Call to a Name (line 191):
    
    # Assigning a Call to a Name (line 191):
    
    # Call to _check_axis(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'axis' (line 191)
    axis_119891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 35), 'axis', False)
    # Getting the type of 'input' (line 191)
    input_119892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 191)
    ndim_119893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 41), input_119892, 'ndim')
    # Processing the call keyword arguments (line 191)
    kwargs_119894 = {}
    # Getting the type of '_ni_support' (line 191)
    _ni_support_119889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 191)
    _check_axis_119890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 11), _ni_support_119889, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 191)
    _check_axis_call_result_119895 = invoke(stypy.reporting.localization.Localization(__file__, 191, 11), _check_axis_119890, *[axis_119891, ndim_119893], **kwargs_119894)
    
    # Assigning a type to the variable 'axis' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'axis', _check_axis_call_result_119895)
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to _normalize_sequence(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'size' (line 192)
    size_119898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 44), 'size', False)
    # Getting the type of 'input' (line 192)
    input_119899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 50), 'input', False)
    # Obtaining the member 'ndim' of a type (line 192)
    ndim_119900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 50), input_119899, 'ndim')
    # Processing the call keyword arguments (line 192)
    kwargs_119901 = {}
    # Getting the type of '_ni_support' (line 192)
    _ni_support_119896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 192)
    _normalize_sequence_119897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), _ni_support_119896, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 192)
    _normalize_sequence_call_result_119902 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), _normalize_sequence_119897, *[size_119898, ndim_119900], **kwargs_119901)
    
    # Assigning a type to the variable 'sizes' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'sizes', _normalize_sequence_call_result_119902)
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to asarray(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'sizes' (line 193)
    sizes_119905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'sizes', False)
    # Processing the call keyword arguments (line 193)
    # Getting the type of 'numpy' (line 193)
    numpy_119906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 39), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 193)
    float64_119907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 39), numpy_119906, 'float64')
    keyword_119908 = float64_119907
    kwargs_119909 = {'dtype': keyword_119908}
    # Getting the type of 'numpy' (line 193)
    numpy_119903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 193)
    asarray_119904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), numpy_119903, 'asarray')
    # Calling asarray(args, kwargs) (line 193)
    asarray_call_result_119910 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), asarray_119904, *[sizes_119905], **kwargs_119909)
    
    # Assigning a type to the variable 'sizes' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'sizes', asarray_call_result_119910)
    
    
    # Getting the type of 'sizes' (line 194)
    sizes_119911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'sizes')
    # Obtaining the member 'flags' of a type (line 194)
    flags_119912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 11), sizes_119911, 'flags')
    # Obtaining the member 'contiguous' of a type (line 194)
    contiguous_119913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 11), flags_119912, 'contiguous')
    # Applying the 'not' unary operator (line 194)
    result_not__119914 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 7), 'not', contiguous_119913)
    
    # Testing the type of an if condition (line 194)
    if_condition_119915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 4), result_not__119914)
    # Assigning a type to the variable 'if_condition_119915' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'if_condition_119915', if_condition_119915)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to copy(...): (line 195)
    # Processing the call keyword arguments (line 195)
    kwargs_119918 = {}
    # Getting the type of 'sizes' (line 195)
    sizes_119916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'sizes', False)
    # Obtaining the member 'copy' of a type (line 195)
    copy_119917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 16), sizes_119916, 'copy')
    # Calling copy(args, kwargs) (line 195)
    copy_call_result_119919 = invoke(stypy.reporting.localization.Localization(__file__, 195, 16), copy_119917, *[], **kwargs_119918)
    
    # Assigning a type to the variable 'sizes' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'sizes', copy_call_result_119919)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fourier_filter(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'input' (line 196)
    input_119922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 29), 'input', False)
    # Getting the type of 'sizes' (line 196)
    sizes_119923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 36), 'sizes', False)
    # Getting the type of 'n' (line 196)
    n_119924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 43), 'n', False)
    # Getting the type of 'axis' (line 196)
    axis_119925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 46), 'axis', False)
    # Getting the type of 'output' (line 196)
    output_119926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 52), 'output', False)
    int_119927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 60), 'int')
    # Processing the call keyword arguments (line 196)
    kwargs_119928 = {}
    # Getting the type of '_nd_image' (line 196)
    _nd_image_119920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), '_nd_image', False)
    # Obtaining the member 'fourier_filter' of a type (line 196)
    fourier_filter_119921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 4), _nd_image_119920, 'fourier_filter')
    # Calling fourier_filter(args, kwargs) (line 196)
    fourier_filter_call_result_119929 = invoke(stypy.reporting.localization.Localization(__file__, 196, 4), fourier_filter_119921, *[input_119922, sizes_119923, n_119924, axis_119925, output_119926, int_119927], **kwargs_119928)
    
    # Getting the type of 'return_value' (line 197)
    return_value_119930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type', return_value_119930)
    
    # ################# End of 'fourier_uniform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fourier_uniform' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_119931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119931)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fourier_uniform'
    return stypy_return_type_119931

# Assigning a type to the variable 'fourier_uniform' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'fourier_uniform', fourier_uniform)

@norecursion
def fourier_ellipsoid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_119932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 37), 'int')
    int_119933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 46), 'int')
    # Getting the type of 'None' (line 200)
    None_119934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 57), 'None')
    defaults = [int_119932, int_119933, None_119934]
    # Create a new context for function 'fourier_ellipsoid'
    module_type_store = module_type_store.open_function_context('fourier_ellipsoid', 200, 0, False)
    
    # Passed parameters checking function
    fourier_ellipsoid.stypy_localization = localization
    fourier_ellipsoid.stypy_type_of_self = None
    fourier_ellipsoid.stypy_type_store = module_type_store
    fourier_ellipsoid.stypy_function_name = 'fourier_ellipsoid'
    fourier_ellipsoid.stypy_param_names_list = ['input', 'size', 'n', 'axis', 'output']
    fourier_ellipsoid.stypy_varargs_param_name = None
    fourier_ellipsoid.stypy_kwargs_param_name = None
    fourier_ellipsoid.stypy_call_defaults = defaults
    fourier_ellipsoid.stypy_call_varargs = varargs
    fourier_ellipsoid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fourier_ellipsoid', ['input', 'size', 'n', 'axis', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fourier_ellipsoid', localization, ['input', 'size', 'n', 'axis', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fourier_ellipsoid(...)' code ##################

    str_119935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, (-1)), 'str', '\n    Multi-dimensional ellipsoid fourier filter.\n\n    The array is multiplied with the fourier transform of a ellipsoid of\n    given sizes.\n\n    Parameters\n    ----------\n    input : array_like\n        The input array.\n    size : float or sequence\n        The size of the box used for filtering.\n        If a float, `size` is the same for all axes. If a sequence, `size` has\n        to contain one value for each axis.\n    n : int, optional\n        If `n` is negative (default), then the input is assumed to be the\n        result of a complex fft.\n        If `n` is larger than or equal to zero, the input is assumed to be the\n        result of a real fft, and `n` gives the length of the array before\n        transformation along the real transform direction.\n    axis : int, optional\n        The axis of the real transform.\n    output : ndarray, optional\n        If given, the result of filtering the input is placed in this array.\n        None is returned in this case.\n\n    Returns\n    -------\n    fourier_ellipsoid : ndarray or None\n        The filtered input. If `output` is given as a parameter, None is\n        returned.\n\n    Notes\n    -----\n    This function is implemented for arrays of rank 1, 2, or 3.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import numpy.fft\n    >>> import matplotlib.pyplot as plt\n    >>> fig, (ax1, ax2) = plt.subplots(1, 2)\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ascent = misc.ascent()\n    >>> input_ = numpy.fft.fft2(ascent)\n    >>> result = ndimage.fourier_ellipsoid(input_, size=20)\n    >>> result = numpy.fft.ifft2(result)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result.real)  # the imaginary part is an artifact\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 252):
    
    # Assigning a Call to a Name (line 252):
    
    # Call to asarray(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'input' (line 252)
    input_119938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'input', False)
    # Processing the call keyword arguments (line 252)
    kwargs_119939 = {}
    # Getting the type of 'numpy' (line 252)
    numpy_119936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 252)
    asarray_119937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), numpy_119936, 'asarray')
    # Calling asarray(args, kwargs) (line 252)
    asarray_call_result_119940 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), asarray_119937, *[input_119938], **kwargs_119939)
    
    # Assigning a type to the variable 'input' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'input', asarray_call_result_119940)
    
    # Assigning a Call to a Tuple (line 253):
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_119941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 4), 'int')
    
    # Call to _get_output_fourier(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'output' (line 253)
    output_119943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 47), 'output', False)
    # Getting the type of 'input' (line 253)
    input_119944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 55), 'input', False)
    # Processing the call keyword arguments (line 253)
    kwargs_119945 = {}
    # Getting the type of '_get_output_fourier' (line 253)
    _get_output_fourier_119942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 27), '_get_output_fourier', False)
    # Calling _get_output_fourier(args, kwargs) (line 253)
    _get_output_fourier_call_result_119946 = invoke(stypy.reporting.localization.Localization(__file__, 253, 27), _get_output_fourier_119942, *[output_119943, input_119944], **kwargs_119945)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___119947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 4), _get_output_fourier_call_result_119946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_119948 = invoke(stypy.reporting.localization.Localization(__file__, 253, 4), getitem___119947, int_119941)
    
    # Assigning a type to the variable 'tuple_var_assignment_119623' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_119623', subscript_call_result_119948)
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_119949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 4), 'int')
    
    # Call to _get_output_fourier(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'output' (line 253)
    output_119951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 47), 'output', False)
    # Getting the type of 'input' (line 253)
    input_119952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 55), 'input', False)
    # Processing the call keyword arguments (line 253)
    kwargs_119953 = {}
    # Getting the type of '_get_output_fourier' (line 253)
    _get_output_fourier_119950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 27), '_get_output_fourier', False)
    # Calling _get_output_fourier(args, kwargs) (line 253)
    _get_output_fourier_call_result_119954 = invoke(stypy.reporting.localization.Localization(__file__, 253, 27), _get_output_fourier_119950, *[output_119951, input_119952], **kwargs_119953)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___119955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 4), _get_output_fourier_call_result_119954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_119956 = invoke(stypy.reporting.localization.Localization(__file__, 253, 4), getitem___119955, int_119949)
    
    # Assigning a type to the variable 'tuple_var_assignment_119624' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_119624', subscript_call_result_119956)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_var_assignment_119623' (line 253)
    tuple_var_assignment_119623_119957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_119623')
    # Assigning a type to the variable 'output' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'output', tuple_var_assignment_119623_119957)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_var_assignment_119624' (line 253)
    tuple_var_assignment_119624_119958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_119624')
    # Assigning a type to the variable 'return_value' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'return_value', tuple_var_assignment_119624_119958)
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to _check_axis(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'axis' (line 254)
    axis_119961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 35), 'axis', False)
    # Getting the type of 'input' (line 254)
    input_119962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 254)
    ndim_119963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 41), input_119962, 'ndim')
    # Processing the call keyword arguments (line 254)
    kwargs_119964 = {}
    # Getting the type of '_ni_support' (line 254)
    _ni_support_119959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 254)
    _check_axis_119960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 11), _ni_support_119959, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 254)
    _check_axis_call_result_119965 = invoke(stypy.reporting.localization.Localization(__file__, 254, 11), _check_axis_119960, *[axis_119961, ndim_119963], **kwargs_119964)
    
    # Assigning a type to the variable 'axis' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'axis', _check_axis_call_result_119965)
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to _normalize_sequence(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'size' (line 255)
    size_119968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 44), 'size', False)
    # Getting the type of 'input' (line 255)
    input_119969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 50), 'input', False)
    # Obtaining the member 'ndim' of a type (line 255)
    ndim_119970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 50), input_119969, 'ndim')
    # Processing the call keyword arguments (line 255)
    kwargs_119971 = {}
    # Getting the type of '_ni_support' (line 255)
    _ni_support_119966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 255)
    _normalize_sequence_119967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), _ni_support_119966, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 255)
    _normalize_sequence_call_result_119972 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), _normalize_sequence_119967, *[size_119968, ndim_119970], **kwargs_119971)
    
    # Assigning a type to the variable 'sizes' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'sizes', _normalize_sequence_call_result_119972)
    
    # Assigning a Call to a Name (line 256):
    
    # Assigning a Call to a Name (line 256):
    
    # Call to asarray(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'sizes' (line 256)
    sizes_119975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'sizes', False)
    # Processing the call keyword arguments (line 256)
    # Getting the type of 'numpy' (line 256)
    numpy_119976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 39), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 256)
    float64_119977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 39), numpy_119976, 'float64')
    keyword_119978 = float64_119977
    kwargs_119979 = {'dtype': keyword_119978}
    # Getting the type of 'numpy' (line 256)
    numpy_119973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 256)
    asarray_119974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), numpy_119973, 'asarray')
    # Calling asarray(args, kwargs) (line 256)
    asarray_call_result_119980 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), asarray_119974, *[sizes_119975], **kwargs_119979)
    
    # Assigning a type to the variable 'sizes' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'sizes', asarray_call_result_119980)
    
    
    # Getting the type of 'sizes' (line 257)
    sizes_119981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'sizes')
    # Obtaining the member 'flags' of a type (line 257)
    flags_119982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), sizes_119981, 'flags')
    # Obtaining the member 'contiguous' of a type (line 257)
    contiguous_119983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), flags_119982, 'contiguous')
    # Applying the 'not' unary operator (line 257)
    result_not__119984 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 7), 'not', contiguous_119983)
    
    # Testing the type of an if condition (line 257)
    if_condition_119985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 4), result_not__119984)
    # Assigning a type to the variable 'if_condition_119985' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'if_condition_119985', if_condition_119985)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to copy(...): (line 258)
    # Processing the call keyword arguments (line 258)
    kwargs_119988 = {}
    # Getting the type of 'sizes' (line 258)
    sizes_119986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 16), 'sizes', False)
    # Obtaining the member 'copy' of a type (line 258)
    copy_119987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 16), sizes_119986, 'copy')
    # Calling copy(args, kwargs) (line 258)
    copy_call_result_119989 = invoke(stypy.reporting.localization.Localization(__file__, 258, 16), copy_119987, *[], **kwargs_119988)
    
    # Assigning a type to the variable 'sizes' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'sizes', copy_call_result_119989)
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fourier_filter(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'input' (line 259)
    input_119992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 29), 'input', False)
    # Getting the type of 'sizes' (line 259)
    sizes_119993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 36), 'sizes', False)
    # Getting the type of 'n' (line 259)
    n_119994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 43), 'n', False)
    # Getting the type of 'axis' (line 259)
    axis_119995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 46), 'axis', False)
    # Getting the type of 'output' (line 259)
    output_119996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 52), 'output', False)
    int_119997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 60), 'int')
    # Processing the call keyword arguments (line 259)
    kwargs_119998 = {}
    # Getting the type of '_nd_image' (line 259)
    _nd_image_119990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), '_nd_image', False)
    # Obtaining the member 'fourier_filter' of a type (line 259)
    fourier_filter_119991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 4), _nd_image_119990, 'fourier_filter')
    # Calling fourier_filter(args, kwargs) (line 259)
    fourier_filter_call_result_119999 = invoke(stypy.reporting.localization.Localization(__file__, 259, 4), fourier_filter_119991, *[input_119992, sizes_119993, n_119994, axis_119995, output_119996, int_119997], **kwargs_119998)
    
    # Getting the type of 'return_value' (line 260)
    return_value_120000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type', return_value_120000)
    
    # ################# End of 'fourier_ellipsoid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fourier_ellipsoid' in the type store
    # Getting the type of 'stypy_return_type' (line 200)
    stypy_return_type_120001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120001)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fourier_ellipsoid'
    return stypy_return_type_120001

# Assigning a type to the variable 'fourier_ellipsoid' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'fourier_ellipsoid', fourier_ellipsoid)

@norecursion
def fourier_shift(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_120002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 34), 'int')
    int_120003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 43), 'int')
    # Getting the type of 'None' (line 263)
    None_120004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 54), 'None')
    defaults = [int_120002, int_120003, None_120004]
    # Create a new context for function 'fourier_shift'
    module_type_store = module_type_store.open_function_context('fourier_shift', 263, 0, False)
    
    # Passed parameters checking function
    fourier_shift.stypy_localization = localization
    fourier_shift.stypy_type_of_self = None
    fourier_shift.stypy_type_store = module_type_store
    fourier_shift.stypy_function_name = 'fourier_shift'
    fourier_shift.stypy_param_names_list = ['input', 'shift', 'n', 'axis', 'output']
    fourier_shift.stypy_varargs_param_name = None
    fourier_shift.stypy_kwargs_param_name = None
    fourier_shift.stypy_call_defaults = defaults
    fourier_shift.stypy_call_varargs = varargs
    fourier_shift.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fourier_shift', ['input', 'shift', 'n', 'axis', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fourier_shift', localization, ['input', 'shift', 'n', 'axis', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fourier_shift(...)' code ##################

    str_120005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, (-1)), 'str', '\n    Multi-dimensional fourier shift filter.\n\n    The array is multiplied with the fourier transform of a shift operation.\n\n    Parameters\n    ----------\n    input : array_like\n        The input array.\n    shift : float or sequence\n        The size of the box used for filtering.\n        If a float, `shift` is the same for all axes. If a sequence, `shift`\n        has to contain one value for each axis.\n    n : int, optional\n        If `n` is negative (default), then the input is assumed to be the\n        result of a complex fft.\n        If `n` is larger than or equal to zero, the input is assumed to be the\n        result of a real fft, and `n` gives the length of the array before\n        transformation along the real transform direction.\n    axis : int, optional\n        The axis of the real transform.\n    output : ndarray, optional\n        If given, the result of shifting the input is placed in this array.\n        None is returned in this case.\n\n    Returns\n    -------\n    fourier_shift : ndarray or None\n        The shifted input. If `output` is given as a parameter, None is\n        returned.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> import numpy.fft\n    >>> fig, (ax1, ax2) = plt.subplots(1, 2)\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ascent = misc.ascent()\n    >>> input_ = numpy.fft.fft2(ascent)\n    >>> result = ndimage.fourier_shift(input_, shift=200)\n    >>> result = numpy.fft.ifft2(result)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result.real)  # the imaginary part is an artifact\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 310):
    
    # Assigning a Call to a Name (line 310):
    
    # Call to asarray(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'input' (line 310)
    input_120008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 26), 'input', False)
    # Processing the call keyword arguments (line 310)
    kwargs_120009 = {}
    # Getting the type of 'numpy' (line 310)
    numpy_120006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 310)
    asarray_120007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), numpy_120006, 'asarray')
    # Calling asarray(args, kwargs) (line 310)
    asarray_call_result_120010 = invoke(stypy.reporting.localization.Localization(__file__, 310, 12), asarray_120007, *[input_120008], **kwargs_120009)
    
    # Assigning a type to the variable 'input' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'input', asarray_call_result_120010)
    
    # Assigning a Call to a Tuple (line 311):
    
    # Assigning a Subscript to a Name (line 311):
    
    # Obtaining the type of the subscript
    int_120011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 4), 'int')
    
    # Call to _get_output_fourier_complex(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'output' (line 311)
    output_120013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 55), 'output', False)
    # Getting the type of 'input' (line 311)
    input_120014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 63), 'input', False)
    # Processing the call keyword arguments (line 311)
    kwargs_120015 = {}
    # Getting the type of '_get_output_fourier_complex' (line 311)
    _get_output_fourier_complex_120012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), '_get_output_fourier_complex', False)
    # Calling _get_output_fourier_complex(args, kwargs) (line 311)
    _get_output_fourier_complex_call_result_120016 = invoke(stypy.reporting.localization.Localization(__file__, 311, 27), _get_output_fourier_complex_120012, *[output_120013, input_120014], **kwargs_120015)
    
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___120017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 4), _get_output_fourier_complex_call_result_120016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_120018 = invoke(stypy.reporting.localization.Localization(__file__, 311, 4), getitem___120017, int_120011)
    
    # Assigning a type to the variable 'tuple_var_assignment_119625' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'tuple_var_assignment_119625', subscript_call_result_120018)
    
    # Assigning a Subscript to a Name (line 311):
    
    # Obtaining the type of the subscript
    int_120019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 4), 'int')
    
    # Call to _get_output_fourier_complex(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'output' (line 311)
    output_120021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 55), 'output', False)
    # Getting the type of 'input' (line 311)
    input_120022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 63), 'input', False)
    # Processing the call keyword arguments (line 311)
    kwargs_120023 = {}
    # Getting the type of '_get_output_fourier_complex' (line 311)
    _get_output_fourier_complex_120020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 27), '_get_output_fourier_complex', False)
    # Calling _get_output_fourier_complex(args, kwargs) (line 311)
    _get_output_fourier_complex_call_result_120024 = invoke(stypy.reporting.localization.Localization(__file__, 311, 27), _get_output_fourier_complex_120020, *[output_120021, input_120022], **kwargs_120023)
    
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___120025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 4), _get_output_fourier_complex_call_result_120024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_120026 = invoke(stypy.reporting.localization.Localization(__file__, 311, 4), getitem___120025, int_120019)
    
    # Assigning a type to the variable 'tuple_var_assignment_119626' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'tuple_var_assignment_119626', subscript_call_result_120026)
    
    # Assigning a Name to a Name (line 311):
    # Getting the type of 'tuple_var_assignment_119625' (line 311)
    tuple_var_assignment_119625_120027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'tuple_var_assignment_119625')
    # Assigning a type to the variable 'output' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'output', tuple_var_assignment_119625_120027)
    
    # Assigning a Name to a Name (line 311):
    # Getting the type of 'tuple_var_assignment_119626' (line 311)
    tuple_var_assignment_119626_120028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'tuple_var_assignment_119626')
    # Assigning a type to the variable 'return_value' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'return_value', tuple_var_assignment_119626_120028)
    
    # Assigning a Call to a Name (line 312):
    
    # Assigning a Call to a Name (line 312):
    
    # Call to _check_axis(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'axis' (line 312)
    axis_120031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 35), 'axis', False)
    # Getting the type of 'input' (line 312)
    input_120032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 312)
    ndim_120033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 41), input_120032, 'ndim')
    # Processing the call keyword arguments (line 312)
    kwargs_120034 = {}
    # Getting the type of '_ni_support' (line 312)
    _ni_support_120029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 312)
    _check_axis_120030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), _ni_support_120029, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 312)
    _check_axis_call_result_120035 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), _check_axis_120030, *[axis_120031, ndim_120033], **kwargs_120034)
    
    # Assigning a type to the variable 'axis' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'axis', _check_axis_call_result_120035)
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to _normalize_sequence(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'shift' (line 313)
    shift_120038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 45), 'shift', False)
    # Getting the type of 'input' (line 313)
    input_120039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 52), 'input', False)
    # Obtaining the member 'ndim' of a type (line 313)
    ndim_120040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 52), input_120039, 'ndim')
    # Processing the call keyword arguments (line 313)
    kwargs_120041 = {}
    # Getting the type of '_ni_support' (line 313)
    _ni_support_120036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 313)
    _normalize_sequence_120037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 13), _ni_support_120036, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 313)
    _normalize_sequence_call_result_120042 = invoke(stypy.reporting.localization.Localization(__file__, 313, 13), _normalize_sequence_120037, *[shift_120038, ndim_120040], **kwargs_120041)
    
    # Assigning a type to the variable 'shifts' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'shifts', _normalize_sequence_call_result_120042)
    
    # Assigning a Call to a Name (line 314):
    
    # Assigning a Call to a Name (line 314):
    
    # Call to asarray(...): (line 314)
    # Processing the call arguments (line 314)
    # Getting the type of 'shifts' (line 314)
    shifts_120045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 27), 'shifts', False)
    # Processing the call keyword arguments (line 314)
    # Getting the type of 'numpy' (line 314)
    numpy_120046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 41), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 314)
    float64_120047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 41), numpy_120046, 'float64')
    keyword_120048 = float64_120047
    kwargs_120049 = {'dtype': keyword_120048}
    # Getting the type of 'numpy' (line 314)
    numpy_120043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 13), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 314)
    asarray_120044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 13), numpy_120043, 'asarray')
    # Calling asarray(args, kwargs) (line 314)
    asarray_call_result_120050 = invoke(stypy.reporting.localization.Localization(__file__, 314, 13), asarray_120044, *[shifts_120045], **kwargs_120049)
    
    # Assigning a type to the variable 'shifts' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'shifts', asarray_call_result_120050)
    
    
    # Getting the type of 'shifts' (line 315)
    shifts_120051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 11), 'shifts')
    # Obtaining the member 'flags' of a type (line 315)
    flags_120052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 11), shifts_120051, 'flags')
    # Obtaining the member 'contiguous' of a type (line 315)
    contiguous_120053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 11), flags_120052, 'contiguous')
    # Applying the 'not' unary operator (line 315)
    result_not__120054 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 7), 'not', contiguous_120053)
    
    # Testing the type of an if condition (line 315)
    if_condition_120055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 4), result_not__120054)
    # Assigning a type to the variable 'if_condition_120055' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'if_condition_120055', if_condition_120055)
    # SSA begins for if statement (line 315)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Call to copy(...): (line 316)
    # Processing the call keyword arguments (line 316)
    kwargs_120058 = {}
    # Getting the type of 'shifts' (line 316)
    shifts_120056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 17), 'shifts', False)
    # Obtaining the member 'copy' of a type (line 316)
    copy_120057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 17), shifts_120056, 'copy')
    # Calling copy(args, kwargs) (line 316)
    copy_call_result_120059 = invoke(stypy.reporting.localization.Localization(__file__, 316, 17), copy_120057, *[], **kwargs_120058)
    
    # Assigning a type to the variable 'shifts' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'shifts', copy_call_result_120059)
    # SSA join for if statement (line 315)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to fourier_shift(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'input' (line 317)
    input_120062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 28), 'input', False)
    # Getting the type of 'shifts' (line 317)
    shifts_120063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 35), 'shifts', False)
    # Getting the type of 'n' (line 317)
    n_120064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 43), 'n', False)
    # Getting the type of 'axis' (line 317)
    axis_120065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 46), 'axis', False)
    # Getting the type of 'output' (line 317)
    output_120066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 52), 'output', False)
    # Processing the call keyword arguments (line 317)
    kwargs_120067 = {}
    # Getting the type of '_nd_image' (line 317)
    _nd_image_120060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), '_nd_image', False)
    # Obtaining the member 'fourier_shift' of a type (line 317)
    fourier_shift_120061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 4), _nd_image_120060, 'fourier_shift')
    # Calling fourier_shift(args, kwargs) (line 317)
    fourier_shift_call_result_120068 = invoke(stypy.reporting.localization.Localization(__file__, 317, 4), fourier_shift_120061, *[input_120062, shifts_120063, n_120064, axis_120065, output_120066], **kwargs_120067)
    
    # Getting the type of 'return_value' (line 318)
    return_value_120069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type', return_value_120069)
    
    # ################# End of 'fourier_shift(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fourier_shift' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_120070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120070)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fourier_shift'
    return stypy_return_type_120070

# Assigning a type to the variable 'fourier_shift' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'fourier_shift', fourier_shift)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

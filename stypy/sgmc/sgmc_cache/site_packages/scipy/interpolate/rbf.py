
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''rbf - Radial basis functions for interpolation/smoothing scattered Nd data.
2: 
3: Written by John Travers <jtravs@gmail.com>, February 2007
4: Based closely on Matlab code by Alex Chirokov
5: Additional, large, improvements by Robert Hetland
6: Some additional alterations by Travis Oliphant
7: 
8: Permission to use, modify, and distribute this software is given under the
9: terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
10: this distribution for specifics.
11: 
12: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
13: 
14: Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
15: Copyright (c) 2007, John Travers <jtravs@gmail.com>
16: 
17: Redistribution and use in source and binary forms, with or without
18: modification, are permitted provided that the following conditions are
19: met:
20: 
21:     * Redistributions of source code must retain the above copyright
22:        notice, this list of conditions and the following disclaimer.
23: 
24:     * Redistributions in binary form must reproduce the above
25:        copyright notice, this list of conditions and the following
26:        disclaimer in the documentation and/or other materials provided
27:        with the distribution.
28: 
29:     * Neither the name of Robert Hetland nor the names of any
30:        contributors may be used to endorse or promote products derived
31:        from this software without specific prior written permission.
32: 
33: THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
34: "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
35: LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
36: A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
37: OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
38: SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
39: LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
40: DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
41: THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
42: (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
43: OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
44: '''
45: from __future__ import division, print_function, absolute_import
46: 
47: import sys
48: import numpy as np
49: 
50: from scipy import linalg
51: from scipy._lib.six import callable, get_method_function, get_function_code
52: from scipy.special import xlogy
53: 
54: __all__ = ['Rbf']
55: 
56: 
57: class Rbf(object):
58:     '''
59:     Rbf(*args)
60: 
61:     A class for radial basis function approximation/interpolation of
62:     n-dimensional scattered data.
63: 
64:     Parameters
65:     ----------
66:     *args : arrays
67:         x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
68:         and d is the array of values at the nodes
69:     function : str or callable, optional
70:         The radial basis function, based on the radius, r, given by the norm
71:         (default is Euclidean distance); the default is 'multiquadric'::
72: 
73:             'multiquadric': sqrt((r/self.epsilon)**2 + 1)
74:             'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
75:             'gaussian': exp(-(r/self.epsilon)**2)
76:             'linear': r
77:             'cubic': r**3
78:             'quintic': r**5
79:             'thin_plate': r**2 * log(r)
80: 
81:         If callable, then it must take 2 arguments (self, r).  The epsilon
82:         parameter will be available as self.epsilon.  Other keyword
83:         arguments passed in will be available as well.
84: 
85:     epsilon : float, optional
86:         Adjustable constant for gaussian or multiquadrics functions
87:         - defaults to approximate average distance between nodes (which is
88:         a good start).
89:     smooth : float, optional
90:         Values greater than zero increase the smoothness of the
91:         approximation.  0 is for interpolation (default), the function will
92:         always go through the nodal points in this case.
93:     norm : callable, optional
94:         A function that returns the 'distance' between two points, with
95:         inputs as arrays of positions (x, y, z, ...), and an output as an
96:         array of distance.  E.g, the default::
97: 
98:             def euclidean_norm(x1, x2):
99:                 return sqrt( ((x1 - x2)**2).sum(axis=0) )
100: 
101:         which is called with ``x1 = x1[ndims, newaxis, :]`` and
102:         ``x2 = x2[ndims, : ,newaxis]`` such that the result is a matrix of the
103:         distances from each point in ``x1`` to each point in ``x2``.
104: 
105:     Examples
106:     --------
107:     >>> from scipy.interpolate import Rbf
108:     >>> x, y, z, d = np.random.rand(4, 50)
109:     >>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
110:     >>> xi = yi = zi = np.linspace(0, 1, 20)
111:     >>> di = rbfi(xi, yi, zi)   # interpolated values
112:     >>> di.shape
113:     (20,)
114: 
115:     '''
116: 
117:     def _euclidean_norm(self, x1, x2):
118:         return np.sqrt(((x1 - x2)**2).sum(axis=0))
119: 
120:     def _h_multiquadric(self, r):
121:         return np.sqrt((1.0/self.epsilon*r)**2 + 1)
122: 
123:     def _h_inverse_multiquadric(self, r):
124:         return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)
125: 
126:     def _h_gaussian(self, r):
127:         return np.exp(-(1.0/self.epsilon*r)**2)
128: 
129:     def _h_linear(self, r):
130:         return r
131: 
132:     def _h_cubic(self, r):
133:         return r**3
134: 
135:     def _h_quintic(self, r):
136:         return r**5
137: 
138:     def _h_thin_plate(self, r):
139:         return xlogy(r**2, r)
140: 
141:     # Setup self._function and do smoke test on initial r
142:     def _init_function(self, r):
143:         if isinstance(self.function, str):
144:             self.function = self.function.lower()
145:             _mapped = {'inverse': 'inverse_multiquadric',
146:                        'inverse multiquadric': 'inverse_multiquadric',
147:                        'thin-plate': 'thin_plate'}
148:             if self.function in _mapped:
149:                 self.function = _mapped[self.function]
150: 
151:             func_name = "_h_" + self.function
152:             if hasattr(self, func_name):
153:                 self._function = getattr(self, func_name)
154:             else:
155:                 functionlist = [x[3:] for x in dir(self) if x.startswith('_h_')]
156:                 raise ValueError("function must be a callable or one of " +
157:                                      ", ".join(functionlist))
158:             self._function = getattr(self, "_h_"+self.function)
159:         elif callable(self.function):
160:             allow_one = False
161:             if hasattr(self.function, 'func_code') or \
162:                    hasattr(self.function, '__code__'):
163:                 val = self.function
164:                 allow_one = True
165:             elif hasattr(self.function, "im_func"):
166:                 val = get_method_function(self.function)
167:             elif hasattr(self.function, "__call__"):
168:                 val = get_method_function(self.function.__call__)
169:             else:
170:                 raise ValueError("Cannot determine number of arguments to function")
171: 
172:             argcount = get_function_code(val).co_argcount
173:             if allow_one and argcount == 1:
174:                 self._function = self.function
175:             elif argcount == 2:
176:                 if sys.version_info[0] >= 3:
177:                     self._function = self.function.__get__(self, Rbf)
178:                 else:
179:                     import new
180:                     self._function = new.instancemethod(self.function, self,
181:                                                         Rbf)
182:             else:
183:                 raise ValueError("Function argument must take 1 or 2 arguments.")
184: 
185:         a0 = self._function(r)
186:         if a0.shape != r.shape:
187:             raise ValueError("Callable must take array and return array of the same shape")
188:         return a0
189: 
190:     def __init__(self, *args, **kwargs):
191:         self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten()
192:                            for a in args[:-1]])
193:         self.N = self.xi.shape[-1]
194:         self.di = np.asarray(args[-1]).flatten()
195: 
196:         if not all([x.size == self.di.size for x in self.xi]):
197:             raise ValueError("All arrays must be equal length.")
198: 
199:         self.norm = kwargs.pop('norm', self._euclidean_norm)
200:         self.epsilon = kwargs.pop('epsilon', None)
201:         if self.epsilon is None:
202:             # default epsilon is the "the average distance between nodes" based
203:             # on a bounding hypercube
204:             dim = self.xi.shape[0]
205:             ximax = np.amax(self.xi, axis=1)
206:             ximin = np.amin(self.xi, axis=1)
207:             edges = ximax-ximin
208:             edges = edges[np.nonzero(edges)]
209:             self.epsilon = np.power(np.prod(edges)/self.N, 1.0/edges.size)
210:         self.smooth = kwargs.pop('smooth', 0.0)
211: 
212:         self.function = kwargs.pop('function', 'multiquadric')
213: 
214:         # attach anything left in kwargs to self
215:         #  for use by any user-callable function or
216:         #  to save on the object returned.
217:         for item, value in kwargs.items():
218:             setattr(self, item, value)
219: 
220:         self.nodes = linalg.solve(self.A, self.di)
221: 
222:     @property
223:     def A(self):
224:         # this only exists for backwards compatibility: self.A was available
225:         # and, at least technically, public.
226:         r = self._call_norm(self.xi, self.xi)
227:         return self._init_function(r) - np.eye(self.N)*self.smooth
228: 
229:     def _call_norm(self, x1, x2):
230:         if len(x1.shape) == 1:
231:             x1 = x1[np.newaxis, :]
232:         if len(x2.shape) == 1:
233:             x2 = x2[np.newaxis, :]
234:         x1 = x1[..., :, np.newaxis]
235:         x2 = x2[..., np.newaxis, :]
236:         return self.norm(x1, x2)
237: 
238:     def __call__(self, *args):
239:         args = [np.asarray(x) for x in args]
240:         if not all([x.shape == y.shape for x in args for y in args]):
241:             raise ValueError("Array lengths must be equal")
242:         shp = args[0].shape
243:         xa = np.asarray([a.flatten() for a in args], dtype=np.float_)
244:         r = self._call_norm(xa, self.xi)
245:         return np.dot(self._function(r), self.nodes).reshape(shp)
246: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_72938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', 'rbf - Radial basis functions for interpolation/smoothing scattered Nd data.\n\nWritten by John Travers <jtravs@gmail.com>, February 2007\nBased closely on Matlab code by Alex Chirokov\nAdditional, large, improvements by Robert Hetland\nSome additional alterations by Travis Oliphant\n\nPermission to use, modify, and distribute this software is given under the\nterms of the SciPy (BSD style) license.  See LICENSE.txt that came with\nthis distribution for specifics.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n\nCopyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>\nCopyright (c) 2007, John Travers <jtravs@gmail.com>\n\nRedistribution and use in source and binary forms, with or without\nmodification, are permitted provided that the following conditions are\nmet:\n\n    * Redistributions of source code must retain the above copyright\n       notice, this list of conditions and the following disclaimer.\n\n    * Redistributions in binary form must reproduce the above\n       copyright notice, this list of conditions and the following\n       disclaimer in the documentation and/or other materials provided\n       with the distribution.\n\n    * Neither the name of Robert Hetland nor the names of any\n       contributors may be used to endorse or promote products derived\n       from this software without specific prior written permission.\n\nTHIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT\nLIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR\nA PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT\nOWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,\nSPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT\nLIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,\nDATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY\nTHEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE\nOF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 0))

# 'import sys' statement (line 47)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'import numpy' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_72939 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy')

if (type(import_72939) is not StypyTypeError):

    if (import_72939 != 'pyd_module'):
        __import__(import_72939)
        sys_modules_72940 = sys.modules[import_72939]
        import_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'np', sys_modules_72940.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy', import_72939)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 0))

# 'from scipy import linalg' statement (line 50)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_72941 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'scipy')

if (type(import_72941) is not StypyTypeError):

    if (import_72941 != 'pyd_module'):
        __import__(import_72941)
        sys_modules_72942 = sys.modules[import_72941]
        import_from_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'scipy', sys_modules_72942.module_type_store, module_type_store, ['linalg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 50, 0), __file__, sys_modules_72942, sys_modules_72942.module_type_store, module_type_store)
    else:
        from scipy import linalg

        import_from_module(stypy.reporting.localization.Localization(__file__, 50, 0), 'scipy', None, module_type_store, ['linalg'], [linalg])

else:
    # Assigning a type to the variable 'scipy' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'scipy', import_72941)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 0))

# 'from scipy._lib.six import callable, get_method_function, get_function_code' statement (line 51)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_72943 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'scipy._lib.six')

if (type(import_72943) is not StypyTypeError):

    if (import_72943 != 'pyd_module'):
        __import__(import_72943)
        sys_modules_72944 = sys.modules[import_72943]
        import_from_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'scipy._lib.six', sys_modules_72944.module_type_store, module_type_store, ['callable', 'get_method_function', 'get_function_code'])
        nest_module(stypy.reporting.localization.Localization(__file__, 51, 0), __file__, sys_modules_72944, sys_modules_72944.module_type_store, module_type_store)
    else:
        from scipy._lib.six import callable, get_method_function, get_function_code

        import_from_module(stypy.reporting.localization.Localization(__file__, 51, 0), 'scipy._lib.six', None, module_type_store, ['callable', 'get_method_function', 'get_function_code'], [callable, get_method_function, get_function_code])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'scipy._lib.six', import_72943)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 0))

# 'from scipy.special import xlogy' statement (line 52)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/')
import_72945 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy.special')

if (type(import_72945) is not StypyTypeError):

    if (import_72945 != 'pyd_module'):
        __import__(import_72945)
        sys_modules_72946 = sys.modules[import_72945]
        import_from_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy.special', sys_modules_72946.module_type_store, module_type_store, ['xlogy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 52, 0), __file__, sys_modules_72946, sys_modules_72946.module_type_store, module_type_store)
    else:
        from scipy.special import xlogy

        import_from_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy.special', None, module_type_store, ['xlogy'], [xlogy])

else:
    # Assigning a type to the variable 'scipy.special' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'scipy.special', import_72945)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/')


# Assigning a List to a Name (line 54):
__all__ = ['Rbf']
module_type_store.set_exportable_members(['Rbf'])

# Obtaining an instance of the builtin type 'list' (line 54)
list_72947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 54)
# Adding element type (line 54)
str_72948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'str', 'Rbf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_72947, str_72948)

# Assigning a type to the variable '__all__' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), '__all__', list_72947)
# Declaration of the 'Rbf' class

class Rbf(object, ):
    str_72949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'str', "\n    Rbf(*args)\n\n    A class for radial basis function approximation/interpolation of\n    n-dimensional scattered data.\n\n    Parameters\n    ----------\n    *args : arrays\n        x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes\n        and d is the array of values at the nodes\n    function : str or callable, optional\n        The radial basis function, based on the radius, r, given by the norm\n        (default is Euclidean distance); the default is 'multiquadric'::\n\n            'multiquadric': sqrt((r/self.epsilon)**2 + 1)\n            'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)\n            'gaussian': exp(-(r/self.epsilon)**2)\n            'linear': r\n            'cubic': r**3\n            'quintic': r**5\n            'thin_plate': r**2 * log(r)\n\n        If callable, then it must take 2 arguments (self, r).  The epsilon\n        parameter will be available as self.epsilon.  Other keyword\n        arguments passed in will be available as well.\n\n    epsilon : float, optional\n        Adjustable constant for gaussian or multiquadrics functions\n        - defaults to approximate average distance between nodes (which is\n        a good start).\n    smooth : float, optional\n        Values greater than zero increase the smoothness of the\n        approximation.  0 is for interpolation (default), the function will\n        always go through the nodal points in this case.\n    norm : callable, optional\n        A function that returns the 'distance' between two points, with\n        inputs as arrays of positions (x, y, z, ...), and an output as an\n        array of distance.  E.g, the default::\n\n            def euclidean_norm(x1, x2):\n                return sqrt( ((x1 - x2)**2).sum(axis=0) )\n\n        which is called with ``x1 = x1[ndims, newaxis, :]`` and\n        ``x2 = x2[ndims, : ,newaxis]`` such that the result is a matrix of the\n        distances from each point in ``x1`` to each point in ``x2``.\n\n    Examples\n    --------\n    >>> from scipy.interpolate import Rbf\n    >>> x, y, z, d = np.random.rand(4, 50)\n    >>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance\n    >>> xi = yi = zi = np.linspace(0, 1, 20)\n    >>> di = rbfi(xi, yi, zi)   # interpolated values\n    >>> di.shape\n    (20,)\n\n    ")

    @norecursion
    def _euclidean_norm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_euclidean_norm'
        module_type_store = module_type_store.open_function_context('_euclidean_norm', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_localization', localization)
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_function_name', 'Rbf._euclidean_norm')
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_param_names_list', ['x1', 'x2'])
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._euclidean_norm.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._euclidean_norm', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_euclidean_norm', localization, ['x1', 'x2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_euclidean_norm(...)' code ##################

        
        # Call to sqrt(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to sum(...): (line 118)
        # Processing the call keyword arguments (line 118)
        int_72958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 47), 'int')
        keyword_72959 = int_72958
        kwargs_72960 = {'axis': keyword_72959}
        # Getting the type of 'x1' (line 118)
        x1_72952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'x1', False)
        # Getting the type of 'x2' (line 118)
        x2_72953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'x2', False)
        # Applying the binary operator '-' (line 118)
        result_sub_72954 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 25), '-', x1_72952, x2_72953)
        
        int_72955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 35), 'int')
        # Applying the binary operator '**' (line 118)
        result_pow_72956 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 24), '**', result_sub_72954, int_72955)
        
        # Obtaining the member 'sum' of a type (line 118)
        sum_72957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 24), result_pow_72956, 'sum')
        # Calling sum(args, kwargs) (line 118)
        sum_call_result_72961 = invoke(stypy.reporting.localization.Localization(__file__, 118, 24), sum_72957, *[], **kwargs_72960)
        
        # Processing the call keyword arguments (line 118)
        kwargs_72962 = {}
        # Getting the type of 'np' (line 118)
        np_72950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 118)
        sqrt_72951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), np_72950, 'sqrt')
        # Calling sqrt(args, kwargs) (line 118)
        sqrt_call_result_72963 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), sqrt_72951, *[sum_call_result_72961], **kwargs_72962)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', sqrt_call_result_72963)
        
        # ################# End of '_euclidean_norm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_euclidean_norm' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_72964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72964)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_euclidean_norm'
        return stypy_return_type_72964


    @norecursion
    def _h_multiquadric(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_h_multiquadric'
        module_type_store = module_type_store.open_function_context('_h_multiquadric', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_localization', localization)
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_function_name', 'Rbf._h_multiquadric')
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_param_names_list', ['r'])
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._h_multiquadric.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._h_multiquadric', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_h_multiquadric', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_h_multiquadric(...)' code ##################

        
        # Call to sqrt(...): (line 121)
        # Processing the call arguments (line 121)
        float_72967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 24), 'float')
        # Getting the type of 'self' (line 121)
        self_72968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'self', False)
        # Obtaining the member 'epsilon' of a type (line 121)
        epsilon_72969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 28), self_72968, 'epsilon')
        # Applying the binary operator 'div' (line 121)
        result_div_72970 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 24), 'div', float_72967, epsilon_72969)
        
        # Getting the type of 'r' (line 121)
        r_72971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 41), 'r', False)
        # Applying the binary operator '*' (line 121)
        result_mul_72972 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 40), '*', result_div_72970, r_72971)
        
        int_72973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 45), 'int')
        # Applying the binary operator '**' (line 121)
        result_pow_72974 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 23), '**', result_mul_72972, int_72973)
        
        int_72975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 49), 'int')
        # Applying the binary operator '+' (line 121)
        result_add_72976 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 23), '+', result_pow_72974, int_72975)
        
        # Processing the call keyword arguments (line 121)
        kwargs_72977 = {}
        # Getting the type of 'np' (line 121)
        np_72965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 121)
        sqrt_72966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), np_72965, 'sqrt')
        # Calling sqrt(args, kwargs) (line 121)
        sqrt_call_result_72978 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), sqrt_72966, *[result_add_72976], **kwargs_72977)
        
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', sqrt_call_result_72978)
        
        # ################# End of '_h_multiquadric(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_h_multiquadric' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_72979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72979)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_h_multiquadric'
        return stypy_return_type_72979


    @norecursion
    def _h_inverse_multiquadric(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_h_inverse_multiquadric'
        module_type_store = module_type_store.open_function_context('_h_inverse_multiquadric', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_localization', localization)
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_function_name', 'Rbf._h_inverse_multiquadric')
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_param_names_list', ['r'])
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._h_inverse_multiquadric.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._h_inverse_multiquadric', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_h_inverse_multiquadric', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_h_inverse_multiquadric(...)' code ##################

        float_72980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'float')
        
        # Call to sqrt(...): (line 124)
        # Processing the call arguments (line 124)
        float_72983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 28), 'float')
        # Getting the type of 'self' (line 124)
        self_72984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 32), 'self', False)
        # Obtaining the member 'epsilon' of a type (line 124)
        epsilon_72985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 32), self_72984, 'epsilon')
        # Applying the binary operator 'div' (line 124)
        result_div_72986 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 28), 'div', float_72983, epsilon_72985)
        
        # Getting the type of 'r' (line 124)
        r_72987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'r', False)
        # Applying the binary operator '*' (line 124)
        result_mul_72988 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 44), '*', result_div_72986, r_72987)
        
        int_72989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 49), 'int')
        # Applying the binary operator '**' (line 124)
        result_pow_72990 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 27), '**', result_mul_72988, int_72989)
        
        int_72991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 53), 'int')
        # Applying the binary operator '+' (line 124)
        result_add_72992 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 27), '+', result_pow_72990, int_72991)
        
        # Processing the call keyword arguments (line 124)
        kwargs_72993 = {}
        # Getting the type of 'np' (line 124)
        np_72981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 124)
        sqrt_72982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 19), np_72981, 'sqrt')
        # Calling sqrt(args, kwargs) (line 124)
        sqrt_call_result_72994 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), sqrt_72982, *[result_add_72992], **kwargs_72993)
        
        # Applying the binary operator 'div' (line 124)
        result_div_72995 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 15), 'div', float_72980, sqrt_call_result_72994)
        
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', result_div_72995)
        
        # ################# End of '_h_inverse_multiquadric(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_h_inverse_multiquadric' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_72996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_72996)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_h_inverse_multiquadric'
        return stypy_return_type_72996


    @norecursion
    def _h_gaussian(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_h_gaussian'
        module_type_store = module_type_store.open_function_context('_h_gaussian', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._h_gaussian.__dict__.__setitem__('stypy_localization', localization)
        Rbf._h_gaussian.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._h_gaussian.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._h_gaussian.__dict__.__setitem__('stypy_function_name', 'Rbf._h_gaussian')
        Rbf._h_gaussian.__dict__.__setitem__('stypy_param_names_list', ['r'])
        Rbf._h_gaussian.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._h_gaussian.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._h_gaussian.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._h_gaussian.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._h_gaussian.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._h_gaussian.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._h_gaussian', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_h_gaussian', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_h_gaussian(...)' code ##################

        
        # Call to exp(...): (line 127)
        # Processing the call arguments (line 127)
        
        float_72999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 24), 'float')
        # Getting the type of 'self' (line 127)
        self_73000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'self', False)
        # Obtaining the member 'epsilon' of a type (line 127)
        epsilon_73001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 28), self_73000, 'epsilon')
        # Applying the binary operator 'div' (line 127)
        result_div_73002 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 24), 'div', float_72999, epsilon_73001)
        
        # Getting the type of 'r' (line 127)
        r_73003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 41), 'r', False)
        # Applying the binary operator '*' (line 127)
        result_mul_73004 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 40), '*', result_div_73002, r_73003)
        
        int_73005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 45), 'int')
        # Applying the binary operator '**' (line 127)
        result_pow_73006 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 23), '**', result_mul_73004, int_73005)
        
        # Applying the 'usub' unary operator (line 127)
        result___neg___73007 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 22), 'usub', result_pow_73006)
        
        # Processing the call keyword arguments (line 127)
        kwargs_73008 = {}
        # Getting the type of 'np' (line 127)
        np_72997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 127)
        exp_72998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), np_72997, 'exp')
        # Calling exp(args, kwargs) (line 127)
        exp_call_result_73009 = invoke(stypy.reporting.localization.Localization(__file__, 127, 15), exp_72998, *[result___neg___73007], **kwargs_73008)
        
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', exp_call_result_73009)
        
        # ################# End of '_h_gaussian(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_h_gaussian' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_73010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73010)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_h_gaussian'
        return stypy_return_type_73010


    @norecursion
    def _h_linear(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_h_linear'
        module_type_store = module_type_store.open_function_context('_h_linear', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._h_linear.__dict__.__setitem__('stypy_localization', localization)
        Rbf._h_linear.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._h_linear.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._h_linear.__dict__.__setitem__('stypy_function_name', 'Rbf._h_linear')
        Rbf._h_linear.__dict__.__setitem__('stypy_param_names_list', ['r'])
        Rbf._h_linear.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._h_linear.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._h_linear.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._h_linear.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._h_linear.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._h_linear.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._h_linear', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_h_linear', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_h_linear(...)' code ##################

        # Getting the type of 'r' (line 130)
        r_73011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'stypy_return_type', r_73011)
        
        # ################# End of '_h_linear(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_h_linear' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_73012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73012)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_h_linear'
        return stypy_return_type_73012


    @norecursion
    def _h_cubic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_h_cubic'
        module_type_store = module_type_store.open_function_context('_h_cubic', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._h_cubic.__dict__.__setitem__('stypy_localization', localization)
        Rbf._h_cubic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._h_cubic.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._h_cubic.__dict__.__setitem__('stypy_function_name', 'Rbf._h_cubic')
        Rbf._h_cubic.__dict__.__setitem__('stypy_param_names_list', ['r'])
        Rbf._h_cubic.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._h_cubic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._h_cubic.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._h_cubic.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._h_cubic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._h_cubic.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._h_cubic', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_h_cubic', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_h_cubic(...)' code ##################

        # Getting the type of 'r' (line 133)
        r_73013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'r')
        int_73014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'int')
        # Applying the binary operator '**' (line 133)
        result_pow_73015 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 15), '**', r_73013, int_73014)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', result_pow_73015)
        
        # ################# End of '_h_cubic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_h_cubic' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_73016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_h_cubic'
        return stypy_return_type_73016


    @norecursion
    def _h_quintic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_h_quintic'
        module_type_store = module_type_store.open_function_context('_h_quintic', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._h_quintic.__dict__.__setitem__('stypy_localization', localization)
        Rbf._h_quintic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._h_quintic.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._h_quintic.__dict__.__setitem__('stypy_function_name', 'Rbf._h_quintic')
        Rbf._h_quintic.__dict__.__setitem__('stypy_param_names_list', ['r'])
        Rbf._h_quintic.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._h_quintic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._h_quintic.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._h_quintic.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._h_quintic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._h_quintic.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._h_quintic', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_h_quintic', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_h_quintic(...)' code ##################

        # Getting the type of 'r' (line 136)
        r_73017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'r')
        int_73018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 18), 'int')
        # Applying the binary operator '**' (line 136)
        result_pow_73019 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 15), '**', r_73017, int_73018)
        
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type', result_pow_73019)
        
        # ################# End of '_h_quintic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_h_quintic' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_73020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73020)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_h_quintic'
        return stypy_return_type_73020


    @norecursion
    def _h_thin_plate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_h_thin_plate'
        module_type_store = module_type_store.open_function_context('_h_thin_plate', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_localization', localization)
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_function_name', 'Rbf._h_thin_plate')
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_param_names_list', ['r'])
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._h_thin_plate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._h_thin_plate', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_h_thin_plate', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_h_thin_plate(...)' code ##################

        
        # Call to xlogy(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'r' (line 139)
        r_73022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'r', False)
        int_73023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 24), 'int')
        # Applying the binary operator '**' (line 139)
        result_pow_73024 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), '**', r_73022, int_73023)
        
        # Getting the type of 'r' (line 139)
        r_73025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 27), 'r', False)
        # Processing the call keyword arguments (line 139)
        kwargs_73026 = {}
        # Getting the type of 'xlogy' (line 139)
        xlogy_73021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'xlogy', False)
        # Calling xlogy(args, kwargs) (line 139)
        xlogy_call_result_73027 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), xlogy_73021, *[result_pow_73024, r_73025], **kwargs_73026)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', xlogy_call_result_73027)
        
        # ################# End of '_h_thin_plate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_h_thin_plate' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_73028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_h_thin_plate'
        return stypy_return_type_73028


    @norecursion
    def _init_function(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_function'
        module_type_store = module_type_store.open_function_context('_init_function', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._init_function.__dict__.__setitem__('stypy_localization', localization)
        Rbf._init_function.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._init_function.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._init_function.__dict__.__setitem__('stypy_function_name', 'Rbf._init_function')
        Rbf._init_function.__dict__.__setitem__('stypy_param_names_list', ['r'])
        Rbf._init_function.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._init_function.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._init_function.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._init_function.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._init_function.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._init_function.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._init_function', ['r'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_function', localization, ['r'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_function(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 143)
        # Getting the type of 'str' (line 143)
        str_73029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 37), 'str')
        # Getting the type of 'self' (line 143)
        self_73030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'self')
        # Obtaining the member 'function' of a type (line 143)
        function_73031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 22), self_73030, 'function')
        
        (may_be_73032, more_types_in_union_73033) = may_be_subtype(str_73029, function_73031)

        if may_be_73032:

            if more_types_in_union_73033:
                # Runtime conditional SSA (line 143)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 143)
            self_73034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
            # Obtaining the member 'function' of a type (line 143)
            function_73035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_73034, 'function')
            # Setting the type of the member 'function' of a type (line 143)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_73034, 'function', remove_not_subtype_from_union(function_73031, str))
            
            # Assigning a Call to a Attribute (line 144):
            
            # Call to lower(...): (line 144)
            # Processing the call keyword arguments (line 144)
            kwargs_73039 = {}
            # Getting the type of 'self' (line 144)
            self_73036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'self', False)
            # Obtaining the member 'function' of a type (line 144)
            function_73037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), self_73036, 'function')
            # Obtaining the member 'lower' of a type (line 144)
            lower_73038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), function_73037, 'lower')
            # Calling lower(args, kwargs) (line 144)
            lower_call_result_73040 = invoke(stypy.reporting.localization.Localization(__file__, 144, 28), lower_73038, *[], **kwargs_73039)
            
            # Getting the type of 'self' (line 144)
            self_73041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'self')
            # Setting the type of the member 'function' of a type (line 144)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), self_73041, 'function', lower_call_result_73040)
            
            # Assigning a Dict to a Name (line 145):
            
            # Obtaining an instance of the builtin type 'dict' (line 145)
            dict_73042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 22), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 145)
            # Adding element type (key, value) (line 145)
            str_73043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'str', 'inverse')
            str_73044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 34), 'str', 'inverse_multiquadric')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 22), dict_73042, (str_73043, str_73044))
            # Adding element type (key, value) (line 145)
            str_73045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 23), 'str', 'inverse multiquadric')
            str_73046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 47), 'str', 'inverse_multiquadric')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 22), dict_73042, (str_73045, str_73046))
            # Adding element type (key, value) (line 145)
            str_73047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'str', 'thin-plate')
            str_73048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 37), 'str', 'thin_plate')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 22), dict_73042, (str_73047, str_73048))
            
            # Assigning a type to the variable '_mapped' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), '_mapped', dict_73042)
            
            
            # Getting the type of 'self' (line 148)
            self_73049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'self')
            # Obtaining the member 'function' of a type (line 148)
            function_73050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 15), self_73049, 'function')
            # Getting the type of '_mapped' (line 148)
            _mapped_73051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), '_mapped')
            # Applying the binary operator 'in' (line 148)
            result_contains_73052 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 15), 'in', function_73050, _mapped_73051)
            
            # Testing the type of an if condition (line 148)
            if_condition_73053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 12), result_contains_73052)
            # Assigning a type to the variable 'if_condition_73053' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'if_condition_73053', if_condition_73053)
            # SSA begins for if statement (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Attribute (line 149):
            
            # Obtaining the type of the subscript
            # Getting the type of 'self' (line 149)
            self_73054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 40), 'self')
            # Obtaining the member 'function' of a type (line 149)
            function_73055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 40), self_73054, 'function')
            # Getting the type of '_mapped' (line 149)
            _mapped_73056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), '_mapped')
            # Obtaining the member '__getitem__' of a type (line 149)
            getitem___73057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 32), _mapped_73056, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 149)
            subscript_call_result_73058 = invoke(stypy.reporting.localization.Localization(__file__, 149, 32), getitem___73057, function_73055)
            
            # Getting the type of 'self' (line 149)
            self_73059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'self')
            # Setting the type of the member 'function' of a type (line 149)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), self_73059, 'function', subscript_call_result_73058)
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a BinOp to a Name (line 151):
            str_73060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 24), 'str', '_h_')
            # Getting the type of 'self' (line 151)
            self_73061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 32), 'self')
            # Obtaining the member 'function' of a type (line 151)
            function_73062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 32), self_73061, 'function')
            # Applying the binary operator '+' (line 151)
            result_add_73063 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 24), '+', str_73060, function_73062)
            
            # Assigning a type to the variable 'func_name' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'func_name', result_add_73063)
            
            
            # Call to hasattr(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'self' (line 152)
            self_73065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'self', False)
            # Getting the type of 'func_name' (line 152)
            func_name_73066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 29), 'func_name', False)
            # Processing the call keyword arguments (line 152)
            kwargs_73067 = {}
            # Getting the type of 'hasattr' (line 152)
            hasattr_73064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 152)
            hasattr_call_result_73068 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), hasattr_73064, *[self_73065, func_name_73066], **kwargs_73067)
            
            # Testing the type of an if condition (line 152)
            if_condition_73069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 12), hasattr_call_result_73068)
            # Assigning a type to the variable 'if_condition_73069' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'if_condition_73069', if_condition_73069)
            # SSA begins for if statement (line 152)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 153):
            
            # Call to getattr(...): (line 153)
            # Processing the call arguments (line 153)
            # Getting the type of 'self' (line 153)
            self_73071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 41), 'self', False)
            # Getting the type of 'func_name' (line 153)
            func_name_73072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 47), 'func_name', False)
            # Processing the call keyword arguments (line 153)
            kwargs_73073 = {}
            # Getting the type of 'getattr' (line 153)
            getattr_73070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'getattr', False)
            # Calling getattr(args, kwargs) (line 153)
            getattr_call_result_73074 = invoke(stypy.reporting.localization.Localization(__file__, 153, 33), getattr_73070, *[self_73071, func_name_73072], **kwargs_73073)
            
            # Getting the type of 'self' (line 153)
            self_73075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'self')
            # Setting the type of the member '_function' of a type (line 153)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 16), self_73075, '_function', getattr_call_result_73074)
            # SSA branch for the else part of an if statement (line 152)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a ListComp to a Name (line 155):
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to dir(...): (line 155)
            # Processing the call arguments (line 155)
            # Getting the type of 'self' (line 155)
            self_73087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 51), 'self', False)
            # Processing the call keyword arguments (line 155)
            kwargs_73088 = {}
            # Getting the type of 'dir' (line 155)
            dir_73086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 47), 'dir', False)
            # Calling dir(args, kwargs) (line 155)
            dir_call_result_73089 = invoke(stypy.reporting.localization.Localization(__file__, 155, 47), dir_73086, *[self_73087], **kwargs_73088)
            
            comprehension_73090 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 32), dir_call_result_73089)
            # Assigning a type to the variable 'x' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'x', comprehension_73090)
            
            # Call to startswith(...): (line 155)
            # Processing the call arguments (line 155)
            str_73083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 73), 'str', '_h_')
            # Processing the call keyword arguments (line 155)
            kwargs_73084 = {}
            # Getting the type of 'x' (line 155)
            x_73081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 60), 'x', False)
            # Obtaining the member 'startswith' of a type (line 155)
            startswith_73082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 60), x_73081, 'startswith')
            # Calling startswith(args, kwargs) (line 155)
            startswith_call_result_73085 = invoke(stypy.reporting.localization.Localization(__file__, 155, 60), startswith_73082, *[str_73083], **kwargs_73084)
            
            
            # Obtaining the type of the subscript
            int_73076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 34), 'int')
            slice_73077 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 155, 32), int_73076, None, None)
            # Getting the type of 'x' (line 155)
            x_73078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'x')
            # Obtaining the member '__getitem__' of a type (line 155)
            getitem___73079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 32), x_73078, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 155)
            subscript_call_result_73080 = invoke(stypy.reporting.localization.Localization(__file__, 155, 32), getitem___73079, slice_73077)
            
            list_73091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 32), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 32), list_73091, subscript_call_result_73080)
            # Assigning a type to the variable 'functionlist' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'functionlist', list_73091)
            
            # Call to ValueError(...): (line 156)
            # Processing the call arguments (line 156)
            str_73093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 33), 'str', 'function must be a callable or one of ')
            
            # Call to join(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'functionlist' (line 157)
            functionlist_73096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 47), 'functionlist', False)
            # Processing the call keyword arguments (line 157)
            kwargs_73097 = {}
            str_73094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 37), 'str', ', ')
            # Obtaining the member 'join' of a type (line 157)
            join_73095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 37), str_73094, 'join')
            # Calling join(args, kwargs) (line 157)
            join_call_result_73098 = invoke(stypy.reporting.localization.Localization(__file__, 157, 37), join_73095, *[functionlist_73096], **kwargs_73097)
            
            # Applying the binary operator '+' (line 156)
            result_add_73099 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 33), '+', str_73093, join_call_result_73098)
            
            # Processing the call keyword arguments (line 156)
            kwargs_73100 = {}
            # Getting the type of 'ValueError' (line 156)
            ValueError_73092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 156)
            ValueError_call_result_73101 = invoke(stypy.reporting.localization.Localization(__file__, 156, 22), ValueError_73092, *[result_add_73099], **kwargs_73100)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 156, 16), ValueError_call_result_73101, 'raise parameter', BaseException)
            # SSA join for if statement (line 152)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Attribute (line 158):
            
            # Call to getattr(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'self' (line 158)
            self_73103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 37), 'self', False)
            str_73104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 43), 'str', '_h_')
            # Getting the type of 'self' (line 158)
            self_73105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 49), 'self', False)
            # Obtaining the member 'function' of a type (line 158)
            function_73106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 49), self_73105, 'function')
            # Applying the binary operator '+' (line 158)
            result_add_73107 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 43), '+', str_73104, function_73106)
            
            # Processing the call keyword arguments (line 158)
            kwargs_73108 = {}
            # Getting the type of 'getattr' (line 158)
            getattr_73102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'getattr', False)
            # Calling getattr(args, kwargs) (line 158)
            getattr_call_result_73109 = invoke(stypy.reporting.localization.Localization(__file__, 158, 29), getattr_73102, *[self_73103, result_add_73107], **kwargs_73108)
            
            # Getting the type of 'self' (line 158)
            self_73110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'self')
            # Setting the type of the member '_function' of a type (line 158)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), self_73110, '_function', getattr_call_result_73109)

            if more_types_in_union_73033:
                # Runtime conditional SSA for else branch (line 143)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_73032) or more_types_in_union_73033):
            # Getting the type of 'self' (line 143)
            self_73111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
            # Obtaining the member 'function' of a type (line 143)
            function_73112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_73111, 'function')
            # Setting the type of the member 'function' of a type (line 143)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_73111, 'function', remove_subtype_from_union(function_73031, str))
            
            
            # Call to callable(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 'self' (line 159)
            self_73114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'self', False)
            # Obtaining the member 'function' of a type (line 159)
            function_73115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), self_73114, 'function')
            # Processing the call keyword arguments (line 159)
            kwargs_73116 = {}
            # Getting the type of 'callable' (line 159)
            callable_73113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'callable', False)
            # Calling callable(args, kwargs) (line 159)
            callable_call_result_73117 = invoke(stypy.reporting.localization.Localization(__file__, 159, 13), callable_73113, *[function_73115], **kwargs_73116)
            
            # Testing the type of an if condition (line 159)
            if_condition_73118 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 13), callable_call_result_73117)
            # Assigning a type to the variable 'if_condition_73118' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'if_condition_73118', if_condition_73118)
            # SSA begins for if statement (line 159)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 160):
            # Getting the type of 'False' (line 160)
            False_73119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'False')
            # Assigning a type to the variable 'allow_one' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'allow_one', False_73119)
            
            
            # Evaluating a boolean operation
            
            # Call to hasattr(...): (line 161)
            # Processing the call arguments (line 161)
            # Getting the type of 'self' (line 161)
            self_73121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'self', False)
            # Obtaining the member 'function' of a type (line 161)
            function_73122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 23), self_73121, 'function')
            str_73123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 38), 'str', 'func_code')
            # Processing the call keyword arguments (line 161)
            kwargs_73124 = {}
            # Getting the type of 'hasattr' (line 161)
            hasattr_73120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 161)
            hasattr_call_result_73125 = invoke(stypy.reporting.localization.Localization(__file__, 161, 15), hasattr_73120, *[function_73122, str_73123], **kwargs_73124)
            
            
            # Call to hasattr(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 'self' (line 162)
            self_73127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'self', False)
            # Obtaining the member 'function' of a type (line 162)
            function_73128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 27), self_73127, 'function')
            str_73129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 42), 'str', '__code__')
            # Processing the call keyword arguments (line 162)
            kwargs_73130 = {}
            # Getting the type of 'hasattr' (line 162)
            hasattr_73126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'hasattr', False)
            # Calling hasattr(args, kwargs) (line 162)
            hasattr_call_result_73131 = invoke(stypy.reporting.localization.Localization(__file__, 162, 19), hasattr_73126, *[function_73128, str_73129], **kwargs_73130)
            
            # Applying the binary operator 'or' (line 161)
            result_or_keyword_73132 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), 'or', hasattr_call_result_73125, hasattr_call_result_73131)
            
            # Testing the type of an if condition (line 161)
            if_condition_73133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 12), result_or_keyword_73132)
            # Assigning a type to the variable 'if_condition_73133' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'if_condition_73133', if_condition_73133)
            # SSA begins for if statement (line 161)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 163):
            # Getting the type of 'self' (line 163)
            self_73134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'self')
            # Obtaining the member 'function' of a type (line 163)
            function_73135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 22), self_73134, 'function')
            # Assigning a type to the variable 'val' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'val', function_73135)
            
            # Assigning a Name to a Name (line 164):
            # Getting the type of 'True' (line 164)
            True_73136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'True')
            # Assigning a type to the variable 'allow_one' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'allow_one', True_73136)
            # SSA branch for the else part of an if statement (line 161)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 165)
            str_73137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 40), 'str', 'im_func')
            # Getting the type of 'self' (line 165)
            self_73138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'self')
            # Obtaining the member 'function' of a type (line 165)
            function_73139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 25), self_73138, 'function')
            
            (may_be_73140, more_types_in_union_73141) = may_provide_member(str_73137, function_73139)

            if may_be_73140:

                if more_types_in_union_73141:
                    # Runtime conditional SSA (line 165)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'self' (line 165)
                self_73142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'self')
                # Obtaining the member 'function' of a type (line 165)
                function_73143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 17), self_73142, 'function')
                # Setting the type of the member 'function' of a type (line 165)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 17), self_73142, 'function', remove_not_member_provider_from_union(function_73139, 'im_func'))
                
                # Assigning a Call to a Name (line 166):
                
                # Call to get_method_function(...): (line 166)
                # Processing the call arguments (line 166)
                # Getting the type of 'self' (line 166)
                self_73145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 42), 'self', False)
                # Obtaining the member 'function' of a type (line 166)
                function_73146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 42), self_73145, 'function')
                # Processing the call keyword arguments (line 166)
                kwargs_73147 = {}
                # Getting the type of 'get_method_function' (line 166)
                get_method_function_73144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'get_method_function', False)
                # Calling get_method_function(args, kwargs) (line 166)
                get_method_function_call_result_73148 = invoke(stypy.reporting.localization.Localization(__file__, 166, 22), get_method_function_73144, *[function_73146], **kwargs_73147)
                
                # Assigning a type to the variable 'val' (line 166)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'val', get_method_function_call_result_73148)

                if more_types_in_union_73141:
                    # Runtime conditional SSA for else branch (line 165)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_73140) or more_types_in_union_73141):
                # Getting the type of 'self' (line 165)
                self_73149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'self')
                # Obtaining the member 'function' of a type (line 165)
                function_73150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 17), self_73149, 'function')
                # Setting the type of the member 'function' of a type (line 165)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 17), self_73149, 'function', remove_member_provider_from_union(function_73139, 'im_func'))
                
                # Type idiom detected: calculating its left and rigth part (line 167)
                str_73151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 40), 'str', '__call__')
                # Getting the type of 'self' (line 167)
                self_73152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'self')
                # Obtaining the member 'function' of a type (line 167)
                function_73153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 25), self_73152, 'function')
                
                (may_be_73154, more_types_in_union_73155) = may_provide_member(str_73151, function_73153)

                if may_be_73154:

                    if more_types_in_union_73155:
                        # Runtime conditional SSA (line 167)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Getting the type of 'self' (line 167)
                    self_73156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'self')
                    # Obtaining the member 'function' of a type (line 167)
                    function_73157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 17), self_73156, 'function')
                    # Setting the type of the member 'function' of a type (line 167)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 17), self_73156, 'function', remove_not_member_provider_from_union(function_73153, '__call__'))
                    
                    # Assigning a Call to a Name (line 168):
                    
                    # Call to get_method_function(...): (line 168)
                    # Processing the call arguments (line 168)
                    # Getting the type of 'self' (line 168)
                    self_73159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'self', False)
                    # Obtaining the member 'function' of a type (line 168)
                    function_73160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 42), self_73159, 'function')
                    # Obtaining the member '__call__' of a type (line 168)
                    call___73161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 42), function_73160, '__call__')
                    # Processing the call keyword arguments (line 168)
                    kwargs_73162 = {}
                    # Getting the type of 'get_method_function' (line 168)
                    get_method_function_73158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'get_method_function', False)
                    # Calling get_method_function(args, kwargs) (line 168)
                    get_method_function_call_result_73163 = invoke(stypy.reporting.localization.Localization(__file__, 168, 22), get_method_function_73158, *[call___73161], **kwargs_73162)
                    
                    # Assigning a type to the variable 'val' (line 168)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'val', get_method_function_call_result_73163)

                    if more_types_in_union_73155:
                        # Runtime conditional SSA for else branch (line 167)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_73154) or more_types_in_union_73155):
                    # Getting the type of 'self' (line 167)
                    self_73164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'self')
                    # Obtaining the member 'function' of a type (line 167)
                    function_73165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 17), self_73164, 'function')
                    # Setting the type of the member 'function' of a type (line 167)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 17), self_73164, 'function', remove_member_provider_from_union(function_73153, '__call__'))
                    
                    # Call to ValueError(...): (line 170)
                    # Processing the call arguments (line 170)
                    str_73167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 33), 'str', 'Cannot determine number of arguments to function')
                    # Processing the call keyword arguments (line 170)
                    kwargs_73168 = {}
                    # Getting the type of 'ValueError' (line 170)
                    ValueError_73166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'ValueError', False)
                    # Calling ValueError(args, kwargs) (line 170)
                    ValueError_call_result_73169 = invoke(stypy.reporting.localization.Localization(__file__, 170, 22), ValueError_73166, *[str_73167], **kwargs_73168)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 170, 16), ValueError_call_result_73169, 'raise parameter', BaseException)

                    if (may_be_73154 and more_types_in_union_73155):
                        # SSA join for if statement (line 167)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_73140 and more_types_in_union_73141):
                    # SSA join for if statement (line 165)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 161)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Attribute to a Name (line 172):
            
            # Call to get_function_code(...): (line 172)
            # Processing the call arguments (line 172)
            # Getting the type of 'val' (line 172)
            val_73171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 41), 'val', False)
            # Processing the call keyword arguments (line 172)
            kwargs_73172 = {}
            # Getting the type of 'get_function_code' (line 172)
            get_function_code_73170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'get_function_code', False)
            # Calling get_function_code(args, kwargs) (line 172)
            get_function_code_call_result_73173 = invoke(stypy.reporting.localization.Localization(__file__, 172, 23), get_function_code_73170, *[val_73171], **kwargs_73172)
            
            # Obtaining the member 'co_argcount' of a type (line 172)
            co_argcount_73174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 23), get_function_code_call_result_73173, 'co_argcount')
            # Assigning a type to the variable 'argcount' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'argcount', co_argcount_73174)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'allow_one' (line 173)
            allow_one_73175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'allow_one')
            
            # Getting the type of 'argcount' (line 173)
            argcount_73176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 29), 'argcount')
            int_73177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 41), 'int')
            # Applying the binary operator '==' (line 173)
            result_eq_73178 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 29), '==', argcount_73176, int_73177)
            
            # Applying the binary operator 'and' (line 173)
            result_and_keyword_73179 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), 'and', allow_one_73175, result_eq_73178)
            
            # Testing the type of an if condition (line 173)
            if_condition_73180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), result_and_keyword_73179)
            # Assigning a type to the variable 'if_condition_73180' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_73180', if_condition_73180)
            # SSA begins for if statement (line 173)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 174):
            # Getting the type of 'self' (line 174)
            self_73181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 33), 'self')
            # Obtaining the member 'function' of a type (line 174)
            function_73182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 33), self_73181, 'function')
            # Getting the type of 'self' (line 174)
            self_73183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'self')
            # Setting the type of the member '_function' of a type (line 174)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), self_73183, '_function', function_73182)
            # SSA branch for the else part of an if statement (line 173)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'argcount' (line 175)
            argcount_73184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'argcount')
            int_73185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'int')
            # Applying the binary operator '==' (line 175)
            result_eq_73186 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 17), '==', argcount_73184, int_73185)
            
            # Testing the type of an if condition (line 175)
            if_condition_73187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 17), result_eq_73186)
            # Assigning a type to the variable 'if_condition_73187' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'if_condition_73187', if_condition_73187)
            # SSA begins for if statement (line 175)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Obtaining the type of the subscript
            int_73188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 36), 'int')
            # Getting the type of 'sys' (line 176)
            sys_73189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'sys')
            # Obtaining the member 'version_info' of a type (line 176)
            version_info_73190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), sys_73189, 'version_info')
            # Obtaining the member '__getitem__' of a type (line 176)
            getitem___73191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), version_info_73190, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 176)
            subscript_call_result_73192 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), getitem___73191, int_73188)
            
            int_73193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 42), 'int')
            # Applying the binary operator '>=' (line 176)
            result_ge_73194 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 19), '>=', subscript_call_result_73192, int_73193)
            
            # Testing the type of an if condition (line 176)
            if_condition_73195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 16), result_ge_73194)
            # Assigning a type to the variable 'if_condition_73195' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'if_condition_73195', if_condition_73195)
            # SSA begins for if statement (line 176)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 177):
            
            # Call to __get__(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'self' (line 177)
            self_73199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 59), 'self', False)
            # Getting the type of 'Rbf' (line 177)
            Rbf_73200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 65), 'Rbf', False)
            # Processing the call keyword arguments (line 177)
            kwargs_73201 = {}
            # Getting the type of 'self' (line 177)
            self_73196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 37), 'self', False)
            # Obtaining the member 'function' of a type (line 177)
            function_73197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 37), self_73196, 'function')
            # Obtaining the member '__get__' of a type (line 177)
            get___73198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 37), function_73197, '__get__')
            # Calling __get__(args, kwargs) (line 177)
            get___call_result_73202 = invoke(stypy.reporting.localization.Localization(__file__, 177, 37), get___73198, *[self_73199, Rbf_73200], **kwargs_73201)
            
            # Getting the type of 'self' (line 177)
            self_73203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'self')
            # Setting the type of the member '_function' of a type (line 177)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 20), self_73203, '_function', get___call_result_73202)
            # SSA branch for the else part of an if statement (line 176)
            module_type_store.open_ssa_branch('else')
            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 179, 20))
            
            # 'import new' statement (line 179)
            import new

            import_module(stypy.reporting.localization.Localization(__file__, 179, 20), 'new', new, module_type_store)
            
            
            # Assigning a Call to a Attribute (line 180):
            
            # Call to instancemethod(...): (line 180)
            # Processing the call arguments (line 180)
            # Getting the type of 'self' (line 180)
            self_73206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 56), 'self', False)
            # Obtaining the member 'function' of a type (line 180)
            function_73207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 56), self_73206, 'function')
            # Getting the type of 'self' (line 180)
            self_73208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 71), 'self', False)
            # Getting the type of 'Rbf' (line 181)
            Rbf_73209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 56), 'Rbf', False)
            # Processing the call keyword arguments (line 180)
            kwargs_73210 = {}
            # Getting the type of 'new' (line 180)
            new_73204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 37), 'new', False)
            # Obtaining the member 'instancemethod' of a type (line 180)
            instancemethod_73205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 37), new_73204, 'instancemethod')
            # Calling instancemethod(args, kwargs) (line 180)
            instancemethod_call_result_73211 = invoke(stypy.reporting.localization.Localization(__file__, 180, 37), instancemethod_73205, *[function_73207, self_73208, Rbf_73209], **kwargs_73210)
            
            # Getting the type of 'self' (line 180)
            self_73212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'self')
            # Setting the type of the member '_function' of a type (line 180)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), self_73212, '_function', instancemethod_call_result_73211)
            # SSA join for if statement (line 176)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 175)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 183)
            # Processing the call arguments (line 183)
            str_73214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 33), 'str', 'Function argument must take 1 or 2 arguments.')
            # Processing the call keyword arguments (line 183)
            kwargs_73215 = {}
            # Getting the type of 'ValueError' (line 183)
            ValueError_73213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 183)
            ValueError_call_result_73216 = invoke(stypy.reporting.localization.Localization(__file__, 183, 22), ValueError_73213, *[str_73214], **kwargs_73215)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 183, 16), ValueError_call_result_73216, 'raise parameter', BaseException)
            # SSA join for if statement (line 175)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 173)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 159)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_73032 and more_types_in_union_73033):
                # SSA join for if statement (line 143)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 185):
        
        # Call to _function(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'r' (line 185)
        r_73219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 28), 'r', False)
        # Processing the call keyword arguments (line 185)
        kwargs_73220 = {}
        # Getting the type of 'self' (line 185)
        self_73217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'self', False)
        # Obtaining the member '_function' of a type (line 185)
        _function_73218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 13), self_73217, '_function')
        # Calling _function(args, kwargs) (line 185)
        _function_call_result_73221 = invoke(stypy.reporting.localization.Localization(__file__, 185, 13), _function_73218, *[r_73219], **kwargs_73220)
        
        # Assigning a type to the variable 'a0' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'a0', _function_call_result_73221)
        
        
        # Getting the type of 'a0' (line 186)
        a0_73222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'a0')
        # Obtaining the member 'shape' of a type (line 186)
        shape_73223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 11), a0_73222, 'shape')
        # Getting the type of 'r' (line 186)
        r_73224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 'r')
        # Obtaining the member 'shape' of a type (line 186)
        shape_73225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 23), r_73224, 'shape')
        # Applying the binary operator '!=' (line 186)
        result_ne_73226 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 11), '!=', shape_73223, shape_73225)
        
        # Testing the type of an if condition (line 186)
        if_condition_73227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 8), result_ne_73226)
        # Assigning a type to the variable 'if_condition_73227' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'if_condition_73227', if_condition_73227)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 187)
        # Processing the call arguments (line 187)
        str_73229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 29), 'str', 'Callable must take array and return array of the same shape')
        # Processing the call keyword arguments (line 187)
        kwargs_73230 = {}
        # Getting the type of 'ValueError' (line 187)
        ValueError_73228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 187)
        ValueError_call_result_73231 = invoke(stypy.reporting.localization.Localization(__file__, 187, 18), ValueError_73228, *[str_73229], **kwargs_73230)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 12), ValueError_call_result_73231, 'raise parameter', BaseException)
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'a0' (line 188)
        a0_73232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'a0')
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', a0_73232)
        
        # ################# End of '_init_function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_function' in the type store
        # Getting the type of 'stypy_return_type' (line 142)
        stypy_return_type_73233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73233)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_function'
        return stypy_return_type_73233


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 191):
        
        # Call to asarray(...): (line 191)
        # Processing the call arguments (line 191)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        int_73247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 42), 'int')
        slice_73248 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 192, 36), None, int_73247, None)
        # Getting the type of 'args' (line 192)
        args_73249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___73250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 36), args_73249, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_73251 = invoke(stypy.reporting.localization.Localization(__file__, 192, 36), getitem___73250, slice_73248)
        
        comprehension_73252 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), subscript_call_result_73251)
        # Assigning a type to the variable 'a' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'a', comprehension_73252)
        
        # Call to flatten(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_73245 = {}
        
        # Call to asarray(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'a' (line 191)
        a_73238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 41), 'a', False)
        # Processing the call keyword arguments (line 191)
        # Getting the type of 'np' (line 191)
        np_73239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 50), 'np', False)
        # Obtaining the member 'float_' of a type (line 191)
        float__73240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 50), np_73239, 'float_')
        keyword_73241 = float__73240
        kwargs_73242 = {'dtype': keyword_73241}
        # Getting the type of 'np' (line 191)
        np_73236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'np', False)
        # Obtaining the member 'asarray' of a type (line 191)
        asarray_73237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 30), np_73236, 'asarray')
        # Calling asarray(args, kwargs) (line 191)
        asarray_call_result_73243 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), asarray_73237, *[a_73238], **kwargs_73242)
        
        # Obtaining the member 'flatten' of a type (line 191)
        flatten_73244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 30), asarray_call_result_73243, 'flatten')
        # Calling flatten(args, kwargs) (line 191)
        flatten_call_result_73246 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), flatten_73244, *[], **kwargs_73245)
        
        list_73253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 30), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), list_73253, flatten_call_result_73246)
        # Processing the call keyword arguments (line 191)
        kwargs_73254 = {}
        # Getting the type of 'np' (line 191)
        np_73234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 191)
        asarray_73235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 18), np_73234, 'asarray')
        # Calling asarray(args, kwargs) (line 191)
        asarray_call_result_73255 = invoke(stypy.reporting.localization.Localization(__file__, 191, 18), asarray_73235, *[list_73253], **kwargs_73254)
        
        # Getting the type of 'self' (line 191)
        self_73256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self')
        # Setting the type of the member 'xi' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_73256, 'xi', asarray_call_result_73255)
        
        # Assigning a Subscript to a Attribute (line 193):
        
        # Obtaining the type of the subscript
        int_73257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 31), 'int')
        # Getting the type of 'self' (line 193)
        self_73258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'self')
        # Obtaining the member 'xi' of a type (line 193)
        xi_73259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 17), self_73258, 'xi')
        # Obtaining the member 'shape' of a type (line 193)
        shape_73260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 17), xi_73259, 'shape')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___73261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 17), shape_73260, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_73262 = invoke(stypy.reporting.localization.Localization(__file__, 193, 17), getitem___73261, int_73257)
        
        # Getting the type of 'self' (line 193)
        self_73263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'self')
        # Setting the type of the member 'N' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), self_73263, 'N', subscript_call_result_73262)
        
        # Assigning a Call to a Attribute (line 194):
        
        # Call to flatten(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_73273 = {}
        
        # Call to asarray(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Obtaining the type of the subscript
        int_73266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 34), 'int')
        # Getting the type of 'args' (line 194)
        args_73267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 29), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 194)
        getitem___73268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 29), args_73267, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 194)
        subscript_call_result_73269 = invoke(stypy.reporting.localization.Localization(__file__, 194, 29), getitem___73268, int_73266)
        
        # Processing the call keyword arguments (line 194)
        kwargs_73270 = {}
        # Getting the type of 'np' (line 194)
        np_73264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 194)
        asarray_73265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 18), np_73264, 'asarray')
        # Calling asarray(args, kwargs) (line 194)
        asarray_call_result_73271 = invoke(stypy.reporting.localization.Localization(__file__, 194, 18), asarray_73265, *[subscript_call_result_73269], **kwargs_73270)
        
        # Obtaining the member 'flatten' of a type (line 194)
        flatten_73272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 18), asarray_call_result_73271, 'flatten')
        # Calling flatten(args, kwargs) (line 194)
        flatten_call_result_73274 = invoke(stypy.reporting.localization.Localization(__file__, 194, 18), flatten_73272, *[], **kwargs_73273)
        
        # Getting the type of 'self' (line 194)
        self_73275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self')
        # Setting the type of the member 'di' of a type (line 194)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_73275, 'di', flatten_call_result_73274)
        
        
        
        # Call to all(...): (line 196)
        # Processing the call arguments (line 196)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 196)
        self_73283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 52), 'self', False)
        # Obtaining the member 'xi' of a type (line 196)
        xi_73284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 52), self_73283, 'xi')
        comprehension_73285 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 20), xi_73284)
        # Assigning a type to the variable 'x' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'x', comprehension_73285)
        
        # Getting the type of 'x' (line 196)
        x_73277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'x', False)
        # Obtaining the member 'size' of a type (line 196)
        size_73278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 20), x_73277, 'size')
        # Getting the type of 'self' (line 196)
        self_73279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 30), 'self', False)
        # Obtaining the member 'di' of a type (line 196)
        di_73280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 30), self_73279, 'di')
        # Obtaining the member 'size' of a type (line 196)
        size_73281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 30), di_73280, 'size')
        # Applying the binary operator '==' (line 196)
        result_eq_73282 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 20), '==', size_73278, size_73281)
        
        list_73286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 20), list_73286, result_eq_73282)
        # Processing the call keyword arguments (line 196)
        kwargs_73287 = {}
        # Getting the type of 'all' (line 196)
        all_73276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'all', False)
        # Calling all(args, kwargs) (line 196)
        all_call_result_73288 = invoke(stypy.reporting.localization.Localization(__file__, 196, 15), all_73276, *[list_73286], **kwargs_73287)
        
        # Applying the 'not' unary operator (line 196)
        result_not__73289 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 11), 'not', all_call_result_73288)
        
        # Testing the type of an if condition (line 196)
        if_condition_73290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 8), result_not__73289)
        # Assigning a type to the variable 'if_condition_73290' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'if_condition_73290', if_condition_73290)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 197)
        # Processing the call arguments (line 197)
        str_73292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 29), 'str', 'All arrays must be equal length.')
        # Processing the call keyword arguments (line 197)
        kwargs_73293 = {}
        # Getting the type of 'ValueError' (line 197)
        ValueError_73291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 197)
        ValueError_call_result_73294 = invoke(stypy.reporting.localization.Localization(__file__, 197, 18), ValueError_73291, *[str_73292], **kwargs_73293)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 197, 12), ValueError_call_result_73294, 'raise parameter', BaseException)
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 199):
        
        # Call to pop(...): (line 199)
        # Processing the call arguments (line 199)
        str_73297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 31), 'str', 'norm')
        # Getting the type of 'self' (line 199)
        self_73298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 39), 'self', False)
        # Obtaining the member '_euclidean_norm' of a type (line 199)
        _euclidean_norm_73299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 39), self_73298, '_euclidean_norm')
        # Processing the call keyword arguments (line 199)
        kwargs_73300 = {}
        # Getting the type of 'kwargs' (line 199)
        kwargs_73295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 199)
        pop_73296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 20), kwargs_73295, 'pop')
        # Calling pop(args, kwargs) (line 199)
        pop_call_result_73301 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), pop_73296, *[str_73297, _euclidean_norm_73299], **kwargs_73300)
        
        # Getting the type of 'self' (line 199)
        self_73302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'self')
        # Setting the type of the member 'norm' of a type (line 199)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), self_73302, 'norm', pop_call_result_73301)
        
        # Assigning a Call to a Attribute (line 200):
        
        # Call to pop(...): (line 200)
        # Processing the call arguments (line 200)
        str_73305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'str', 'epsilon')
        # Getting the type of 'None' (line 200)
        None_73306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 45), 'None', False)
        # Processing the call keyword arguments (line 200)
        kwargs_73307 = {}
        # Getting the type of 'kwargs' (line 200)
        kwargs_73303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 200)
        pop_73304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 23), kwargs_73303, 'pop')
        # Calling pop(args, kwargs) (line 200)
        pop_call_result_73308 = invoke(stypy.reporting.localization.Localization(__file__, 200, 23), pop_73304, *[str_73305, None_73306], **kwargs_73307)
        
        # Getting the type of 'self' (line 200)
        self_73309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Setting the type of the member 'epsilon' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_73309, 'epsilon', pop_call_result_73308)
        
        # Type idiom detected: calculating its left and rigth part (line 201)
        # Getting the type of 'self' (line 201)
        self_73310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'self')
        # Obtaining the member 'epsilon' of a type (line 201)
        epsilon_73311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), self_73310, 'epsilon')
        # Getting the type of 'None' (line 201)
        None_73312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'None')
        
        (may_be_73313, more_types_in_union_73314) = may_be_none(epsilon_73311, None_73312)

        if may_be_73313:

            if more_types_in_union_73314:
                # Runtime conditional SSA (line 201)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 204):
            
            # Obtaining the type of the subscript
            int_73315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 32), 'int')
            # Getting the type of 'self' (line 204)
            self_73316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'self')
            # Obtaining the member 'xi' of a type (line 204)
            xi_73317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), self_73316, 'xi')
            # Obtaining the member 'shape' of a type (line 204)
            shape_73318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), xi_73317, 'shape')
            # Obtaining the member '__getitem__' of a type (line 204)
            getitem___73319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), shape_73318, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 204)
            subscript_call_result_73320 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), getitem___73319, int_73315)
            
            # Assigning a type to the variable 'dim' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'dim', subscript_call_result_73320)
            
            # Assigning a Call to a Name (line 205):
            
            # Call to amax(...): (line 205)
            # Processing the call arguments (line 205)
            # Getting the type of 'self' (line 205)
            self_73323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 28), 'self', False)
            # Obtaining the member 'xi' of a type (line 205)
            xi_73324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 28), self_73323, 'xi')
            # Processing the call keyword arguments (line 205)
            int_73325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 42), 'int')
            keyword_73326 = int_73325
            kwargs_73327 = {'axis': keyword_73326}
            # Getting the type of 'np' (line 205)
            np_73321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'np', False)
            # Obtaining the member 'amax' of a type (line 205)
            amax_73322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 20), np_73321, 'amax')
            # Calling amax(args, kwargs) (line 205)
            amax_call_result_73328 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), amax_73322, *[xi_73324], **kwargs_73327)
            
            # Assigning a type to the variable 'ximax' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'ximax', amax_call_result_73328)
            
            # Assigning a Call to a Name (line 206):
            
            # Call to amin(...): (line 206)
            # Processing the call arguments (line 206)
            # Getting the type of 'self' (line 206)
            self_73331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 28), 'self', False)
            # Obtaining the member 'xi' of a type (line 206)
            xi_73332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 28), self_73331, 'xi')
            # Processing the call keyword arguments (line 206)
            int_73333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 42), 'int')
            keyword_73334 = int_73333
            kwargs_73335 = {'axis': keyword_73334}
            # Getting the type of 'np' (line 206)
            np_73329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'np', False)
            # Obtaining the member 'amin' of a type (line 206)
            amin_73330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 20), np_73329, 'amin')
            # Calling amin(args, kwargs) (line 206)
            amin_call_result_73336 = invoke(stypy.reporting.localization.Localization(__file__, 206, 20), amin_73330, *[xi_73332], **kwargs_73335)
            
            # Assigning a type to the variable 'ximin' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'ximin', amin_call_result_73336)
            
            # Assigning a BinOp to a Name (line 207):
            # Getting the type of 'ximax' (line 207)
            ximax_73337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'ximax')
            # Getting the type of 'ximin' (line 207)
            ximin_73338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'ximin')
            # Applying the binary operator '-' (line 207)
            result_sub_73339 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 20), '-', ximax_73337, ximin_73338)
            
            # Assigning a type to the variable 'edges' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'edges', result_sub_73339)
            
            # Assigning a Subscript to a Name (line 208):
            
            # Obtaining the type of the subscript
            
            # Call to nonzero(...): (line 208)
            # Processing the call arguments (line 208)
            # Getting the type of 'edges' (line 208)
            edges_73342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 37), 'edges', False)
            # Processing the call keyword arguments (line 208)
            kwargs_73343 = {}
            # Getting the type of 'np' (line 208)
            np_73340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'np', False)
            # Obtaining the member 'nonzero' of a type (line 208)
            nonzero_73341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 26), np_73340, 'nonzero')
            # Calling nonzero(args, kwargs) (line 208)
            nonzero_call_result_73344 = invoke(stypy.reporting.localization.Localization(__file__, 208, 26), nonzero_73341, *[edges_73342], **kwargs_73343)
            
            # Getting the type of 'edges' (line 208)
            edges_73345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'edges')
            # Obtaining the member '__getitem__' of a type (line 208)
            getitem___73346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), edges_73345, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 208)
            subscript_call_result_73347 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), getitem___73346, nonzero_call_result_73344)
            
            # Assigning a type to the variable 'edges' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'edges', subscript_call_result_73347)
            
            # Assigning a Call to a Attribute (line 209):
            
            # Call to power(...): (line 209)
            # Processing the call arguments (line 209)
            
            # Call to prod(...): (line 209)
            # Processing the call arguments (line 209)
            # Getting the type of 'edges' (line 209)
            edges_73352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 44), 'edges', False)
            # Processing the call keyword arguments (line 209)
            kwargs_73353 = {}
            # Getting the type of 'np' (line 209)
            np_73350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'np', False)
            # Obtaining the member 'prod' of a type (line 209)
            prod_73351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 36), np_73350, 'prod')
            # Calling prod(args, kwargs) (line 209)
            prod_call_result_73354 = invoke(stypy.reporting.localization.Localization(__file__, 209, 36), prod_73351, *[edges_73352], **kwargs_73353)
            
            # Getting the type of 'self' (line 209)
            self_73355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 51), 'self', False)
            # Obtaining the member 'N' of a type (line 209)
            N_73356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 51), self_73355, 'N')
            # Applying the binary operator 'div' (line 209)
            result_div_73357 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 36), 'div', prod_call_result_73354, N_73356)
            
            float_73358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 59), 'float')
            # Getting the type of 'edges' (line 209)
            edges_73359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 63), 'edges', False)
            # Obtaining the member 'size' of a type (line 209)
            size_73360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 63), edges_73359, 'size')
            # Applying the binary operator 'div' (line 209)
            result_div_73361 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 59), 'div', float_73358, size_73360)
            
            # Processing the call keyword arguments (line 209)
            kwargs_73362 = {}
            # Getting the type of 'np' (line 209)
            np_73348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'np', False)
            # Obtaining the member 'power' of a type (line 209)
            power_73349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 27), np_73348, 'power')
            # Calling power(args, kwargs) (line 209)
            power_call_result_73363 = invoke(stypy.reporting.localization.Localization(__file__, 209, 27), power_73349, *[result_div_73357, result_div_73361], **kwargs_73362)
            
            # Getting the type of 'self' (line 209)
            self_73364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'self')
            # Setting the type of the member 'epsilon' of a type (line 209)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), self_73364, 'epsilon', power_call_result_73363)

            if more_types_in_union_73314:
                # SSA join for if statement (line 201)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 210):
        
        # Call to pop(...): (line 210)
        # Processing the call arguments (line 210)
        str_73367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 33), 'str', 'smooth')
        float_73368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 43), 'float')
        # Processing the call keyword arguments (line 210)
        kwargs_73369 = {}
        # Getting the type of 'kwargs' (line 210)
        kwargs_73365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 210)
        pop_73366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), kwargs_73365, 'pop')
        # Calling pop(args, kwargs) (line 210)
        pop_call_result_73370 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), pop_73366, *[str_73367, float_73368], **kwargs_73369)
        
        # Getting the type of 'self' (line 210)
        self_73371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self')
        # Setting the type of the member 'smooth' of a type (line 210)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_73371, 'smooth', pop_call_result_73370)
        
        # Assigning a Call to a Attribute (line 212):
        
        # Call to pop(...): (line 212)
        # Processing the call arguments (line 212)
        str_73374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 35), 'str', 'function')
        str_73375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 47), 'str', 'multiquadric')
        # Processing the call keyword arguments (line 212)
        kwargs_73376 = {}
        # Getting the type of 'kwargs' (line 212)
        kwargs_73372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 212)
        pop_73373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 24), kwargs_73372, 'pop')
        # Calling pop(args, kwargs) (line 212)
        pop_call_result_73377 = invoke(stypy.reporting.localization.Localization(__file__, 212, 24), pop_73373, *[str_73374, str_73375], **kwargs_73376)
        
        # Getting the type of 'self' (line 212)
        self_73378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self')
        # Setting the type of the member 'function' of a type (line 212)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_73378, 'function', pop_call_result_73377)
        
        
        # Call to items(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_73381 = {}
        # Getting the type of 'kwargs' (line 217)
        kwargs_73379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 27), 'kwargs', False)
        # Obtaining the member 'items' of a type (line 217)
        items_73380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 27), kwargs_73379, 'items')
        # Calling items(args, kwargs) (line 217)
        items_call_result_73382 = invoke(stypy.reporting.localization.Localization(__file__, 217, 27), items_73380, *[], **kwargs_73381)
        
        # Testing the type of a for loop iterable (line 217)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 217, 8), items_call_result_73382)
        # Getting the type of the for loop variable (line 217)
        for_loop_var_73383 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 217, 8), items_call_result_73382)
        # Assigning a type to the variable 'item' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'item', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 8), for_loop_var_73383))
        # Assigning a type to the variable 'value' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 8), for_loop_var_73383))
        # SSA begins for a for statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'self' (line 218)
        self_73385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 20), 'self', False)
        # Getting the type of 'item' (line 218)
        item_73386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 26), 'item', False)
        # Getting the type of 'value' (line 218)
        value_73387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 32), 'value', False)
        # Processing the call keyword arguments (line 218)
        kwargs_73388 = {}
        # Getting the type of 'setattr' (line 218)
        setattr_73384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 218)
        setattr_call_result_73389 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), setattr_73384, *[self_73385, item_73386, value_73387], **kwargs_73388)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 220):
        
        # Call to solve(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'self' (line 220)
        self_73392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 'self', False)
        # Obtaining the member 'A' of a type (line 220)
        A_73393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 34), self_73392, 'A')
        # Getting the type of 'self' (line 220)
        self_73394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 42), 'self', False)
        # Obtaining the member 'di' of a type (line 220)
        di_73395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 42), self_73394, 'di')
        # Processing the call keyword arguments (line 220)
        kwargs_73396 = {}
        # Getting the type of 'linalg' (line 220)
        linalg_73390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'linalg', False)
        # Obtaining the member 'solve' of a type (line 220)
        solve_73391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), linalg_73390, 'solve')
        # Calling solve(args, kwargs) (line 220)
        solve_call_result_73397 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), solve_73391, *[A_73393, di_73395], **kwargs_73396)
        
        # Getting the type of 'self' (line 220)
        self_73398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self')
        # Setting the type of the member 'nodes' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_73398, 'nodes', solve_call_result_73397)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def A(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'A'
        module_type_store = module_type_store.open_function_context('A', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf.A.__dict__.__setitem__('stypy_localization', localization)
        Rbf.A.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf.A.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf.A.__dict__.__setitem__('stypy_function_name', 'Rbf.A')
        Rbf.A.__dict__.__setitem__('stypy_param_names_list', [])
        Rbf.A.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf.A.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf.A.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf.A.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf.A.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf.A.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf.A', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'A', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'A(...)' code ##################

        
        # Assigning a Call to a Name (line 226):
        
        # Call to _call_norm(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'self' (line 226)
        self_73401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'self', False)
        # Obtaining the member 'xi' of a type (line 226)
        xi_73402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 28), self_73401, 'xi')
        # Getting the type of 'self' (line 226)
        self_73403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 37), 'self', False)
        # Obtaining the member 'xi' of a type (line 226)
        xi_73404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 37), self_73403, 'xi')
        # Processing the call keyword arguments (line 226)
        kwargs_73405 = {}
        # Getting the type of 'self' (line 226)
        self_73399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'self', False)
        # Obtaining the member '_call_norm' of a type (line 226)
        _call_norm_73400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), self_73399, '_call_norm')
        # Calling _call_norm(args, kwargs) (line 226)
        _call_norm_call_result_73406 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), _call_norm_73400, *[xi_73402, xi_73404], **kwargs_73405)
        
        # Assigning a type to the variable 'r' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'r', _call_norm_call_result_73406)
        
        # Call to _init_function(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'r' (line 227)
        r_73409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 35), 'r', False)
        # Processing the call keyword arguments (line 227)
        kwargs_73410 = {}
        # Getting the type of 'self' (line 227)
        self_73407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'self', False)
        # Obtaining the member '_init_function' of a type (line 227)
        _init_function_73408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 15), self_73407, '_init_function')
        # Calling _init_function(args, kwargs) (line 227)
        _init_function_call_result_73411 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), _init_function_73408, *[r_73409], **kwargs_73410)
        
        
        # Call to eye(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'self' (line 227)
        self_73414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'self', False)
        # Obtaining the member 'N' of a type (line 227)
        N_73415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 47), self_73414, 'N')
        # Processing the call keyword arguments (line 227)
        kwargs_73416 = {}
        # Getting the type of 'np' (line 227)
        np_73412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 40), 'np', False)
        # Obtaining the member 'eye' of a type (line 227)
        eye_73413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 40), np_73412, 'eye')
        # Calling eye(args, kwargs) (line 227)
        eye_call_result_73417 = invoke(stypy.reporting.localization.Localization(__file__, 227, 40), eye_73413, *[N_73415], **kwargs_73416)
        
        # Getting the type of 'self' (line 227)
        self_73418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 55), 'self')
        # Obtaining the member 'smooth' of a type (line 227)
        smooth_73419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 55), self_73418, 'smooth')
        # Applying the binary operator '*' (line 227)
        result_mul_73420 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 40), '*', eye_call_result_73417, smooth_73419)
        
        # Applying the binary operator '-' (line 227)
        result_sub_73421 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 15), '-', _init_function_call_result_73411, result_mul_73420)
        
        # Assigning a type to the variable 'stypy_return_type' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'stypy_return_type', result_sub_73421)
        
        # ################# End of 'A(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'A' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_73422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73422)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'A'
        return stypy_return_type_73422


    @norecursion
    def _call_norm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_call_norm'
        module_type_store = module_type_store.open_function_context('_call_norm', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf._call_norm.__dict__.__setitem__('stypy_localization', localization)
        Rbf._call_norm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf._call_norm.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf._call_norm.__dict__.__setitem__('stypy_function_name', 'Rbf._call_norm')
        Rbf._call_norm.__dict__.__setitem__('stypy_param_names_list', ['x1', 'x2'])
        Rbf._call_norm.__dict__.__setitem__('stypy_varargs_param_name', None)
        Rbf._call_norm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf._call_norm.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf._call_norm.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf._call_norm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf._call_norm.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf._call_norm', ['x1', 'x2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_call_norm', localization, ['x1', 'x2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_call_norm(...)' code ##################

        
        
        
        # Call to len(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'x1' (line 230)
        x1_73424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'x1', False)
        # Obtaining the member 'shape' of a type (line 230)
        shape_73425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 15), x1_73424, 'shape')
        # Processing the call keyword arguments (line 230)
        kwargs_73426 = {}
        # Getting the type of 'len' (line 230)
        len_73423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'len', False)
        # Calling len(args, kwargs) (line 230)
        len_call_result_73427 = invoke(stypy.reporting.localization.Localization(__file__, 230, 11), len_73423, *[shape_73425], **kwargs_73426)
        
        int_73428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 28), 'int')
        # Applying the binary operator '==' (line 230)
        result_eq_73429 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), '==', len_call_result_73427, int_73428)
        
        # Testing the type of an if condition (line 230)
        if_condition_73430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_eq_73429)
        # Assigning a type to the variable 'if_condition_73430' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_73430', if_condition_73430)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        # Getting the type of 'np' (line 231)
        np_73431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'np')
        # Obtaining the member 'newaxis' of a type (line 231)
        newaxis_73432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 20), np_73431, 'newaxis')
        slice_73433 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 231, 17), None, None, None)
        # Getting the type of 'x1' (line 231)
        x1_73434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'x1')
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___73435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 17), x1_73434, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_73436 = invoke(stypy.reporting.localization.Localization(__file__, 231, 17), getitem___73435, (newaxis_73432, slice_73433))
        
        # Assigning a type to the variable 'x1' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'x1', subscript_call_result_73436)
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'x2' (line 232)
        x2_73438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'x2', False)
        # Obtaining the member 'shape' of a type (line 232)
        shape_73439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), x2_73438, 'shape')
        # Processing the call keyword arguments (line 232)
        kwargs_73440 = {}
        # Getting the type of 'len' (line 232)
        len_73437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'len', False)
        # Calling len(args, kwargs) (line 232)
        len_call_result_73441 = invoke(stypy.reporting.localization.Localization(__file__, 232, 11), len_73437, *[shape_73439], **kwargs_73440)
        
        int_73442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 28), 'int')
        # Applying the binary operator '==' (line 232)
        result_eq_73443 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 11), '==', len_call_result_73441, int_73442)
        
        # Testing the type of an if condition (line 232)
        if_condition_73444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), result_eq_73443)
        # Assigning a type to the variable 'if_condition_73444' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_73444', if_condition_73444)
        # SSA begins for if statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 233):
        
        # Obtaining the type of the subscript
        # Getting the type of 'np' (line 233)
        np_73445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'np')
        # Obtaining the member 'newaxis' of a type (line 233)
        newaxis_73446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 20), np_73445, 'newaxis')
        slice_73447 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 233, 17), None, None, None)
        # Getting the type of 'x2' (line 233)
        x2_73448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 17), 'x2')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___73449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 17), x2_73448, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_73450 = invoke(stypy.reporting.localization.Localization(__file__, 233, 17), getitem___73449, (newaxis_73446, slice_73447))
        
        # Assigning a type to the variable 'x2' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'x2', subscript_call_result_73450)
        # SSA join for if statement (line 232)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 234):
        
        # Obtaining the type of the subscript
        Ellipsis_73451 = Ellipsis
        slice_73452 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 234, 13), None, None, None)
        # Getting the type of 'np' (line 234)
        np_73453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'np')
        # Obtaining the member 'newaxis' of a type (line 234)
        newaxis_73454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 24), np_73453, 'newaxis')
        # Getting the type of 'x1' (line 234)
        x1_73455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 13), 'x1')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___73456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 13), x1_73455, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_73457 = invoke(stypy.reporting.localization.Localization(__file__, 234, 13), getitem___73456, (Ellipsis_73451, slice_73452, newaxis_73454))
        
        # Assigning a type to the variable 'x1' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'x1', subscript_call_result_73457)
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        Ellipsis_73458 = Ellipsis
        # Getting the type of 'np' (line 235)
        np_73459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 21), 'np')
        # Obtaining the member 'newaxis' of a type (line 235)
        newaxis_73460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 21), np_73459, 'newaxis')
        slice_73461 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 235, 13), None, None, None)
        # Getting the type of 'x2' (line 235)
        x2_73462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'x2')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___73463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 13), x2_73462, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_73464 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), getitem___73463, (Ellipsis_73458, newaxis_73460, slice_73461))
        
        # Assigning a type to the variable 'x2' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'x2', subscript_call_result_73464)
        
        # Call to norm(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'x1' (line 236)
        x1_73467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'x1', False)
        # Getting the type of 'x2' (line 236)
        x2_73468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 29), 'x2', False)
        # Processing the call keyword arguments (line 236)
        kwargs_73469 = {}
        # Getting the type of 'self' (line 236)
        self_73465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'self', False)
        # Obtaining the member 'norm' of a type (line 236)
        norm_73466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), self_73465, 'norm')
        # Calling norm(args, kwargs) (line 236)
        norm_call_result_73470 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), norm_73466, *[x1_73467, x2_73468], **kwargs_73469)
        
        # Assigning a type to the variable 'stypy_return_type' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'stypy_return_type', norm_call_result_73470)
        
        # ################# End of '_call_norm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_call_norm' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_73471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73471)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_call_norm'
        return stypy_return_type_73471


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Rbf.__call__.__dict__.__setitem__('stypy_localization', localization)
        Rbf.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Rbf.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Rbf.__call__.__dict__.__setitem__('stypy_function_name', 'Rbf.__call__')
        Rbf.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        Rbf.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Rbf.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Rbf.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Rbf.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Rbf.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Rbf.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Rbf.__call__', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a ListComp to a Name (line 239):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'args' (line 239)
        args_73477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'args')
        comprehension_73478 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 16), args_73477)
        # Assigning a type to the variable 'x' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'x', comprehension_73478)
        
        # Call to asarray(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'x' (line 239)
        x_73474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'x', False)
        # Processing the call keyword arguments (line 239)
        kwargs_73475 = {}
        # Getting the type of 'np' (line 239)
        np_73472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 239)
        asarray_73473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 16), np_73472, 'asarray')
        # Calling asarray(args, kwargs) (line 239)
        asarray_call_result_73476 = invoke(stypy.reporting.localization.Localization(__file__, 239, 16), asarray_73473, *[x_73474], **kwargs_73475)
        
        list_73479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 16), list_73479, asarray_call_result_73476)
        # Assigning a type to the variable 'args' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'args', list_73479)
        
        
        
        # Call to all(...): (line 240)
        # Processing the call arguments (line 240)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'args' (line 240)
        args_73486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 48), 'args', False)
        comprehension_73487 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 20), args_73486)
        # Assigning a type to the variable 'x' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'x', comprehension_73487)
        # Calculating comprehension expression
        # Getting the type of 'args' (line 240)
        args_73488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 62), 'args', False)
        comprehension_73489 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 20), args_73488)
        # Assigning a type to the variable 'y' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'y', comprehension_73489)
        
        # Getting the type of 'x' (line 240)
        x_73481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'x', False)
        # Obtaining the member 'shape' of a type (line 240)
        shape_73482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 20), x_73481, 'shape')
        # Getting the type of 'y' (line 240)
        y_73483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'y', False)
        # Obtaining the member 'shape' of a type (line 240)
        shape_73484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 31), y_73483, 'shape')
        # Applying the binary operator '==' (line 240)
        result_eq_73485 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 20), '==', shape_73482, shape_73484)
        
        list_73490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 20), list_73490, result_eq_73485)
        # Processing the call keyword arguments (line 240)
        kwargs_73491 = {}
        # Getting the type of 'all' (line 240)
        all_73480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'all', False)
        # Calling all(args, kwargs) (line 240)
        all_call_result_73492 = invoke(stypy.reporting.localization.Localization(__file__, 240, 15), all_73480, *[list_73490], **kwargs_73491)
        
        # Applying the 'not' unary operator (line 240)
        result_not__73493 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), 'not', all_call_result_73492)
        
        # Testing the type of an if condition (line 240)
        if_condition_73494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_not__73493)
        # Assigning a type to the variable 'if_condition_73494' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_73494', if_condition_73494)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 241)
        # Processing the call arguments (line 241)
        str_73496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 29), 'str', 'Array lengths must be equal')
        # Processing the call keyword arguments (line 241)
        kwargs_73497 = {}
        # Getting the type of 'ValueError' (line 241)
        ValueError_73495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 241)
        ValueError_call_result_73498 = invoke(stypy.reporting.localization.Localization(__file__, 241, 18), ValueError_73495, *[str_73496], **kwargs_73497)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 241, 12), ValueError_call_result_73498, 'raise parameter', BaseException)
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 242):
        
        # Obtaining the type of the subscript
        int_73499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 19), 'int')
        # Getting the type of 'args' (line 242)
        args_73500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'args')
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___73501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 14), args_73500, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_73502 = invoke(stypy.reporting.localization.Localization(__file__, 242, 14), getitem___73501, int_73499)
        
        # Obtaining the member 'shape' of a type (line 242)
        shape_73503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 14), subscript_call_result_73502, 'shape')
        # Assigning a type to the variable 'shp' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'shp', shape_73503)
        
        # Assigning a Call to a Name (line 243):
        
        # Call to asarray(...): (line 243)
        # Processing the call arguments (line 243)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'args' (line 243)
        args_73510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 46), 'args', False)
        comprehension_73511 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 25), args_73510)
        # Assigning a type to the variable 'a' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 25), 'a', comprehension_73511)
        
        # Call to flatten(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_73508 = {}
        # Getting the type of 'a' (line 243)
        a_73506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 25), 'a', False)
        # Obtaining the member 'flatten' of a type (line 243)
        flatten_73507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 25), a_73506, 'flatten')
        # Calling flatten(args, kwargs) (line 243)
        flatten_call_result_73509 = invoke(stypy.reporting.localization.Localization(__file__, 243, 25), flatten_73507, *[], **kwargs_73508)
        
        list_73512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 25), list_73512, flatten_call_result_73509)
        # Processing the call keyword arguments (line 243)
        # Getting the type of 'np' (line 243)
        np_73513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 59), 'np', False)
        # Obtaining the member 'float_' of a type (line 243)
        float__73514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 59), np_73513, 'float_')
        keyword_73515 = float__73514
        kwargs_73516 = {'dtype': keyword_73515}
        # Getting the type of 'np' (line 243)
        np_73504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 13), 'np', False)
        # Obtaining the member 'asarray' of a type (line 243)
        asarray_73505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 13), np_73504, 'asarray')
        # Calling asarray(args, kwargs) (line 243)
        asarray_call_result_73517 = invoke(stypy.reporting.localization.Localization(__file__, 243, 13), asarray_73505, *[list_73512], **kwargs_73516)
        
        # Assigning a type to the variable 'xa' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'xa', asarray_call_result_73517)
        
        # Assigning a Call to a Name (line 244):
        
        # Call to _call_norm(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'xa' (line 244)
        xa_73520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 28), 'xa', False)
        # Getting the type of 'self' (line 244)
        self_73521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 32), 'self', False)
        # Obtaining the member 'xi' of a type (line 244)
        xi_73522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 32), self_73521, 'xi')
        # Processing the call keyword arguments (line 244)
        kwargs_73523 = {}
        # Getting the type of 'self' (line 244)
        self_73518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self', False)
        # Obtaining the member '_call_norm' of a type (line 244)
        _call_norm_73519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), self_73518, '_call_norm')
        # Calling _call_norm(args, kwargs) (line 244)
        _call_norm_call_result_73524 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), _call_norm_73519, *[xa_73520, xi_73522], **kwargs_73523)
        
        # Assigning a type to the variable 'r' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'r', _call_norm_call_result_73524)
        
        # Call to reshape(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'shp' (line 245)
        shp_73537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 61), 'shp', False)
        # Processing the call keyword arguments (line 245)
        kwargs_73538 = {}
        
        # Call to dot(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Call to _function(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'r' (line 245)
        r_73529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 37), 'r', False)
        # Processing the call keyword arguments (line 245)
        kwargs_73530 = {}
        # Getting the type of 'self' (line 245)
        self_73527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 22), 'self', False)
        # Obtaining the member '_function' of a type (line 245)
        _function_73528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 22), self_73527, '_function')
        # Calling _function(args, kwargs) (line 245)
        _function_call_result_73531 = invoke(stypy.reporting.localization.Localization(__file__, 245, 22), _function_73528, *[r_73529], **kwargs_73530)
        
        # Getting the type of 'self' (line 245)
        self_73532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 41), 'self', False)
        # Obtaining the member 'nodes' of a type (line 245)
        nodes_73533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 41), self_73532, 'nodes')
        # Processing the call keyword arguments (line 245)
        kwargs_73534 = {}
        # Getting the type of 'np' (line 245)
        np_73525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 245)
        dot_73526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), np_73525, 'dot')
        # Calling dot(args, kwargs) (line 245)
        dot_call_result_73535 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), dot_73526, *[_function_call_result_73531, nodes_73533], **kwargs_73534)
        
        # Obtaining the member 'reshape' of a type (line 245)
        reshape_73536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), dot_call_result_73535, 'reshape')
        # Calling reshape(args, kwargs) (line 245)
        reshape_call_result_73539 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), reshape_73536, *[shp_73537], **kwargs_73538)
        
        # Assigning a type to the variable 'stypy_return_type' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', reshape_call_result_73539)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_73540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_73540


# Assigning a type to the variable 'Rbf' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'Rbf', Rbf)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()


# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Functions which are common and require SciPy Base and Level 1 SciPy
3: (special, linalg)
4: '''
5: 
6: from __future__ import division, print_function, absolute_import
7: 
8: from numpy import arange, newaxis, hstack, product, array, fromstring
9: 
10: __all__ = ['central_diff_weights', 'derivative', 'ascent', 'face']
11: 
12: 
13: def central_diff_weights(Np, ndiv=1):
14:     '''
15:     Return weights for an Np-point central derivative.
16: 
17:     Assumes equally-spaced function points.
18: 
19:     If weights are in the vector w, then
20:     derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)
21: 
22:     Parameters
23:     ----------
24:     Np : int
25:         Number of points for the central derivative.
26:     ndiv : int, optional
27:         Number of divisions.  Default is 1.
28: 
29:     Notes
30:     -----
31:     Can be inaccurate for large number of points.
32: 
33:     '''
34:     if Np < ndiv + 1:
35:         raise ValueError("Number of points must be at least the derivative order + 1.")
36:     if Np % 2 == 0:
37:         raise ValueError("The number of points must be odd.")
38:     from scipy import linalg
39:     ho = Np >> 1
40:     x = arange(-ho,ho+1.0)
41:     x = x[:,newaxis]
42:     X = x**0.0
43:     for k in range(1,Np):
44:         X = hstack([X,x**k])
45:     w = product(arange(1,ndiv+1),axis=0)*linalg.inv(X)[ndiv]
46:     return w
47: 
48: 
49: def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
50:     '''
51:     Find the n-th derivative of a function at a point.
52: 
53:     Given a function, use a central difference formula with spacing `dx` to
54:     compute the `n`-th derivative at `x0`.
55: 
56:     Parameters
57:     ----------
58:     func : function
59:         Input function.
60:     x0 : float
61:         The point at which `n`-th derivative is found.
62:     dx : float, optional
63:         Spacing.
64:     n : int, optional
65:         Order of the derivative. Default is 1.
66:     args : tuple, optional
67:         Arguments
68:     order : int, optional
69:         Number of points to use, must be odd.
70: 
71:     Notes
72:     -----
73:     Decreasing the step size too small can result in round-off error.
74: 
75:     Examples
76:     --------
77:     >>> from scipy.misc import derivative
78:     >>> def f(x):
79:     ...     return x**3 + x**2
80:     >>> derivative(f, 1.0, dx=1e-6)
81:     4.9999999999217337
82: 
83:     '''
84:     if order < n + 1:
85:         raise ValueError("'order' (the number of points used to compute the derivative), "
86:                          "must be at least the derivative order 'n' + 1.")
87:     if order % 2 == 0:
88:         raise ValueError("'order' (the number of points used to compute the derivative) "
89:                          "must be odd.")
90:     # pre-computed for n=1 and 2 and low-order for speed.
91:     if n == 1:
92:         if order == 3:
93:             weights = array([-1,0,1])/2.0
94:         elif order == 5:
95:             weights = array([1,-8,0,8,-1])/12.0
96:         elif order == 7:
97:             weights = array([-1,9,-45,0,45,-9,1])/60.0
98:         elif order == 9:
99:             weights = array([3,-32,168,-672,0,672,-168,32,-3])/840.0
100:         else:
101:             weights = central_diff_weights(order,1)
102:     elif n == 2:
103:         if order == 3:
104:             weights = array([1,-2.0,1])
105:         elif order == 5:
106:             weights = array([-1,16,-30,16,-1])/12.0
107:         elif order == 7:
108:             weights = array([2,-27,270,-490,270,-27,2])/180.0
109:         elif order == 9:
110:             weights = array([-9,128,-1008,8064,-14350,8064,-1008,128,-9])/5040.0
111:         else:
112:             weights = central_diff_weights(order,2)
113:     else:
114:         weights = central_diff_weights(order, n)
115:     val = 0.0
116:     ho = order >> 1
117:     for k in range(order):
118:         val += weights[k]*func(x0+(k-ho)*dx,*args)
119:     return val / product((dx,)*n,axis=0)
120: 
121: 
122: def ascent():
123:     '''
124:     Get an 8-bit grayscale bit-depth, 512 x 512 derived image for easy use in demos
125: 
126:     The image is derived from accent-to-the-top.jpg at
127:     http://www.public-domain-image.com/people-public-domain-images-pictures/
128: 
129:     Parameters
130:     ----------
131:     None
132: 
133:     Returns
134:     -------
135:     ascent : ndarray
136:        convenient image to use for testing and demonstration
137: 
138:     Examples
139:     --------
140:     >>> import scipy.misc
141:     >>> ascent = scipy.misc.ascent()
142:     >>> ascent.shape
143:     (512, 512)
144:     >>> ascent.max()
145:     255
146: 
147:     >>> import matplotlib.pyplot as plt
148:     >>> plt.gray()
149:     >>> plt.imshow(ascent)
150:     >>> plt.show()
151: 
152:     '''
153:     import pickle
154:     import os
155:     fname = os.path.join(os.path.dirname(__file__),'ascent.dat')
156:     with open(fname, 'rb') as f:
157:         ascent = array(pickle.load(f))
158:     return ascent
159: 
160: 
161: def face(gray=False):
162:     '''
163:     Get a 1024 x 768, color image of a raccoon face.
164: 
165:     raccoon-procyon-lotor.jpg at http://www.public-domain-image.com
166: 
167:     Parameters
168:     ----------
169:     gray : bool, optional
170:         If True return 8-bit grey-scale image, otherwise return a color image
171: 
172:     Returns
173:     -------
174:     face : ndarray
175:         image of a racoon face
176: 
177:     Examples
178:     --------
179:     >>> import scipy.misc
180:     >>> face = scipy.misc.face()
181:     >>> face.shape
182:     (768, 1024, 3)
183:     >>> face.max()
184:     255
185:     >>> face.dtype
186:     dtype('uint8')
187: 
188:     >>> import matplotlib.pyplot as plt
189:     >>> plt.gray()
190:     >>> plt.imshow(face)
191:     >>> plt.show()
192: 
193:     '''
194:     import bz2
195:     import os
196:     with open(os.path.join(os.path.dirname(__file__), 'face.dat'), 'rb') as f:
197:         rawdata = f.read()
198:     data = bz2.decompress(rawdata)
199:     face = fromstring(data, dtype='uint8')
200:     face.shape = (768, 1024, 3)
201:     if gray is True:
202:         face = (0.21 * face[:,:,0] + 0.71 * face[:,:,1] + 0.07 * face[:,:,2]).astype('uint8')
203:     return face
204: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_113610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nFunctions which are common and require SciPy Base and Level 1 SciPy\n(special, linalg)\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy import arange, newaxis, hstack, product, array, fromstring' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_113611 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_113611) is not StypyTypeError):

    if (import_113611 != 'pyd_module'):
        __import__(import_113611)
        sys_modules_113612 = sys.modules[import_113611]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', sys_modules_113612.module_type_store, module_type_store, ['arange', 'newaxis', 'hstack', 'product', 'array', 'fromstring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_113612, sys_modules_113612.module_type_store, module_type_store)
    else:
        from numpy import arange, newaxis, hstack, product, array, fromstring

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', None, module_type_store, ['arange', 'newaxis', 'hstack', 'product', 'array', 'fromstring'], [arange, newaxis, hstack, product, array, fromstring])

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_113611)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')


# Assigning a List to a Name (line 10):
__all__ = ['central_diff_weights', 'derivative', 'ascent', 'face']
module_type_store.set_exportable_members(['central_diff_weights', 'derivative', 'ascent', 'face'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_113613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_113614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'central_diff_weights')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_113613, str_113614)
# Adding element type (line 10)
str_113615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 35), 'str', 'derivative')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_113613, str_113615)
# Adding element type (line 10)
str_113616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 49), 'str', 'ascent')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_113613, str_113616)
# Adding element type (line 10)
str_113617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 59), 'str', 'face')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_113613, str_113617)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_113613)

@norecursion
def central_diff_weights(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_113618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 34), 'int')
    defaults = [int_113618]
    # Create a new context for function 'central_diff_weights'
    module_type_store = module_type_store.open_function_context('central_diff_weights', 13, 0, False)
    
    # Passed parameters checking function
    central_diff_weights.stypy_localization = localization
    central_diff_weights.stypy_type_of_self = None
    central_diff_weights.stypy_type_store = module_type_store
    central_diff_weights.stypy_function_name = 'central_diff_weights'
    central_diff_weights.stypy_param_names_list = ['Np', 'ndiv']
    central_diff_weights.stypy_varargs_param_name = None
    central_diff_weights.stypy_kwargs_param_name = None
    central_diff_weights.stypy_call_defaults = defaults
    central_diff_weights.stypy_call_varargs = varargs
    central_diff_weights.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'central_diff_weights', ['Np', 'ndiv'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'central_diff_weights', localization, ['Np', 'ndiv'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'central_diff_weights(...)' code ##################

    str_113619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\n    Return weights for an Np-point central derivative.\n\n    Assumes equally-spaced function points.\n\n    If weights are in the vector w, then\n    derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)\n\n    Parameters\n    ----------\n    Np : int\n        Number of points for the central derivative.\n    ndiv : int, optional\n        Number of divisions.  Default is 1.\n\n    Notes\n    -----\n    Can be inaccurate for large number of points.\n\n    ')
    
    
    # Getting the type of 'Np' (line 34)
    Np_113620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 7), 'Np')
    # Getting the type of 'ndiv' (line 34)
    ndiv_113621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'ndiv')
    int_113622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'int')
    # Applying the binary operator '+' (line 34)
    result_add_113623 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), '+', ndiv_113621, int_113622)
    
    # Applying the binary operator '<' (line 34)
    result_lt_113624 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 7), '<', Np_113620, result_add_113623)
    
    # Testing the type of an if condition (line 34)
    if_condition_113625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_lt_113624)
    # Assigning a type to the variable 'if_condition_113625' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'if_condition_113625', if_condition_113625)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 35)
    # Processing the call arguments (line 35)
    str_113627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'str', 'Number of points must be at least the derivative order + 1.')
    # Processing the call keyword arguments (line 35)
    kwargs_113628 = {}
    # Getting the type of 'ValueError' (line 35)
    ValueError_113626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 35)
    ValueError_call_result_113629 = invoke(stypy.reporting.localization.Localization(__file__, 35, 14), ValueError_113626, *[str_113627], **kwargs_113628)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 35, 8), ValueError_call_result_113629, 'raise parameter', BaseException)
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'Np' (line 36)
    Np_113630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 7), 'Np')
    int_113631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 12), 'int')
    # Applying the binary operator '%' (line 36)
    result_mod_113632 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), '%', Np_113630, int_113631)
    
    int_113633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'int')
    # Applying the binary operator '==' (line 36)
    result_eq_113634 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), '==', result_mod_113632, int_113633)
    
    # Testing the type of an if condition (line 36)
    if_condition_113635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 4), result_eq_113634)
    # Assigning a type to the variable 'if_condition_113635' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'if_condition_113635', if_condition_113635)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 37)
    # Processing the call arguments (line 37)
    str_113637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'str', 'The number of points must be odd.')
    # Processing the call keyword arguments (line 37)
    kwargs_113638 = {}
    # Getting the type of 'ValueError' (line 37)
    ValueError_113636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 37)
    ValueError_call_result_113639 = invoke(stypy.reporting.localization.Localization(__file__, 37, 14), ValueError_113636, *[str_113637], **kwargs_113638)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 37, 8), ValueError_call_result_113639, 'raise parameter', BaseException)
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 4))
    
    # 'from scipy import linalg' statement (line 38)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
    import_113640 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 4), 'scipy')

    if (type(import_113640) is not StypyTypeError):

        if (import_113640 != 'pyd_module'):
            __import__(import_113640)
            sys_modules_113641 = sys.modules[import_113640]
            import_from_module(stypy.reporting.localization.Localization(__file__, 38, 4), 'scipy', sys_modules_113641.module_type_store, module_type_store, ['linalg'])
            nest_module(stypy.reporting.localization.Localization(__file__, 38, 4), __file__, sys_modules_113641, sys_modules_113641.module_type_store, module_type_store)
        else:
            from scipy import linalg

            import_from_module(stypy.reporting.localization.Localization(__file__, 38, 4), 'scipy', None, module_type_store, ['linalg'], [linalg])

    else:
        # Assigning a type to the variable 'scipy' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'scipy', import_113640)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')
    
    
    # Assigning a BinOp to a Name (line 39):
    # Getting the type of 'Np' (line 39)
    Np_113642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'Np')
    int_113643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 15), 'int')
    # Applying the binary operator '>>' (line 39)
    result_rshift_113644 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 9), '>>', Np_113642, int_113643)
    
    # Assigning a type to the variable 'ho' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'ho', result_rshift_113644)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to arange(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Getting the type of 'ho' (line 40)
    ho_113646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'ho', False)
    # Applying the 'usub' unary operator (line 40)
    result___neg___113647 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 15), 'usub', ho_113646)
    
    # Getting the type of 'ho' (line 40)
    ho_113648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'ho', False)
    float_113649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'float')
    # Applying the binary operator '+' (line 40)
    result_add_113650 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 19), '+', ho_113648, float_113649)
    
    # Processing the call keyword arguments (line 40)
    kwargs_113651 = {}
    # Getting the type of 'arange' (line 40)
    arange_113645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'arange', False)
    # Calling arange(args, kwargs) (line 40)
    arange_call_result_113652 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), arange_113645, *[result___neg___113647, result_add_113650], **kwargs_113651)
    
    # Assigning a type to the variable 'x' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'x', arange_call_result_113652)
    
    # Assigning a Subscript to a Name (line 41):
    
    # Obtaining the type of the subscript
    slice_113653 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 41, 8), None, None, None)
    # Getting the type of 'newaxis' (line 41)
    newaxis_113654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'newaxis')
    # Getting the type of 'x' (line 41)
    x_113655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___113656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), x_113655, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_113657 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), getitem___113656, (slice_113653, newaxis_113654))
    
    # Assigning a type to the variable 'x' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'x', subscript_call_result_113657)
    
    # Assigning a BinOp to a Name (line 42):
    # Getting the type of 'x' (line 42)
    x_113658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'x')
    float_113659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'float')
    # Applying the binary operator '**' (line 42)
    result_pow_113660 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 8), '**', x_113658, float_113659)
    
    # Assigning a type to the variable 'X' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'X', result_pow_113660)
    
    
    # Call to range(...): (line 43)
    # Processing the call arguments (line 43)
    int_113662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    # Getting the type of 'Np' (line 43)
    Np_113663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'Np', False)
    # Processing the call keyword arguments (line 43)
    kwargs_113664 = {}
    # Getting the type of 'range' (line 43)
    range_113661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'range', False)
    # Calling range(args, kwargs) (line 43)
    range_call_result_113665 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), range_113661, *[int_113662, Np_113663], **kwargs_113664)
    
    # Testing the type of a for loop iterable (line 43)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 4), range_call_result_113665)
    # Getting the type of the for loop variable (line 43)
    for_loop_var_113666 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 4), range_call_result_113665)
    # Assigning a type to the variable 'k' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'k', for_loop_var_113666)
    # SSA begins for a for statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 44):
    
    # Call to hstack(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_113668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    # Getting the type of 'X' (line 44)
    X_113669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'X', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), list_113668, X_113669)
    # Adding element type (line 44)
    # Getting the type of 'x' (line 44)
    x_113670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'x', False)
    # Getting the type of 'k' (line 44)
    k_113671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'k', False)
    # Applying the binary operator '**' (line 44)
    result_pow_113672 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 22), '**', x_113670, k_113671)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 19), list_113668, result_pow_113672)
    
    # Processing the call keyword arguments (line 44)
    kwargs_113673 = {}
    # Getting the type of 'hstack' (line 44)
    hstack_113667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'hstack', False)
    # Calling hstack(args, kwargs) (line 44)
    hstack_call_result_113674 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), hstack_113667, *[list_113668], **kwargs_113673)
    
    # Assigning a type to the variable 'X' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'X', hstack_call_result_113674)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 45):
    
    # Call to product(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to arange(...): (line 45)
    # Processing the call arguments (line 45)
    int_113677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'int')
    # Getting the type of 'ndiv' (line 45)
    ndiv_113678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'ndiv', False)
    int_113679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 30), 'int')
    # Applying the binary operator '+' (line 45)
    result_add_113680 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 25), '+', ndiv_113678, int_113679)
    
    # Processing the call keyword arguments (line 45)
    kwargs_113681 = {}
    # Getting the type of 'arange' (line 45)
    arange_113676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'arange', False)
    # Calling arange(args, kwargs) (line 45)
    arange_call_result_113682 = invoke(stypy.reporting.localization.Localization(__file__, 45, 16), arange_113676, *[int_113677, result_add_113680], **kwargs_113681)
    
    # Processing the call keyword arguments (line 45)
    int_113683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 38), 'int')
    keyword_113684 = int_113683
    kwargs_113685 = {'axis': keyword_113684}
    # Getting the type of 'product' (line 45)
    product_113675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'product', False)
    # Calling product(args, kwargs) (line 45)
    product_call_result_113686 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), product_113675, *[arange_call_result_113682], **kwargs_113685)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ndiv' (line 45)
    ndiv_113687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 55), 'ndiv')
    
    # Call to inv(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'X' (line 45)
    X_113690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 52), 'X', False)
    # Processing the call keyword arguments (line 45)
    kwargs_113691 = {}
    # Getting the type of 'linalg' (line 45)
    linalg_113688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 41), 'linalg', False)
    # Obtaining the member 'inv' of a type (line 45)
    inv_113689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 41), linalg_113688, 'inv')
    # Calling inv(args, kwargs) (line 45)
    inv_call_result_113692 = invoke(stypy.reporting.localization.Localization(__file__, 45, 41), inv_113689, *[X_113690], **kwargs_113691)
    
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___113693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 41), inv_call_result_113692, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_113694 = invoke(stypy.reporting.localization.Localization(__file__, 45, 41), getitem___113693, ndiv_113687)
    
    # Applying the binary operator '*' (line 45)
    result_mul_113695 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 8), '*', product_call_result_113686, subscript_call_result_113694)
    
    # Assigning a type to the variable 'w' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'w', result_mul_113695)
    # Getting the type of 'w' (line 46)
    w_113696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', w_113696)
    
    # ################# End of 'central_diff_weights(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'central_diff_weights' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_113697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_113697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'central_diff_weights'
    return stypy_return_type_113697

# Assigning a type to the variable 'central_diff_weights' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'central_diff_weights', central_diff_weights)

@norecursion
def derivative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_113698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'float')
    int_113699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 49)
    tuple_113700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 49)
    
    int_113701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 53), 'int')
    defaults = [float_113698, int_113699, tuple_113700, int_113701]
    # Create a new context for function 'derivative'
    module_type_store = module_type_store.open_function_context('derivative', 49, 0, False)
    
    # Passed parameters checking function
    derivative.stypy_localization = localization
    derivative.stypy_type_of_self = None
    derivative.stypy_type_store = module_type_store
    derivative.stypy_function_name = 'derivative'
    derivative.stypy_param_names_list = ['func', 'x0', 'dx', 'n', 'args', 'order']
    derivative.stypy_varargs_param_name = None
    derivative.stypy_kwargs_param_name = None
    derivative.stypy_call_defaults = defaults
    derivative.stypy_call_varargs = varargs
    derivative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'derivative', ['func', 'x0', 'dx', 'n', 'args', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'derivative', localization, ['func', 'x0', 'dx', 'n', 'args', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'derivative(...)' code ##################

    str_113702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'str', '\n    Find the n-th derivative of a function at a point.\n\n    Given a function, use a central difference formula with spacing `dx` to\n    compute the `n`-th derivative at `x0`.\n\n    Parameters\n    ----------\n    func : function\n        Input function.\n    x0 : float\n        The point at which `n`-th derivative is found.\n    dx : float, optional\n        Spacing.\n    n : int, optional\n        Order of the derivative. Default is 1.\n    args : tuple, optional\n        Arguments\n    order : int, optional\n        Number of points to use, must be odd.\n\n    Notes\n    -----\n    Decreasing the step size too small can result in round-off error.\n\n    Examples\n    --------\n    >>> from scipy.misc import derivative\n    >>> def f(x):\n    ...     return x**3 + x**2\n    >>> derivative(f, 1.0, dx=1e-6)\n    4.9999999999217337\n\n    ')
    
    
    # Getting the type of 'order' (line 84)
    order_113703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 7), 'order')
    # Getting the type of 'n' (line 84)
    n_113704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'n')
    int_113705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 19), 'int')
    # Applying the binary operator '+' (line 84)
    result_add_113706 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), '+', n_113704, int_113705)
    
    # Applying the binary operator '<' (line 84)
    result_lt_113707 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 7), '<', order_113703, result_add_113706)
    
    # Testing the type of an if condition (line 84)
    if_condition_113708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 4), result_lt_113707)
    # Assigning a type to the variable 'if_condition_113708' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'if_condition_113708', if_condition_113708)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 85)
    # Processing the call arguments (line 85)
    str_113710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'str', "'order' (the number of points used to compute the derivative), must be at least the derivative order 'n' + 1.")
    # Processing the call keyword arguments (line 85)
    kwargs_113711 = {}
    # Getting the type of 'ValueError' (line 85)
    ValueError_113709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 85)
    ValueError_call_result_113712 = invoke(stypy.reporting.localization.Localization(__file__, 85, 14), ValueError_113709, *[str_113710], **kwargs_113711)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 85, 8), ValueError_call_result_113712, 'raise parameter', BaseException)
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'order' (line 87)
    order_113713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 7), 'order')
    int_113714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'int')
    # Applying the binary operator '%' (line 87)
    result_mod_113715 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 7), '%', order_113713, int_113714)
    
    int_113716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'int')
    # Applying the binary operator '==' (line 87)
    result_eq_113717 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 7), '==', result_mod_113715, int_113716)
    
    # Testing the type of an if condition (line 87)
    if_condition_113718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 4), result_eq_113717)
    # Assigning a type to the variable 'if_condition_113718' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'if_condition_113718', if_condition_113718)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 88)
    # Processing the call arguments (line 88)
    str_113720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'str', "'order' (the number of points used to compute the derivative) must be odd.")
    # Processing the call keyword arguments (line 88)
    kwargs_113721 = {}
    # Getting the type of 'ValueError' (line 88)
    ValueError_113719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 88)
    ValueError_call_result_113722 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), ValueError_113719, *[str_113720], **kwargs_113721)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 8), ValueError_call_result_113722, 'raise parameter', BaseException)
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 91)
    n_113723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 7), 'n')
    int_113724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 12), 'int')
    # Applying the binary operator '==' (line 91)
    result_eq_113725 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 7), '==', n_113723, int_113724)
    
    # Testing the type of an if condition (line 91)
    if_condition_113726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 4), result_eq_113725)
    # Assigning a type to the variable 'if_condition_113726' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'if_condition_113726', if_condition_113726)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'order' (line 92)
    order_113727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'order')
    int_113728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'int')
    # Applying the binary operator '==' (line 92)
    result_eq_113729 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 11), '==', order_113727, int_113728)
    
    # Testing the type of an if condition (line 92)
    if_condition_113730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_eq_113729)
    # Assigning a type to the variable 'if_condition_113730' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_113730', if_condition_113730)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 93):
    
    # Call to array(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Obtaining an instance of the builtin type 'list' (line 93)
    list_113732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 93)
    # Adding element type (line 93)
    int_113733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 28), list_113732, int_113733)
    # Adding element type (line 93)
    int_113734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 28), list_113732, int_113734)
    # Adding element type (line 93)
    int_113735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 28), list_113732, int_113735)
    
    # Processing the call keyword arguments (line 93)
    kwargs_113736 = {}
    # Getting the type of 'array' (line 93)
    array_113731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'array', False)
    # Calling array(args, kwargs) (line 93)
    array_call_result_113737 = invoke(stypy.reporting.localization.Localization(__file__, 93, 22), array_113731, *[list_113732], **kwargs_113736)
    
    float_113738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 38), 'float')
    # Applying the binary operator 'div' (line 93)
    result_div_113739 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 22), 'div', array_call_result_113737, float_113738)
    
    # Assigning a type to the variable 'weights' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'weights', result_div_113739)
    # SSA branch for the else part of an if statement (line 92)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'order' (line 94)
    order_113740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'order')
    int_113741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'int')
    # Applying the binary operator '==' (line 94)
    result_eq_113742 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 13), '==', order_113740, int_113741)
    
    # Testing the type of an if condition (line 94)
    if_condition_113743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 13), result_eq_113742)
    # Assigning a type to the variable 'if_condition_113743' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'if_condition_113743', if_condition_113743)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 95):
    
    # Call to array(...): (line 95)
    # Processing the call arguments (line 95)
    
    # Obtaining an instance of the builtin type 'list' (line 95)
    list_113745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 95)
    # Adding element type (line 95)
    int_113746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), list_113745, int_113746)
    # Adding element type (line 95)
    int_113747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), list_113745, int_113747)
    # Adding element type (line 95)
    int_113748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), list_113745, int_113748)
    # Adding element type (line 95)
    int_113749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), list_113745, int_113749)
    # Adding element type (line 95)
    int_113750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), list_113745, int_113750)
    
    # Processing the call keyword arguments (line 95)
    kwargs_113751 = {}
    # Getting the type of 'array' (line 95)
    array_113744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'array', False)
    # Calling array(args, kwargs) (line 95)
    array_call_result_113752 = invoke(stypy.reporting.localization.Localization(__file__, 95, 22), array_113744, *[list_113745], **kwargs_113751)
    
    float_113753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 43), 'float')
    # Applying the binary operator 'div' (line 95)
    result_div_113754 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 22), 'div', array_call_result_113752, float_113753)
    
    # Assigning a type to the variable 'weights' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'weights', result_div_113754)
    # SSA branch for the else part of an if statement (line 94)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'order' (line 96)
    order_113755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'order')
    int_113756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'int')
    # Applying the binary operator '==' (line 96)
    result_eq_113757 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 13), '==', order_113755, int_113756)
    
    # Testing the type of an if condition (line 96)
    if_condition_113758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 13), result_eq_113757)
    # Assigning a type to the variable 'if_condition_113758' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'if_condition_113758', if_condition_113758)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 97):
    
    # Call to array(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_113760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    # Adding element type (line 97)
    int_113761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_113760, int_113761)
    # Adding element type (line 97)
    int_113762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_113760, int_113762)
    # Adding element type (line 97)
    int_113763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_113760, int_113763)
    # Adding element type (line 97)
    int_113764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_113760, int_113764)
    # Adding element type (line 97)
    int_113765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_113760, int_113765)
    # Adding element type (line 97)
    int_113766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_113760, int_113766)
    # Adding element type (line 97)
    int_113767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 28), list_113760, int_113767)
    
    # Processing the call keyword arguments (line 97)
    kwargs_113768 = {}
    # Getting the type of 'array' (line 97)
    array_113759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'array', False)
    # Calling array(args, kwargs) (line 97)
    array_call_result_113769 = invoke(stypy.reporting.localization.Localization(__file__, 97, 22), array_113759, *[list_113760], **kwargs_113768)
    
    float_113770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 50), 'float')
    # Applying the binary operator 'div' (line 97)
    result_div_113771 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 22), 'div', array_call_result_113769, float_113770)
    
    # Assigning a type to the variable 'weights' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'weights', result_div_113771)
    # SSA branch for the else part of an if statement (line 96)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'order' (line 98)
    order_113772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'order')
    int_113773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 22), 'int')
    # Applying the binary operator '==' (line 98)
    result_eq_113774 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 13), '==', order_113772, int_113773)
    
    # Testing the type of an if condition (line 98)
    if_condition_113775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 13), result_eq_113774)
    # Assigning a type to the variable 'if_condition_113775' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'if_condition_113775', if_condition_113775)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 99):
    
    # Call to array(...): (line 99)
    # Processing the call arguments (line 99)
    
    # Obtaining an instance of the builtin type 'list' (line 99)
    list_113777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 99)
    # Adding element type (line 99)
    int_113778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113778)
    # Adding element type (line 99)
    int_113779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113779)
    # Adding element type (line 99)
    int_113780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113780)
    # Adding element type (line 99)
    int_113781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113781)
    # Adding element type (line 99)
    int_113782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113782)
    # Adding element type (line 99)
    int_113783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113783)
    # Adding element type (line 99)
    int_113784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 50), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113784)
    # Adding element type (line 99)
    int_113785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113785)
    # Adding element type (line 99)
    int_113786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 58), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_113777, int_113786)
    
    # Processing the call keyword arguments (line 99)
    kwargs_113787 = {}
    # Getting the type of 'array' (line 99)
    array_113776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'array', False)
    # Calling array(args, kwargs) (line 99)
    array_call_result_113788 = invoke(stypy.reporting.localization.Localization(__file__, 99, 22), array_113776, *[list_113777], **kwargs_113787)
    
    float_113789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 63), 'float')
    # Applying the binary operator 'div' (line 99)
    result_div_113790 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 22), 'div', array_call_result_113788, float_113789)
    
    # Assigning a type to the variable 'weights' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'weights', result_div_113790)
    # SSA branch for the else part of an if statement (line 98)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 101):
    
    # Call to central_diff_weights(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'order' (line 101)
    order_113792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 43), 'order', False)
    int_113793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 49), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_113794 = {}
    # Getting the type of 'central_diff_weights' (line 101)
    central_diff_weights_113791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'central_diff_weights', False)
    # Calling central_diff_weights(args, kwargs) (line 101)
    central_diff_weights_call_result_113795 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), central_diff_weights_113791, *[order_113792, int_113793], **kwargs_113794)
    
    # Assigning a type to the variable 'weights' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'weights', central_diff_weights_call_result_113795)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 91)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'n' (line 102)
    n_113796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 9), 'n')
    int_113797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 14), 'int')
    # Applying the binary operator '==' (line 102)
    result_eq_113798 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 9), '==', n_113796, int_113797)
    
    # Testing the type of an if condition (line 102)
    if_condition_113799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 9), result_eq_113798)
    # Assigning a type to the variable 'if_condition_113799' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 9), 'if_condition_113799', if_condition_113799)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'order' (line 103)
    order_113800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'order')
    int_113801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 20), 'int')
    # Applying the binary operator '==' (line 103)
    result_eq_113802 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), '==', order_113800, int_113801)
    
    # Testing the type of an if condition (line 103)
    if_condition_113803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_eq_113802)
    # Assigning a type to the variable 'if_condition_113803' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_113803', if_condition_113803)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 104):
    
    # Call to array(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining an instance of the builtin type 'list' (line 104)
    list_113805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 104)
    # Adding element type (line 104)
    int_113806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 28), list_113805, int_113806)
    # Adding element type (line 104)
    float_113807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 28), list_113805, float_113807)
    # Adding element type (line 104)
    int_113808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 28), list_113805, int_113808)
    
    # Processing the call keyword arguments (line 104)
    kwargs_113809 = {}
    # Getting the type of 'array' (line 104)
    array_113804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'array', False)
    # Calling array(args, kwargs) (line 104)
    array_call_result_113810 = invoke(stypy.reporting.localization.Localization(__file__, 104, 22), array_113804, *[list_113805], **kwargs_113809)
    
    # Assigning a type to the variable 'weights' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'weights', array_call_result_113810)
    # SSA branch for the else part of an if statement (line 103)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'order' (line 105)
    order_113811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'order')
    int_113812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'int')
    # Applying the binary operator '==' (line 105)
    result_eq_113813 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 13), '==', order_113811, int_113812)
    
    # Testing the type of an if condition (line 105)
    if_condition_113814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 13), result_eq_113813)
    # Assigning a type to the variable 'if_condition_113814' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'if_condition_113814', if_condition_113814)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 106):
    
    # Call to array(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_113816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    # Adding element type (line 106)
    int_113817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), list_113816, int_113817)
    # Adding element type (line 106)
    int_113818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), list_113816, int_113818)
    # Adding element type (line 106)
    int_113819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), list_113816, int_113819)
    # Adding element type (line 106)
    int_113820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), list_113816, int_113820)
    # Adding element type (line 106)
    int_113821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), list_113816, int_113821)
    
    # Processing the call keyword arguments (line 106)
    kwargs_113822 = {}
    # Getting the type of 'array' (line 106)
    array_113815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'array', False)
    # Calling array(args, kwargs) (line 106)
    array_call_result_113823 = invoke(stypy.reporting.localization.Localization(__file__, 106, 22), array_113815, *[list_113816], **kwargs_113822)
    
    float_113824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 47), 'float')
    # Applying the binary operator 'div' (line 106)
    result_div_113825 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 22), 'div', array_call_result_113823, float_113824)
    
    # Assigning a type to the variable 'weights' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'weights', result_div_113825)
    # SSA branch for the else part of an if statement (line 105)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'order' (line 107)
    order_113826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'order')
    int_113827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'int')
    # Applying the binary operator '==' (line 107)
    result_eq_113828 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 13), '==', order_113826, int_113827)
    
    # Testing the type of an if condition (line 107)
    if_condition_113829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 13), result_eq_113828)
    # Assigning a type to the variable 'if_condition_113829' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'if_condition_113829', if_condition_113829)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 108):
    
    # Call to array(...): (line 108)
    # Processing the call arguments (line 108)
    
    # Obtaining an instance of the builtin type 'list' (line 108)
    list_113831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 108)
    # Adding element type (line 108)
    int_113832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_113831, int_113832)
    # Adding element type (line 108)
    int_113833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_113831, int_113833)
    # Adding element type (line 108)
    int_113834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_113831, int_113834)
    # Adding element type (line 108)
    int_113835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_113831, int_113835)
    # Adding element type (line 108)
    int_113836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_113831, int_113836)
    # Adding element type (line 108)
    int_113837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_113831, int_113837)
    # Adding element type (line 108)
    int_113838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 28), list_113831, int_113838)
    
    # Processing the call keyword arguments (line 108)
    kwargs_113839 = {}
    # Getting the type of 'array' (line 108)
    array_113830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'array', False)
    # Calling array(args, kwargs) (line 108)
    array_call_result_113840 = invoke(stypy.reporting.localization.Localization(__file__, 108, 22), array_113830, *[list_113831], **kwargs_113839)
    
    float_113841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 56), 'float')
    # Applying the binary operator 'div' (line 108)
    result_div_113842 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 22), 'div', array_call_result_113840, float_113841)
    
    # Assigning a type to the variable 'weights' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'weights', result_div_113842)
    # SSA branch for the else part of an if statement (line 107)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'order' (line 109)
    order_113843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'order')
    int_113844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'int')
    # Applying the binary operator '==' (line 109)
    result_eq_113845 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 13), '==', order_113843, int_113844)
    
    # Testing the type of an if condition (line 109)
    if_condition_113846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 13), result_eq_113845)
    # Assigning a type to the variable 'if_condition_113846' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'if_condition_113846', if_condition_113846)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 110):
    
    # Call to array(...): (line 110)
    # Processing the call arguments (line 110)
    
    # Obtaining an instance of the builtin type 'list' (line 110)
    list_113848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 110)
    # Adding element type (line 110)
    int_113849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113849)
    # Adding element type (line 110)
    int_113850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113850)
    # Adding element type (line 110)
    int_113851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113851)
    # Adding element type (line 110)
    int_113852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 42), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113852)
    # Adding element type (line 110)
    int_113853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 47), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113853)
    # Adding element type (line 110)
    int_113854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113854)
    # Adding element type (line 110)
    int_113855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113855)
    # Adding element type (line 110)
    int_113856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 65), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113856)
    # Adding element type (line 110)
    int_113857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 69), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 28), list_113848, int_113857)
    
    # Processing the call keyword arguments (line 110)
    kwargs_113858 = {}
    # Getting the type of 'array' (line 110)
    array_113847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'array', False)
    # Calling array(args, kwargs) (line 110)
    array_call_result_113859 = invoke(stypy.reporting.localization.Localization(__file__, 110, 22), array_113847, *[list_113848], **kwargs_113858)
    
    float_113860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 74), 'float')
    # Applying the binary operator 'div' (line 110)
    result_div_113861 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 22), 'div', array_call_result_113859, float_113860)
    
    # Assigning a type to the variable 'weights' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'weights', result_div_113861)
    # SSA branch for the else part of an if statement (line 109)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 112):
    
    # Call to central_diff_weights(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'order' (line 112)
    order_113863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 'order', False)
    int_113864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 49), 'int')
    # Processing the call keyword arguments (line 112)
    kwargs_113865 = {}
    # Getting the type of 'central_diff_weights' (line 112)
    central_diff_weights_113862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'central_diff_weights', False)
    # Calling central_diff_weights(args, kwargs) (line 112)
    central_diff_weights_call_result_113866 = invoke(stypy.reporting.localization.Localization(__file__, 112, 22), central_diff_weights_113862, *[order_113863, int_113864], **kwargs_113865)
    
    # Assigning a type to the variable 'weights' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'weights', central_diff_weights_call_result_113866)
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 102)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 114):
    
    # Call to central_diff_weights(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'order' (line 114)
    order_113868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 39), 'order', False)
    # Getting the type of 'n' (line 114)
    n_113869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'n', False)
    # Processing the call keyword arguments (line 114)
    kwargs_113870 = {}
    # Getting the type of 'central_diff_weights' (line 114)
    central_diff_weights_113867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'central_diff_weights', False)
    # Calling central_diff_weights(args, kwargs) (line 114)
    central_diff_weights_call_result_113871 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), central_diff_weights_113867, *[order_113868, n_113869], **kwargs_113870)
    
    # Assigning a type to the variable 'weights' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'weights', central_diff_weights_call_result_113871)
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 115):
    float_113872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 10), 'float')
    # Assigning a type to the variable 'val' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'val', float_113872)
    
    # Assigning a BinOp to a Name (line 116):
    # Getting the type of 'order' (line 116)
    order_113873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'order')
    int_113874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 18), 'int')
    # Applying the binary operator '>>' (line 116)
    result_rshift_113875 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 9), '>>', order_113873, int_113874)
    
    # Assigning a type to the variable 'ho' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'ho', result_rshift_113875)
    
    
    # Call to range(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'order' (line 117)
    order_113877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 19), 'order', False)
    # Processing the call keyword arguments (line 117)
    kwargs_113878 = {}
    # Getting the type of 'range' (line 117)
    range_113876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'range', False)
    # Calling range(args, kwargs) (line 117)
    range_call_result_113879 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), range_113876, *[order_113877], **kwargs_113878)
    
    # Testing the type of a for loop iterable (line 117)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 4), range_call_result_113879)
    # Getting the type of the for loop variable (line 117)
    for_loop_var_113880 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 4), range_call_result_113879)
    # Assigning a type to the variable 'k' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'k', for_loop_var_113880)
    # SSA begins for a for statement (line 117)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'val' (line 118)
    val_113881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'val')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 118)
    k_113882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'k')
    # Getting the type of 'weights' (line 118)
    weights_113883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'weights')
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___113884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), weights_113883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 118)
    subscript_call_result_113885 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), getitem___113884, k_113882)
    
    
    # Call to func(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'x0' (line 118)
    x0_113887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'x0', False)
    # Getting the type of 'k' (line 118)
    k_113888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 35), 'k', False)
    # Getting the type of 'ho' (line 118)
    ho_113889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 37), 'ho', False)
    # Applying the binary operator '-' (line 118)
    result_sub_113890 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 35), '-', k_113888, ho_113889)
    
    # Getting the type of 'dx' (line 118)
    dx_113891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 41), 'dx', False)
    # Applying the binary operator '*' (line 118)
    result_mul_113892 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 34), '*', result_sub_113890, dx_113891)
    
    # Applying the binary operator '+' (line 118)
    result_add_113893 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 31), '+', x0_113887, result_mul_113892)
    
    # Getting the type of 'args' (line 118)
    args_113894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 45), 'args', False)
    # Processing the call keyword arguments (line 118)
    kwargs_113895 = {}
    # Getting the type of 'func' (line 118)
    func_113886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'func', False)
    # Calling func(args, kwargs) (line 118)
    func_call_result_113896 = invoke(stypy.reporting.localization.Localization(__file__, 118, 26), func_113886, *[result_add_113893, args_113894], **kwargs_113895)
    
    # Applying the binary operator '*' (line 118)
    result_mul_113897 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), '*', subscript_call_result_113885, func_call_result_113896)
    
    # Applying the binary operator '+=' (line 118)
    result_iadd_113898 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 8), '+=', val_113881, result_mul_113897)
    # Assigning a type to the variable 'val' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'val', result_iadd_113898)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'val' (line 119)
    val_113899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'val')
    
    # Call to product(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Obtaining an instance of the builtin type 'tuple' (line 119)
    tuple_113901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 119)
    # Adding element type (line 119)
    # Getting the type of 'dx' (line 119)
    dx_113902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'dx', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), tuple_113901, dx_113902)
    
    # Getting the type of 'n' (line 119)
    n_113903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'n', False)
    # Applying the binary operator '*' (line 119)
    result_mul_113904 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 25), '*', tuple_113901, n_113903)
    
    # Processing the call keyword arguments (line 119)
    int_113905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 38), 'int')
    keyword_113906 = int_113905
    kwargs_113907 = {'axis': keyword_113906}
    # Getting the type of 'product' (line 119)
    product_113900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'product', False)
    # Calling product(args, kwargs) (line 119)
    product_call_result_113908 = invoke(stypy.reporting.localization.Localization(__file__, 119, 17), product_113900, *[result_mul_113904], **kwargs_113907)
    
    # Applying the binary operator 'div' (line 119)
    result_div_113909 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), 'div', val_113899, product_call_result_113908)
    
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type', result_div_113909)
    
    # ################# End of 'derivative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'derivative' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_113910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_113910)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'derivative'
    return stypy_return_type_113910

# Assigning a type to the variable 'derivative' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'derivative', derivative)

@norecursion
def ascent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ascent'
    module_type_store = module_type_store.open_function_context('ascent', 122, 0, False)
    
    # Passed parameters checking function
    ascent.stypy_localization = localization
    ascent.stypy_type_of_self = None
    ascent.stypy_type_store = module_type_store
    ascent.stypy_function_name = 'ascent'
    ascent.stypy_param_names_list = []
    ascent.stypy_varargs_param_name = None
    ascent.stypy_kwargs_param_name = None
    ascent.stypy_call_defaults = defaults
    ascent.stypy_call_varargs = varargs
    ascent.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ascent', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ascent', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ascent(...)' code ##################

    str_113911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, (-1)), 'str', '\n    Get an 8-bit grayscale bit-depth, 512 x 512 derived image for easy use in demos\n\n    The image is derived from accent-to-the-top.jpg at\n    http://www.public-domain-image.com/people-public-domain-images-pictures/\n\n    Parameters\n    ----------\n    None\n\n    Returns\n    -------\n    ascent : ndarray\n       convenient image to use for testing and demonstration\n\n    Examples\n    --------\n    >>> import scipy.misc\n    >>> ascent = scipy.misc.ascent()\n    >>> ascent.shape\n    (512, 512)\n    >>> ascent.max()\n    255\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.gray()\n    >>> plt.imshow(ascent)\n    >>> plt.show()\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 153, 4))
    
    # 'import pickle' statement (line 153)
    import pickle

    import_module(stypy.reporting.localization.Localization(__file__, 153, 4), 'pickle', pickle, module_type_store)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 154, 4))
    
    # 'import os' statement (line 154)
    import os

    import_module(stypy.reporting.localization.Localization(__file__, 154, 4), 'os', os, module_type_store)
    
    
    # Assigning a Call to a Name (line 155):
    
    # Call to join(...): (line 155)
    # Processing the call arguments (line 155)
    
    # Call to dirname(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of '__file__' (line 155)
    file___113918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), '__file__', False)
    # Processing the call keyword arguments (line 155)
    kwargs_113919 = {}
    # Getting the type of 'os' (line 155)
    os_113915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), 'os', False)
    # Obtaining the member 'path' of a type (line 155)
    path_113916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 25), os_113915, 'path')
    # Obtaining the member 'dirname' of a type (line 155)
    dirname_113917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 25), path_113916, 'dirname')
    # Calling dirname(args, kwargs) (line 155)
    dirname_call_result_113920 = invoke(stypy.reporting.localization.Localization(__file__, 155, 25), dirname_113917, *[file___113918], **kwargs_113919)
    
    str_113921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 51), 'str', 'ascent.dat')
    # Processing the call keyword arguments (line 155)
    kwargs_113922 = {}
    # Getting the type of 'os' (line 155)
    os_113912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'os', False)
    # Obtaining the member 'path' of a type (line 155)
    path_113913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), os_113912, 'path')
    # Obtaining the member 'join' of a type (line 155)
    join_113914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), path_113913, 'join')
    # Calling join(args, kwargs) (line 155)
    join_call_result_113923 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), join_113914, *[dirname_call_result_113920, str_113921], **kwargs_113922)
    
    # Assigning a type to the variable 'fname' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'fname', join_call_result_113923)
    
    # Call to open(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'fname' (line 156)
    fname_113925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'fname', False)
    str_113926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'str', 'rb')
    # Processing the call keyword arguments (line 156)
    kwargs_113927 = {}
    # Getting the type of 'open' (line 156)
    open_113924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 9), 'open', False)
    # Calling open(args, kwargs) (line 156)
    open_call_result_113928 = invoke(stypy.reporting.localization.Localization(__file__, 156, 9), open_113924, *[fname_113925, str_113926], **kwargs_113927)
    
    with_113929 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 156, 9), open_call_result_113928, 'with parameter', '__enter__', '__exit__')

    if with_113929:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 156)
        enter___113930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 9), open_call_result_113928, '__enter__')
        with_enter_113931 = invoke(stypy.reporting.localization.Localization(__file__, 156, 9), enter___113930)
        # Assigning a type to the variable 'f' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 9), 'f', with_enter_113931)
        
        # Assigning a Call to a Name (line 157):
        
        # Call to array(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Call to load(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'f' (line 157)
        f_113935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 35), 'f', False)
        # Processing the call keyword arguments (line 157)
        kwargs_113936 = {}
        # Getting the type of 'pickle' (line 157)
        pickle_113933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 23), 'pickle', False)
        # Obtaining the member 'load' of a type (line 157)
        load_113934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 23), pickle_113933, 'load')
        # Calling load(args, kwargs) (line 157)
        load_call_result_113937 = invoke(stypy.reporting.localization.Localization(__file__, 157, 23), load_113934, *[f_113935], **kwargs_113936)
        
        # Processing the call keyword arguments (line 157)
        kwargs_113938 = {}
        # Getting the type of 'array' (line 157)
        array_113932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'array', False)
        # Calling array(args, kwargs) (line 157)
        array_call_result_113939 = invoke(stypy.reporting.localization.Localization(__file__, 157, 17), array_113932, *[load_call_result_113937], **kwargs_113938)
        
        # Assigning a type to the variable 'ascent' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'ascent', array_call_result_113939)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 156)
        exit___113940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 9), open_call_result_113928, '__exit__')
        with_exit_113941 = invoke(stypy.reporting.localization.Localization(__file__, 156, 9), exit___113940, None, None, None)

    # Getting the type of 'ascent' (line 158)
    ascent_113942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'ascent')
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type', ascent_113942)
    
    # ################# End of 'ascent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ascent' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_113943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_113943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ascent'
    return stypy_return_type_113943

# Assigning a type to the variable 'ascent' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'ascent', ascent)

@norecursion
def face(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 161)
    False_113944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 14), 'False')
    defaults = [False_113944]
    # Create a new context for function 'face'
    module_type_store = module_type_store.open_function_context('face', 161, 0, False)
    
    # Passed parameters checking function
    face.stypy_localization = localization
    face.stypy_type_of_self = None
    face.stypy_type_store = module_type_store
    face.stypy_function_name = 'face'
    face.stypy_param_names_list = ['gray']
    face.stypy_varargs_param_name = None
    face.stypy_kwargs_param_name = None
    face.stypy_call_defaults = defaults
    face.stypy_call_varargs = varargs
    face.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'face', ['gray'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'face', localization, ['gray'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'face(...)' code ##################

    str_113945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', "\n    Get a 1024 x 768, color image of a raccoon face.\n\n    raccoon-procyon-lotor.jpg at http://www.public-domain-image.com\n\n    Parameters\n    ----------\n    gray : bool, optional\n        If True return 8-bit grey-scale image, otherwise return a color image\n\n    Returns\n    -------\n    face : ndarray\n        image of a racoon face\n\n    Examples\n    --------\n    >>> import scipy.misc\n    >>> face = scipy.misc.face()\n    >>> face.shape\n    (768, 1024, 3)\n    >>> face.max()\n    255\n    >>> face.dtype\n    dtype('uint8')\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.gray()\n    >>> plt.imshow(face)\n    >>> plt.show()\n\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 194, 4))
    
    # 'import bz2' statement (line 194)
    import bz2

    import_module(stypy.reporting.localization.Localization(__file__, 194, 4), 'bz2', bz2, module_type_store)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 195, 4))
    
    # 'import os' statement (line 195)
    import os

    import_module(stypy.reporting.localization.Localization(__file__, 195, 4), 'os', os, module_type_store)
    
    
    # Call to open(...): (line 196)
    # Processing the call arguments (line 196)
    
    # Call to join(...): (line 196)
    # Processing the call arguments (line 196)
    
    # Call to dirname(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of '__file__' (line 196)
    file___113953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 43), '__file__', False)
    # Processing the call keyword arguments (line 196)
    kwargs_113954 = {}
    # Getting the type of 'os' (line 196)
    os_113950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 196)
    path_113951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 27), os_113950, 'path')
    # Obtaining the member 'dirname' of a type (line 196)
    dirname_113952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 27), path_113951, 'dirname')
    # Calling dirname(args, kwargs) (line 196)
    dirname_call_result_113955 = invoke(stypy.reporting.localization.Localization(__file__, 196, 27), dirname_113952, *[file___113953], **kwargs_113954)
    
    str_113956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 54), 'str', 'face.dat')
    # Processing the call keyword arguments (line 196)
    kwargs_113957 = {}
    # Getting the type of 'os' (line 196)
    os_113947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 196)
    path_113948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 14), os_113947, 'path')
    # Obtaining the member 'join' of a type (line 196)
    join_113949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 14), path_113948, 'join')
    # Calling join(args, kwargs) (line 196)
    join_call_result_113958 = invoke(stypy.reporting.localization.Localization(__file__, 196, 14), join_113949, *[dirname_call_result_113955, str_113956], **kwargs_113957)
    
    str_113959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 67), 'str', 'rb')
    # Processing the call keyword arguments (line 196)
    kwargs_113960 = {}
    # Getting the type of 'open' (line 196)
    open_113946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 9), 'open', False)
    # Calling open(args, kwargs) (line 196)
    open_call_result_113961 = invoke(stypy.reporting.localization.Localization(__file__, 196, 9), open_113946, *[join_call_result_113958, str_113959], **kwargs_113960)
    
    with_113962 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 196, 9), open_call_result_113961, 'with parameter', '__enter__', '__exit__')

    if with_113962:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 196)
        enter___113963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 9), open_call_result_113961, '__enter__')
        with_enter_113964 = invoke(stypy.reporting.localization.Localization(__file__, 196, 9), enter___113963)
        # Assigning a type to the variable 'f' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 9), 'f', with_enter_113964)
        
        # Assigning a Call to a Name (line 197):
        
        # Call to read(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_113967 = {}
        # Getting the type of 'f' (line 197)
        f_113965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 18), 'f', False)
        # Obtaining the member 'read' of a type (line 197)
        read_113966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 18), f_113965, 'read')
        # Calling read(args, kwargs) (line 197)
        read_call_result_113968 = invoke(stypy.reporting.localization.Localization(__file__, 197, 18), read_113966, *[], **kwargs_113967)
        
        # Assigning a type to the variable 'rawdata' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'rawdata', read_call_result_113968)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 196)
        exit___113969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 9), open_call_result_113961, '__exit__')
        with_exit_113970 = invoke(stypy.reporting.localization.Localization(__file__, 196, 9), exit___113969, None, None, None)

    
    # Assigning a Call to a Name (line 198):
    
    # Call to decompress(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'rawdata' (line 198)
    rawdata_113973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 26), 'rawdata', False)
    # Processing the call keyword arguments (line 198)
    kwargs_113974 = {}
    # Getting the type of 'bz2' (line 198)
    bz2_113971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'bz2', False)
    # Obtaining the member 'decompress' of a type (line 198)
    decompress_113972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), bz2_113971, 'decompress')
    # Calling decompress(args, kwargs) (line 198)
    decompress_call_result_113975 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), decompress_113972, *[rawdata_113973], **kwargs_113974)
    
    # Assigning a type to the variable 'data' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'data', decompress_call_result_113975)
    
    # Assigning a Call to a Name (line 199):
    
    # Call to fromstring(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'data' (line 199)
    data_113977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 22), 'data', False)
    # Processing the call keyword arguments (line 199)
    str_113978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 34), 'str', 'uint8')
    keyword_113979 = str_113978
    kwargs_113980 = {'dtype': keyword_113979}
    # Getting the type of 'fromstring' (line 199)
    fromstring_113976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'fromstring', False)
    # Calling fromstring(args, kwargs) (line 199)
    fromstring_call_result_113981 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), fromstring_113976, *[data_113977], **kwargs_113980)
    
    # Assigning a type to the variable 'face' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'face', fromstring_call_result_113981)
    
    # Assigning a Tuple to a Attribute (line 200):
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_113982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    int_113983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 18), tuple_113982, int_113983)
    # Adding element type (line 200)
    int_113984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 18), tuple_113982, int_113984)
    # Adding element type (line 200)
    int_113985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 18), tuple_113982, int_113985)
    
    # Getting the type of 'face' (line 200)
    face_113986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'face')
    # Setting the type of the member 'shape' of a type (line 200)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 4), face_113986, 'shape', tuple_113982)
    
    
    # Getting the type of 'gray' (line 201)
    gray_113987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 7), 'gray')
    # Getting the type of 'True' (line 201)
    True_113988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'True')
    # Applying the binary operator 'is' (line 201)
    result_is__113989 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 7), 'is', gray_113987, True_113988)
    
    # Testing the type of an if condition (line 201)
    if_condition_113990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 4), result_is__113989)
    # Assigning a type to the variable 'if_condition_113990' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'if_condition_113990', if_condition_113990)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 202):
    
    # Call to astype(...): (line 202)
    # Processing the call arguments (line 202)
    str_114018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 85), 'str', 'uint8')
    # Processing the call keyword arguments (line 202)
    kwargs_114019 = {}
    float_113991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'float')
    
    # Obtaining the type of the subscript
    slice_113992 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 202, 23), None, None, None)
    slice_113993 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 202, 23), None, None, None)
    int_113994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 32), 'int')
    # Getting the type of 'face' (line 202)
    face_113995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'face', False)
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___113996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 23), face_113995, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_113997 = invoke(stypy.reporting.localization.Localization(__file__, 202, 23), getitem___113996, (slice_113992, slice_113993, int_113994))
    
    # Applying the binary operator '*' (line 202)
    result_mul_113998 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 16), '*', float_113991, subscript_call_result_113997)
    
    float_113999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 37), 'float')
    
    # Obtaining the type of the subscript
    slice_114000 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 202, 44), None, None, None)
    slice_114001 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 202, 44), None, None, None)
    int_114002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 53), 'int')
    # Getting the type of 'face' (line 202)
    face_114003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 44), 'face', False)
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___114004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 44), face_114003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_114005 = invoke(stypy.reporting.localization.Localization(__file__, 202, 44), getitem___114004, (slice_114000, slice_114001, int_114002))
    
    # Applying the binary operator '*' (line 202)
    result_mul_114006 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 37), '*', float_113999, subscript_call_result_114005)
    
    # Applying the binary operator '+' (line 202)
    result_add_114007 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 16), '+', result_mul_113998, result_mul_114006)
    
    float_114008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 58), 'float')
    
    # Obtaining the type of the subscript
    slice_114009 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 202, 65), None, None, None)
    slice_114010 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 202, 65), None, None, None)
    int_114011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 74), 'int')
    # Getting the type of 'face' (line 202)
    face_114012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 65), 'face', False)
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___114013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 65), face_114012, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_114014 = invoke(stypy.reporting.localization.Localization(__file__, 202, 65), getitem___114013, (slice_114009, slice_114010, int_114011))
    
    # Applying the binary operator '*' (line 202)
    result_mul_114015 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 58), '*', float_114008, subscript_call_result_114014)
    
    # Applying the binary operator '+' (line 202)
    result_add_114016 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 56), '+', result_add_114007, result_mul_114015)
    
    # Obtaining the member 'astype' of a type (line 202)
    astype_114017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 56), result_add_114016, 'astype')
    # Calling astype(args, kwargs) (line 202)
    astype_call_result_114020 = invoke(stypy.reporting.localization.Localization(__file__, 202, 56), astype_114017, *[str_114018], **kwargs_114019)
    
    # Assigning a type to the variable 'face' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'face', astype_call_result_114020)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'face' (line 203)
    face_114021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'face')
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type', face_114021)
    
    # ################# End of 'face(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'face' in the type store
    # Getting the type of 'stypy_return_type' (line 161)
    stypy_return_type_114022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'face'
    return stypy_return_type_114022

# Assigning a type to the variable 'face' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'face', face)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

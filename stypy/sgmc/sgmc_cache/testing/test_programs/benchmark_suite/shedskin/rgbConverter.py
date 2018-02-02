
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Conversion functions between RGB and other color systems.
2: 
3: This modules provides two functions for each color system ABC:
4: 
5:   rgb_to_abc(r, g, b) --> a, b, c
6:   abc_to_rgb(a, b, c) --> r, g, b
7: 
8: All inputs and outputs are triples of floats in the range [0.0...1.0]
9: (with the exception of I and Q, which covers a slightly larger range).
10: Inputs outside the valid range may cause exceptions or invalid outputs.
11: 
12: Supported color systems:
13: RGB: Red, Green, Blue components
14: YIQ: Luminance, Chrominance (used by composite video signals)
15: HLS: Hue, Luminance, Saturation
16: HSV: Hue, Saturation, Value
17: '''
18: 
19: # References:
20: # http://en.wikipedia.org/wiki/YIQ
21: # http://en.wikipedia.org/wiki/HLS_color_space
22: # http://en.wikipedia.org/wiki/HSV_color_space
23: 
24: __all__ = ["rgb_to_yiq", "yiq_to_rgb", "rgb_to_hls", "hls_to_rgb",
25:            "rgb_to_hsv", "hsv_to_rgb"]
26: 
27: # Some floating point constants
28: 
29: ONE_THIRD = 1.0 / 3.0
30: ONE_SIXTH = 1.0 / 6.0
31: TWO_THIRD = 2.0 / 3.0
32: 
33: 
34: # YIQ: used by composite video signals (linear combinations of RGB)
35: # Y: perceived grey level (0.0 == black, 1.0 == white)
36: # I, Q: color components
37: 
38: def rgb_to_yiq(r, g, b):
39:     y = 0.30 * r + 0.59 * g + 0.11 * b
40:     i = 0.60 * r - 0.28 * g - 0.32 * b
41:     q = 0.21 * r - 0.52 * g + 0.31 * b
42:     return (y, i, q)
43: 
44: 
45: def yiq_to_rgb(y, i, q):
46:     r = y + 0.948262 * i + 0.624013 * q
47:     g = y - 0.276066 * i - 0.639810 * q
48:     b = y - 1.105450 * i + 1.729860 * q
49:     if r < 0.0:
50:         r = 0.0
51:     if g < 0.0:
52:         g = 0.0
53:     if b < 0.0:
54:         b = 0.0
55:     if r > 1.0:
56:         r = 1.0
57:     if g > 1.0:
58:         g = 1.0
59:     if b > 1.0:
60:         b = 1.0
61:     return (r, g, b)
62: 
63: 
64: # HLS: Hue, Luminance, Saturation
65: # H: position in the spectrum
66: # L: color lightness
67: # S: color saturation
68: 
69: def rgb_to_hls(r, g, b):
70:     maxc = max(r, g, b)
71:     minc = min(r, g, b)
72:     # XXX Can optimize (maxc+minc) and (maxc-minc)
73:     l = (minc + maxc) / 2.0
74:     if minc == maxc:
75:         return 0.0, l, 0.0
76:     if l <= 0.5:
77:         s = (maxc - minc) / (maxc + minc)
78:     else:
79:         s = (maxc - minc) / (2.0 - maxc - minc)
80:     rc = (maxc - r) / (maxc - minc)
81:     gc = (maxc - g) / (maxc - minc)
82:     bc = (maxc - b) / (maxc - minc)
83:     if r == maxc:
84:         h = bc - gc
85:     elif g == maxc:
86:         h = 2.0 + rc - bc
87:     else:
88:         h = 4.0 + gc - rc
89:     h = (h / 6.0) % 1.0
90:     return h, l, s
91: 
92: 
93: def hls_to_rgb(h, l, s):
94:     if s == 0.0:
95:         return l, l, l
96:     if l <= 0.5:
97:         m2 = l * (1.0 + s)
98:     else:
99:         m2 = l + s - (l * s)
100:     m1 = 2.0 * l - m2
101:     return (_v(m1, m2, h + ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h - ONE_THIRD))
102: 
103: 
104: def _v(m1, m2, hue):
105:     hue = hue % 1.0
106:     if hue < ONE_SIXTH:
107:         return m1 + (m2 - m1) * hue * 6.0
108:     if hue < 0.5:
109:         return m2
110:     if hue < TWO_THIRD:
111:         return m1 + (m2 - m1) * (TWO_THIRD - hue) * 6.0
112:     return m1
113: 
114: 
115: # HSV: Hue, Saturation, Value
116: # H: position in the spectrum
117: # S: color saturation ("purity")
118: # V: color brightness
119: 
120: def rgb_to_hsv(r, g, b):
121:     maxc = max(r, g, b)
122:     minc = min(r, g, b)
123:     v = maxc
124:     if minc == maxc:
125:         return 0.0, 0.0, v
126:     s = (maxc - minc) / maxc
127:     rc = (maxc - r) / (maxc - minc)
128:     gc = (maxc - g) / (maxc - minc)
129:     bc = (maxc - b) / (maxc - minc)
130:     if r == maxc:
131:         h = bc - gc
132:     elif g == maxc:
133:         h = 2.0 + rc - bc
134:     else:
135:         h = 4.0 + gc - rc
136:     h = (h / 6.0) % 1.0
137:     return h, s, v
138: 
139: 
140: def hsv_to_rgb(h, s, v):
141:     if s == 0.0:
142:         return v, v, v
143:     i = int(h * 6.0)  # XXX assume int() truncates!
144:     f = (h * 6.0) - i
145:     p = v * (1.0 - s)
146:     q = v * (1.0 - s * f)
147:     t = v * (1.0 - s * (1.0 - f))
148:     i = i % 6
149:     if i == 0:
150:         return v, t, p
151:     if i == 1:
152:         return q, v, p
153:     if i == 2:
154:         return p, v, t
155:     if i == 3:
156:         return p, q, v
157:     if i == 4:
158:         return t, p, v
159:     if i == 5:
160:         return v, p, q
161:     # Cannot get here
162: 
163: 
164: def run():
165:     for i in range(50000):
166:         hls_to_rgb(1.0, 0.5, 0.7)
167:         rgb_to_hls(1.0, 0.5, 0.7)
168:         yiq_to_rgb(1.0, 0.5, 0.7)
169:         rgb_to_yiq(1.0, 0.5, 0.7)
170:         hsv_to_rgb(1.0, 0.5, 0.7)
171:         rgb_to_hsv(1.0, 0.5, 0.7)
172: 
173:     return True
174: 
175: 
176: run()
177: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', 'Conversion functions between RGB and other color systems.\n\nThis modules provides two functions for each color system ABC:\n\n  rgb_to_abc(r, g, b) --> a, b, c\n  abc_to_rgb(a, b, c) --> r, g, b\n\nAll inputs and outputs are triples of floats in the range [0.0...1.0]\n(with the exception of I and Q, which covers a slightly larger range).\nInputs outside the valid range may cause exceptions or invalid outputs.\n\nSupported color systems:\nRGB: Red, Green, Blue components\nYIQ: Luminance, Chrominance (used by composite video signals)\nHLS: Hue, Luminance, Saturation\nHSV: Hue, Saturation, Value\n')

# Assigning a List to a Name (line 24):
__all__ = ['rgb_to_yiq', 'yiq_to_rgb', 'rgb_to_hls', 'hls_to_rgb', 'rgb_to_hsv', 'hsv_to_rgb']
module_type_store.set_exportable_members(['rgb_to_yiq', 'yiq_to_rgb', 'rgb_to_hls', 'hls_to_rgb', 'rgb_to_hsv', 'hsv_to_rgb'])

# Obtaining an instance of the builtin type 'list' (line 24)
list_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'str', 'rgb_to_yiq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_2, str_3)
# Adding element type (line 24)
str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', 'yiq_to_rgb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_2, str_4)
# Adding element type (line 24)
str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 39), 'str', 'rgb_to_hls')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_2, str_5)
# Adding element type (line 24)
str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 53), 'str', 'hls_to_rgb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_2, str_6)
# Adding element type (line 24)
str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'str', 'rgb_to_hsv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_2, str_7)
# Adding element type (line 24)
str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'str', 'hsv_to_rgb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_2, str_8)

# Assigning a type to the variable '__all__' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '__all__', list_2)

# Assigning a BinOp to a Name (line 29):
float_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 12), 'float')
float_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'float')
# Applying the binary operator 'div' (line 29)
result_div_11 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 12), 'div', float_9, float_10)

# Assigning a type to the variable 'ONE_THIRD' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'ONE_THIRD', result_div_11)

# Assigning a BinOp to a Name (line 30):
float_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'float')
float_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 18), 'float')
# Applying the binary operator 'div' (line 30)
result_div_14 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), 'div', float_12, float_13)

# Assigning a type to the variable 'ONE_SIXTH' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'ONE_SIXTH', result_div_14)

# Assigning a BinOp to a Name (line 31):
float_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'float')
float_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'float')
# Applying the binary operator 'div' (line 31)
result_div_17 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 12), 'div', float_15, float_16)

# Assigning a type to the variable 'TWO_THIRD' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'TWO_THIRD', result_div_17)

@norecursion
def rgb_to_yiq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rgb_to_yiq'
    module_type_store = module_type_store.open_function_context('rgb_to_yiq', 38, 0, False)
    
    # Passed parameters checking function
    rgb_to_yiq.stypy_localization = localization
    rgb_to_yiq.stypy_type_of_self = None
    rgb_to_yiq.stypy_type_store = module_type_store
    rgb_to_yiq.stypy_function_name = 'rgb_to_yiq'
    rgb_to_yiq.stypy_param_names_list = ['r', 'g', 'b']
    rgb_to_yiq.stypy_varargs_param_name = None
    rgb_to_yiq.stypy_kwargs_param_name = None
    rgb_to_yiq.stypy_call_defaults = defaults
    rgb_to_yiq.stypy_call_varargs = varargs
    rgb_to_yiq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rgb_to_yiq', ['r', 'g', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rgb_to_yiq', localization, ['r', 'g', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rgb_to_yiq(...)' code ##################

    
    # Assigning a BinOp to a Name (line 39):
    float_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'float')
    # Getting the type of 'r' (line 39)
    r_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'r')
    # Applying the binary operator '*' (line 39)
    result_mul_20 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 8), '*', float_18, r_19)
    
    float_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'float')
    # Getting the type of 'g' (line 39)
    g_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'g')
    # Applying the binary operator '*' (line 39)
    result_mul_23 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 19), '*', float_21, g_22)
    
    # Applying the binary operator '+' (line 39)
    result_add_24 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 8), '+', result_mul_20, result_mul_23)
    
    float_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'float')
    # Getting the type of 'b' (line 39)
    b_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'b')
    # Applying the binary operator '*' (line 39)
    result_mul_27 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 30), '*', float_25, b_26)
    
    # Applying the binary operator '+' (line 39)
    result_add_28 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 28), '+', result_add_24, result_mul_27)
    
    # Assigning a type to the variable 'y' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'y', result_add_28)
    
    # Assigning a BinOp to a Name (line 40):
    float_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 8), 'float')
    # Getting the type of 'r' (line 40)
    r_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'r')
    # Applying the binary operator '*' (line 40)
    result_mul_31 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 8), '*', float_29, r_30)
    
    float_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'float')
    # Getting the type of 'g' (line 40)
    g_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'g')
    # Applying the binary operator '*' (line 40)
    result_mul_34 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 19), '*', float_32, g_33)
    
    # Applying the binary operator '-' (line 40)
    result_sub_35 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 8), '-', result_mul_31, result_mul_34)
    
    float_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'float')
    # Getting the type of 'b' (line 40)
    b_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 37), 'b')
    # Applying the binary operator '*' (line 40)
    result_mul_38 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 30), '*', float_36, b_37)
    
    # Applying the binary operator '-' (line 40)
    result_sub_39 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 28), '-', result_sub_35, result_mul_38)
    
    # Assigning a type to the variable 'i' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'i', result_sub_39)
    
    # Assigning a BinOp to a Name (line 41):
    float_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 8), 'float')
    # Getting the type of 'r' (line 41)
    r_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'r')
    # Applying the binary operator '*' (line 41)
    result_mul_42 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 8), '*', float_40, r_41)
    
    float_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'float')
    # Getting the type of 'g' (line 41)
    g_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'g')
    # Applying the binary operator '*' (line 41)
    result_mul_45 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 19), '*', float_43, g_44)
    
    # Applying the binary operator '-' (line 41)
    result_sub_46 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 8), '-', result_mul_42, result_mul_45)
    
    float_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'float')
    # Getting the type of 'b' (line 41)
    b_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'b')
    # Applying the binary operator '*' (line 41)
    result_mul_49 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 30), '*', float_47, b_48)
    
    # Applying the binary operator '+' (line 41)
    result_add_50 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 28), '+', result_sub_46, result_mul_49)
    
    # Assigning a type to the variable 'q' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'q', result_add_50)
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'y' (line 42)
    y_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 12), tuple_51, y_52)
    # Adding element type (line 42)
    # Getting the type of 'i' (line 42)
    i_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 12), tuple_51, i_53)
    # Adding element type (line 42)
    # Getting the type of 'q' (line 42)
    q_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 12), tuple_51, q_54)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', tuple_51)
    
    # ################# End of 'rgb_to_yiq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rgb_to_yiq' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_55)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rgb_to_yiq'
    return stypy_return_type_55

# Assigning a type to the variable 'rgb_to_yiq' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'rgb_to_yiq', rgb_to_yiq)

@norecursion
def yiq_to_rgb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'yiq_to_rgb'
    module_type_store = module_type_store.open_function_context('yiq_to_rgb', 45, 0, False)
    
    # Passed parameters checking function
    yiq_to_rgb.stypy_localization = localization
    yiq_to_rgb.stypy_type_of_self = None
    yiq_to_rgb.stypy_type_store = module_type_store
    yiq_to_rgb.stypy_function_name = 'yiq_to_rgb'
    yiq_to_rgb.stypy_param_names_list = ['y', 'i', 'q']
    yiq_to_rgb.stypy_varargs_param_name = None
    yiq_to_rgb.stypy_kwargs_param_name = None
    yiq_to_rgb.stypy_call_defaults = defaults
    yiq_to_rgb.stypy_call_varargs = varargs
    yiq_to_rgb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'yiq_to_rgb', ['y', 'i', 'q'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'yiq_to_rgb', localization, ['y', 'i', 'q'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'yiq_to_rgb(...)' code ##################

    
    # Assigning a BinOp to a Name (line 46):
    # Getting the type of 'y' (line 46)
    y_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'y')
    float_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'float')
    # Getting the type of 'i' (line 46)
    i_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'i')
    # Applying the binary operator '*' (line 46)
    result_mul_59 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 12), '*', float_57, i_58)
    
    # Applying the binary operator '+' (line 46)
    result_add_60 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 8), '+', y_56, result_mul_59)
    
    float_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'float')
    # Getting the type of 'q' (line 46)
    q_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 38), 'q')
    # Applying the binary operator '*' (line 46)
    result_mul_63 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 27), '*', float_61, q_62)
    
    # Applying the binary operator '+' (line 46)
    result_add_64 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 25), '+', result_add_60, result_mul_63)
    
    # Assigning a type to the variable 'r' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'r', result_add_64)
    
    # Assigning a BinOp to a Name (line 47):
    # Getting the type of 'y' (line 47)
    y_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'y')
    float_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 12), 'float')
    # Getting the type of 'i' (line 47)
    i_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'i')
    # Applying the binary operator '*' (line 47)
    result_mul_68 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 12), '*', float_66, i_67)
    
    # Applying the binary operator '-' (line 47)
    result_sub_69 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 8), '-', y_65, result_mul_68)
    
    float_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'float')
    # Getting the type of 'q' (line 47)
    q_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 38), 'q')
    # Applying the binary operator '*' (line 47)
    result_mul_72 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 27), '*', float_70, q_71)
    
    # Applying the binary operator '-' (line 47)
    result_sub_73 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 25), '-', result_sub_69, result_mul_72)
    
    # Assigning a type to the variable 'g' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'g', result_sub_73)
    
    # Assigning a BinOp to a Name (line 48):
    # Getting the type of 'y' (line 48)
    y_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'y')
    float_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 12), 'float')
    # Getting the type of 'i' (line 48)
    i_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'i')
    # Applying the binary operator '*' (line 48)
    result_mul_77 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 12), '*', float_75, i_76)
    
    # Applying the binary operator '-' (line 48)
    result_sub_78 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 8), '-', y_74, result_mul_77)
    
    float_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'float')
    # Getting the type of 'q' (line 48)
    q_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 38), 'q')
    # Applying the binary operator '*' (line 48)
    result_mul_81 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 27), '*', float_79, q_80)
    
    # Applying the binary operator '+' (line 48)
    result_add_82 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 25), '+', result_sub_78, result_mul_81)
    
    # Assigning a type to the variable 'b' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'b', result_add_82)
    
    # Getting the type of 'r' (line 49)
    r_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 7), 'r')
    float_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 11), 'float')
    # Applying the binary operator '<' (line 49)
    result_lt_85 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 7), '<', r_83, float_84)
    
    # Testing if the type of an if condition is none (line 49)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 4), result_lt_85):
        pass
    else:
        
        # Testing the type of an if condition (line 49)
        if_condition_86 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 4), result_lt_85)
        # Assigning a type to the variable 'if_condition_86' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'if_condition_86', if_condition_86)
        # SSA begins for if statement (line 49)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 50):
        float_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'float')
        # Assigning a type to the variable 'r' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'r', float_87)
        # SSA join for if statement (line 49)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'g' (line 51)
    g_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'g')
    float_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'float')
    # Applying the binary operator '<' (line 51)
    result_lt_90 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 7), '<', g_88, float_89)
    
    # Testing if the type of an if condition is none (line 51)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 51, 4), result_lt_90):
        pass
    else:
        
        # Testing the type of an if condition (line 51)
        if_condition_91 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 4), result_lt_90)
        # Assigning a type to the variable 'if_condition_91' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'if_condition_91', if_condition_91)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 52):
        float_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'float')
        # Assigning a type to the variable 'g' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'g', float_92)
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'b' (line 53)
    b_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'b')
    float_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 11), 'float')
    # Applying the binary operator '<' (line 53)
    result_lt_95 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), '<', b_93, float_94)
    
    # Testing if the type of an if condition is none (line 53)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 53, 4), result_lt_95):
        pass
    else:
        
        # Testing the type of an if condition (line 53)
        if_condition_96 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), result_lt_95)
        # Assigning a type to the variable 'if_condition_96' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_96', if_condition_96)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 54):
        float_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'float')
        # Assigning a type to the variable 'b' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'b', float_97)
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'r' (line 55)
    r_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 7), 'r')
    float_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'float')
    # Applying the binary operator '>' (line 55)
    result_gt_100 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 7), '>', r_98, float_99)
    
    # Testing if the type of an if condition is none (line 55)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 4), result_gt_100):
        pass
    else:
        
        # Testing the type of an if condition (line 55)
        if_condition_101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 4), result_gt_100)
        # Assigning a type to the variable 'if_condition_101' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'if_condition_101', if_condition_101)
        # SSA begins for if statement (line 55)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 56):
        float_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'float')
        # Assigning a type to the variable 'r' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'r', float_102)
        # SSA join for if statement (line 55)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'g' (line 57)
    g_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'g')
    float_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'float')
    # Applying the binary operator '>' (line 57)
    result_gt_105 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), '>', g_103, float_104)
    
    # Testing if the type of an if condition is none (line 57)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 4), result_gt_105):
        pass
    else:
        
        # Testing the type of an if condition (line 57)
        if_condition_106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), result_gt_105)
        # Assigning a type to the variable 'if_condition_106' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_106', if_condition_106)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 58):
        float_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 12), 'float')
        # Assigning a type to the variable 'g' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'g', float_107)
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'b' (line 59)
    b_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), 'b')
    float_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 11), 'float')
    # Applying the binary operator '>' (line 59)
    result_gt_110 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 7), '>', b_108, float_109)
    
    # Testing if the type of an if condition is none (line 59)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 4), result_gt_110):
        pass
    else:
        
        # Testing the type of an if condition (line 59)
        if_condition_111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 4), result_gt_110)
        # Assigning a type to the variable 'if_condition_111' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'if_condition_111', if_condition_111)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 60):
        float_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 12), 'float')
        # Assigning a type to the variable 'b' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'b', float_112)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    # Getting the type of 'r' (line 61)
    r_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 12), tuple_113, r_114)
    # Adding element type (line 61)
    # Getting the type of 'g' (line 61)
    g_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'g')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 12), tuple_113, g_115)
    # Adding element type (line 61)
    # Getting the type of 'b' (line 61)
    b_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 12), tuple_113, b_116)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', tuple_113)
    
    # ################# End of 'yiq_to_rgb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'yiq_to_rgb' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'yiq_to_rgb'
    return stypy_return_type_117

# Assigning a type to the variable 'yiq_to_rgb' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'yiq_to_rgb', yiq_to_rgb)

@norecursion
def rgb_to_hls(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rgb_to_hls'
    module_type_store = module_type_store.open_function_context('rgb_to_hls', 69, 0, False)
    
    # Passed parameters checking function
    rgb_to_hls.stypy_localization = localization
    rgb_to_hls.stypy_type_of_self = None
    rgb_to_hls.stypy_type_store = module_type_store
    rgb_to_hls.stypy_function_name = 'rgb_to_hls'
    rgb_to_hls.stypy_param_names_list = ['r', 'g', 'b']
    rgb_to_hls.stypy_varargs_param_name = None
    rgb_to_hls.stypy_kwargs_param_name = None
    rgb_to_hls.stypy_call_defaults = defaults
    rgb_to_hls.stypy_call_varargs = varargs
    rgb_to_hls.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rgb_to_hls', ['r', 'g', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rgb_to_hls', localization, ['r', 'g', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rgb_to_hls(...)' code ##################

    
    # Assigning a Call to a Name (line 70):
    
    # Call to max(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'r' (line 70)
    r_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'r', False)
    # Getting the type of 'g' (line 70)
    g_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'g', False)
    # Getting the type of 'b' (line 70)
    b_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'b', False)
    # Processing the call keyword arguments (line 70)
    kwargs_122 = {}
    # Getting the type of 'max' (line 70)
    max_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'max', False)
    # Calling max(args, kwargs) (line 70)
    max_call_result_123 = invoke(stypy.reporting.localization.Localization(__file__, 70, 11), max_118, *[r_119, g_120, b_121], **kwargs_122)
    
    # Assigning a type to the variable 'maxc' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'maxc', max_call_result_123)
    
    # Assigning a Call to a Name (line 71):
    
    # Call to min(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'r' (line 71)
    r_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'r', False)
    # Getting the type of 'g' (line 71)
    g_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'g', False)
    # Getting the type of 'b' (line 71)
    b_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'b', False)
    # Processing the call keyword arguments (line 71)
    kwargs_128 = {}
    # Getting the type of 'min' (line 71)
    min_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'min', False)
    # Calling min(args, kwargs) (line 71)
    min_call_result_129 = invoke(stypy.reporting.localization.Localization(__file__, 71, 11), min_124, *[r_125, g_126, b_127], **kwargs_128)
    
    # Assigning a type to the variable 'minc' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'minc', min_call_result_129)
    
    # Assigning a BinOp to a Name (line 73):
    # Getting the type of 'minc' (line 73)
    minc_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 9), 'minc')
    # Getting the type of 'maxc' (line 73)
    maxc_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'maxc')
    # Applying the binary operator '+' (line 73)
    result_add_132 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 9), '+', minc_130, maxc_131)
    
    float_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 24), 'float')
    # Applying the binary operator 'div' (line 73)
    result_div_134 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 8), 'div', result_add_132, float_133)
    
    # Assigning a type to the variable 'l' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'l', result_div_134)
    
    # Getting the type of 'minc' (line 74)
    minc_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'minc')
    # Getting the type of 'maxc' (line 74)
    maxc_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'maxc')
    # Applying the binary operator '==' (line 74)
    result_eq_137 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 7), '==', minc_135, maxc_136)
    
    # Testing if the type of an if condition is none (line 74)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 4), result_eq_137):
        pass
    else:
        
        # Testing the type of an if condition (line 74)
        if_condition_138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 4), result_eq_137)
        # Assigning a type to the variable 'if_condition_138' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'if_condition_138', if_condition_138)
        # SSA begins for if statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        float_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 15), tuple_139, float_140)
        # Adding element type (line 75)
        # Getting the type of 'l' (line 75)
        l_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 15), tuple_139, l_141)
        # Adding element type (line 75)
        float_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 15), tuple_139, float_142)
        
        # Assigning a type to the variable 'stypy_return_type' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', tuple_139)
        # SSA join for if statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'l' (line 76)
    l_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'l')
    float_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'float')
    # Applying the binary operator '<=' (line 76)
    result_le_145 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), '<=', l_143, float_144)
    
    # Testing if the type of an if condition is none (line 76)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 4), result_le_145):
        
        # Assigning a BinOp to a Name (line 79):
        # Getting the type of 'maxc' (line 79)
        maxc_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'maxc')
        # Getting the type of 'minc' (line 79)
        minc_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'minc')
        # Applying the binary operator '-' (line 79)
        result_sub_156 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 13), '-', maxc_154, minc_155)
        
        float_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'float')
        # Getting the type of 'maxc' (line 79)
        maxc_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 35), 'maxc')
        # Applying the binary operator '-' (line 79)
        result_sub_159 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 29), '-', float_157, maxc_158)
        
        # Getting the type of 'minc' (line 79)
        minc_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'minc')
        # Applying the binary operator '-' (line 79)
        result_sub_161 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 40), '-', result_sub_159, minc_160)
        
        # Applying the binary operator 'div' (line 79)
        result_div_162 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 12), 'div', result_sub_156, result_sub_161)
        
        # Assigning a type to the variable 's' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 's', result_div_162)
    else:
        
        # Testing the type of an if condition (line 76)
        if_condition_146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_le_145)
        # Assigning a type to the variable 'if_condition_146' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_146', if_condition_146)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 77):
        # Getting the type of 'maxc' (line 77)
        maxc_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'maxc')
        # Getting the type of 'minc' (line 77)
        minc_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'minc')
        # Applying the binary operator '-' (line 77)
        result_sub_149 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 13), '-', maxc_147, minc_148)
        
        # Getting the type of 'maxc' (line 77)
        maxc_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'maxc')
        # Getting the type of 'minc' (line 77)
        minc_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'minc')
        # Applying the binary operator '+' (line 77)
        result_add_152 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 29), '+', maxc_150, minc_151)
        
        # Applying the binary operator 'div' (line 77)
        result_div_153 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 12), 'div', result_sub_149, result_add_152)
        
        # Assigning a type to the variable 's' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 's', result_div_153)
        # SSA branch for the else part of an if statement (line 76)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 79):
        # Getting the type of 'maxc' (line 79)
        maxc_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'maxc')
        # Getting the type of 'minc' (line 79)
        minc_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'minc')
        # Applying the binary operator '-' (line 79)
        result_sub_156 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 13), '-', maxc_154, minc_155)
        
        float_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'float')
        # Getting the type of 'maxc' (line 79)
        maxc_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 35), 'maxc')
        # Applying the binary operator '-' (line 79)
        result_sub_159 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 29), '-', float_157, maxc_158)
        
        # Getting the type of 'minc' (line 79)
        minc_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'minc')
        # Applying the binary operator '-' (line 79)
        result_sub_161 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 40), '-', result_sub_159, minc_160)
        
        # Applying the binary operator 'div' (line 79)
        result_div_162 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 12), 'div', result_sub_156, result_sub_161)
        
        # Assigning a type to the variable 's' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 's', result_div_162)
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 80):
    # Getting the type of 'maxc' (line 80)
    maxc_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 10), 'maxc')
    # Getting the type of 'r' (line 80)
    r_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'r')
    # Applying the binary operator '-' (line 80)
    result_sub_165 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 10), '-', maxc_163, r_164)
    
    # Getting the type of 'maxc' (line 80)
    maxc_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'maxc')
    # Getting the type of 'minc' (line 80)
    minc_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'minc')
    # Applying the binary operator '-' (line 80)
    result_sub_168 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 23), '-', maxc_166, minc_167)
    
    # Applying the binary operator 'div' (line 80)
    result_div_169 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 9), 'div', result_sub_165, result_sub_168)
    
    # Assigning a type to the variable 'rc' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'rc', result_div_169)
    
    # Assigning a BinOp to a Name (line 81):
    # Getting the type of 'maxc' (line 81)
    maxc_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 10), 'maxc')
    # Getting the type of 'g' (line 81)
    g_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'g')
    # Applying the binary operator '-' (line 81)
    result_sub_172 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 10), '-', maxc_170, g_171)
    
    # Getting the type of 'maxc' (line 81)
    maxc_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'maxc')
    # Getting the type of 'minc' (line 81)
    minc_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'minc')
    # Applying the binary operator '-' (line 81)
    result_sub_175 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 23), '-', maxc_173, minc_174)
    
    # Applying the binary operator 'div' (line 81)
    result_div_176 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 9), 'div', result_sub_172, result_sub_175)
    
    # Assigning a type to the variable 'gc' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'gc', result_div_176)
    
    # Assigning a BinOp to a Name (line 82):
    # Getting the type of 'maxc' (line 82)
    maxc_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 10), 'maxc')
    # Getting the type of 'b' (line 82)
    b_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'b')
    # Applying the binary operator '-' (line 82)
    result_sub_179 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 10), '-', maxc_177, b_178)
    
    # Getting the type of 'maxc' (line 82)
    maxc_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'maxc')
    # Getting the type of 'minc' (line 82)
    minc_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'minc')
    # Applying the binary operator '-' (line 82)
    result_sub_182 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 23), '-', maxc_180, minc_181)
    
    # Applying the binary operator 'div' (line 82)
    result_div_183 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 9), 'div', result_sub_179, result_sub_182)
    
    # Assigning a type to the variable 'bc' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'bc', result_div_183)
    
    # Getting the type of 'r' (line 83)
    r_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 7), 'r')
    # Getting the type of 'maxc' (line 83)
    maxc_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'maxc')
    # Applying the binary operator '==' (line 83)
    result_eq_186 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 7), '==', r_184, maxc_185)
    
    # Testing if the type of an if condition is none (line 83)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 4), result_eq_186):
        
        # Getting the type of 'g' (line 85)
        g_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'g')
        # Getting the type of 'maxc' (line 85)
        maxc_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'maxc')
        # Applying the binary operator '==' (line 85)
        result_eq_193 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 9), '==', g_191, maxc_192)
        
        # Testing if the type of an if condition is none (line 85)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 9), result_eq_193):
            
            # Assigning a BinOp to a Name (line 88):
            float_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'float')
            # Getting the type of 'gc' (line 88)
            gc_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'gc')
            # Applying the binary operator '+' (line 88)
            result_add_202 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '+', float_200, gc_201)
            
            # Getting the type of 'rc' (line 88)
            rc_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'rc')
            # Applying the binary operator '-' (line 88)
            result_sub_204 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '-', result_add_202, rc_203)
            
            # Assigning a type to the variable 'h' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'h', result_sub_204)
        else:
            
            # Testing the type of an if condition (line 85)
            if_condition_194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 9), result_eq_193)
            # Assigning a type to the variable 'if_condition_194' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'if_condition_194', if_condition_194)
            # SSA begins for if statement (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 86):
            float_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'float')
            # Getting the type of 'rc' (line 86)
            rc_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'rc')
            # Applying the binary operator '+' (line 86)
            result_add_197 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 12), '+', float_195, rc_196)
            
            # Getting the type of 'bc' (line 86)
            bc_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'bc')
            # Applying the binary operator '-' (line 86)
            result_sub_199 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '-', result_add_197, bc_198)
            
            # Assigning a type to the variable 'h' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'h', result_sub_199)
            # SSA branch for the else part of an if statement (line 85)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 88):
            float_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'float')
            # Getting the type of 'gc' (line 88)
            gc_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'gc')
            # Applying the binary operator '+' (line 88)
            result_add_202 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '+', float_200, gc_201)
            
            # Getting the type of 'rc' (line 88)
            rc_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'rc')
            # Applying the binary operator '-' (line 88)
            result_sub_204 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '-', result_add_202, rc_203)
            
            # Assigning a type to the variable 'h' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'h', result_sub_204)
            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 83)
        if_condition_187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 4), result_eq_186)
        # Assigning a type to the variable 'if_condition_187' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'if_condition_187', if_condition_187)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 84):
        # Getting the type of 'bc' (line 84)
        bc_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'bc')
        # Getting the type of 'gc' (line 84)
        gc_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'gc')
        # Applying the binary operator '-' (line 84)
        result_sub_190 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 12), '-', bc_188, gc_189)
        
        # Assigning a type to the variable 'h' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'h', result_sub_190)
        # SSA branch for the else part of an if statement (line 83)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'g' (line 85)
        g_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'g')
        # Getting the type of 'maxc' (line 85)
        maxc_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'maxc')
        # Applying the binary operator '==' (line 85)
        result_eq_193 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 9), '==', g_191, maxc_192)
        
        # Testing if the type of an if condition is none (line 85)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 9), result_eq_193):
            
            # Assigning a BinOp to a Name (line 88):
            float_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'float')
            # Getting the type of 'gc' (line 88)
            gc_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'gc')
            # Applying the binary operator '+' (line 88)
            result_add_202 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '+', float_200, gc_201)
            
            # Getting the type of 'rc' (line 88)
            rc_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'rc')
            # Applying the binary operator '-' (line 88)
            result_sub_204 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '-', result_add_202, rc_203)
            
            # Assigning a type to the variable 'h' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'h', result_sub_204)
        else:
            
            # Testing the type of an if condition (line 85)
            if_condition_194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 9), result_eq_193)
            # Assigning a type to the variable 'if_condition_194' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 9), 'if_condition_194', if_condition_194)
            # SSA begins for if statement (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 86):
            float_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'float')
            # Getting the type of 'rc' (line 86)
            rc_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'rc')
            # Applying the binary operator '+' (line 86)
            result_add_197 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 12), '+', float_195, rc_196)
            
            # Getting the type of 'bc' (line 86)
            bc_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'bc')
            # Applying the binary operator '-' (line 86)
            result_sub_199 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '-', result_add_197, bc_198)
            
            # Assigning a type to the variable 'h' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'h', result_sub_199)
            # SSA branch for the else part of an if statement (line 85)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 88):
            float_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'float')
            # Getting the type of 'gc' (line 88)
            gc_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'gc')
            # Applying the binary operator '+' (line 88)
            result_add_202 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '+', float_200, gc_201)
            
            # Getting the type of 'rc' (line 88)
            rc_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'rc')
            # Applying the binary operator '-' (line 88)
            result_sub_204 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 21), '-', result_add_202, rc_203)
            
            # Assigning a type to the variable 'h' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'h', result_sub_204)
            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 89):
    # Getting the type of 'h' (line 89)
    h_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 9), 'h')
    float_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 13), 'float')
    # Applying the binary operator 'div' (line 89)
    result_div_207 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 9), 'div', h_205, float_206)
    
    float_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 20), 'float')
    # Applying the binary operator '%' (line 89)
    result_mod_209 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 8), '%', result_div_207, float_208)
    
    # Assigning a type to the variable 'h' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'h', result_mod_209)
    
    # Obtaining an instance of the builtin type 'tuple' (line 90)
    tuple_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 90)
    # Adding element type (line 90)
    # Getting the type of 'h' (line 90)
    h_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), tuple_210, h_211)
    # Adding element type (line 90)
    # Getting the type of 'l' (line 90)
    l_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 14), 'l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), tuple_210, l_212)
    # Adding element type (line 90)
    # Getting the type of 's' (line 90)
    s_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), tuple_210, s_213)
    
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type', tuple_210)
    
    # ################# End of 'rgb_to_hls(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rgb_to_hls' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_214)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rgb_to_hls'
    return stypy_return_type_214

# Assigning a type to the variable 'rgb_to_hls' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'rgb_to_hls', rgb_to_hls)

@norecursion
def hls_to_rgb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hls_to_rgb'
    module_type_store = module_type_store.open_function_context('hls_to_rgb', 93, 0, False)
    
    # Passed parameters checking function
    hls_to_rgb.stypy_localization = localization
    hls_to_rgb.stypy_type_of_self = None
    hls_to_rgb.stypy_type_store = module_type_store
    hls_to_rgb.stypy_function_name = 'hls_to_rgb'
    hls_to_rgb.stypy_param_names_list = ['h', 'l', 's']
    hls_to_rgb.stypy_varargs_param_name = None
    hls_to_rgb.stypy_kwargs_param_name = None
    hls_to_rgb.stypy_call_defaults = defaults
    hls_to_rgb.stypy_call_varargs = varargs
    hls_to_rgb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hls_to_rgb', ['h', 'l', 's'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hls_to_rgb', localization, ['h', 'l', 's'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hls_to_rgb(...)' code ##################

    
    # Getting the type of 's' (line 94)
    s_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 's')
    float_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 12), 'float')
    # Applying the binary operator '==' (line 94)
    result_eq_217 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), '==', s_215, float_216)
    
    # Testing if the type of an if condition is none (line 94)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 94, 4), result_eq_217):
        pass
    else:
        
        # Testing the type of an if condition (line 94)
        if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_eq_217)
        # Assigning a type to the variable 'if_condition_218' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_218', if_condition_218)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 95)
        tuple_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 95)
        # Adding element type (line 95)
        # Getting the type of 'l' (line 95)
        l_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 15), tuple_219, l_220)
        # Adding element type (line 95)
        # Getting the type of 'l' (line 95)
        l_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 15), tuple_219, l_221)
        # Adding element type (line 95)
        # Getting the type of 'l' (line 95)
        l_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 15), tuple_219, l_222)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', tuple_219)
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'l' (line 96)
    l_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'l')
    float_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'float')
    # Applying the binary operator '<=' (line 96)
    result_le_225 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '<=', l_223, float_224)
    
    # Testing if the type of an if condition is none (line 96)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 4), result_le_225):
        
        # Assigning a BinOp to a Name (line 99):
        # Getting the type of 'l' (line 99)
        l_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'l')
        # Getting the type of 's' (line 99)
        s_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 's')
        # Applying the binary operator '+' (line 99)
        result_add_234 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 13), '+', l_232, s_233)
        
        # Getting the type of 'l' (line 99)
        l_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'l')
        # Getting the type of 's' (line 99)
        s_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 's')
        # Applying the binary operator '*' (line 99)
        result_mul_237 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 22), '*', l_235, s_236)
        
        # Applying the binary operator '-' (line 99)
        result_sub_238 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 19), '-', result_add_234, result_mul_237)
        
        # Assigning a type to the variable 'm2' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'm2', result_sub_238)
    else:
        
        # Testing the type of an if condition (line 96)
        if_condition_226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_le_225)
        # Assigning a type to the variable 'if_condition_226' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_226', if_condition_226)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 97):
        # Getting the type of 'l' (line 97)
        l_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 13), 'l')
        float_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 18), 'float')
        # Getting the type of 's' (line 97)
        s_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 's')
        # Applying the binary operator '+' (line 97)
        result_add_230 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 18), '+', float_228, s_229)
        
        # Applying the binary operator '*' (line 97)
        result_mul_231 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 13), '*', l_227, result_add_230)
        
        # Assigning a type to the variable 'm2' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'm2', result_mul_231)
        # SSA branch for the else part of an if statement (line 96)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 99):
        # Getting the type of 'l' (line 99)
        l_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'l')
        # Getting the type of 's' (line 99)
        s_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 's')
        # Applying the binary operator '+' (line 99)
        result_add_234 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 13), '+', l_232, s_233)
        
        # Getting the type of 'l' (line 99)
        l_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'l')
        # Getting the type of 's' (line 99)
        s_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 's')
        # Applying the binary operator '*' (line 99)
        result_mul_237 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 22), '*', l_235, s_236)
        
        # Applying the binary operator '-' (line 99)
        result_sub_238 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 19), '-', result_add_234, result_mul_237)
        
        # Assigning a type to the variable 'm2' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'm2', result_sub_238)
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 100):
    float_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'float')
    # Getting the type of 'l' (line 100)
    l_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'l')
    # Applying the binary operator '*' (line 100)
    result_mul_241 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 9), '*', float_239, l_240)
    
    # Getting the type of 'm2' (line 100)
    m2_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'm2')
    # Applying the binary operator '-' (line 100)
    result_sub_243 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 9), '-', result_mul_241, m2_242)
    
    # Assigning a type to the variable 'm1' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'm1', result_sub_243)
    
    # Obtaining an instance of the builtin type 'tuple' (line 101)
    tuple_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 101)
    # Adding element type (line 101)
    
    # Call to _v(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'm1' (line 101)
    m1_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'm1', False)
    # Getting the type of 'm2' (line 101)
    m2_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'm2', False)
    # Getting the type of 'h' (line 101)
    h_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'h', False)
    # Getting the type of 'ONE_THIRD' (line 101)
    ONE_THIRD_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'ONE_THIRD', False)
    # Applying the binary operator '+' (line 101)
    result_add_250 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 23), '+', h_248, ONE_THIRD_249)
    
    # Processing the call keyword arguments (line 101)
    kwargs_251 = {}
    # Getting the type of '_v' (line 101)
    _v_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), '_v', False)
    # Calling _v(args, kwargs) (line 101)
    _v_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), _v_245, *[m1_246, m2_247, result_add_250], **kwargs_251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 12), tuple_244, _v_call_result_252)
    # Adding element type (line 101)
    
    # Call to _v(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'm1' (line 101)
    m1_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'm1', False)
    # Getting the type of 'm2' (line 101)
    m2_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'm2', False)
    # Getting the type of 'h' (line 101)
    h_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'h', False)
    # Processing the call keyword arguments (line 101)
    kwargs_257 = {}
    # Getting the type of '_v' (line 101)
    _v_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), '_v', False)
    # Calling _v(args, kwargs) (line 101)
    _v_call_result_258 = invoke(stypy.reporting.localization.Localization(__file__, 101, 39), _v_253, *[m1_254, m2_255, h_256], **kwargs_257)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 12), tuple_244, _v_call_result_258)
    # Adding element type (line 101)
    
    # Call to _v(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'm1' (line 101)
    m1_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 57), 'm1', False)
    # Getting the type of 'm2' (line 101)
    m2_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 61), 'm2', False)
    # Getting the type of 'h' (line 101)
    h_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 65), 'h', False)
    # Getting the type of 'ONE_THIRD' (line 101)
    ONE_THIRD_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 69), 'ONE_THIRD', False)
    # Applying the binary operator '-' (line 101)
    result_sub_264 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 65), '-', h_262, ONE_THIRD_263)
    
    # Processing the call keyword arguments (line 101)
    kwargs_265 = {}
    # Getting the type of '_v' (line 101)
    _v_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 54), '_v', False)
    # Calling _v(args, kwargs) (line 101)
    _v_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 101, 54), _v_259, *[m1_260, m2_261, result_sub_264], **kwargs_265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 12), tuple_244, _v_call_result_266)
    
    # Assigning a type to the variable 'stypy_return_type' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type', tuple_244)
    
    # ################# End of 'hls_to_rgb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hls_to_rgb' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_267)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hls_to_rgb'
    return stypy_return_type_267

# Assigning a type to the variable 'hls_to_rgb' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'hls_to_rgb', hls_to_rgb)

@norecursion
def _v(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_v'
    module_type_store = module_type_store.open_function_context('_v', 104, 0, False)
    
    # Passed parameters checking function
    _v.stypy_localization = localization
    _v.stypy_type_of_self = None
    _v.stypy_type_store = module_type_store
    _v.stypy_function_name = '_v'
    _v.stypy_param_names_list = ['m1', 'm2', 'hue']
    _v.stypy_varargs_param_name = None
    _v.stypy_kwargs_param_name = None
    _v.stypy_call_defaults = defaults
    _v.stypy_call_varargs = varargs
    _v.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_v', ['m1', 'm2', 'hue'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_v', localization, ['m1', 'm2', 'hue'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_v(...)' code ##################

    
    # Assigning a BinOp to a Name (line 105):
    # Getting the type of 'hue' (line 105)
    hue_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 10), 'hue')
    float_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 16), 'float')
    # Applying the binary operator '%' (line 105)
    result_mod_270 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 10), '%', hue_268, float_269)
    
    # Assigning a type to the variable 'hue' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'hue', result_mod_270)
    
    # Getting the type of 'hue' (line 106)
    hue_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 7), 'hue')
    # Getting the type of 'ONE_SIXTH' (line 106)
    ONE_SIXTH_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'ONE_SIXTH')
    # Applying the binary operator '<' (line 106)
    result_lt_273 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 7), '<', hue_271, ONE_SIXTH_272)
    
    # Testing if the type of an if condition is none (line 106)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 4), result_lt_273):
        pass
    else:
        
        # Testing the type of an if condition (line 106)
        if_condition_274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 4), result_lt_273)
        # Assigning a type to the variable 'if_condition_274' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'if_condition_274', if_condition_274)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'm1' (line 107)
        m1_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'm1')
        # Getting the type of 'm2' (line 107)
        m2_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'm2')
        # Getting the type of 'm1' (line 107)
        m1_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 26), 'm1')
        # Applying the binary operator '-' (line 107)
        result_sub_278 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 21), '-', m2_276, m1_277)
        
        # Getting the type of 'hue' (line 107)
        hue_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'hue')
        # Applying the binary operator '*' (line 107)
        result_mul_280 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 20), '*', result_sub_278, hue_279)
        
        float_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 38), 'float')
        # Applying the binary operator '*' (line 107)
        result_mul_282 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 36), '*', result_mul_280, float_281)
        
        # Applying the binary operator '+' (line 107)
        result_add_283 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), '+', m1_275, result_mul_282)
        
        # Assigning a type to the variable 'stypy_return_type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', result_add_283)
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'hue' (line 108)
    hue_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'hue')
    float_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 13), 'float')
    # Applying the binary operator '<' (line 108)
    result_lt_286 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 7), '<', hue_284, float_285)
    
    # Testing if the type of an if condition is none (line 108)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 108, 4), result_lt_286):
        pass
    else:
        
        # Testing the type of an if condition (line 108)
        if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), result_lt_286)
        # Assigning a type to the variable 'if_condition_287' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_287', if_condition_287)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'm2' (line 109)
        m2_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'm2')
        # Assigning a type to the variable 'stypy_return_type' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'stypy_return_type', m2_288)
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'hue' (line 110)
    hue_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), 'hue')
    # Getting the type of 'TWO_THIRD' (line 110)
    TWO_THIRD_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'TWO_THIRD')
    # Applying the binary operator '<' (line 110)
    result_lt_291 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), '<', hue_289, TWO_THIRD_290)
    
    # Testing if the type of an if condition is none (line 110)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 110, 4), result_lt_291):
        pass
    else:
        
        # Testing the type of an if condition (line 110)
        if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), result_lt_291)
        # Assigning a type to the variable 'if_condition_292' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_292', if_condition_292)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'm1' (line 111)
        m1_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'm1')
        # Getting the type of 'm2' (line 111)
        m2_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'm2')
        # Getting the type of 'm1' (line 111)
        m1_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'm1')
        # Applying the binary operator '-' (line 111)
        result_sub_296 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 21), '-', m2_294, m1_295)
        
        # Getting the type of 'TWO_THIRD' (line 111)
        TWO_THIRD_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'TWO_THIRD')
        # Getting the type of 'hue' (line 111)
        hue_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 45), 'hue')
        # Applying the binary operator '-' (line 111)
        result_sub_299 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 33), '-', TWO_THIRD_297, hue_298)
        
        # Applying the binary operator '*' (line 111)
        result_mul_300 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 20), '*', result_sub_296, result_sub_299)
        
        float_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 52), 'float')
        # Applying the binary operator '*' (line 111)
        result_mul_302 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 50), '*', result_mul_300, float_301)
        
        # Applying the binary operator '+' (line 111)
        result_add_303 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '+', m1_293, result_mul_302)
        
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', result_add_303)
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'm1' (line 112)
    m1_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'm1')
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', m1_304)
    
    # ################# End of '_v(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_v' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_v'
    return stypy_return_type_305

# Assigning a type to the variable '_v' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), '_v', _v)

@norecursion
def rgb_to_hsv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rgb_to_hsv'
    module_type_store = module_type_store.open_function_context('rgb_to_hsv', 120, 0, False)
    
    # Passed parameters checking function
    rgb_to_hsv.stypy_localization = localization
    rgb_to_hsv.stypy_type_of_self = None
    rgb_to_hsv.stypy_type_store = module_type_store
    rgb_to_hsv.stypy_function_name = 'rgb_to_hsv'
    rgb_to_hsv.stypy_param_names_list = ['r', 'g', 'b']
    rgb_to_hsv.stypy_varargs_param_name = None
    rgb_to_hsv.stypy_kwargs_param_name = None
    rgb_to_hsv.stypy_call_defaults = defaults
    rgb_to_hsv.stypy_call_varargs = varargs
    rgb_to_hsv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rgb_to_hsv', ['r', 'g', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rgb_to_hsv', localization, ['r', 'g', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rgb_to_hsv(...)' code ##################

    
    # Assigning a Call to a Name (line 121):
    
    # Call to max(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'r' (line 121)
    r_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'r', False)
    # Getting the type of 'g' (line 121)
    g_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'g', False)
    # Getting the type of 'b' (line 121)
    b_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 21), 'b', False)
    # Processing the call keyword arguments (line 121)
    kwargs_310 = {}
    # Getting the type of 'max' (line 121)
    max_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'max', False)
    # Calling max(args, kwargs) (line 121)
    max_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), max_306, *[r_307, g_308, b_309], **kwargs_310)
    
    # Assigning a type to the variable 'maxc' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'maxc', max_call_result_311)
    
    # Assigning a Call to a Name (line 122):
    
    # Call to min(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'r' (line 122)
    r_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'r', False)
    # Getting the type of 'g' (line 122)
    g_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 18), 'g', False)
    # Getting the type of 'b' (line 122)
    b_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'b', False)
    # Processing the call keyword arguments (line 122)
    kwargs_316 = {}
    # Getting the type of 'min' (line 122)
    min_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'min', False)
    # Calling min(args, kwargs) (line 122)
    min_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 122, 11), min_312, *[r_313, g_314, b_315], **kwargs_316)
    
    # Assigning a type to the variable 'minc' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'minc', min_call_result_317)
    
    # Assigning a Name to a Name (line 123):
    # Getting the type of 'maxc' (line 123)
    maxc_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'maxc')
    # Assigning a type to the variable 'v' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'v', maxc_318)
    
    # Getting the type of 'minc' (line 124)
    minc_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'minc')
    # Getting the type of 'maxc' (line 124)
    maxc_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'maxc')
    # Applying the binary operator '==' (line 124)
    result_eq_321 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 7), '==', minc_319, maxc_320)
    
    # Testing if the type of an if condition is none (line 124)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 124, 4), result_eq_321):
        pass
    else:
        
        # Testing the type of an if condition (line 124)
        if_condition_322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), result_eq_321)
        # Assigning a type to the variable 'if_condition_322' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_322', if_condition_322)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        float_324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_323, float_324)
        # Adding element type (line 125)
        float_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_323, float_325)
        # Adding element type (line 125)
        # Getting the type of 'v' (line 125)
        v_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_323, v_326)
        
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', tuple_323)
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 126):
    # Getting the type of 'maxc' (line 126)
    maxc_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 9), 'maxc')
    # Getting the type of 'minc' (line 126)
    minc_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'minc')
    # Applying the binary operator '-' (line 126)
    result_sub_329 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 9), '-', maxc_327, minc_328)
    
    # Getting the type of 'maxc' (line 126)
    maxc_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'maxc')
    # Applying the binary operator 'div' (line 126)
    result_div_331 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 8), 'div', result_sub_329, maxc_330)
    
    # Assigning a type to the variable 's' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 's', result_div_331)
    
    # Assigning a BinOp to a Name (line 127):
    # Getting the type of 'maxc' (line 127)
    maxc_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 10), 'maxc')
    # Getting the type of 'r' (line 127)
    r_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'r')
    # Applying the binary operator '-' (line 127)
    result_sub_334 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 10), '-', maxc_332, r_333)
    
    # Getting the type of 'maxc' (line 127)
    maxc_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'maxc')
    # Getting the type of 'minc' (line 127)
    minc_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 30), 'minc')
    # Applying the binary operator '-' (line 127)
    result_sub_337 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 23), '-', maxc_335, minc_336)
    
    # Applying the binary operator 'div' (line 127)
    result_div_338 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 9), 'div', result_sub_334, result_sub_337)
    
    # Assigning a type to the variable 'rc' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'rc', result_div_338)
    
    # Assigning a BinOp to a Name (line 128):
    # Getting the type of 'maxc' (line 128)
    maxc_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 10), 'maxc')
    # Getting the type of 'g' (line 128)
    g_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'g')
    # Applying the binary operator '-' (line 128)
    result_sub_341 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 10), '-', maxc_339, g_340)
    
    # Getting the type of 'maxc' (line 128)
    maxc_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'maxc')
    # Getting the type of 'minc' (line 128)
    minc_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 'minc')
    # Applying the binary operator '-' (line 128)
    result_sub_344 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 23), '-', maxc_342, minc_343)
    
    # Applying the binary operator 'div' (line 128)
    result_div_345 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 9), 'div', result_sub_341, result_sub_344)
    
    # Assigning a type to the variable 'gc' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'gc', result_div_345)
    
    # Assigning a BinOp to a Name (line 129):
    # Getting the type of 'maxc' (line 129)
    maxc_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 10), 'maxc')
    # Getting the type of 'b' (line 129)
    b_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'b')
    # Applying the binary operator '-' (line 129)
    result_sub_348 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 10), '-', maxc_346, b_347)
    
    # Getting the type of 'maxc' (line 129)
    maxc_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'maxc')
    # Getting the type of 'minc' (line 129)
    minc_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'minc')
    # Applying the binary operator '-' (line 129)
    result_sub_351 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 23), '-', maxc_349, minc_350)
    
    # Applying the binary operator 'div' (line 129)
    result_div_352 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 9), 'div', result_sub_348, result_sub_351)
    
    # Assigning a type to the variable 'bc' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'bc', result_div_352)
    
    # Getting the type of 'r' (line 130)
    r_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'r')
    # Getting the type of 'maxc' (line 130)
    maxc_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'maxc')
    # Applying the binary operator '==' (line 130)
    result_eq_355 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), '==', r_353, maxc_354)
    
    # Testing if the type of an if condition is none (line 130)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 4), result_eq_355):
        
        # Getting the type of 'g' (line 132)
        g_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 9), 'g')
        # Getting the type of 'maxc' (line 132)
        maxc_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'maxc')
        # Applying the binary operator '==' (line 132)
        result_eq_362 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 9), '==', g_360, maxc_361)
        
        # Testing if the type of an if condition is none (line 132)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 132, 9), result_eq_362):
            
            # Assigning a BinOp to a Name (line 135):
            float_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 12), 'float')
            # Getting the type of 'gc' (line 135)
            gc_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'gc')
            # Applying the binary operator '+' (line 135)
            result_add_371 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 12), '+', float_369, gc_370)
            
            # Getting the type of 'rc' (line 135)
            rc_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'rc')
            # Applying the binary operator '-' (line 135)
            result_sub_373 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 21), '-', result_add_371, rc_372)
            
            # Assigning a type to the variable 'h' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'h', result_sub_373)
        else:
            
            # Testing the type of an if condition (line 132)
            if_condition_363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 9), result_eq_362)
            # Assigning a type to the variable 'if_condition_363' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 9), 'if_condition_363', if_condition_363)
            # SSA begins for if statement (line 132)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 133):
            float_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 12), 'float')
            # Getting the type of 'rc' (line 133)
            rc_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'rc')
            # Applying the binary operator '+' (line 133)
            result_add_366 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 12), '+', float_364, rc_365)
            
            # Getting the type of 'bc' (line 133)
            bc_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'bc')
            # Applying the binary operator '-' (line 133)
            result_sub_368 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 21), '-', result_add_366, bc_367)
            
            # Assigning a type to the variable 'h' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'h', result_sub_368)
            # SSA branch for the else part of an if statement (line 132)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 135):
            float_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 12), 'float')
            # Getting the type of 'gc' (line 135)
            gc_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'gc')
            # Applying the binary operator '+' (line 135)
            result_add_371 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 12), '+', float_369, gc_370)
            
            # Getting the type of 'rc' (line 135)
            rc_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'rc')
            # Applying the binary operator '-' (line 135)
            result_sub_373 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 21), '-', result_add_371, rc_372)
            
            # Assigning a type to the variable 'h' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'h', result_sub_373)
            # SSA join for if statement (line 132)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 130)
        if_condition_356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), result_eq_355)
        # Assigning a type to the variable 'if_condition_356' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_356', if_condition_356)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 131):
        # Getting the type of 'bc' (line 131)
        bc_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'bc')
        # Getting the type of 'gc' (line 131)
        gc_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), 'gc')
        # Applying the binary operator '-' (line 131)
        result_sub_359 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 12), '-', bc_357, gc_358)
        
        # Assigning a type to the variable 'h' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'h', result_sub_359)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'g' (line 132)
        g_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 9), 'g')
        # Getting the type of 'maxc' (line 132)
        maxc_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'maxc')
        # Applying the binary operator '==' (line 132)
        result_eq_362 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 9), '==', g_360, maxc_361)
        
        # Testing if the type of an if condition is none (line 132)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 132, 9), result_eq_362):
            
            # Assigning a BinOp to a Name (line 135):
            float_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 12), 'float')
            # Getting the type of 'gc' (line 135)
            gc_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'gc')
            # Applying the binary operator '+' (line 135)
            result_add_371 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 12), '+', float_369, gc_370)
            
            # Getting the type of 'rc' (line 135)
            rc_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'rc')
            # Applying the binary operator '-' (line 135)
            result_sub_373 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 21), '-', result_add_371, rc_372)
            
            # Assigning a type to the variable 'h' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'h', result_sub_373)
        else:
            
            # Testing the type of an if condition (line 132)
            if_condition_363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 9), result_eq_362)
            # Assigning a type to the variable 'if_condition_363' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 9), 'if_condition_363', if_condition_363)
            # SSA begins for if statement (line 132)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 133):
            float_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 12), 'float')
            # Getting the type of 'rc' (line 133)
            rc_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'rc')
            # Applying the binary operator '+' (line 133)
            result_add_366 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 12), '+', float_364, rc_365)
            
            # Getting the type of 'bc' (line 133)
            bc_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'bc')
            # Applying the binary operator '-' (line 133)
            result_sub_368 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 21), '-', result_add_366, bc_367)
            
            # Assigning a type to the variable 'h' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'h', result_sub_368)
            # SSA branch for the else part of an if statement (line 132)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 135):
            float_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 12), 'float')
            # Getting the type of 'gc' (line 135)
            gc_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'gc')
            # Applying the binary operator '+' (line 135)
            result_add_371 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 12), '+', float_369, gc_370)
            
            # Getting the type of 'rc' (line 135)
            rc_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'rc')
            # Applying the binary operator '-' (line 135)
            result_sub_373 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 21), '-', result_add_371, rc_372)
            
            # Assigning a type to the variable 'h' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'h', result_sub_373)
            # SSA join for if statement (line 132)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a BinOp to a Name (line 136):
    # Getting the type of 'h' (line 136)
    h_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 9), 'h')
    float_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 13), 'float')
    # Applying the binary operator 'div' (line 136)
    result_div_376 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 9), 'div', h_374, float_375)
    
    float_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'float')
    # Applying the binary operator '%' (line 136)
    result_mod_378 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 8), '%', result_div_376, float_377)
    
    # Assigning a type to the variable 'h' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'h', result_mod_378)
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'h' (line 137)
    h_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_379, h_380)
    # Adding element type (line 137)
    # Getting the type of 's' (line 137)
    s_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 14), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_379, s_381)
    # Adding element type (line 137)
    # Getting the type of 'v' (line 137)
    v_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 11), tuple_379, v_382)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', tuple_379)
    
    # ################# End of 'rgb_to_hsv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rgb_to_hsv' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_383)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rgb_to_hsv'
    return stypy_return_type_383

# Assigning a type to the variable 'rgb_to_hsv' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'rgb_to_hsv', rgb_to_hsv)

@norecursion
def hsv_to_rgb(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hsv_to_rgb'
    module_type_store = module_type_store.open_function_context('hsv_to_rgb', 140, 0, False)
    
    # Passed parameters checking function
    hsv_to_rgb.stypy_localization = localization
    hsv_to_rgb.stypy_type_of_self = None
    hsv_to_rgb.stypy_type_store = module_type_store
    hsv_to_rgb.stypy_function_name = 'hsv_to_rgb'
    hsv_to_rgb.stypy_param_names_list = ['h', 's', 'v']
    hsv_to_rgb.stypy_varargs_param_name = None
    hsv_to_rgb.stypy_kwargs_param_name = None
    hsv_to_rgb.stypy_call_defaults = defaults
    hsv_to_rgb.stypy_call_varargs = varargs
    hsv_to_rgb.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hsv_to_rgb', ['h', 's', 'v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hsv_to_rgb', localization, ['h', 's', 'v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hsv_to_rgb(...)' code ##################

    
    # Getting the type of 's' (line 141)
    s_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 7), 's')
    float_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 12), 'float')
    # Applying the binary operator '==' (line 141)
    result_eq_386 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 7), '==', s_384, float_385)
    
    # Testing if the type of an if condition is none (line 141)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 4), result_eq_386):
        pass
    else:
        
        # Testing the type of an if condition (line 141)
        if_condition_387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 4), result_eq_386)
        # Assigning a type to the variable 'if_condition_387' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'if_condition_387', if_condition_387)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 142)
        tuple_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 142)
        # Adding element type (line 142)
        # Getting the type of 'v' (line 142)
        v_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 15), tuple_388, v_389)
        # Adding element type (line 142)
        # Getting the type of 'v' (line 142)
        v_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 15), tuple_388, v_390)
        # Adding element type (line 142)
        # Getting the type of 'v' (line 142)
        v_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 15), tuple_388, v_391)
        
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', tuple_388)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 143):
    
    # Call to int(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'h' (line 143)
    h_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'h', False)
    float_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 16), 'float')
    # Applying the binary operator '*' (line 143)
    result_mul_395 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 12), '*', h_393, float_394)
    
    # Processing the call keyword arguments (line 143)
    kwargs_396 = {}
    # Getting the type of 'int' (line 143)
    int_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'int', False)
    # Calling int(args, kwargs) (line 143)
    int_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), int_392, *[result_mul_395], **kwargs_396)
    
    # Assigning a type to the variable 'i' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'i', int_call_result_397)
    
    # Assigning a BinOp to a Name (line 144):
    # Getting the type of 'h' (line 144)
    h_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 9), 'h')
    float_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 13), 'float')
    # Applying the binary operator '*' (line 144)
    result_mul_400 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 9), '*', h_398, float_399)
    
    # Getting the type of 'i' (line 144)
    i_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'i')
    # Applying the binary operator '-' (line 144)
    result_sub_402 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 8), '-', result_mul_400, i_401)
    
    # Assigning a type to the variable 'f' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'f', result_sub_402)
    
    # Assigning a BinOp to a Name (line 145):
    # Getting the type of 'v' (line 145)
    v_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'v')
    float_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 13), 'float')
    # Getting the type of 's' (line 145)
    s_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 's')
    # Applying the binary operator '-' (line 145)
    result_sub_406 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 13), '-', float_404, s_405)
    
    # Applying the binary operator '*' (line 145)
    result_mul_407 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 8), '*', v_403, result_sub_406)
    
    # Assigning a type to the variable 'p' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'p', result_mul_407)
    
    # Assigning a BinOp to a Name (line 146):
    # Getting the type of 'v' (line 146)
    v_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'v')
    float_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 13), 'float')
    # Getting the type of 's' (line 146)
    s_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 's')
    # Getting the type of 'f' (line 146)
    f_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'f')
    # Applying the binary operator '*' (line 146)
    result_mul_412 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 19), '*', s_410, f_411)
    
    # Applying the binary operator '-' (line 146)
    result_sub_413 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 13), '-', float_409, result_mul_412)
    
    # Applying the binary operator '*' (line 146)
    result_mul_414 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 8), '*', v_408, result_sub_413)
    
    # Assigning a type to the variable 'q' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'q', result_mul_414)
    
    # Assigning a BinOp to a Name (line 147):
    # Getting the type of 'v' (line 147)
    v_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'v')
    float_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 13), 'float')
    # Getting the type of 's' (line 147)
    s_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 's')
    float_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'float')
    # Getting the type of 'f' (line 147)
    f_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'f')
    # Applying the binary operator '-' (line 147)
    result_sub_420 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 24), '-', float_418, f_419)
    
    # Applying the binary operator '*' (line 147)
    result_mul_421 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), '*', s_417, result_sub_420)
    
    # Applying the binary operator '-' (line 147)
    result_sub_422 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 13), '-', float_416, result_mul_421)
    
    # Applying the binary operator '*' (line 147)
    result_mul_423 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 8), '*', v_415, result_sub_422)
    
    # Assigning a type to the variable 't' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 't', result_mul_423)
    
    # Assigning a BinOp to a Name (line 148):
    # Getting the type of 'i' (line 148)
    i_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'i')
    int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'int')
    # Applying the binary operator '%' (line 148)
    result_mod_426 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 8), '%', i_424, int_425)
    
    # Assigning a type to the variable 'i' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'i', result_mod_426)
    
    # Getting the type of 'i' (line 149)
    i_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 'i')
    int_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'int')
    # Applying the binary operator '==' (line 149)
    result_eq_429 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 7), '==', i_427, int_428)
    
    # Testing if the type of an if condition is none (line 149)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 4), result_eq_429):
        pass
    else:
        
        # Testing the type of an if condition (line 149)
        if_condition_430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 4), result_eq_429)
        # Assigning a type to the variable 'if_condition_430' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'if_condition_430', if_condition_430)
        # SSA begins for if statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 150)
        tuple_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 150)
        # Adding element type (line 150)
        # Getting the type of 'v' (line 150)
        v_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 15), tuple_431, v_432)
        # Adding element type (line 150)
        # Getting the type of 't' (line 150)
        t_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 15), tuple_431, t_433)
        # Adding element type (line 150)
        # Getting the type of 'p' (line 150)
        p_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 15), tuple_431, p_434)
        
        # Assigning a type to the variable 'stypy_return_type' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'stypy_return_type', tuple_431)
        # SSA join for if statement (line 149)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'i' (line 151)
    i_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'i')
    int_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
    # Applying the binary operator '==' (line 151)
    result_eq_437 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 7), '==', i_435, int_436)
    
    # Testing if the type of an if condition is none (line 151)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 4), result_eq_437):
        pass
    else:
        
        # Testing the type of an if condition (line 151)
        if_condition_438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), result_eq_437)
        # Assigning a type to the variable 'if_condition_438' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_438', if_condition_438)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        # Getting the type of 'q' (line 152)
        q_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_439, q_440)
        # Adding element type (line 152)
        # Getting the type of 'v' (line 152)
        v_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_439, v_441)
        # Adding element type (line 152)
        # Getting the type of 'p' (line 152)
        p_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 21), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 15), tuple_439, p_442)
        
        # Assigning a type to the variable 'stypy_return_type' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'stypy_return_type', tuple_439)
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'i' (line 153)
    i_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'i')
    int_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 12), 'int')
    # Applying the binary operator '==' (line 153)
    result_eq_445 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 7), '==', i_443, int_444)
    
    # Testing if the type of an if condition is none (line 153)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 4), result_eq_445):
        pass
    else:
        
        # Testing the type of an if condition (line 153)
        if_condition_446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 4), result_eq_445)
        # Assigning a type to the variable 'if_condition_446' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'if_condition_446', if_condition_446)
        # SSA begins for if statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 154)
        tuple_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 154)
        # Adding element type (line 154)
        # Getting the type of 'p' (line 154)
        p_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 15), tuple_447, p_448)
        # Adding element type (line 154)
        # Getting the type of 'v' (line 154)
        v_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 18), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 15), tuple_447, v_449)
        # Adding element type (line 154)
        # Getting the type of 't' (line 154)
        t_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 15), tuple_447, t_450)
        
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', tuple_447)
        # SSA join for if statement (line 153)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'i' (line 155)
    i_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 7), 'i')
    int_452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 12), 'int')
    # Applying the binary operator '==' (line 155)
    result_eq_453 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 7), '==', i_451, int_452)
    
    # Testing if the type of an if condition is none (line 155)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 155, 4), result_eq_453):
        pass
    else:
        
        # Testing the type of an if condition (line 155)
        if_condition_454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 4), result_eq_453)
        # Assigning a type to the variable 'if_condition_454' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'if_condition_454', if_condition_454)
        # SSA begins for if statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 156)
        tuple_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 156)
        # Adding element type (line 156)
        # Getting the type of 'p' (line 156)
        p_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 15), tuple_455, p_456)
        # Adding element type (line 156)
        # Getting the type of 'q' (line 156)
        q_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 18), 'q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 15), tuple_455, q_457)
        # Adding element type (line 156)
        # Getting the type of 'v' (line 156)
        v_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 15), tuple_455, v_458)
        
        # Assigning a type to the variable 'stypy_return_type' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', tuple_455)
        # SSA join for if statement (line 155)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'i' (line 157)
    i_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 7), 'i')
    int_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 12), 'int')
    # Applying the binary operator '==' (line 157)
    result_eq_461 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 7), '==', i_459, int_460)
    
    # Testing if the type of an if condition is none (line 157)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 157, 4), result_eq_461):
        pass
    else:
        
        # Testing the type of an if condition (line 157)
        if_condition_462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 4), result_eq_461)
        # Assigning a type to the variable 'if_condition_462' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'if_condition_462', if_condition_462)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 158)
        tuple_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 158)
        # Adding element type (line 158)
        # Getting the type of 't' (line 158)
        t_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 't')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 15), tuple_463, t_464)
        # Adding element type (line 158)
        # Getting the type of 'p' (line 158)
        p_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 15), tuple_463, p_465)
        # Adding element type (line 158)
        # Getting the type of 'v' (line 158)
        v_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 15), tuple_463, v_466)
        
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', tuple_463)
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'i' (line 159)
    i_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'i')
    int_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'int')
    # Applying the binary operator '==' (line 159)
    result_eq_469 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), '==', i_467, int_468)
    
    # Testing if the type of an if condition is none (line 159)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 4), result_eq_469):
        pass
    else:
        
        # Testing the type of an if condition (line 159)
        if_condition_470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 4), result_eq_469)
        # Assigning a type to the variable 'if_condition_470' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'if_condition_470', if_condition_470)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 160)
        tuple_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 160)
        # Adding element type (line 160)
        # Getting the type of 'v' (line 160)
        v_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'v')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 15), tuple_471, v_472)
        # Adding element type (line 160)
        # Getting the type of 'p' (line 160)
        p_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 18), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 15), tuple_471, p_473)
        # Adding element type (line 160)
        # Getting the type of 'q' (line 160)
        q_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 15), tuple_471, q_474)
        
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', tuple_471)
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'hsv_to_rgb(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hsv_to_rgb' in the type store
    # Getting the type of 'stypy_return_type' (line 140)
    stypy_return_type_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_475)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hsv_to_rgb'
    return stypy_return_type_475

# Assigning a type to the variable 'hsv_to_rgb' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'hsv_to_rgb', hsv_to_rgb)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 164, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = []
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    
    # Call to range(...): (line 165)
    # Processing the call arguments (line 165)
    int_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 19), 'int')
    # Processing the call keyword arguments (line 165)
    kwargs_478 = {}
    # Getting the type of 'range' (line 165)
    range_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'range', False)
    # Calling range(args, kwargs) (line 165)
    range_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 165, 13), range_476, *[int_477], **kwargs_478)
    
    # Testing if the for loop is going to be iterated (line 165)
    # Testing the type of a for loop iterable (line 165)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 165, 4), range_call_result_479)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 165, 4), range_call_result_479):
        # Getting the type of the for loop variable (line 165)
        for_loop_var_480 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 165, 4), range_call_result_479)
        # Assigning a type to the variable 'i' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'i', for_loop_var_480)
        # SSA begins for a for statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to hls_to_rgb(...): (line 166)
        # Processing the call arguments (line 166)
        float_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 19), 'float')
        float_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 24), 'float')
        float_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 29), 'float')
        # Processing the call keyword arguments (line 166)
        kwargs_485 = {}
        # Getting the type of 'hls_to_rgb' (line 166)
        hls_to_rgb_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'hls_to_rgb', False)
        # Calling hls_to_rgb(args, kwargs) (line 166)
        hls_to_rgb_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), hls_to_rgb_481, *[float_482, float_483, float_484], **kwargs_485)
        
        
        # Call to rgb_to_hls(...): (line 167)
        # Processing the call arguments (line 167)
        float_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 19), 'float')
        float_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 24), 'float')
        float_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 29), 'float')
        # Processing the call keyword arguments (line 167)
        kwargs_491 = {}
        # Getting the type of 'rgb_to_hls' (line 167)
        rgb_to_hls_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'rgb_to_hls', False)
        # Calling rgb_to_hls(args, kwargs) (line 167)
        rgb_to_hls_call_result_492 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), rgb_to_hls_487, *[float_488, float_489, float_490], **kwargs_491)
        
        
        # Call to yiq_to_rgb(...): (line 168)
        # Processing the call arguments (line 168)
        float_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'float')
        float_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 24), 'float')
        float_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 29), 'float')
        # Processing the call keyword arguments (line 168)
        kwargs_497 = {}
        # Getting the type of 'yiq_to_rgb' (line 168)
        yiq_to_rgb_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'yiq_to_rgb', False)
        # Calling yiq_to_rgb(args, kwargs) (line 168)
        yiq_to_rgb_call_result_498 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), yiq_to_rgb_493, *[float_494, float_495, float_496], **kwargs_497)
        
        
        # Call to rgb_to_yiq(...): (line 169)
        # Processing the call arguments (line 169)
        float_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 19), 'float')
        float_501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'float')
        float_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 29), 'float')
        # Processing the call keyword arguments (line 169)
        kwargs_503 = {}
        # Getting the type of 'rgb_to_yiq' (line 169)
        rgb_to_yiq_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'rgb_to_yiq', False)
        # Calling rgb_to_yiq(args, kwargs) (line 169)
        rgb_to_yiq_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), rgb_to_yiq_499, *[float_500, float_501, float_502], **kwargs_503)
        
        
        # Call to hsv_to_rgb(...): (line 170)
        # Processing the call arguments (line 170)
        float_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'float')
        float_507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 24), 'float')
        float_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 29), 'float')
        # Processing the call keyword arguments (line 170)
        kwargs_509 = {}
        # Getting the type of 'hsv_to_rgb' (line 170)
        hsv_to_rgb_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'hsv_to_rgb', False)
        # Calling hsv_to_rgb(args, kwargs) (line 170)
        hsv_to_rgb_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), hsv_to_rgb_505, *[float_506, float_507, float_508], **kwargs_509)
        
        
        # Call to rgb_to_hsv(...): (line 171)
        # Processing the call arguments (line 171)
        float_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 19), 'float')
        float_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 24), 'float')
        float_514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 29), 'float')
        # Processing the call keyword arguments (line 171)
        kwargs_515 = {}
        # Getting the type of 'rgb_to_hsv' (line 171)
        rgb_to_hsv_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'rgb_to_hsv', False)
        # Calling rgb_to_hsv(args, kwargs) (line 171)
        rgb_to_hsv_call_result_516 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), rgb_to_hsv_511, *[float_512, float_513, float_514], **kwargs_515)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 173)
    True_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type', True_517)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 164)
    stypy_return_type_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_518)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_518

# Assigning a type to the variable 'run' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'run', run)

# Call to run(...): (line 176)
# Processing the call keyword arguments (line 176)
kwargs_520 = {}
# Getting the type of 'run' (line 176)
run_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'run', False)
# Calling run(args, kwargs) (line 176)
run_call_result_521 = invoke(stypy.reporting.localization.Localization(__file__, 176, 0), run_519, *[], **kwargs_520)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

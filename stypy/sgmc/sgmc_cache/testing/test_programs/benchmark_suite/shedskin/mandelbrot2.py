
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # interactive mandelbrot program
2: # copyright Tony Veijalainen, tony.veijalainen@gmail.com
3: 
4: from __future__ import print_function
5: import sys
6: import time
7: import colorsys
8: 
9: 
10: class kohn_bmp:
11:     '''py_kohn_bmp - Copyright 2007 by Michael Kohn
12:        http://www.mikekohn.net/
13:        mike@mikekohn.net'''
14: 
15:     def __init__(self, filename, width, height, depth):
16:         self.width = width
17:         self.height = height
18:         self.depth = depth
19:         self.xpos = 0
20: 
21:         self.width_bytes = width * depth
22:         if (self.width_bytes % 4) != 0:
23:             self.width_bytes = self.width_bytes + (4 - (self.width_bytes % 4))
24: 
25:         self.out = open(filename, "wb")
26:         self.out.write("BM")  # magic number
27: 
28:         self.write_int(self.width_bytes * height + 54 + (1024 if depth == 1 else 0))
29: 
30:         self.write_word(0)
31:         self.write_word(0)
32:         self.write_int(54 + (1024 if depth == 1 else 0))
33:         self.write_int(40)  # header_size
34:         self.write_int(width)  # width
35:         self.write_int(height)  # height
36:         self.write_word(1)  # planes
37:         self.write_word(depth * 8)  # bits per pixel
38:         self.write_int(0)  # compression
39:         self.write_int(self.width_bytes * height * depth)  # image_size
40:         self.write_int(0)  # biXPelsperMetre
41:         self.write_int(0)  # biYPelsperMetre
42: 
43:         if depth == 1:
44:             self.write_int(256)  # colors used
45:             self.write_int(256)  # colors important
46:             self.out.write(''.join(chr(c) * 3 + chr(0) for c in range(256)))
47:         else:
48:             self.write_int(0)  # colors used - 0 since 24 bit
49:             self.write_int(0)  # colors important - 0 since 24 bit
50: 
51:     def write_int(self, n):
52:         self.out.write('%c%c%c%c' % ((n & 255), (n >> 8) & 255, (n >> 16) & 255, (n >> 24) & 255))
53: 
54:     def write_word(self, n):
55:         self.out.write('%c%c' % ((n & 255), (n >> 8) & 255))
56: 
57:     def write_pixel_bw(self, y):
58:         self.out.write(chr(y))
59:         self.xpos = self.xpos + 1
60:         if self.xpos == self.width:
61:             while self.xpos < self.width_bytes:
62:                 self.out.write(chr(0))
63:                 self.xpos = self.xpos + 1
64:             self.xpos = 0
65: 
66:     def write_pixel(self, red, green, blue):
67:         self.out.write(chr((blue & 255)))
68:         self.out.write(chr((green & 255)))
69:         self.out.write(chr((red & 255)))
70:         self.xpos = self.xpos + 1
71:         if self.xpos == self.width:
72:             self.xpos = self.xpos * 3
73:             while self.xpos < self.width_bytes:
74:                 self.out.write(chr(0))
75:                 self.xpos = self.xpos + 1
76:             self.xpos = 0
77: 
78:     def close(self):
79:         self.out.close()
80: 
81: 
82: def mandel(real, imag, max_iterations=20):
83:     '''determines if a point is in the Mandelbrot set based on deciding if,
84:        after a maximum allowed number of iterations, the absolute value of
85:        the resulting number is greater or equal to 2.'''
86:     z_real, z_imag = 0.0, 0.0
87:     for i in range(0, max_iterations):
88:         z_real, z_imag = (z_real * z_real - z_imag * z_imag + real,
89:                           2 * z_real * z_imag + imag)
90:         if (z_real * z_real + z_imag * z_imag) >= 4:
91:             return i % max_iterations
92:     return -1
93: 
94: 
95: def make_colors(number_of_colors, saturation=0.8, value=0.9):
96:     number_of_colors -= 1  # first reserved for black
97:     tuples = [colorsys.hsv_to_rgb(x * 1.0 / number_of_colors, saturation, value) for x in range(number_of_colors)]
98:     return [(0, 0, 0)] + [(int(256 * r), int(256 * g), int(256 * b)) for r, g, b in tuples]
99: 
100: 
101: # colors can be writen over to module by user
102: colors = make_colors(1024)
103: 
104: 
105: # Changing the values below will change the resulting image
106: def mandel_file(cx=-0.7, cy=0.0, size=3.2, max_iterations=512, width=640, height=480):
107:     t0 = time.clock()
108:     increment = min(size / width, size / height)
109:     proportion = 1.0 * width / height
110:     start_real, start_imag = cx - increment * width / 2, cy - increment * height / 2
111: 
112:     mandel_pos = "%g %gi_%g_%i" % (cx, cy, size, max_iterations)
113:     fname = "m%s.bmp" % mandel_pos
114:     my_bmp = kohn_bmp(fname, width, height, 3)
115: 
116:     current_y = start_imag
117:     for y in range(height):
118:         if not y % 10:
119:             pass  # sys.stdout.write('\rrow %i / %i'  % (y + 1, height))
120:         pass  # sys.stdout.flush()
121:         current_x = start_real
122: 
123:         for x in range(width):
124:             c = mandel(current_x, current_y, max_iterations)
125:             c = (c % (len(colors) - 1) + 1) if c != -1 else 0
126:             current_x += increment
127:             my_bmp.write_pixel(colors[c][0], colors[c][1], colors[c][2])
128:         current_y += increment
129: 
130:     ##    print("\r%.3f s             " % (time.clock() - t0))
131:     my_bmp.close()
132:     return fname
133: 
134: 
135: def run():
136:     res = 0
137:     res = mandel(1.0, 1.0, 128)
138:     mandel_file(max_iterations=256)
139:     return True
140: 
141: 
142: run()
143: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import time' statement (line 6)
import time

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import colorsys' statement (line 7)
import colorsys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'colorsys', colorsys, module_type_store)

# Declaration of the 'kohn_bmp' class

class kohn_bmp:
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', 'py_kohn_bmp - Copyright 2007 by Michael Kohn\n       http://www.mikekohn.net/\n       mike@mikekohn.net')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'kohn_bmp.__init__', ['filename', 'width', 'height', 'depth'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['filename', 'width', 'height', 'depth'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 16):
        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of 'width' (line 16)
        width_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 21), 'width')
        # Getting the type of 'self' (line 16)
        self_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'width' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_9, 'width', width_8)
        
        # Assigning a Name to a Attribute (line 17):
        
        # Assigning a Name to a Attribute (line 17):
        # Getting the type of 'height' (line 17)
        height_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'height')
        # Getting the type of 'self' (line 17)
        self_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'height' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_11, 'height', height_10)
        
        # Assigning a Name to a Attribute (line 18):
        
        # Assigning a Name to a Attribute (line 18):
        # Getting the type of 'depth' (line 18)
        depth_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'depth')
        # Getting the type of 'self' (line 18)
        self_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'depth' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_13, 'depth', depth_12)
        
        # Assigning a Num to a Attribute (line 19):
        
        # Assigning a Num to a Attribute (line 19):
        int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'int')
        # Getting the type of 'self' (line 19)
        self_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member 'xpos' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_15, 'xpos', int_14)
        
        # Assigning a BinOp to a Attribute (line 21):
        
        # Assigning a BinOp to a Attribute (line 21):
        # Getting the type of 'width' (line 21)
        width_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'width')
        # Getting the type of 'depth' (line 21)
        depth_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 35), 'depth')
        # Applying the binary operator '*' (line 21)
        result_mul_18 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 27), '*', width_16, depth_17)
        
        # Getting the type of 'self' (line 21)
        self_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member 'width_bytes' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_19, 'width_bytes', result_mul_18)
        
        # Getting the type of 'self' (line 22)
        self_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'self')
        # Obtaining the member 'width_bytes' of a type (line 22)
        width_bytes_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), self_20, 'width_bytes')
        int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'int')
        # Applying the binary operator '%' (line 22)
        result_mod_23 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), '%', width_bytes_21, int_22)
        
        int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 37), 'int')
        # Applying the binary operator '!=' (line 22)
        result_ne_25 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '!=', result_mod_23, int_24)
        
        # Testing if the type of an if condition is none (line 22)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 22, 8), result_ne_25):
            pass
        else:
            
            # Testing the type of an if condition (line 22)
            if_condition_26 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 8), result_ne_25)
            # Assigning a type to the variable 'if_condition_26' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'if_condition_26', if_condition_26)
            # SSA begins for if statement (line 22)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Attribute (line 23):
            
            # Assigning a BinOp to a Attribute (line 23):
            # Getting the type of 'self' (line 23)
            self_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 31), 'self')
            # Obtaining the member 'width_bytes' of a type (line 23)
            width_bytes_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 31), self_27, 'width_bytes')
            int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 51), 'int')
            # Getting the type of 'self' (line 23)
            self_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 56), 'self')
            # Obtaining the member 'width_bytes' of a type (line 23)
            width_bytes_31 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 56), self_30, 'width_bytes')
            int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 75), 'int')
            # Applying the binary operator '%' (line 23)
            result_mod_33 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 56), '%', width_bytes_31, int_32)
            
            # Applying the binary operator '-' (line 23)
            result_sub_34 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 51), '-', int_29, result_mod_33)
            
            # Applying the binary operator '+' (line 23)
            result_add_35 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 31), '+', width_bytes_28, result_sub_34)
            
            # Getting the type of 'self' (line 23)
            self_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'self')
            # Setting the type of the member 'width_bytes' of a type (line 23)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), self_36, 'width_bytes', result_add_35)
            # SSA join for if statement (line 22)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Attribute (line 25):
        
        # Assigning a Call to a Attribute (line 25):
        
        # Call to open(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'filename' (line 25)
        filename_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'filename', False)
        str_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 34), 'str', 'wb')
        # Processing the call keyword arguments (line 25)
        kwargs_40 = {}
        # Getting the type of 'open' (line 25)
        open_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'open', False)
        # Calling open(args, kwargs) (line 25)
        open_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), open_37, *[filename_38, str_39], **kwargs_40)
        
        # Getting the type of 'self' (line 25)
        self_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'out' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_42, 'out', open_call_result_41)
        
        # Call to write(...): (line 26)
        # Processing the call arguments (line 26)
        str_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'str', 'BM')
        # Processing the call keyword arguments (line 26)
        kwargs_47 = {}
        # Getting the type of 'self' (line 26)
        self_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self', False)
        # Obtaining the member 'out' of a type (line 26)
        out_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_43, 'out')
        # Obtaining the member 'write' of a type (line 26)
        write_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), out_44, 'write')
        # Calling write(args, kwargs) (line 26)
        write_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), write_45, *[str_46], **kwargs_47)
        
        
        # Call to write_int(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'self', False)
        # Obtaining the member 'width_bytes' of a type (line 28)
        width_bytes_52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 23), self_51, 'width_bytes')
        # Getting the type of 'height' (line 28)
        height_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'height', False)
        # Applying the binary operator '*' (line 28)
        result_mul_54 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 23), '*', width_bytes_52, height_53)
        
        int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 51), 'int')
        # Applying the binary operator '+' (line 28)
        result_add_56 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 23), '+', result_mul_54, int_55)
        
        
        
        # Getting the type of 'depth' (line 28)
        depth_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 65), 'depth', False)
        int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 74), 'int')
        # Applying the binary operator '==' (line 28)
        result_eq_59 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 65), '==', depth_57, int_58)
        
        # Testing the type of an if expression (line 28)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 57), result_eq_59)
        # SSA begins for if expression (line 28)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 57), 'int')
        # SSA branch for the else part of an if expression (line 28)
        module_type_store.open_ssa_branch('if expression else')
        int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 81), 'int')
        # SSA join for if expression (line 28)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_62 = union_type.UnionType.add(int_60, int_61)
        
        # Applying the binary operator '+' (line 28)
        result_add_63 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 54), '+', result_add_56, if_exp_62)
        
        # Processing the call keyword arguments (line 28)
        kwargs_64 = {}
        # Getting the type of 'self' (line 28)
        self_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 28)
        write_int_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_49, 'write_int')
        # Calling write_int(args, kwargs) (line 28)
        write_int_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), write_int_50, *[result_add_63], **kwargs_64)
        
        
        # Call to write_word(...): (line 30)
        # Processing the call arguments (line 30)
        int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'int')
        # Processing the call keyword arguments (line 30)
        kwargs_69 = {}
        # Getting the type of 'self' (line 30)
        self_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', False)
        # Obtaining the member 'write_word' of a type (line 30)
        write_word_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_66, 'write_word')
        # Calling write_word(args, kwargs) (line 30)
        write_word_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), write_word_67, *[int_68], **kwargs_69)
        
        
        # Call to write_word(...): (line 31)
        # Processing the call arguments (line 31)
        int_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_74 = {}
        # Getting the type of 'self' (line 31)
        self_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self', False)
        # Obtaining the member 'write_word' of a type (line 31)
        write_word_72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_71, 'write_word')
        # Calling write_word(args, kwargs) (line 31)
        write_word_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), write_word_72, *[int_73], **kwargs_74)
        
        
        # Call to write_int(...): (line 32)
        # Processing the call arguments (line 32)
        int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'int')
        
        
        # Getting the type of 'depth' (line 32)
        depth_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 37), 'depth', False)
        int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 46), 'int')
        # Applying the binary operator '==' (line 32)
        result_eq_81 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 37), '==', depth_79, int_80)
        
        # Testing the type of an if expression (line 32)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 29), result_eq_81)
        # SSA begins for if expression (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'int')
        # SSA branch for the else part of an if expression (line 32)
        module_type_store.open_ssa_branch('if expression else')
        int_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 53), 'int')
        # SSA join for if expression (line 32)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_84 = union_type.UnionType.add(int_82, int_83)
        
        # Applying the binary operator '+' (line 32)
        result_add_85 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 23), '+', int_78, if_exp_84)
        
        # Processing the call keyword arguments (line 32)
        kwargs_86 = {}
        # Getting the type of 'self' (line 32)
        self_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 32)
        write_int_77 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_76, 'write_int')
        # Calling write_int(args, kwargs) (line 32)
        write_int_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), write_int_77, *[result_add_85], **kwargs_86)
        
        
        # Call to write_int(...): (line 33)
        # Processing the call arguments (line 33)
        int_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'int')
        # Processing the call keyword arguments (line 33)
        kwargs_91 = {}
        # Getting the type of 'self' (line 33)
        self_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 33)
        write_int_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_88, 'write_int')
        # Calling write_int(args, kwargs) (line 33)
        write_int_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), write_int_89, *[int_90], **kwargs_91)
        
        
        # Call to write_int(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'width' (line 34)
        width_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'width', False)
        # Processing the call keyword arguments (line 34)
        kwargs_96 = {}
        # Getting the type of 'self' (line 34)
        self_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 34)
        write_int_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_93, 'write_int')
        # Calling write_int(args, kwargs) (line 34)
        write_int_call_result_97 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), write_int_94, *[width_95], **kwargs_96)
        
        
        # Call to write_int(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'height' (line 35)
        height_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'height', False)
        # Processing the call keyword arguments (line 35)
        kwargs_101 = {}
        # Getting the type of 'self' (line 35)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 35)
        write_int_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_98, 'write_int')
        # Calling write_int(args, kwargs) (line 35)
        write_int_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), write_int_99, *[height_100], **kwargs_101)
        
        
        # Call to write_word(...): (line 36)
        # Processing the call arguments (line 36)
        int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 24), 'int')
        # Processing the call keyword arguments (line 36)
        kwargs_106 = {}
        # Getting the type of 'self' (line 36)
        self_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', False)
        # Obtaining the member 'write_word' of a type (line 36)
        write_word_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_103, 'write_word')
        # Calling write_word(args, kwargs) (line 36)
        write_word_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), write_word_104, *[int_105], **kwargs_106)
        
        
        # Call to write_word(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'depth' (line 37)
        depth_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'depth', False)
        int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'int')
        # Applying the binary operator '*' (line 37)
        result_mul_112 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 24), '*', depth_110, int_111)
        
        # Processing the call keyword arguments (line 37)
        kwargs_113 = {}
        # Getting the type of 'self' (line 37)
        self_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self', False)
        # Obtaining the member 'write_word' of a type (line 37)
        write_word_109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_108, 'write_word')
        # Calling write_word(args, kwargs) (line 37)
        write_word_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), write_word_109, *[result_mul_112], **kwargs_113)
        
        
        # Call to write_int(...): (line 38)
        # Processing the call arguments (line 38)
        int_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'int')
        # Processing the call keyword arguments (line 38)
        kwargs_118 = {}
        # Getting the type of 'self' (line 38)
        self_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 38)
        write_int_116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_115, 'write_int')
        # Calling write_int(args, kwargs) (line 38)
        write_int_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), write_int_116, *[int_117], **kwargs_118)
        
        
        # Call to write_int(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'self', False)
        # Obtaining the member 'width_bytes' of a type (line 39)
        width_bytes_123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 23), self_122, 'width_bytes')
        # Getting the type of 'height' (line 39)
        height_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 42), 'height', False)
        # Applying the binary operator '*' (line 39)
        result_mul_125 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '*', width_bytes_123, height_124)
        
        # Getting the type of 'depth' (line 39)
        depth_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 51), 'depth', False)
        # Applying the binary operator '*' (line 39)
        result_mul_127 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 49), '*', result_mul_125, depth_126)
        
        # Processing the call keyword arguments (line 39)
        kwargs_128 = {}
        # Getting the type of 'self' (line 39)
        self_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 39)
        write_int_121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_120, 'write_int')
        # Calling write_int(args, kwargs) (line 39)
        write_int_call_result_129 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), write_int_121, *[result_mul_127], **kwargs_128)
        
        
        # Call to write_int(...): (line 40)
        # Processing the call arguments (line 40)
        int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'int')
        # Processing the call keyword arguments (line 40)
        kwargs_133 = {}
        # Getting the type of 'self' (line 40)
        self_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 40)
        write_int_131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_130, 'write_int')
        # Calling write_int(args, kwargs) (line 40)
        write_int_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), write_int_131, *[int_132], **kwargs_133)
        
        
        # Call to write_int(...): (line 41)
        # Processing the call arguments (line 41)
        int_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'int')
        # Processing the call keyword arguments (line 41)
        kwargs_138 = {}
        # Getting the type of 'self' (line 41)
        self_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self', False)
        # Obtaining the member 'write_int' of a type (line 41)
        write_int_136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_135, 'write_int')
        # Calling write_int(args, kwargs) (line 41)
        write_int_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), write_int_136, *[int_137], **kwargs_138)
        
        
        # Getting the type of 'depth' (line 43)
        depth_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'depth')
        int_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'int')
        # Applying the binary operator '==' (line 43)
        result_eq_142 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), '==', depth_140, int_141)
        
        # Testing if the type of an if condition is none (line 43)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 43, 8), result_eq_142):
            
            # Call to write_int(...): (line 48)
            # Processing the call arguments (line 48)
            int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
            # Processing the call keyword arguments (line 48)
            kwargs_183 = {}
            # Getting the type of 'self' (line 48)
            self_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self', False)
            # Obtaining the member 'write_int' of a type (line 48)
            write_int_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_180, 'write_int')
            # Calling write_int(args, kwargs) (line 48)
            write_int_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), write_int_181, *[int_182], **kwargs_183)
            
            
            # Call to write_int(...): (line 49)
            # Processing the call arguments (line 49)
            int_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'int')
            # Processing the call keyword arguments (line 49)
            kwargs_188 = {}
            # Getting the type of 'self' (line 49)
            self_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self', False)
            # Obtaining the member 'write_int' of a type (line 49)
            write_int_186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_185, 'write_int')
            # Calling write_int(args, kwargs) (line 49)
            write_int_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), write_int_186, *[int_187], **kwargs_188)
            
        else:
            
            # Testing the type of an if condition (line 43)
            if_condition_143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), result_eq_142)
            # Assigning a type to the variable 'if_condition_143' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_143', if_condition_143)
            # SSA begins for if statement (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write_int(...): (line 44)
            # Processing the call arguments (line 44)
            int_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'int')
            # Processing the call keyword arguments (line 44)
            kwargs_147 = {}
            # Getting the type of 'self' (line 44)
            self_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'self', False)
            # Obtaining the member 'write_int' of a type (line 44)
            write_int_145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), self_144, 'write_int')
            # Calling write_int(args, kwargs) (line 44)
            write_int_call_result_148 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), write_int_145, *[int_146], **kwargs_147)
            
            
            # Call to write_int(...): (line 45)
            # Processing the call arguments (line 45)
            int_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 27), 'int')
            # Processing the call keyword arguments (line 45)
            kwargs_152 = {}
            # Getting the type of 'self' (line 45)
            self_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'self', False)
            # Obtaining the member 'write_int' of a type (line 45)
            write_int_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), self_149, 'write_int')
            # Calling write_int(args, kwargs) (line 45)
            write_int_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), write_int_150, *[int_151], **kwargs_152)
            
            
            # Call to write(...): (line 46)
            # Processing the call arguments (line 46)
            
            # Call to join(...): (line 46)
            # Processing the call arguments (line 46)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 46, 35, True)
            # Calculating comprehension expression
            
            # Call to range(...): (line 46)
            # Processing the call arguments (line 46)
            int_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 70), 'int')
            # Processing the call keyword arguments (line 46)
            kwargs_172 = {}
            # Getting the type of 'range' (line 46)
            range_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 64), 'range', False)
            # Calling range(args, kwargs) (line 46)
            range_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 46, 64), range_170, *[int_171], **kwargs_172)
            
            comprehension_174 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 35), range_call_result_173)
            # Assigning a type to the variable 'c' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'c', comprehension_174)
            
            # Call to chr(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'c' (line 46)
            c_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), 'c', False)
            # Processing the call keyword arguments (line 46)
            kwargs_161 = {}
            # Getting the type of 'chr' (line 46)
            chr_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'chr', False)
            # Calling chr(args, kwargs) (line 46)
            chr_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 46, 35), chr_159, *[c_160], **kwargs_161)
            
            int_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 44), 'int')
            # Applying the binary operator '*' (line 46)
            result_mul_164 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 35), '*', chr_call_result_162, int_163)
            
            
            # Call to chr(...): (line 46)
            # Processing the call arguments (line 46)
            int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 52), 'int')
            # Processing the call keyword arguments (line 46)
            kwargs_167 = {}
            # Getting the type of 'chr' (line 46)
            chr_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 48), 'chr', False)
            # Calling chr(args, kwargs) (line 46)
            chr_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 46, 48), chr_165, *[int_166], **kwargs_167)
            
            # Applying the binary operator '+' (line 46)
            result_add_169 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 35), '+', result_mul_164, chr_call_result_168)
            
            list_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 35), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 35), list_175, result_add_169)
            # Processing the call keyword arguments (line 46)
            kwargs_176 = {}
            str_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 27), 'str', '')
            # Obtaining the member 'join' of a type (line 46)
            join_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 27), str_157, 'join')
            # Calling join(args, kwargs) (line 46)
            join_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 46, 27), join_158, *[list_175], **kwargs_176)
            
            # Processing the call keyword arguments (line 46)
            kwargs_178 = {}
            # Getting the type of 'self' (line 46)
            self_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'self', False)
            # Obtaining the member 'out' of a type (line 46)
            out_155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), self_154, 'out')
            # Obtaining the member 'write' of a type (line 46)
            write_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), out_155, 'write')
            # Calling write(args, kwargs) (line 46)
            write_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), write_156, *[join_call_result_177], **kwargs_178)
            
            # SSA branch for the else part of an if statement (line 43)
            module_type_store.open_ssa_branch('else')
            
            # Call to write_int(...): (line 48)
            # Processing the call arguments (line 48)
            int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'int')
            # Processing the call keyword arguments (line 48)
            kwargs_183 = {}
            # Getting the type of 'self' (line 48)
            self_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self', False)
            # Obtaining the member 'write_int' of a type (line 48)
            write_int_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_180, 'write_int')
            # Calling write_int(args, kwargs) (line 48)
            write_int_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), write_int_181, *[int_182], **kwargs_183)
            
            
            # Call to write_int(...): (line 49)
            # Processing the call arguments (line 49)
            int_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 27), 'int')
            # Processing the call keyword arguments (line 49)
            kwargs_188 = {}
            # Getting the type of 'self' (line 49)
            self_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'self', False)
            # Obtaining the member 'write_int' of a type (line 49)
            write_int_186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), self_185, 'write_int')
            # Calling write_int(args, kwargs) (line 49)
            write_int_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), write_int_186, *[int_187], **kwargs_188)
            
            # SSA join for if statement (line 43)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def write_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_int'
        module_type_store = module_type_store.open_function_context('write_int', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        kohn_bmp.write_int.__dict__.__setitem__('stypy_localization', localization)
        kohn_bmp.write_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        kohn_bmp.write_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        kohn_bmp.write_int.__dict__.__setitem__('stypy_function_name', 'kohn_bmp.write_int')
        kohn_bmp.write_int.__dict__.__setitem__('stypy_param_names_list', ['n'])
        kohn_bmp.write_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        kohn_bmp.write_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        kohn_bmp.write_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        kohn_bmp.write_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        kohn_bmp.write_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        kohn_bmp.write_int.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'kohn_bmp.write_int', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_int', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_int(...)' code ##################

        
        # Call to write(...): (line 52)
        # Processing the call arguments (line 52)
        str_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'str', '%c%c%c%c')
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        # Getting the type of 'n' (line 52)
        n_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'n', False)
        int_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 42), 'int')
        # Applying the binary operator '&' (line 52)
        result_and__197 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 38), '&', n_195, int_196)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), tuple_194, result_and__197)
        # Adding element type (line 52)
        # Getting the type of 'n' (line 52)
        n_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 49), 'n', False)
        int_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 54), 'int')
        # Applying the binary operator '>>' (line 52)
        result_rshift_200 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 49), '>>', n_198, int_199)
        
        int_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 59), 'int')
        # Applying the binary operator '&' (line 52)
        result_and__202 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 48), '&', result_rshift_200, int_201)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), tuple_194, result_and__202)
        # Adding element type (line 52)
        # Getting the type of 'n' (line 52)
        n_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 65), 'n', False)
        int_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 70), 'int')
        # Applying the binary operator '>>' (line 52)
        result_rshift_205 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 65), '>>', n_203, int_204)
        
        int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 76), 'int')
        # Applying the binary operator '&' (line 52)
        result_and__207 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 64), '&', result_rshift_205, int_206)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), tuple_194, result_and__207)
        # Adding element type (line 52)
        # Getting the type of 'n' (line 52)
        n_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 82), 'n', False)
        int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 87), 'int')
        # Applying the binary operator '>>' (line 52)
        result_rshift_210 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 82), '>>', n_208, int_209)
        
        int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 93), 'int')
        # Applying the binary operator '&' (line 52)
        result_and__212 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 81), '&', result_rshift_210, int_211)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 37), tuple_194, result_and__212)
        
        # Applying the binary operator '%' (line 52)
        result_mod_213 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 23), '%', str_193, tuple_194)
        
        # Processing the call keyword arguments (line 52)
        kwargs_214 = {}
        # Getting the type of 'self' (line 52)
        self_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'out' of a type (line 52)
        out_191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_190, 'out')
        # Obtaining the member 'write' of a type (line 52)
        write_192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), out_191, 'write')
        # Calling write(args, kwargs) (line 52)
        write_call_result_215 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), write_192, *[result_mod_213], **kwargs_214)
        
        
        # ################# End of 'write_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_int' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_216)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_int'
        return stypy_return_type_216


    @norecursion
    def write_word(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_word'
        module_type_store = module_type_store.open_function_context('write_word', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        kohn_bmp.write_word.__dict__.__setitem__('stypy_localization', localization)
        kohn_bmp.write_word.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        kohn_bmp.write_word.__dict__.__setitem__('stypy_type_store', module_type_store)
        kohn_bmp.write_word.__dict__.__setitem__('stypy_function_name', 'kohn_bmp.write_word')
        kohn_bmp.write_word.__dict__.__setitem__('stypy_param_names_list', ['n'])
        kohn_bmp.write_word.__dict__.__setitem__('stypy_varargs_param_name', None)
        kohn_bmp.write_word.__dict__.__setitem__('stypy_kwargs_param_name', None)
        kohn_bmp.write_word.__dict__.__setitem__('stypy_call_defaults', defaults)
        kohn_bmp.write_word.__dict__.__setitem__('stypy_call_varargs', varargs)
        kohn_bmp.write_word.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        kohn_bmp.write_word.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'kohn_bmp.write_word', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_word', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_word(...)' code ##################

        
        # Call to write(...): (line 55)
        # Processing the call arguments (line 55)
        str_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 23), 'str', '%c%c')
        
        # Obtaining an instance of the builtin type 'tuple' (line 55)
        tuple_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 55)
        # Adding element type (line 55)
        # Getting the type of 'n' (line 55)
        n_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 34), 'n', False)
        int_223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'int')
        # Applying the binary operator '&' (line 55)
        result_and__224 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 34), '&', n_222, int_223)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 33), tuple_221, result_and__224)
        # Adding element type (line 55)
        # Getting the type of 'n' (line 55)
        n_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 45), 'n', False)
        int_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 50), 'int')
        # Applying the binary operator '>>' (line 55)
        result_rshift_227 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 45), '>>', n_225, int_226)
        
        int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 55), 'int')
        # Applying the binary operator '&' (line 55)
        result_and__229 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 44), '&', result_rshift_227, int_228)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 33), tuple_221, result_and__229)
        
        # Applying the binary operator '%' (line 55)
        result_mod_230 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 23), '%', str_220, tuple_221)
        
        # Processing the call keyword arguments (line 55)
        kwargs_231 = {}
        # Getting the type of 'self' (line 55)
        self_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self', False)
        # Obtaining the member 'out' of a type (line 55)
        out_218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_217, 'out')
        # Obtaining the member 'write' of a type (line 55)
        write_219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), out_218, 'write')
        # Calling write(args, kwargs) (line 55)
        write_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), write_219, *[result_mod_230], **kwargs_231)
        
        
        # ################# End of 'write_word(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_word' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_233)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_word'
        return stypy_return_type_233


    @norecursion
    def write_pixel_bw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_pixel_bw'
        module_type_store = module_type_store.open_function_context('write_pixel_bw', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_localization', localization)
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_type_store', module_type_store)
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_function_name', 'kohn_bmp.write_pixel_bw')
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_param_names_list', ['y'])
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_varargs_param_name', None)
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_call_defaults', defaults)
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_call_varargs', varargs)
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        kohn_bmp.write_pixel_bw.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'kohn_bmp.write_pixel_bw', ['y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_pixel_bw', localization, ['y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_pixel_bw(...)' code ##################

        
        # Call to write(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to chr(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'y' (line 58)
        y_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'y', False)
        # Processing the call keyword arguments (line 58)
        kwargs_239 = {}
        # Getting the type of 'chr' (line 58)
        chr_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'chr', False)
        # Calling chr(args, kwargs) (line 58)
        chr_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), chr_237, *[y_238], **kwargs_239)
        
        # Processing the call keyword arguments (line 58)
        kwargs_241 = {}
        # Getting the type of 'self' (line 58)
        self_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', False)
        # Obtaining the member 'out' of a type (line 58)
        out_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_234, 'out')
        # Obtaining the member 'write' of a type (line 58)
        write_236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), out_235, 'write')
        # Calling write(args, kwargs) (line 58)
        write_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), write_236, *[chr_call_result_240], **kwargs_241)
        
        
        # Assigning a BinOp to a Attribute (line 59):
        
        # Assigning a BinOp to a Attribute (line 59):
        # Getting the type of 'self' (line 59)
        self_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'self')
        # Obtaining the member 'xpos' of a type (line 59)
        xpos_244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), self_243, 'xpos')
        int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 32), 'int')
        # Applying the binary operator '+' (line 59)
        result_add_246 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 20), '+', xpos_244, int_245)
        
        # Getting the type of 'self' (line 59)
        self_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'xpos' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_247, 'xpos', result_add_246)
        
        # Getting the type of 'self' (line 60)
        self_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'self')
        # Obtaining the member 'xpos' of a type (line 60)
        xpos_249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), self_248, 'xpos')
        # Getting the type of 'self' (line 60)
        self_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'self')
        # Obtaining the member 'width' of a type (line 60)
        width_251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 24), self_250, 'width')
        # Applying the binary operator '==' (line 60)
        result_eq_252 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 11), '==', xpos_249, width_251)
        
        # Testing if the type of an if condition is none (line 60)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 8), result_eq_252):
            pass
        else:
            
            # Testing the type of an if condition (line 60)
            if_condition_253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), result_eq_252)
            # Assigning a type to the variable 'if_condition_253' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_253', if_condition_253)
            # SSA begins for if statement (line 60)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Getting the type of 'self' (line 61)
            self_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'self')
            # Obtaining the member 'xpos' of a type (line 61)
            xpos_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 18), self_254, 'xpos')
            # Getting the type of 'self' (line 61)
            self_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'self')
            # Obtaining the member 'width_bytes' of a type (line 61)
            width_bytes_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 30), self_256, 'width_bytes')
            # Applying the binary operator '<' (line 61)
            result_lt_258 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 18), '<', xpos_255, width_bytes_257)
            
            # Assigning a type to the variable 'result_lt_258' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'result_lt_258', result_lt_258)
            # Testing if the while is going to be iterated (line 61)
            # Testing the type of an if condition (line 61)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 12), result_lt_258)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 12), result_lt_258):
                # SSA begins for while statement (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Call to write(...): (line 62)
                # Processing the call arguments (line 62)
                
                # Call to chr(...): (line 62)
                # Processing the call arguments (line 62)
                int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 35), 'int')
                # Processing the call keyword arguments (line 62)
                kwargs_264 = {}
                # Getting the type of 'chr' (line 62)
                chr_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'chr', False)
                # Calling chr(args, kwargs) (line 62)
                chr_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 62, 31), chr_262, *[int_263], **kwargs_264)
                
                # Processing the call keyword arguments (line 62)
                kwargs_266 = {}
                # Getting the type of 'self' (line 62)
                self_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'self', False)
                # Obtaining the member 'out' of a type (line 62)
                out_260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), self_259, 'out')
                # Obtaining the member 'write' of a type (line 62)
                write_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), out_260, 'write')
                # Calling write(args, kwargs) (line 62)
                write_call_result_267 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), write_261, *[chr_call_result_265], **kwargs_266)
                
                
                # Assigning a BinOp to a Attribute (line 63):
                
                # Assigning a BinOp to a Attribute (line 63):
                # Getting the type of 'self' (line 63)
                self_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'self')
                # Obtaining the member 'xpos' of a type (line 63)
                xpos_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 28), self_268, 'xpos')
                int_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 40), 'int')
                # Applying the binary operator '+' (line 63)
                result_add_271 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 28), '+', xpos_269, int_270)
                
                # Getting the type of 'self' (line 63)
                self_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'self')
                # Setting the type of the member 'xpos' of a type (line 63)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), self_272, 'xpos', result_add_271)
                # SSA join for while statement (line 61)
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Num to a Attribute (line 64):
            
            # Assigning a Num to a Attribute (line 64):
            int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 24), 'int')
            # Getting the type of 'self' (line 64)
            self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self')
            # Setting the type of the member 'xpos' of a type (line 64)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_274, 'xpos', int_273)
            # SSA join for if statement (line 60)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'write_pixel_bw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_pixel_bw' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_275)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_pixel_bw'
        return stypy_return_type_275


    @norecursion
    def write_pixel(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write_pixel'
        module_type_store = module_type_store.open_function_context('write_pixel', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_localization', localization)
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_type_store', module_type_store)
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_function_name', 'kohn_bmp.write_pixel')
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_param_names_list', ['red', 'green', 'blue'])
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_varargs_param_name', None)
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_kwargs_param_name', None)
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_call_defaults', defaults)
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_call_varargs', varargs)
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        kohn_bmp.write_pixel.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'kohn_bmp.write_pixel', ['red', 'green', 'blue'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_pixel', localization, ['red', 'green', 'blue'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_pixel(...)' code ##################

        
        # Call to write(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to chr(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'blue' (line 67)
        blue_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'blue', False)
        int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 35), 'int')
        # Applying the binary operator '&' (line 67)
        result_and__282 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 28), '&', blue_280, int_281)
        
        # Processing the call keyword arguments (line 67)
        kwargs_283 = {}
        # Getting the type of 'chr' (line 67)
        chr_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'chr', False)
        # Calling chr(args, kwargs) (line 67)
        chr_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 67, 23), chr_279, *[result_and__282], **kwargs_283)
        
        # Processing the call keyword arguments (line 67)
        kwargs_285 = {}
        # Getting the type of 'self' (line 67)
        self_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'out' of a type (line 67)
        out_277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_276, 'out')
        # Obtaining the member 'write' of a type (line 67)
        write_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), out_277, 'write')
        # Calling write(args, kwargs) (line 67)
        write_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), write_278, *[chr_call_result_284], **kwargs_285)
        
        
        # Call to write(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to chr(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'green' (line 68)
        green_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'green', False)
        int_292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 36), 'int')
        # Applying the binary operator '&' (line 68)
        result_and__293 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 28), '&', green_291, int_292)
        
        # Processing the call keyword arguments (line 68)
        kwargs_294 = {}
        # Getting the type of 'chr' (line 68)
        chr_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'chr', False)
        # Calling chr(args, kwargs) (line 68)
        chr_call_result_295 = invoke(stypy.reporting.localization.Localization(__file__, 68, 23), chr_290, *[result_and__293], **kwargs_294)
        
        # Processing the call keyword arguments (line 68)
        kwargs_296 = {}
        # Getting the type of 'self' (line 68)
        self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self', False)
        # Obtaining the member 'out' of a type (line 68)
        out_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_287, 'out')
        # Obtaining the member 'write' of a type (line 68)
        write_289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), out_288, 'write')
        # Calling write(args, kwargs) (line 68)
        write_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), write_289, *[chr_call_result_295], **kwargs_296)
        
        
        # Call to write(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to chr(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'red' (line 69)
        red_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'red', False)
        int_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'int')
        # Applying the binary operator '&' (line 69)
        result_and__304 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 28), '&', red_302, int_303)
        
        # Processing the call keyword arguments (line 69)
        kwargs_305 = {}
        # Getting the type of 'chr' (line 69)
        chr_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'chr', False)
        # Calling chr(args, kwargs) (line 69)
        chr_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), chr_301, *[result_and__304], **kwargs_305)
        
        # Processing the call keyword arguments (line 69)
        kwargs_307 = {}
        # Getting the type of 'self' (line 69)
        self_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self', False)
        # Obtaining the member 'out' of a type (line 69)
        out_299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_298, 'out')
        # Obtaining the member 'write' of a type (line 69)
        write_300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), out_299, 'write')
        # Calling write(args, kwargs) (line 69)
        write_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), write_300, *[chr_call_result_306], **kwargs_307)
        
        
        # Assigning a BinOp to a Attribute (line 70):
        
        # Assigning a BinOp to a Attribute (line 70):
        # Getting the type of 'self' (line 70)
        self_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'self')
        # Obtaining the member 'xpos' of a type (line 70)
        xpos_310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 20), self_309, 'xpos')
        int_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 32), 'int')
        # Applying the binary operator '+' (line 70)
        result_add_312 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 20), '+', xpos_310, int_311)
        
        # Getting the type of 'self' (line 70)
        self_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'xpos' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_313, 'xpos', result_add_312)
        
        # Getting the type of 'self' (line 71)
        self_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'self')
        # Obtaining the member 'xpos' of a type (line 71)
        xpos_315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), self_314, 'xpos')
        # Getting the type of 'self' (line 71)
        self_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'self')
        # Obtaining the member 'width' of a type (line 71)
        width_317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 24), self_316, 'width')
        # Applying the binary operator '==' (line 71)
        result_eq_318 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 11), '==', xpos_315, width_317)
        
        # Testing if the type of an if condition is none (line 71)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 8), result_eq_318):
            pass
        else:
            
            # Testing the type of an if condition (line 71)
            if_condition_319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 8), result_eq_318)
            # Assigning a type to the variable 'if_condition_319' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'if_condition_319', if_condition_319)
            # SSA begins for if statement (line 71)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Attribute (line 72):
            
            # Assigning a BinOp to a Attribute (line 72):
            # Getting the type of 'self' (line 72)
            self_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'self')
            # Obtaining the member 'xpos' of a type (line 72)
            xpos_321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), self_320, 'xpos')
            int_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 36), 'int')
            # Applying the binary operator '*' (line 72)
            result_mul_323 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 24), '*', xpos_321, int_322)
            
            # Getting the type of 'self' (line 72)
            self_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'self')
            # Setting the type of the member 'xpos' of a type (line 72)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), self_324, 'xpos', result_mul_323)
            
            
            # Getting the type of 'self' (line 73)
            self_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'self')
            # Obtaining the member 'xpos' of a type (line 73)
            xpos_326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 18), self_325, 'xpos')
            # Getting the type of 'self' (line 73)
            self_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'self')
            # Obtaining the member 'width_bytes' of a type (line 73)
            width_bytes_328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 30), self_327, 'width_bytes')
            # Applying the binary operator '<' (line 73)
            result_lt_329 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 18), '<', xpos_326, width_bytes_328)
            
            # Assigning a type to the variable 'result_lt_329' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'result_lt_329', result_lt_329)
            # Testing if the while is going to be iterated (line 73)
            # Testing the type of an if condition (line 73)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 12), result_lt_329)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 73, 12), result_lt_329):
                # SSA begins for while statement (line 73)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Call to write(...): (line 74)
                # Processing the call arguments (line 74)
                
                # Call to chr(...): (line 74)
                # Processing the call arguments (line 74)
                int_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 35), 'int')
                # Processing the call keyword arguments (line 74)
                kwargs_335 = {}
                # Getting the type of 'chr' (line 74)
                chr_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 31), 'chr', False)
                # Calling chr(args, kwargs) (line 74)
                chr_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 74, 31), chr_333, *[int_334], **kwargs_335)
                
                # Processing the call keyword arguments (line 74)
                kwargs_337 = {}
                # Getting the type of 'self' (line 74)
                self_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'self', False)
                # Obtaining the member 'out' of a type (line 74)
                out_331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), self_330, 'out')
                # Obtaining the member 'write' of a type (line 74)
                write_332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), out_331, 'write')
                # Calling write(args, kwargs) (line 74)
                write_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), write_332, *[chr_call_result_336], **kwargs_337)
                
                
                # Assigning a BinOp to a Attribute (line 75):
                
                # Assigning a BinOp to a Attribute (line 75):
                # Getting the type of 'self' (line 75)
                self_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'self')
                # Obtaining the member 'xpos' of a type (line 75)
                xpos_340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 28), self_339, 'xpos')
                int_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 40), 'int')
                # Applying the binary operator '+' (line 75)
                result_add_342 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 28), '+', xpos_340, int_341)
                
                # Getting the type of 'self' (line 75)
                self_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'self')
                # Setting the type of the member 'xpos' of a type (line 75)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), self_343, 'xpos', result_add_342)
                # SSA join for while statement (line 73)
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Num to a Attribute (line 76):
            
            # Assigning a Num to a Attribute (line 76):
            int_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'int')
            # Getting the type of 'self' (line 76)
            self_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'self')
            # Setting the type of the member 'xpos' of a type (line 76)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), self_345, 'xpos', int_344)
            # SSA join for if statement (line 71)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'write_pixel(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_pixel' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_346)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_pixel'
        return stypy_return_type_346


    @norecursion
    def close(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'close'
        module_type_store = module_type_store.open_function_context('close', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        kohn_bmp.close.__dict__.__setitem__('stypy_localization', localization)
        kohn_bmp.close.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        kohn_bmp.close.__dict__.__setitem__('stypy_type_store', module_type_store)
        kohn_bmp.close.__dict__.__setitem__('stypy_function_name', 'kohn_bmp.close')
        kohn_bmp.close.__dict__.__setitem__('stypy_param_names_list', [])
        kohn_bmp.close.__dict__.__setitem__('stypy_varargs_param_name', None)
        kohn_bmp.close.__dict__.__setitem__('stypy_kwargs_param_name', None)
        kohn_bmp.close.__dict__.__setitem__('stypy_call_defaults', defaults)
        kohn_bmp.close.__dict__.__setitem__('stypy_call_varargs', varargs)
        kohn_bmp.close.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        kohn_bmp.close.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'kohn_bmp.close', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'close', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'close(...)' code ##################

        
        # Call to close(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_350 = {}
        # Getting the type of 'self' (line 79)
        self_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', False)
        # Obtaining the member 'out' of a type (line 79)
        out_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_347, 'out')
        # Obtaining the member 'close' of a type (line 79)
        close_349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), out_348, 'close')
        # Calling close(args, kwargs) (line 79)
        close_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), close_349, *[], **kwargs_350)
        
        
        # ################# End of 'close(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'close' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'close'
        return stypy_return_type_352


# Assigning a type to the variable 'kohn_bmp' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'kohn_bmp', kohn_bmp)

@norecursion
def mandel(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 38), 'int')
    defaults = [int_353]
    # Create a new context for function 'mandel'
    module_type_store = module_type_store.open_function_context('mandel', 82, 0, False)
    
    # Passed parameters checking function
    mandel.stypy_localization = localization
    mandel.stypy_type_of_self = None
    mandel.stypy_type_store = module_type_store
    mandel.stypy_function_name = 'mandel'
    mandel.stypy_param_names_list = ['real', 'imag', 'max_iterations']
    mandel.stypy_varargs_param_name = None
    mandel.stypy_kwargs_param_name = None
    mandel.stypy_call_defaults = defaults
    mandel.stypy_call_varargs = varargs
    mandel.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mandel', ['real', 'imag', 'max_iterations'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mandel', localization, ['real', 'imag', 'max_iterations'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mandel(...)' code ##################

    str_354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', 'determines if a point is in the Mandelbrot set based on deciding if,\n       after a maximum allowed number of iterations, the absolute value of\n       the resulting number is greater or equal to 2.')
    
    # Assigning a Tuple to a Tuple (line 86):
    
    # Assigning a Num to a Name (line 86):
    float_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'float')
    # Assigning a type to the variable 'tuple_assignment_1' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'tuple_assignment_1', float_355)
    
    # Assigning a Num to a Name (line 86):
    float_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'float')
    # Assigning a type to the variable 'tuple_assignment_2' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'tuple_assignment_2', float_356)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_assignment_1' (line 86)
    tuple_assignment_1_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'tuple_assignment_1')
    # Assigning a type to the variable 'z_real' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'z_real', tuple_assignment_1_357)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_assignment_2' (line 86)
    tuple_assignment_2_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'tuple_assignment_2')
    # Assigning a type to the variable 'z_imag' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'z_imag', tuple_assignment_2_358)
    
    
    # Call to range(...): (line 87)
    # Processing the call arguments (line 87)
    int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'int')
    # Getting the type of 'max_iterations' (line 87)
    max_iterations_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'max_iterations', False)
    # Processing the call keyword arguments (line 87)
    kwargs_362 = {}
    # Getting the type of 'range' (line 87)
    range_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'range', False)
    # Calling range(args, kwargs) (line 87)
    range_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 87, 13), range_359, *[int_360, max_iterations_361], **kwargs_362)
    
    # Assigning a type to the variable 'range_call_result_363' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'range_call_result_363', range_call_result_363)
    # Testing if the for loop is going to be iterated (line 87)
    # Testing the type of a for loop iterable (line 87)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 4), range_call_result_363)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 4), range_call_result_363):
        # Getting the type of the for loop variable (line 87)
        for_loop_var_364 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 4), range_call_result_363)
        # Assigning a type to the variable 'i' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'i', for_loop_var_364)
        # SSA begins for a for statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Tuple (line 88):
        
        # Assigning a BinOp to a Name (line 88):
        # Getting the type of 'z_real' (line 88)
        z_real_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'z_real')
        # Getting the type of 'z_real' (line 88)
        z_real_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 35), 'z_real')
        # Applying the binary operator '*' (line 88)
        result_mul_367 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 26), '*', z_real_365, z_real_366)
        
        # Getting the type of 'z_imag' (line 88)
        z_imag_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 44), 'z_imag')
        # Getting the type of 'z_imag' (line 88)
        z_imag_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 53), 'z_imag')
        # Applying the binary operator '*' (line 88)
        result_mul_370 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 44), '*', z_imag_368, z_imag_369)
        
        # Applying the binary operator '-' (line 88)
        result_sub_371 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 26), '-', result_mul_367, result_mul_370)
        
        # Getting the type of 'real' (line 88)
        real_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 62), 'real')
        # Applying the binary operator '+' (line 88)
        result_add_373 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 60), '+', result_sub_371, real_372)
        
        # Assigning a type to the variable 'tuple_assignment_3' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'tuple_assignment_3', result_add_373)
        
        # Assigning a BinOp to a Name (line 88):
        int_374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 26), 'int')
        # Getting the type of 'z_real' (line 89)
        z_real_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'z_real')
        # Applying the binary operator '*' (line 89)
        result_mul_376 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 26), '*', int_374, z_real_375)
        
        # Getting the type of 'z_imag' (line 89)
        z_imag_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'z_imag')
        # Applying the binary operator '*' (line 89)
        result_mul_378 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 37), '*', result_mul_376, z_imag_377)
        
        # Getting the type of 'imag' (line 89)
        imag_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 48), 'imag')
        # Applying the binary operator '+' (line 89)
        result_add_380 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 26), '+', result_mul_378, imag_379)
        
        # Assigning a type to the variable 'tuple_assignment_4' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'tuple_assignment_4', result_add_380)
        
        # Assigning a Name to a Name (line 88):
        # Getting the type of 'tuple_assignment_3' (line 88)
        tuple_assignment_3_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'tuple_assignment_3')
        # Assigning a type to the variable 'z_real' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'z_real', tuple_assignment_3_381)
        
        # Assigning a Name to a Name (line 88):
        # Getting the type of 'tuple_assignment_4' (line 88)
        tuple_assignment_4_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'tuple_assignment_4')
        # Assigning a type to the variable 'z_imag' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'z_imag', tuple_assignment_4_382)
        
        # Getting the type of 'z_real' (line 90)
        z_real_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'z_real')
        # Getting the type of 'z_real' (line 90)
        z_real_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'z_real')
        # Applying the binary operator '*' (line 90)
        result_mul_385 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '*', z_real_383, z_real_384)
        
        # Getting the type of 'z_imag' (line 90)
        z_imag_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'z_imag')
        # Getting the type of 'z_imag' (line 90)
        z_imag_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 39), 'z_imag')
        # Applying the binary operator '*' (line 90)
        result_mul_388 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 30), '*', z_imag_386, z_imag_387)
        
        # Applying the binary operator '+' (line 90)
        result_add_389 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '+', result_mul_385, result_mul_388)
        
        int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 50), 'int')
        # Applying the binary operator '>=' (line 90)
        result_ge_391 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 11), '>=', result_add_389, int_390)
        
        # Testing if the type of an if condition is none (line 90)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 8), result_ge_391):
            pass
        else:
            
            # Testing the type of an if condition (line 90)
            if_condition_392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 8), result_ge_391)
            # Assigning a type to the variable 'if_condition_392' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'if_condition_392', if_condition_392)
            # SSA begins for if statement (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'i' (line 91)
            i_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'i')
            # Getting the type of 'max_iterations' (line 91)
            max_iterations_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'max_iterations')
            # Applying the binary operator '%' (line 91)
            result_mod_395 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 19), '%', i_393, max_iterations_394)
            
            # Assigning a type to the variable 'stypy_return_type' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'stypy_return_type', result_mod_395)
            # SSA join for if statement (line 90)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    int_396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', int_396)
    
    # ################# End of 'mandel(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mandel' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_397)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mandel'
    return stypy_return_type_397

# Assigning a type to the variable 'mandel' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'mandel', mandel)

@norecursion
def make_colors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 45), 'float')
    float_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 56), 'float')
    defaults = [float_398, float_399]
    # Create a new context for function 'make_colors'
    module_type_store = module_type_store.open_function_context('make_colors', 95, 0, False)
    
    # Passed parameters checking function
    make_colors.stypy_localization = localization
    make_colors.stypy_type_of_self = None
    make_colors.stypy_type_store = module_type_store
    make_colors.stypy_function_name = 'make_colors'
    make_colors.stypy_param_names_list = ['number_of_colors', 'saturation', 'value']
    make_colors.stypy_varargs_param_name = None
    make_colors.stypy_kwargs_param_name = None
    make_colors.stypy_call_defaults = defaults
    make_colors.stypy_call_varargs = varargs
    make_colors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_colors', ['number_of_colors', 'saturation', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_colors', localization, ['number_of_colors', 'saturation', 'value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_colors(...)' code ##################

    
    # Getting the type of 'number_of_colors' (line 96)
    number_of_colors_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'number_of_colors')
    int_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 24), 'int')
    # Applying the binary operator '-=' (line 96)
    result_isub_402 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 4), '-=', number_of_colors_400, int_401)
    # Assigning a type to the variable 'number_of_colors' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'number_of_colors', result_isub_402)
    
    
    # Assigning a ListComp to a Name (line 97):
    
    # Assigning a ListComp to a Name (line 97):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'number_of_colors' (line 97)
    number_of_colors_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 96), 'number_of_colors', False)
    # Processing the call keyword arguments (line 97)
    kwargs_416 = {}
    # Getting the type of 'range' (line 97)
    range_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 90), 'range', False)
    # Calling range(args, kwargs) (line 97)
    range_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 97, 90), range_414, *[number_of_colors_415], **kwargs_416)
    
    comprehension_418 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 14), range_call_result_417)
    # Assigning a type to the variable 'x' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'x', comprehension_418)
    
    # Call to hsv_to_rgb(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'x' (line 97)
    x_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 34), 'x', False)
    float_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 38), 'float')
    # Applying the binary operator '*' (line 97)
    result_mul_407 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 34), '*', x_405, float_406)
    
    # Getting the type of 'number_of_colors' (line 97)
    number_of_colors_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 44), 'number_of_colors', False)
    # Applying the binary operator 'div' (line 97)
    result_div_409 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 42), 'div', result_mul_407, number_of_colors_408)
    
    # Getting the type of 'saturation' (line 97)
    saturation_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 62), 'saturation', False)
    # Getting the type of 'value' (line 97)
    value_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 74), 'value', False)
    # Processing the call keyword arguments (line 97)
    kwargs_412 = {}
    # Getting the type of 'colorsys' (line 97)
    colorsys_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'colorsys', False)
    # Obtaining the member 'hsv_to_rgb' of a type (line 97)
    hsv_to_rgb_404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 14), colorsys_403, 'hsv_to_rgb')
    # Calling hsv_to_rgb(args, kwargs) (line 97)
    hsv_to_rgb_call_result_413 = invoke(stypy.reporting.localization.Localization(__file__, 97, 14), hsv_to_rgb_404, *[result_div_409, saturation_410, value_411], **kwargs_412)
    
    list_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 14), list_419, hsv_to_rgb_call_result_413)
    # Assigning a type to the variable 'tuples' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'tuples', list_419)
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    
    # Obtaining an instance of the builtin type 'tuple' (line 98)
    tuple_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 98)
    # Adding element type (line 98)
    int_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 13), tuple_421, int_422)
    # Adding element type (line 98)
    int_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 13), tuple_421, int_423)
    # Adding element type (line 98)
    int_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 13), tuple_421, int_424)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 11), list_420, tuple_421)
    
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'tuples' (line 98)
    tuples_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 84), 'tuples')
    comprehension_445 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 26), tuples_444)
    # Assigning a type to the variable 'r' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 26), comprehension_445))
    # Assigning a type to the variable 'g' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'g', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 26), comprehension_445))
    # Assigning a type to the variable 'b' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'b', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 26), comprehension_445))
    
    # Obtaining an instance of the builtin type 'tuple' (line 98)
    tuple_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 98)
    # Adding element type (line 98)
    
    # Call to int(...): (line 98)
    # Processing the call arguments (line 98)
    int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'int')
    # Getting the type of 'r' (line 98)
    r_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'r', False)
    # Applying the binary operator '*' (line 98)
    result_mul_429 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 31), '*', int_427, r_428)
    
    # Processing the call keyword arguments (line 98)
    kwargs_430 = {}
    # Getting the type of 'int' (line 98)
    int_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'int', False)
    # Calling int(args, kwargs) (line 98)
    int_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 98, 27), int_426, *[result_mul_429], **kwargs_430)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 27), tuple_425, int_call_result_431)
    # Adding element type (line 98)
    
    # Call to int(...): (line 98)
    # Processing the call arguments (line 98)
    int_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 45), 'int')
    # Getting the type of 'g' (line 98)
    g_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 51), 'g', False)
    # Applying the binary operator '*' (line 98)
    result_mul_435 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 45), '*', int_433, g_434)
    
    # Processing the call keyword arguments (line 98)
    kwargs_436 = {}
    # Getting the type of 'int' (line 98)
    int_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 41), 'int', False)
    # Calling int(args, kwargs) (line 98)
    int_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 98, 41), int_432, *[result_mul_435], **kwargs_436)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 27), tuple_425, int_call_result_437)
    # Adding element type (line 98)
    
    # Call to int(...): (line 98)
    # Processing the call arguments (line 98)
    int_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 59), 'int')
    # Getting the type of 'b' (line 98)
    b_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 65), 'b', False)
    # Applying the binary operator '*' (line 98)
    result_mul_441 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 59), '*', int_439, b_440)
    
    # Processing the call keyword arguments (line 98)
    kwargs_442 = {}
    # Getting the type of 'int' (line 98)
    int_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 55), 'int', False)
    # Calling int(args, kwargs) (line 98)
    int_call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 98, 55), int_438, *[result_mul_441], **kwargs_442)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 27), tuple_425, int_call_result_443)
    
    list_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 26), list_446, tuple_425)
    # Applying the binary operator '+' (line 98)
    result_add_447 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), '+', list_420, list_446)
    
    # Assigning a type to the variable 'stypy_return_type' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type', result_add_447)
    
    # ################# End of 'make_colors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_colors' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_448)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_colors'
    return stypy_return_type_448

# Assigning a type to the variable 'make_colors' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'make_colors', make_colors)

# Assigning a Call to a Name (line 102):

# Assigning a Call to a Name (line 102):

# Call to make_colors(...): (line 102)
# Processing the call arguments (line 102)
int_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'int')
# Processing the call keyword arguments (line 102)
kwargs_451 = {}
# Getting the type of 'make_colors' (line 102)
make_colors_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 9), 'make_colors', False)
# Calling make_colors(args, kwargs) (line 102)
make_colors_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 102, 9), make_colors_449, *[int_450], **kwargs_451)

# Assigning a type to the variable 'colors' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'colors', make_colors_call_result_452)

@norecursion
def mandel_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'float')
    float_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'float')
    float_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 38), 'float')
    int_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 58), 'int')
    int_457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 69), 'int')
    int_458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 81), 'int')
    defaults = [float_453, float_454, float_455, int_456, int_457, int_458]
    # Create a new context for function 'mandel_file'
    module_type_store = module_type_store.open_function_context('mandel_file', 106, 0, False)
    
    # Passed parameters checking function
    mandel_file.stypy_localization = localization
    mandel_file.stypy_type_of_self = None
    mandel_file.stypy_type_store = module_type_store
    mandel_file.stypy_function_name = 'mandel_file'
    mandel_file.stypy_param_names_list = ['cx', 'cy', 'size', 'max_iterations', 'width', 'height']
    mandel_file.stypy_varargs_param_name = None
    mandel_file.stypy_kwargs_param_name = None
    mandel_file.stypy_call_defaults = defaults
    mandel_file.stypy_call_varargs = varargs
    mandel_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mandel_file', ['cx', 'cy', 'size', 'max_iterations', 'width', 'height'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mandel_file', localization, ['cx', 'cy', 'size', 'max_iterations', 'width', 'height'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mandel_file(...)' code ##################

    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to clock(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_461 = {}
    # Getting the type of 'time' (line 107)
    time_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 9), 'time', False)
    # Obtaining the member 'clock' of a type (line 107)
    clock_460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 9), time_459, 'clock')
    # Calling clock(args, kwargs) (line 107)
    clock_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 107, 9), clock_460, *[], **kwargs_461)
    
    # Assigning a type to the variable 't0' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 't0', clock_call_result_462)
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to min(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'size' (line 108)
    size_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'size', False)
    # Getting the type of 'width' (line 108)
    width_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'width', False)
    # Applying the binary operator 'div' (line 108)
    result_div_466 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 20), 'div', size_464, width_465)
    
    # Getting the type of 'size' (line 108)
    size_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'size', False)
    # Getting the type of 'height' (line 108)
    height_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 41), 'height', False)
    # Applying the binary operator 'div' (line 108)
    result_div_469 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 34), 'div', size_467, height_468)
    
    # Processing the call keyword arguments (line 108)
    kwargs_470 = {}
    # Getting the type of 'min' (line 108)
    min_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'min', False)
    # Calling min(args, kwargs) (line 108)
    min_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), min_463, *[result_div_466, result_div_469], **kwargs_470)
    
    # Assigning a type to the variable 'increment' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'increment', min_call_result_471)
    
    # Assigning a BinOp to a Name (line 109):
    
    # Assigning a BinOp to a Name (line 109):
    float_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'float')
    # Getting the type of 'width' (line 109)
    width_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'width')
    # Applying the binary operator '*' (line 109)
    result_mul_474 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 17), '*', float_472, width_473)
    
    # Getting the type of 'height' (line 109)
    height_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'height')
    # Applying the binary operator 'div' (line 109)
    result_div_476 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 29), 'div', result_mul_474, height_475)
    
    # Assigning a type to the variable 'proportion' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'proportion', result_div_476)
    
    # Assigning a Tuple to a Tuple (line 110):
    
    # Assigning a BinOp to a Name (line 110):
    # Getting the type of 'cx' (line 110)
    cx_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'cx')
    # Getting the type of 'increment' (line 110)
    increment_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 34), 'increment')
    # Getting the type of 'width' (line 110)
    width_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 46), 'width')
    # Applying the binary operator '*' (line 110)
    result_mul_480 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 34), '*', increment_478, width_479)
    
    int_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 54), 'int')
    # Applying the binary operator 'div' (line 110)
    result_div_482 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 52), 'div', result_mul_480, int_481)
    
    # Applying the binary operator '-' (line 110)
    result_sub_483 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 29), '-', cx_477, result_div_482)
    
    # Assigning a type to the variable 'tuple_assignment_5' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'tuple_assignment_5', result_sub_483)
    
    # Assigning a BinOp to a Name (line 110):
    # Getting the type of 'cy' (line 110)
    cy_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 57), 'cy')
    # Getting the type of 'increment' (line 110)
    increment_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 62), 'increment')
    # Getting the type of 'height' (line 110)
    height_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 74), 'height')
    # Applying the binary operator '*' (line 110)
    result_mul_487 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 62), '*', increment_485, height_486)
    
    int_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 83), 'int')
    # Applying the binary operator 'div' (line 110)
    result_div_489 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 81), 'div', result_mul_487, int_488)
    
    # Applying the binary operator '-' (line 110)
    result_sub_490 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 57), '-', cy_484, result_div_489)
    
    # Assigning a type to the variable 'tuple_assignment_6' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'tuple_assignment_6', result_sub_490)
    
    # Assigning a Name to a Name (line 110):
    # Getting the type of 'tuple_assignment_5' (line 110)
    tuple_assignment_5_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'tuple_assignment_5')
    # Assigning a type to the variable 'start_real' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'start_real', tuple_assignment_5_491)
    
    # Assigning a Name to a Name (line 110):
    # Getting the type of 'tuple_assignment_6' (line 110)
    tuple_assignment_6_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'tuple_assignment_6')
    # Assigning a type to the variable 'start_imag' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'start_imag', tuple_assignment_6_492)
    
    # Assigning a BinOp to a Name (line 112):
    
    # Assigning a BinOp to a Name (line 112):
    str_493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 17), 'str', '%g %gi_%g_%i')
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'cx' (line 112)
    cx_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'cx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_494, cx_495)
    # Adding element type (line 112)
    # Getting the type of 'cy' (line 112)
    cy_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 39), 'cy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_494, cy_496)
    # Adding element type (line 112)
    # Getting the type of 'size' (line 112)
    size_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_494, size_497)
    # Adding element type (line 112)
    # Getting the type of 'max_iterations' (line 112)
    max_iterations_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 49), 'max_iterations')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_494, max_iterations_498)
    
    # Applying the binary operator '%' (line 112)
    result_mod_499 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 17), '%', str_493, tuple_494)
    
    # Assigning a type to the variable 'mandel_pos' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'mandel_pos', result_mod_499)
    
    # Assigning a BinOp to a Name (line 113):
    
    # Assigning a BinOp to a Name (line 113):
    str_500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 12), 'str', 'm%s.bmp')
    # Getting the type of 'mandel_pos' (line 113)
    mandel_pos_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'mandel_pos')
    # Applying the binary operator '%' (line 113)
    result_mod_502 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 12), '%', str_500, mandel_pos_501)
    
    # Assigning a type to the variable 'fname' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'fname', result_mod_502)
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to kohn_bmp(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'fname' (line 114)
    fname_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'fname', False)
    # Getting the type of 'width' (line 114)
    width_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'width', False)
    # Getting the type of 'height' (line 114)
    height_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 36), 'height', False)
    int_507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 44), 'int')
    # Processing the call keyword arguments (line 114)
    kwargs_508 = {}
    # Getting the type of 'kohn_bmp' (line 114)
    kohn_bmp_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'kohn_bmp', False)
    # Calling kohn_bmp(args, kwargs) (line 114)
    kohn_bmp_call_result_509 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), kohn_bmp_503, *[fname_504, width_505, height_506, int_507], **kwargs_508)
    
    # Assigning a type to the variable 'my_bmp' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'my_bmp', kohn_bmp_call_result_509)
    
    # Assigning a Name to a Name (line 116):
    
    # Assigning a Name to a Name (line 116):
    # Getting the type of 'start_imag' (line 116)
    start_imag_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'start_imag')
    # Assigning a type to the variable 'current_y' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'current_y', start_imag_510)
    
    
    # Call to range(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'height' (line 117)
    height_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 19), 'height', False)
    # Processing the call keyword arguments (line 117)
    kwargs_513 = {}
    # Getting the type of 'range' (line 117)
    range_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'range', False)
    # Calling range(args, kwargs) (line 117)
    range_call_result_514 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), range_511, *[height_512], **kwargs_513)
    
    # Assigning a type to the variable 'range_call_result_514' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'range_call_result_514', range_call_result_514)
    # Testing if the for loop is going to be iterated (line 117)
    # Testing the type of a for loop iterable (line 117)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 4), range_call_result_514)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 117, 4), range_call_result_514):
        # Getting the type of the for loop variable (line 117)
        for_loop_var_515 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 4), range_call_result_514)
        # Assigning a type to the variable 'y' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'y', for_loop_var_515)
        # SSA begins for a for statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'y' (line 118)
        y_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'y')
        int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 19), 'int')
        # Applying the binary operator '%' (line 118)
        result_mod_518 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), '%', y_516, int_517)
        
        # Applying the 'not' unary operator (line 118)
        result_not__519 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 11), 'not', result_mod_518)
        
        # Testing if the type of an if condition is none (line 118)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 118, 8), result_not__519):
            pass
        else:
            
            # Testing the type of an if condition (line 118)
            if_condition_520 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 8), result_not__519)
            # Assigning a type to the variable 'if_condition_520' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'if_condition_520', if_condition_520)
            # SSA begins for if statement (line 118)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 118)
            module_type_store = module_type_store.join_ssa_context()
            

        pass
        
        # Assigning a Name to a Name (line 121):
        
        # Assigning a Name to a Name (line 121):
        # Getting the type of 'start_real' (line 121)
        start_real_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'start_real')
        # Assigning a type to the variable 'current_x' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'current_x', start_real_521)
        
        
        # Call to range(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'width' (line 123)
        width_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'width', False)
        # Processing the call keyword arguments (line 123)
        kwargs_524 = {}
        # Getting the type of 'range' (line 123)
        range_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 17), 'range', False)
        # Calling range(args, kwargs) (line 123)
        range_call_result_525 = invoke(stypy.reporting.localization.Localization(__file__, 123, 17), range_522, *[width_523], **kwargs_524)
        
        # Assigning a type to the variable 'range_call_result_525' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'range_call_result_525', range_call_result_525)
        # Testing if the for loop is going to be iterated (line 123)
        # Testing the type of a for loop iterable (line 123)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 8), range_call_result_525)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 123, 8), range_call_result_525):
            # Getting the type of the for loop variable (line 123)
            for_loop_var_526 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 8), range_call_result_525)
            # Assigning a type to the variable 'x' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'x', for_loop_var_526)
            # SSA begins for a for statement (line 123)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 124):
            
            # Assigning a Call to a Name (line 124):
            
            # Call to mandel(...): (line 124)
            # Processing the call arguments (line 124)
            # Getting the type of 'current_x' (line 124)
            current_x_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'current_x', False)
            # Getting the type of 'current_y' (line 124)
            current_y_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'current_y', False)
            # Getting the type of 'max_iterations' (line 124)
            max_iterations_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 45), 'max_iterations', False)
            # Processing the call keyword arguments (line 124)
            kwargs_531 = {}
            # Getting the type of 'mandel' (line 124)
            mandel_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'mandel', False)
            # Calling mandel(args, kwargs) (line 124)
            mandel_call_result_532 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), mandel_527, *[current_x_528, current_y_529, max_iterations_530], **kwargs_531)
            
            # Assigning a type to the variable 'c' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'c', mandel_call_result_532)
            
            # Assigning a IfExp to a Name (line 125):
            
            # Assigning a IfExp to a Name (line 125):
            
            
            # Getting the type of 'c' (line 125)
            c_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'c')
            int_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 52), 'int')
            # Applying the binary operator '!=' (line 125)
            result_ne_535 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 47), '!=', c_533, int_534)
            
            # Testing the type of an if expression (line 125)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 16), result_ne_535)
            # SSA begins for if expression (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            # Getting the type of 'c' (line 125)
            c_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'c')
            
            # Call to len(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'colors' (line 125)
            colors_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'colors', False)
            # Processing the call keyword arguments (line 125)
            kwargs_539 = {}
            # Getting the type of 'len' (line 125)
            len_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'len', False)
            # Calling len(args, kwargs) (line 125)
            len_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 125, 22), len_537, *[colors_538], **kwargs_539)
            
            int_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 36), 'int')
            # Applying the binary operator '-' (line 125)
            result_sub_542 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 22), '-', len_call_result_540, int_541)
            
            # Applying the binary operator '%' (line 125)
            result_mod_543 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 17), '%', c_536, result_sub_542)
            
            int_544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 41), 'int')
            # Applying the binary operator '+' (line 125)
            result_add_545 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 17), '+', result_mod_543, int_544)
            
            # SSA branch for the else part of an if expression (line 125)
            module_type_store.open_ssa_branch('if expression else')
            int_546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 60), 'int')
            # SSA join for if expression (line 125)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_547 = union_type.UnionType.add(result_add_545, int_546)
            
            # Assigning a type to the variable 'c' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'c', if_exp_547)
            
            # Getting the type of 'current_x' (line 126)
            current_x_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'current_x')
            # Getting the type of 'increment' (line 126)
            increment_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'increment')
            # Applying the binary operator '+=' (line 126)
            result_iadd_550 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 12), '+=', current_x_548, increment_549)
            # Assigning a type to the variable 'current_x' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'current_x', result_iadd_550)
            
            
            # Call to write_pixel(...): (line 127)
            # Processing the call arguments (line 127)
            
            # Obtaining the type of the subscript
            int_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 41), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'c' (line 127)
            c_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 38), 'c', False)
            # Getting the type of 'colors' (line 127)
            colors_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'colors', False)
            # Obtaining the member '__getitem__' of a type (line 127)
            getitem___556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 31), colors_555, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 127)
            subscript_call_result_557 = invoke(stypy.reporting.localization.Localization(__file__, 127, 31), getitem___556, c_554)
            
            # Obtaining the member '__getitem__' of a type (line 127)
            getitem___558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 31), subscript_call_result_557, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 127)
            subscript_call_result_559 = invoke(stypy.reporting.localization.Localization(__file__, 127, 31), getitem___558, int_553)
            
            
            # Obtaining the type of the subscript
            int_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 55), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'c' (line 127)
            c_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 52), 'c', False)
            # Getting the type of 'colors' (line 127)
            colors_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 45), 'colors', False)
            # Obtaining the member '__getitem__' of a type (line 127)
            getitem___563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 45), colors_562, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 127)
            subscript_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 127, 45), getitem___563, c_561)
            
            # Obtaining the member '__getitem__' of a type (line 127)
            getitem___565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 45), subscript_call_result_564, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 127)
            subscript_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 127, 45), getitem___565, int_560)
            
            
            # Obtaining the type of the subscript
            int_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 69), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'c' (line 127)
            c_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 66), 'c', False)
            # Getting the type of 'colors' (line 127)
            colors_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'colors', False)
            # Obtaining the member '__getitem__' of a type (line 127)
            getitem___570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), colors_569, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 127)
            subscript_call_result_571 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___570, c_568)
            
            # Obtaining the member '__getitem__' of a type (line 127)
            getitem___572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 59), subscript_call_result_571, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 127)
            subscript_call_result_573 = invoke(stypy.reporting.localization.Localization(__file__, 127, 59), getitem___572, int_567)
            
            # Processing the call keyword arguments (line 127)
            kwargs_574 = {}
            # Getting the type of 'my_bmp' (line 127)
            my_bmp_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'my_bmp', False)
            # Obtaining the member 'write_pixel' of a type (line 127)
            write_pixel_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), my_bmp_551, 'write_pixel')
            # Calling write_pixel(args, kwargs) (line 127)
            write_pixel_call_result_575 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), write_pixel_552, *[subscript_call_result_559, subscript_call_result_566, subscript_call_result_573], **kwargs_574)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'current_y' (line 128)
        current_y_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'current_y')
        # Getting the type of 'increment' (line 128)
        increment_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'increment')
        # Applying the binary operator '+=' (line 128)
        result_iadd_578 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 8), '+=', current_y_576, increment_577)
        # Assigning a type to the variable 'current_y' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'current_y', result_iadd_578)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to close(...): (line 131)
    # Processing the call keyword arguments (line 131)
    kwargs_581 = {}
    # Getting the type of 'my_bmp' (line 131)
    my_bmp_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'my_bmp', False)
    # Obtaining the member 'close' of a type (line 131)
    close_580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), my_bmp_579, 'close')
    # Calling close(args, kwargs) (line 131)
    close_call_result_582 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), close_580, *[], **kwargs_581)
    
    # Getting the type of 'fname' (line 132)
    fname_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'fname')
    # Assigning a type to the variable 'stypy_return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type', fname_583)
    
    # ################# End of 'mandel_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mandel_file' in the type store
    # Getting the type of 'stypy_return_type' (line 106)
    stypy_return_type_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_584)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mandel_file'
    return stypy_return_type_584

# Assigning a type to the variable 'mandel_file' (line 106)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 0), 'mandel_file', mandel_file)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 135, 0, False)
    
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

    
    # Assigning a Num to a Name (line 136):
    
    # Assigning a Num to a Name (line 136):
    int_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 10), 'int')
    # Assigning a type to the variable 'res' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'res', int_585)
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to mandel(...): (line 137)
    # Processing the call arguments (line 137)
    float_587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 17), 'float')
    float_588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'float')
    int_589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 27), 'int')
    # Processing the call keyword arguments (line 137)
    kwargs_590 = {}
    # Getting the type of 'mandel' (line 137)
    mandel_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 10), 'mandel', False)
    # Calling mandel(args, kwargs) (line 137)
    mandel_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 137, 10), mandel_586, *[float_587, float_588, int_589], **kwargs_590)
    
    # Assigning a type to the variable 'res' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'res', mandel_call_result_591)
    
    # Call to mandel_file(...): (line 138)
    # Processing the call keyword arguments (line 138)
    int_593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 31), 'int')
    keyword_594 = int_593
    kwargs_595 = {'max_iterations': keyword_594}
    # Getting the type of 'mandel_file' (line 138)
    mandel_file_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'mandel_file', False)
    # Calling mandel_file(args, kwargs) (line 138)
    mandel_file_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), mandel_file_592, *[], **kwargs_595)
    
    # Getting the type of 'True' (line 139)
    True_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type', True_597)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_598)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_598

# Assigning a type to the variable 'run' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'run', run)

# Call to run(...): (line 142)
# Processing the call keyword arguments (line 142)
kwargs_600 = {}
# Getting the type of 'run' (line 142)
run_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'run', False)
# Calling run(args, kwargs) (line 142)
run_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 142, 0), run_599, *[], **kwargs_600)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()

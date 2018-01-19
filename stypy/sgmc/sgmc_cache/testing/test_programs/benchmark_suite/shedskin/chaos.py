
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #   Copyright (C) 2005 Carl Friedrich Bolz
2: 
3: #   MIT license
4: 
5: '''create chaosgame-like fractals
6: '''
7: 
8: import random
9: import math
10: random.seed(1234)
11: import sys
12: import time
13: import os
14: 
15: def Relative(path):
16:     return os.path.join(os.path.dirname(__file__), path)
17: 
18: class GVector(object):
19:     def __init__(self, x = 0, y = 0, z = 0):
20:         self.x = x
21:         self.y = y
22:         self.z = z
23: 
24:     def Mag(self):
25:         return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
26: 
27:     def dist(self, other):
28:         return math.sqrt((self.x - other.x) ** 2 +
29:                          (self.y - other.y) ** 2 +
30:                          (self.z - other.z) ** 2)
31: 
32:     def __add__(self, other):
33:         v = GVector(self.x + other.x, self.y + other.y, self.z + other.z)
34:         return v
35: 
36:     def __sub__(self, other):
37:         return self + other * -1
38: 
39:     def __mul__(self, other):
40:         v = GVector(self.x * other, self.y * other, self.z * other)
41:         return v
42:     
43:     def linear_combination(self, other, l1, l2):
44:         v = GVector(self.x * l1 + other.x * l2,
45:                     self.y * l1 + other.y * l2,
46:                     self.z * l1 + other.z * l2)
47:         return v
48: 
49:     def __str__(self):
50:         return "<%f, %f, %f>" % (self.x, self.y, self.z)
51: 
52:     def __repr__(self):
53:         return "GVector(%f, %f, %f)" % (self.x, self.y, self.z)
54: 
55: def GetKnots(points, degree):
56:     knots = [0] * degree + range(1, len(points) - degree)
57:     knots += [len(points) - degree] * degree
58:     return knots
59: 
60: class Spline(object):
61:     '''Class for representing B-Splines and NURBS of arbitrary degree'''
62:     def __init__(self, points, degree = 3, knots = None):
63:         '''Creates a Spline. points is a list of GVector, degree is the
64: degree of the Spline.'''
65:         if knots == None:
66:             self.knots = GetKnots(points, degree)
67:         else:
68:             if len(points) > len(knots) - degree + 1:
69:                 raise ValueError("too many control points")
70:             elif len(points) < len(knots) - degree + 1:
71:                 raise ValueError("not enough control points")
72:             last = knots[0]
73:             for cur in knots[1:]:
74:                 if cur < last:
75:                     raise ValueError( "knots not strictly increasing")
76:                 last = cur
77:             self.knots = knots
78:         self.points = points
79:         self.degree = degree
80: 
81:     def GetDomain(self):
82:         '''Returns the domain of the B-Spline'''
83:         return (self.knots[self.degree - 1],
84:                 self.knots[len(self.knots) - self.degree])
85: 
86:     def call(self, u):
87:         '''Calculates a point of the B-Spline using de Boors Algorithm'''
88:         dom = self.GetDomain()
89:         if u < dom[0] or u > dom[1]:
90:             raise ValueError("Function value not in domain")
91:         if u == dom[0]:
92:             return self.points[0]
93:         if u == dom[1]:
94:             return self.points[-1]
95:         I = self.GetIndex(u)
96:         d = [self.points[I - self.degree + 1 + ii]
97:              for ii in range(self.degree + 1)]
98:         U = self.knots
99:         for ik in range(1, self.degree + 1):
100:             for ii in range(I - self.degree + ik + 1, I + 2):
101:                 ua = U[ii + self.degree - ik]
102:                 ub = U[ii - 1]
103:                 co1 = (ua - u) / float(ua - ub)
104:                 co2 = (u - ub) / float(ua - ub)
105:                 index = ii - I + self.degree - ik - 1
106:                 d[index] = d[index].linear_combination(d[index + 1], co1, co2)
107:         return d[0]
108: 
109:     def GetIndex(self, u):
110:         dom = self.GetDomain()
111:         for ii in range(self.degree - 1, len(self.knots) - self.degree):
112:             if u >= self.knots[ii] and u < self.knots[ii + 1]:
113:                 I = ii
114:                 break
115:         else:
116:              I = dom[1] - 1
117:         return I
118: 
119:     def __len__(self):
120:         return len(self.points)
121: 
122:     def __repr__(self):
123:         return "Spline(%r, %r, %r)" % (self.points, self.degree, self.knots)
124: 
125:         
126: def save_im(im, fn):
127:     f = open(fn, "wb")
128:     magic = 'P6\n'
129:     maxval = 255
130:     w = len(im)
131:     h = len(im[0])
132:     f.write(magic)
133:     f.write('%i %i\n%i\n' % (w, h, maxval))
134:     for j in range(h):
135:         for i in range(w):
136:             val = im[i][j]
137:             c = val * 255
138:             f.write('%c%c%c' % (c, c, c))
139:     f.close()
140: 
141: class Chaosgame(object):
142:     def __init__(self, splines, thickness=0.1):
143:         self.splines = splines
144:         self.thickness = thickness
145:         self.minx = min([p.x for spl in splines for p in spl.points])
146:         self.miny = min([p.y for spl in splines for p in spl.points])
147:         self.maxx = max([p.x for spl in splines for p in spl.points])
148:         self.maxy = max([p.y for spl in splines for p in spl.points])
149:         self.height = self.maxy - self.miny
150:         self.width = self.maxx - self.minx
151:         self.num_trafos = []
152:         maxlength = thickness * self.width / float(self.height)
153:         for spl in splines:
154:             length = 0
155:             curr = spl.call(0)
156:             for i in range(1, 1000):
157:                 last = curr
158:                 t = 1.0 / 999 * i
159:                 curr = spl.call(t)
160:                 length += curr.dist(last)
161:             self.num_trafos.append(max(1, int(length / maxlength * 1.5)))
162:         self.num_total = reduce(lambda a,b: a+b, self.num_trafos, 0)
163: 
164: 
165:     def get_random_trafo(self):
166:         r = random.randrange(int(self.num_total) + 1)
167:         l = 0
168:         for i in range(len(self.num_trafos)):
169:             if r >= l and r < l + self.num_trafos[i]:
170:                 return i, random.randrange(self.num_trafos[i])
171:             l += self.num_trafos[i]
172:         return len(self.num_trafos) - 1, random.randrange(self.num_trafos[-1])
173: 
174:     def transform_point(self, point, trafo=None):
175:         x = (point.x - self.minx) / float(self.width)
176:         y = (point.y - self.miny) / float(self.height)
177:         if trafo is None:
178:             trafo = self.get_random_trafo()
179:         start, end = self.splines[trafo[0]].GetDomain()
180:         length = end - start
181:         seg_length = length / float(self.num_trafos[trafo[0]])
182:         t = start + seg_length * trafo[1] + seg_length * x
183:         basepoint = self.splines[trafo[0]].call(t)
184:         if t + 1.0/50000 > end:
185:             neighbour = self.splines[trafo[0]].call(t - 1.0/50000)
186:             derivative = neighbour - basepoint
187:         else:
188:             neighbour = self.splines[trafo[0]].call(t + 1.0/50000)
189:             derivative = basepoint - neighbour
190:         if derivative.Mag() != 0:
191:             basepoint.x += derivative.y / derivative.Mag() * (y - 0.5) * \
192:                            self.thickness
193:             basepoint.y += -derivative.x / derivative.Mag() * (y - 0.5) * \
194:                            self.thickness
195:         else:
196:             pass#print "r",
197:         self.truncate(basepoint)
198:         return basepoint
199: 
200:     def truncate(self, point):
201:         if point.x >= self.maxx:
202:             point.x = self.maxx
203:         if point.y >= self.maxy:
204:             point.y = self.maxy
205:         if point.x < self.minx:
206:             point.x = self.minx
207:         if point.y < self.miny:
208:             point.y = self.miny
209: 
210:     def create_image_chaos(self, w, h, name, n):
211:         im = [[1] * h for i in range(w)]
212:         point = GVector((self.maxx + self.minx) / 2.0,
213:                         (self.maxy + self.miny) / 2.0, 0)
214:         colored = 0
215:         times = []
216:         for _ in range(n):
217:             t1 = time.time()
218:             for i in xrange(5000):
219:                 point = self.transform_point(point)
220:                 x = (point.x - self.minx) / self.width * w
221:                 y = (point.y - self.miny) / self.height * h
222:                 x = int(x)
223:                 y = int(y)
224:                 if x == w:
225:                     x -= 1
226:                 if y == h:
227:                     y -= 1
228:                 im[x][h - y - 1] = 0
229:             t2 = time.time()
230:             times.append(t2 - t1)
231:         save_im(im, name)
232:         return times
233: 
234: 
235: def main(n):
236:     splines = [
237:         Spline([
238:             GVector(1.597350, 3.304460, 0.000000),
239:             GVector(1.575810, 4.123260, 0.000000),
240:             GVector(1.313210, 5.288350, 0.000000),
241:             GVector(1.618900, 5.329910, 0.000000),
242:             GVector(2.889940, 5.502700, 0.000000),
243:             GVector(2.373060, 4.381830, 0.000000),
244:             GVector(1.662000, 4.360280, 0.000000)],
245:             3, [0, 0, 0, 1, 1, 1, 2, 2, 2]),
246:         Spline([
247:             GVector(2.804500, 4.017350, 0.000000),
248:             GVector(2.550500, 3.525230, 0.000000),
249:             GVector(1.979010, 2.620360, 0.000000),
250:             GVector(1.979010, 2.620360, 0.000000)],
251:             3, [0, 0, 0, 1, 1, 1]),
252:         Spline([
253:             GVector(2.001670, 4.011320, 0.000000),
254:             GVector(2.335040, 3.312830, 0.000000),
255:             GVector(2.366800, 3.233460, 0.000000),
256:             GVector(2.366800, 3.233460, 0.000000)],
257:             3, [0, 0, 0, 1, 1, 1])
258:         ]
259:     c = Chaosgame(splines, 0.25)
260:     return c.create_image_chaos(1000, 1200, Relative("py.ppm"), n)
261: 
262: 
263: def run():
264:     main(50)
265:     return True
266: 
267: run()
268: 
269: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', 'create chaosgame-like fractals\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import random' statement (line 8)
import random

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import math' statement (line 9)
import math

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'math', math, module_type_store)


# Call to seed(...): (line 10)
# Processing the call arguments (line 10)
int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 12), 'int')
# Processing the call keyword arguments (line 10)
kwargs_7 = {}
# Getting the type of 'random' (line 10)
random_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'random', False)
# Obtaining the member 'seed' of a type (line 10)
seed_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), random_4, 'seed')
# Calling seed(args, kwargs) (line 10)
seed_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), seed_5, *[int_6], **kwargs_7)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import sys' statement (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import time' statement (line 12)
import time

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import os' statement (line 13)
import os

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 15, 0, False)
    
    # Passed parameters checking function
    Relative.stypy_localization = localization
    Relative.stypy_type_of_self = None
    Relative.stypy_type_store = module_type_store
    Relative.stypy_function_name = 'Relative'
    Relative.stypy_param_names_list = ['path']
    Relative.stypy_varargs_param_name = None
    Relative.stypy_kwargs_param_name = None
    Relative.stypy_call_defaults = defaults
    Relative.stypy_call_varargs = varargs
    Relative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Relative', ['path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Relative', localization, ['path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Relative(...)' code ##################

    
    # Call to join(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to dirname(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of '__file__' (line 16)
    file___15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 40), '__file__', False)
    # Processing the call keyword arguments (line 16)
    kwargs_16 = {}
    # Getting the type of 'os' (line 16)
    os_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 16)
    path_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), os_12, 'path')
    # Obtaining the member 'dirname' of a type (line 16)
    dirname_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), path_13, 'dirname')
    # Calling dirname(args, kwargs) (line 16)
    dirname_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), dirname_14, *[file___15], **kwargs_16)
    
    # Getting the type of 'path' (line 16)
    path_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 51), 'path', False)
    # Processing the call keyword arguments (line 16)
    kwargs_19 = {}
    # Getting the type of 'os' (line 16)
    os_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 16)
    path_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), os_9, 'path')
    # Obtaining the member 'join' of a type (line 16)
    join_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), path_10, 'join')
    # Calling join(args, kwargs) (line 16)
    join_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), join_11, *[dirname_call_result_17, path_18], **kwargs_19)
    
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', join_call_result_20)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_21

# Assigning a type to the variable 'Relative' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Relative', Relative)
# Declaration of the 'GVector' class

class GVector(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'int')
        int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'int')
        int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'int')
        defaults = [int_22, int_23, int_24]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.__init__', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 20):
        
        # Assigning a Name to a Attribute (line 20):
        # Getting the type of 'x' (line 20)
        x_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'x')
        # Getting the type of 'self' (line 20)
        self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'x' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_26, 'x', x_25)
        
        # Assigning a Name to a Attribute (line 21):
        
        # Assigning a Name to a Attribute (line 21):
        # Getting the type of 'y' (line 21)
        y_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'y')
        # Getting the type of 'self' (line 21)
        self_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member 'y' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_28, 'y', y_27)
        
        # Assigning a Name to a Attribute (line 22):
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'z' (line 22)
        z_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'z')
        # Getting the type of 'self' (line 22)
        self_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'z' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_30, 'z', z_29)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def Mag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'Mag'
        module_type_store = module_type_store.open_function_context('Mag', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.Mag.__dict__.__setitem__('stypy_localization', localization)
        GVector.Mag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.Mag.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.Mag.__dict__.__setitem__('stypy_function_name', 'GVector.Mag')
        GVector.Mag.__dict__.__setitem__('stypy_param_names_list', [])
        GVector.Mag.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.Mag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.Mag.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.Mag.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.Mag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.Mag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.Mag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'Mag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'Mag(...)' code ##################

        
        # Call to sqrt(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'self' (line 25)
        self_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'self', False)
        # Obtaining the member 'x' of a type (line 25)
        x_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), self_33, 'x')
        int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'int')
        # Applying the binary operator '**' (line 25)
        result_pow_36 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '**', x_34, int_35)
        
        # Getting the type of 'self' (line 25)
        self_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'self', False)
        # Obtaining the member 'y' of a type (line 25)
        y_38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), self_37, 'y')
        int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 49), 'int')
        # Applying the binary operator '**' (line 25)
        result_pow_40 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 39), '**', y_38, int_39)
        
        # Applying the binary operator '+' (line 25)
        result_add_41 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '+', result_pow_36, result_pow_40)
        
        # Getting the type of 'self' (line 25)
        self_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 53), 'self', False)
        # Obtaining the member 'z' of a type (line 25)
        z_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 53), self_42, 'z')
        int_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 63), 'int')
        # Applying the binary operator '**' (line 25)
        result_pow_45 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 53), '**', z_43, int_44)
        
        # Applying the binary operator '+' (line 25)
        result_add_46 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 51), '+', result_add_41, result_pow_45)
        
        # Processing the call keyword arguments (line 25)
        kwargs_47 = {}
        # Getting the type of 'math' (line 25)
        math_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 25)
        sqrt_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), math_31, 'sqrt')
        # Calling sqrt(args, kwargs) (line 25)
        sqrt_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), sqrt_32, *[result_add_46], **kwargs_47)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', sqrt_call_result_48)
        
        # ################# End of 'Mag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'Mag' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_49)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'Mag'
        return stypy_return_type_49


    @norecursion
    def dist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dist'
        module_type_store = module_type_store.open_function_context('dist', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.dist.__dict__.__setitem__('stypy_localization', localization)
        GVector.dist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.dist.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.dist.__dict__.__setitem__('stypy_function_name', 'GVector.dist')
        GVector.dist.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GVector.dist.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.dist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.dist.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.dist.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.dist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.dist.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.dist', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dist', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dist(...)' code ##################

        
        # Call to sqrt(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'self', False)
        # Obtaining the member 'x' of a type (line 28)
        x_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 26), self_52, 'x')
        # Getting the type of 'other' (line 28)
        other_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'other', False)
        # Obtaining the member 'x' of a type (line 28)
        x_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 35), other_54, 'x')
        # Applying the binary operator '-' (line 28)
        result_sub_56 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 26), '-', x_53, x_55)
        
        int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 47), 'int')
        # Applying the binary operator '**' (line 28)
        result_pow_58 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 25), '**', result_sub_56, int_57)
        
        # Getting the type of 'self' (line 29)
        self_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'self', False)
        # Obtaining the member 'y' of a type (line 29)
        y_60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 26), self_59, 'y')
        # Getting the type of 'other' (line 29)
        other_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'other', False)
        # Obtaining the member 'y' of a type (line 29)
        y_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 35), other_61, 'y')
        # Applying the binary operator '-' (line 29)
        result_sub_63 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 26), '-', y_60, y_62)
        
        int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 47), 'int')
        # Applying the binary operator '**' (line 29)
        result_pow_65 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 25), '**', result_sub_63, int_64)
        
        # Applying the binary operator '+' (line 28)
        result_add_66 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 25), '+', result_pow_58, result_pow_65)
        
        # Getting the type of 'self' (line 30)
        self_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'self', False)
        # Obtaining the member 'z' of a type (line 30)
        z_68 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 26), self_67, 'z')
        # Getting the type of 'other' (line 30)
        other_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'other', False)
        # Obtaining the member 'z' of a type (line 30)
        z_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 35), other_69, 'z')
        # Applying the binary operator '-' (line 30)
        result_sub_71 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 26), '-', z_68, z_70)
        
        int_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 47), 'int')
        # Applying the binary operator '**' (line 30)
        result_pow_73 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 25), '**', result_sub_71, int_72)
        
        # Applying the binary operator '+' (line 29)
        result_add_74 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 49), '+', result_add_66, result_pow_73)
        
        # Processing the call keyword arguments (line 28)
        kwargs_75 = {}
        # Getting the type of 'math' (line 28)
        math_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 28)
        sqrt_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), math_50, 'sqrt')
        # Calling sqrt(args, kwargs) (line 28)
        sqrt_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), sqrt_51, *[result_add_74], **kwargs_75)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', sqrt_call_result_76)
        
        # ################# End of 'dist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dist' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_77)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dist'
        return stypy_return_type_77


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.__add__.__dict__.__setitem__('stypy_localization', localization)
        GVector.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.__add__.__dict__.__setitem__('stypy_function_name', 'GVector.__add__')
        GVector.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GVector.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to GVector(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'self' (line 33)
        self_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'self', False)
        # Obtaining the member 'x' of a type (line 33)
        x_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 20), self_79, 'x')
        # Getting the type of 'other' (line 33)
        other_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'other', False)
        # Obtaining the member 'x' of a type (line 33)
        x_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 29), other_81, 'x')
        # Applying the binary operator '+' (line 33)
        result_add_83 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 20), '+', x_80, x_82)
        
        # Getting the type of 'self' (line 33)
        self_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'self', False)
        # Obtaining the member 'y' of a type (line 33)
        y_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 38), self_84, 'y')
        # Getting the type of 'other' (line 33)
        other_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 47), 'other', False)
        # Obtaining the member 'y' of a type (line 33)
        y_87 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 47), other_86, 'y')
        # Applying the binary operator '+' (line 33)
        result_add_88 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 38), '+', y_85, y_87)
        
        # Getting the type of 'self' (line 33)
        self_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 56), 'self', False)
        # Obtaining the member 'z' of a type (line 33)
        z_90 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 56), self_89, 'z')
        # Getting the type of 'other' (line 33)
        other_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 65), 'other', False)
        # Obtaining the member 'z' of a type (line 33)
        z_92 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 65), other_91, 'z')
        # Applying the binary operator '+' (line 33)
        result_add_93 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 56), '+', z_90, z_92)
        
        # Processing the call keyword arguments (line 33)
        kwargs_94 = {}
        # Getting the type of 'GVector' (line 33)
        GVector_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'GVector', False)
        # Calling GVector(args, kwargs) (line 33)
        GVector_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), GVector_78, *[result_add_83, result_add_88, result_add_93], **kwargs_94)
        
        # Assigning a type to the variable 'v' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'v', GVector_call_result_95)
        # Getting the type of 'v' (line 34)
        v_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', v_96)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_97)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_97


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.__sub__.__dict__.__setitem__('stypy_localization', localization)
        GVector.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.__sub__.__dict__.__setitem__('stypy_function_name', 'GVector.__sub__')
        GVector.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GVector.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.__sub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        # Getting the type of 'self' (line 37)
        self_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'self')
        # Getting the type of 'other' (line 37)
        other_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'other')
        int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'int')
        # Applying the binary operator '*' (line 37)
        result_mul_101 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 22), '*', other_99, int_100)
        
        # Applying the binary operator '+' (line 37)
        result_add_102 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 15), '+', self_98, result_mul_101)
        
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', result_add_102)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_103)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_103


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.__mul__.__dict__.__setitem__('stypy_localization', localization)
        GVector.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.__mul__.__dict__.__setitem__('stypy_function_name', 'GVector.__mul__')
        GVector.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GVector.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.__mul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to GVector(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'self', False)
        # Obtaining the member 'x' of a type (line 40)
        x_106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 20), self_105, 'x')
        # Getting the type of 'other' (line 40)
        other_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'other', False)
        # Applying the binary operator '*' (line 40)
        result_mul_108 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '*', x_106, other_107)
        
        # Getting the type of 'self' (line 40)
        self_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'self', False)
        # Obtaining the member 'y' of a type (line 40)
        y_110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 36), self_109, 'y')
        # Getting the type of 'other' (line 40)
        other_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 45), 'other', False)
        # Applying the binary operator '*' (line 40)
        result_mul_112 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 36), '*', y_110, other_111)
        
        # Getting the type of 'self' (line 40)
        self_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 52), 'self', False)
        # Obtaining the member 'z' of a type (line 40)
        z_114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 52), self_113, 'z')
        # Getting the type of 'other' (line 40)
        other_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 61), 'other', False)
        # Applying the binary operator '*' (line 40)
        result_mul_116 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 52), '*', z_114, other_115)
        
        # Processing the call keyword arguments (line 40)
        kwargs_117 = {}
        # Getting the type of 'GVector' (line 40)
        GVector_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'GVector', False)
        # Calling GVector(args, kwargs) (line 40)
        GVector_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), GVector_104, *[result_mul_108, result_mul_112, result_mul_116], **kwargs_117)
        
        # Assigning a type to the variable 'v' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'v', GVector_call_result_118)
        # Getting the type of 'v' (line 41)
        v_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', v_119)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_120)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_120


    @norecursion
    def linear_combination(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'linear_combination'
        module_type_store = module_type_store.open_function_context('linear_combination', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.linear_combination.__dict__.__setitem__('stypy_localization', localization)
        GVector.linear_combination.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.linear_combination.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.linear_combination.__dict__.__setitem__('stypy_function_name', 'GVector.linear_combination')
        GVector.linear_combination.__dict__.__setitem__('stypy_param_names_list', ['other', 'l1', 'l2'])
        GVector.linear_combination.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.linear_combination.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.linear_combination.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.linear_combination.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.linear_combination.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.linear_combination.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.linear_combination', ['other', 'l1', 'l2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'linear_combination', localization, ['other', 'l1', 'l2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'linear_combination(...)' code ##################

        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to GVector(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'self' (line 44)
        self_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'self', False)
        # Obtaining the member 'x' of a type (line 44)
        x_123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), self_122, 'x')
        # Getting the type of 'l1' (line 44)
        l1_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'l1', False)
        # Applying the binary operator '*' (line 44)
        result_mul_125 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 20), '*', x_123, l1_124)
        
        # Getting the type of 'other' (line 44)
        other_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'other', False)
        # Obtaining the member 'x' of a type (line 44)
        x_127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 34), other_126, 'x')
        # Getting the type of 'l2' (line 44)
        l2_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 44), 'l2', False)
        # Applying the binary operator '*' (line 44)
        result_mul_129 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 34), '*', x_127, l2_128)
        
        # Applying the binary operator '+' (line 44)
        result_add_130 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 20), '+', result_mul_125, result_mul_129)
        
        # Getting the type of 'self' (line 45)
        self_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'self', False)
        # Obtaining the member 'y' of a type (line 45)
        y_132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 20), self_131, 'y')
        # Getting the type of 'l1' (line 45)
        l1_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'l1', False)
        # Applying the binary operator '*' (line 45)
        result_mul_134 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 20), '*', y_132, l1_133)
        
        # Getting the type of 'other' (line 45)
        other_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'other', False)
        # Obtaining the member 'y' of a type (line 45)
        y_136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 34), other_135, 'y')
        # Getting the type of 'l2' (line 45)
        l2_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'l2', False)
        # Applying the binary operator '*' (line 45)
        result_mul_138 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 34), '*', y_136, l2_137)
        
        # Applying the binary operator '+' (line 45)
        result_add_139 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 20), '+', result_mul_134, result_mul_138)
        
        # Getting the type of 'self' (line 46)
        self_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'self', False)
        # Obtaining the member 'z' of a type (line 46)
        z_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), self_140, 'z')
        # Getting the type of 'l1' (line 46)
        l1_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'l1', False)
        # Applying the binary operator '*' (line 46)
        result_mul_143 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 20), '*', z_141, l1_142)
        
        # Getting the type of 'other' (line 46)
        other_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'other', False)
        # Obtaining the member 'z' of a type (line 46)
        z_145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 34), other_144, 'z')
        # Getting the type of 'l2' (line 46)
        l2_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 44), 'l2', False)
        # Applying the binary operator '*' (line 46)
        result_mul_147 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 34), '*', z_145, l2_146)
        
        # Applying the binary operator '+' (line 46)
        result_add_148 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 20), '+', result_mul_143, result_mul_147)
        
        # Processing the call keyword arguments (line 44)
        kwargs_149 = {}
        # Getting the type of 'GVector' (line 44)
        GVector_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'GVector', False)
        # Calling GVector(args, kwargs) (line 44)
        GVector_call_result_150 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), GVector_121, *[result_add_130, result_add_139, result_add_148], **kwargs_149)
        
        # Assigning a type to the variable 'v' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'v', GVector_call_result_150)
        # Getting the type of 'v' (line 47)
        v_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', v_151)
        
        # ################# End of 'linear_combination(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'linear_combination' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_152)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'linear_combination'
        return stypy_return_type_152


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        GVector.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.stypy__str__.__dict__.__setitem__('stypy_function_name', 'GVector.stypy__str__')
        GVector.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        GVector.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        str_153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 15), 'str', '<%f, %f, %f>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 33), 'self')
        # Obtaining the member 'x' of a type (line 50)
        x_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 33), self_155, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_154, x_156)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), 'self')
        # Obtaining the member 'y' of a type (line 50)
        y_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 41), self_157, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_154, y_158)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 49), 'self')
        # Obtaining the member 'z' of a type (line 50)
        z_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 49), self_159, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_154, z_160)
        
        # Applying the binary operator '%' (line 50)
        result_mod_161 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), '%', str_153, tuple_154)
        
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', result_mod_161)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_162


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'GVector.stypy__repr__')
        GVector.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        GVector.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'str', 'GVector(%f, %f, %f)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        # Getting the type of 'self' (line 53)
        self_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 40), 'self')
        # Obtaining the member 'x' of a type (line 53)
        x_166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 40), self_165, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 40), tuple_164, x_166)
        # Adding element type (line 53)
        # Getting the type of 'self' (line 53)
        self_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 48), 'self')
        # Obtaining the member 'y' of a type (line 53)
        y_168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 48), self_167, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 40), tuple_164, y_168)
        # Adding element type (line 53)
        # Getting the type of 'self' (line 53)
        self_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 56), 'self')
        # Obtaining the member 'z' of a type (line 53)
        z_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 56), self_169, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 40), tuple_164, z_170)
        
        # Applying the binary operator '%' (line 53)
        result_mod_171 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 15), '%', str_163, tuple_164)
        
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', result_mod_171)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_172)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_172


# Assigning a type to the variable 'GVector' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'GVector', GVector)

@norecursion
def GetKnots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'GetKnots'
    module_type_store = module_type_store.open_function_context('GetKnots', 55, 0, False)
    
    # Passed parameters checking function
    GetKnots.stypy_localization = localization
    GetKnots.stypy_type_of_self = None
    GetKnots.stypy_type_store = module_type_store
    GetKnots.stypy_function_name = 'GetKnots'
    GetKnots.stypy_param_names_list = ['points', 'degree']
    GetKnots.stypy_varargs_param_name = None
    GetKnots.stypy_kwargs_param_name = None
    GetKnots.stypy_call_defaults = defaults
    GetKnots.stypy_call_varargs = varargs
    GetKnots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'GetKnots', ['points', 'degree'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'GetKnots', localization, ['points', 'degree'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'GetKnots(...)' code ##################

    
    # Assigning a BinOp to a Name (line 56):
    
    # Assigning a BinOp to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), list_173, int_174)
    
    # Getting the type of 'degree' (line 56)
    degree_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'degree')
    # Applying the binary operator '*' (line 56)
    result_mul_176 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 12), '*', list_173, degree_175)
    
    
    # Call to range(...): (line 56)
    # Processing the call arguments (line 56)
    int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'int')
    
    # Call to len(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'points' (line 56)
    points_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 40), 'points', False)
    # Processing the call keyword arguments (line 56)
    kwargs_181 = {}
    # Getting the type of 'len' (line 56)
    len_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'len', False)
    # Calling len(args, kwargs) (line 56)
    len_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 56, 36), len_179, *[points_180], **kwargs_181)
    
    # Getting the type of 'degree' (line 56)
    degree_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 50), 'degree', False)
    # Applying the binary operator '-' (line 56)
    result_sub_184 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 36), '-', len_call_result_182, degree_183)
    
    # Processing the call keyword arguments (line 56)
    kwargs_185 = {}
    # Getting the type of 'range' (line 56)
    range_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'range', False)
    # Calling range(args, kwargs) (line 56)
    range_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 56, 27), range_177, *[int_178, result_sub_184], **kwargs_185)
    
    # Applying the binary operator '+' (line 56)
    result_add_187 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 12), '+', result_mul_176, range_call_result_186)
    
    # Assigning a type to the variable 'knots' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'knots', result_add_187)
    
    # Getting the type of 'knots' (line 57)
    knots_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'knots')
    
    # Obtaining an instance of the builtin type 'list' (line 57)
    list_189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 57)
    # Adding element type (line 57)
    
    # Call to len(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'points' (line 57)
    points_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'points', False)
    # Processing the call keyword arguments (line 57)
    kwargs_192 = {}
    # Getting the type of 'len' (line 57)
    len_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'len', False)
    # Calling len(args, kwargs) (line 57)
    len_call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 57, 14), len_190, *[points_191], **kwargs_192)
    
    # Getting the type of 'degree' (line 57)
    degree_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'degree')
    # Applying the binary operator '-' (line 57)
    result_sub_195 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 14), '-', len_call_result_193, degree_194)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 13), list_189, result_sub_195)
    
    # Getting the type of 'degree' (line 57)
    degree_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 38), 'degree')
    # Applying the binary operator '*' (line 57)
    result_mul_197 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 13), '*', list_189, degree_196)
    
    # Applying the binary operator '+=' (line 57)
    result_iadd_198 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 4), '+=', knots_188, result_mul_197)
    # Assigning a type to the variable 'knots' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'knots', result_iadd_198)
    
    # Getting the type of 'knots' (line 58)
    knots_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'knots')
    # Assigning a type to the variable 'stypy_return_type' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type', knots_199)
    
    # ################# End of 'GetKnots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'GetKnots' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_200)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'GetKnots'
    return stypy_return_type_200

# Assigning a type to the variable 'GetKnots' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'GetKnots', GetKnots)
# Declaration of the 'Spline' class

class Spline(object, ):
    str_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'str', 'Class for representing B-Splines and NURBS of arbitrary degree')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 40), 'int')
        # Getting the type of 'None' (line 62)
        None_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'None')
        defaults = [int_202, None_203]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.__init__', ['points', 'degree', 'knots'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['points', 'degree', 'knots'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', 'Creates a Spline. points is a list of GVector, degree is the\ndegree of the Spline.')
        
        # Type idiom detected: calculating its left and rigth part (line 65)
        # Getting the type of 'knots' (line 65)
        knots_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'knots')
        # Getting the type of 'None' (line 65)
        None_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'None')
        
        (may_be_207, more_types_in_union_208) = may_be_none(knots_205, None_206)

        if may_be_207:

            if more_types_in_union_208:
                # Runtime conditional SSA (line 65)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 66):
            
            # Assigning a Call to a Attribute (line 66):
            
            # Call to GetKnots(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'points' (line 66)
            points_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'points', False)
            # Getting the type of 'degree' (line 66)
            degree_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'degree', False)
            # Processing the call keyword arguments (line 66)
            kwargs_212 = {}
            # Getting the type of 'GetKnots' (line 66)
            GetKnots_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'GetKnots', False)
            # Calling GetKnots(args, kwargs) (line 66)
            GetKnots_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), GetKnots_209, *[points_210, degree_211], **kwargs_212)
            
            # Getting the type of 'self' (line 66)
            self_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'self')
            # Setting the type of the member 'knots' of a type (line 66)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), self_214, 'knots', GetKnots_call_result_213)

            if more_types_in_union_208:
                # Runtime conditional SSA for else branch (line 65)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_207) or more_types_in_union_208):
            
            
            
            # Call to len(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'points' (line 68)
            points_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'points', False)
            # Processing the call keyword arguments (line 68)
            kwargs_217 = {}
            # Getting the type of 'len' (line 68)
            len_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'len', False)
            # Calling len(args, kwargs) (line 68)
            len_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), len_215, *[points_216], **kwargs_217)
            
            
            # Call to len(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'knots' (line 68)
            knots_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'knots', False)
            # Processing the call keyword arguments (line 68)
            kwargs_221 = {}
            # Getting the type of 'len' (line 68)
            len_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'len', False)
            # Calling len(args, kwargs) (line 68)
            len_call_result_222 = invoke(stypy.reporting.localization.Localization(__file__, 68, 29), len_219, *[knots_220], **kwargs_221)
            
            # Getting the type of 'degree' (line 68)
            degree_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 42), 'degree')
            # Applying the binary operator '-' (line 68)
            result_sub_224 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 29), '-', len_call_result_222, degree_223)
            
            int_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 51), 'int')
            # Applying the binary operator '+' (line 68)
            result_add_226 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 49), '+', result_sub_224, int_225)
            
            # Applying the binary operator '>' (line 68)
            result_gt_227 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), '>', len_call_result_218, result_add_226)
            
            # Testing the type of an if condition (line 68)
            if_condition_228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 12), result_gt_227)
            # Assigning a type to the variable 'if_condition_228' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'if_condition_228', if_condition_228)
            # SSA begins for if statement (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 69)
            # Processing the call arguments (line 69)
            str_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'str', 'too many control points')
            # Processing the call keyword arguments (line 69)
            kwargs_231 = {}
            # Getting the type of 'ValueError' (line 69)
            ValueError_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 69)
            ValueError_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 69, 22), ValueError_229, *[str_230], **kwargs_231)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 69, 16), ValueError_call_result_232, 'raise parameter', BaseException)
            # SSA branch for the else part of an if statement (line 68)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to len(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'points' (line 70)
            points_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'points', False)
            # Processing the call keyword arguments (line 70)
            kwargs_235 = {}
            # Getting the type of 'len' (line 70)
            len_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'len', False)
            # Calling len(args, kwargs) (line 70)
            len_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), len_233, *[points_234], **kwargs_235)
            
            
            # Call to len(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'knots' (line 70)
            knots_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'knots', False)
            # Processing the call keyword arguments (line 70)
            kwargs_239 = {}
            # Getting the type of 'len' (line 70)
            len_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'len', False)
            # Calling len(args, kwargs) (line 70)
            len_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 70, 31), len_237, *[knots_238], **kwargs_239)
            
            # Getting the type of 'degree' (line 70)
            degree_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 44), 'degree')
            # Applying the binary operator '-' (line 70)
            result_sub_242 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 31), '-', len_call_result_240, degree_241)
            
            int_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 53), 'int')
            # Applying the binary operator '+' (line 70)
            result_add_244 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 51), '+', result_sub_242, int_243)
            
            # Applying the binary operator '<' (line 70)
            result_lt_245 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 17), '<', len_call_result_236, result_add_244)
            
            # Testing the type of an if condition (line 70)
            if_condition_246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 17), result_lt_245)
            # Assigning a type to the variable 'if_condition_246' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'if_condition_246', if_condition_246)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 71)
            # Processing the call arguments (line 71)
            str_248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'str', 'not enough control points')
            # Processing the call keyword arguments (line 71)
            kwargs_249 = {}
            # Getting the type of 'ValueError' (line 71)
            ValueError_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 71)
            ValueError_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), ValueError_247, *[str_248], **kwargs_249)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 16), ValueError_call_result_250, 'raise parameter', BaseException)
            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 68)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Subscript to a Name (line 72):
            
            # Assigning a Subscript to a Name (line 72):
            
            # Obtaining the type of the subscript
            int_251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'int')
            # Getting the type of 'knots' (line 72)
            knots_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'knots')
            # Obtaining the member '__getitem__' of a type (line 72)
            getitem___253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 19), knots_252, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 72)
            subscript_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 72, 19), getitem___253, int_251)
            
            # Assigning a type to the variable 'last' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'last', subscript_call_result_254)
            
            
            # Obtaining the type of the subscript
            int_255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'int')
            slice_256 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 73, 23), int_255, None, None)
            # Getting the type of 'knots' (line 73)
            knots_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'knots')
            # Obtaining the member '__getitem__' of a type (line 73)
            getitem___258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), knots_257, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 73)
            subscript_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___258, slice_256)
            
            # Testing the type of a for loop iterable (line 73)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 12), subscript_call_result_259)
            # Getting the type of the for loop variable (line 73)
            for_loop_var_260 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 12), subscript_call_result_259)
            # Assigning a type to the variable 'cur' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'cur', for_loop_var_260)
            # SSA begins for a for statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'cur' (line 74)
            cur_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'cur')
            # Getting the type of 'last' (line 74)
            last_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'last')
            # Applying the binary operator '<' (line 74)
            result_lt_263 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 19), '<', cur_261, last_262)
            
            # Testing the type of an if condition (line 74)
            if_condition_264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 16), result_lt_263)
            # Assigning a type to the variable 'if_condition_264' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'if_condition_264', if_condition_264)
            # SSA begins for if statement (line 74)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 75)
            # Processing the call arguments (line 75)
            str_266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 38), 'str', 'knots not strictly increasing')
            # Processing the call keyword arguments (line 75)
            kwargs_267 = {}
            # Getting the type of 'ValueError' (line 75)
            ValueError_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 75)
            ValueError_call_result_268 = invoke(stypy.reporting.localization.Localization(__file__, 75, 26), ValueError_265, *[str_266], **kwargs_267)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 75, 20), ValueError_call_result_268, 'raise parameter', BaseException)
            # SSA join for if statement (line 74)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Name (line 76):
            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'cur' (line 76)
            cur_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'cur')
            # Assigning a type to the variable 'last' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'last', cur_269)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 77):
            
            # Assigning a Name to a Attribute (line 77):
            # Getting the type of 'knots' (line 77)
            knots_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'knots')
            # Getting the type of 'self' (line 77)
            self_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'self')
            # Setting the type of the member 'knots' of a type (line 77)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), self_271, 'knots', knots_270)

            if (may_be_207 and more_types_in_union_208):
                # SSA join for if statement (line 65)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'points' (line 78)
        points_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'points')
        # Getting the type of 'self' (line 78)
        self_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'points' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_273, 'points', points_272)
        
        # Assigning a Name to a Attribute (line 79):
        
        # Assigning a Name to a Attribute (line 79):
        # Getting the type of 'degree' (line 79)
        degree_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'degree')
        # Getting the type of 'self' (line 79)
        self_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'degree' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_275, 'degree', degree_274)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def GetDomain(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'GetDomain'
        module_type_store = module_type_store.open_function_context('GetDomain', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.GetDomain.__dict__.__setitem__('stypy_localization', localization)
        Spline.GetDomain.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.GetDomain.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.GetDomain.__dict__.__setitem__('stypy_function_name', 'Spline.GetDomain')
        Spline.GetDomain.__dict__.__setitem__('stypy_param_names_list', [])
        Spline.GetDomain.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.GetDomain.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.GetDomain.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.GetDomain.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.GetDomain.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.GetDomain.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.GetDomain', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'GetDomain', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'GetDomain(...)' code ##################

        str_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'str', 'Returns the domain of the B-Spline')
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 83)
        self_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'self')
        # Obtaining the member 'degree' of a type (line 83)
        degree_279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 27), self_278, 'degree')
        int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 41), 'int')
        # Applying the binary operator '-' (line 83)
        result_sub_281 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 27), '-', degree_279, int_280)
        
        # Getting the type of 'self' (line 83)
        self_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'self')
        # Obtaining the member 'knots' of a type (line 83)
        knots_283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), self_282, 'knots')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), knots_283, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), getitem___284, result_sub_281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 16), tuple_277, subscript_call_result_285)
        # Adding element type (line 83)
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'self' (line 84)
        self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'self', False)
        # Obtaining the member 'knots' of a type (line 84)
        knots_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 31), self_287, 'knots')
        # Processing the call keyword arguments (line 84)
        kwargs_289 = {}
        # Getting the type of 'len' (line 84)
        len_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'len', False)
        # Calling len(args, kwargs) (line 84)
        len_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), len_286, *[knots_288], **kwargs_289)
        
        # Getting the type of 'self' (line 84)
        self_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 45), 'self')
        # Obtaining the member 'degree' of a type (line 84)
        degree_292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 45), self_291, 'degree')
        # Applying the binary operator '-' (line 84)
        result_sub_293 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 27), '-', len_call_result_290, degree_292)
        
        # Getting the type of 'self' (line 84)
        self_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'self')
        # Obtaining the member 'knots' of a type (line 84)
        knots_295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 16), self_294, 'knots')
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 16), knots_295, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 84, 16), getitem___296, result_sub_293)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 16), tuple_277, subscript_call_result_297)
        
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', tuple_277)
        
        # ################# End of 'GetDomain(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'GetDomain' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_298)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'GetDomain'
        return stypy_return_type_298


    @norecursion
    def call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'call'
        module_type_store = module_type_store.open_function_context('call', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.call.__dict__.__setitem__('stypy_localization', localization)
        Spline.call.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.call.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.call.__dict__.__setitem__('stypy_function_name', 'Spline.call')
        Spline.call.__dict__.__setitem__('stypy_param_names_list', ['u'])
        Spline.call.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.call.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.call.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.call.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.call.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.call.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.call', ['u'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'call', localization, ['u'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'call(...)' code ##################

        str_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'str', 'Calculates a point of the B-Spline using de Boors Algorithm')
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to GetDomain(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_302 = {}
        # Getting the type of 'self' (line 88)
        self_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'self', False)
        # Obtaining the member 'GetDomain' of a type (line 88)
        GetDomain_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 14), self_300, 'GetDomain')
        # Calling GetDomain(args, kwargs) (line 88)
        GetDomain_call_result_303 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), GetDomain_301, *[], **kwargs_302)
        
        # Assigning a type to the variable 'dom' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'dom', GetDomain_call_result_303)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'u' (line 89)
        u_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'u')
        
        # Obtaining the type of the subscript
        int_305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'int')
        # Getting the type of 'dom' (line 89)
        dom_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'dom')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), dom_306, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), getitem___307, int_305)
        
        # Applying the binary operator '<' (line 89)
        result_lt_309 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), '<', u_304, subscript_call_result_308)
        
        
        # Getting the type of 'u' (line 89)
        u_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'u')
        
        # Obtaining the type of the subscript
        int_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 33), 'int')
        # Getting the type of 'dom' (line 89)
        dom_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'dom')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 29), dom_312, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_314 = invoke(stypy.reporting.localization.Localization(__file__, 89, 29), getitem___313, int_311)
        
        # Applying the binary operator '>' (line 89)
        result_gt_315 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 25), '>', u_310, subscript_call_result_314)
        
        # Applying the binary operator 'or' (line 89)
        result_or_keyword_316 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), 'or', result_lt_309, result_gt_315)
        
        # Testing the type of an if condition (line 89)
        if_condition_317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), result_or_keyword_316)
        # Assigning a type to the variable 'if_condition_317' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_317', if_condition_317)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 90)
        # Processing the call arguments (line 90)
        str_319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'str', 'Function value not in domain')
        # Processing the call keyword arguments (line 90)
        kwargs_320 = {}
        # Getting the type of 'ValueError' (line 90)
        ValueError_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 90)
        ValueError_call_result_321 = invoke(stypy.reporting.localization.Localization(__file__, 90, 18), ValueError_318, *[str_319], **kwargs_320)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 90, 12), ValueError_call_result_321, 'raise parameter', BaseException)
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'u' (line 91)
        u_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'u')
        
        # Obtaining the type of the subscript
        int_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 20), 'int')
        # Getting the type of 'dom' (line 91)
        dom_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'dom')
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 16), dom_324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 91, 16), getitem___325, int_323)
        
        # Applying the binary operator '==' (line 91)
        result_eq_327 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), '==', u_322, subscript_call_result_326)
        
        # Testing the type of an if condition (line 91)
        if_condition_328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), result_eq_327)
        # Assigning a type to the variable 'if_condition_328' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_328', if_condition_328)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        int_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 31), 'int')
        # Getting the type of 'self' (line 92)
        self_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'self')
        # Obtaining the member 'points' of a type (line 92)
        points_331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), self_330, 'points')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), points_331, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 92, 19), getitem___332, int_329)
        
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'stypy_return_type', subscript_call_result_333)
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'u' (line 93)
        u_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'u')
        
        # Obtaining the type of the subscript
        int_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'int')
        # Getting the type of 'dom' (line 93)
        dom_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'dom')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), dom_336, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), getitem___337, int_335)
        
        # Applying the binary operator '==' (line 93)
        result_eq_339 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), '==', u_334, subscript_call_result_338)
        
        # Testing the type of an if condition (line 93)
        if_condition_340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 8), result_eq_339)
        # Assigning a type to the variable 'if_condition_340' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'if_condition_340', if_condition_340)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        int_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 31), 'int')
        # Getting the type of 'self' (line 94)
        self_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'self')
        # Obtaining the member 'points' of a type (line 94)
        points_343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 19), self_342, 'points')
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 19), points_343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 94, 19), getitem___344, int_341)
        
        # Assigning a type to the variable 'stypy_return_type' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'stypy_return_type', subscript_call_result_345)
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to GetIndex(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'u' (line 95)
        u_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'u', False)
        # Processing the call keyword arguments (line 95)
        kwargs_349 = {}
        # Getting the type of 'self' (line 95)
        self_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'self', False)
        # Obtaining the member 'GetIndex' of a type (line 95)
        GetIndex_347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), self_346, 'GetIndex')
        # Calling GetIndex(args, kwargs) (line 95)
        GetIndex_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), GetIndex_347, *[u_348], **kwargs_349)
        
        # Assigning a type to the variable 'I' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'I', GetIndex_call_result_350)
        
        # Assigning a ListComp to a Name (line 96):
        
        # Assigning a ListComp to a Name (line 96):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'self' (line 97)
        self_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'self', False)
        # Obtaining the member 'degree' of a type (line 97)
        degree_365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 29), self_364, 'degree')
        int_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 43), 'int')
        # Applying the binary operator '+' (line 97)
        result_add_367 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 29), '+', degree_365, int_366)
        
        # Processing the call keyword arguments (line 97)
        kwargs_368 = {}
        # Getting the type of 'range' (line 97)
        range_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'range', False)
        # Calling range(args, kwargs) (line 97)
        range_call_result_369 = invoke(stypy.reporting.localization.Localization(__file__, 97, 23), range_363, *[result_add_367], **kwargs_368)
        
        comprehension_370 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 13), range_call_result_369)
        # Assigning a type to the variable 'ii' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'ii', comprehension_370)
        
        # Obtaining the type of the subscript
        # Getting the type of 'I' (line 96)
        I_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'I')
        # Getting the type of 'self' (line 96)
        self_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'self')
        # Obtaining the member 'degree' of a type (line 96)
        degree_353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 29), self_352, 'degree')
        # Applying the binary operator '-' (line 96)
        result_sub_354 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 25), '-', I_351, degree_353)
        
        int_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 43), 'int')
        # Applying the binary operator '+' (line 96)
        result_add_356 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 41), '+', result_sub_354, int_355)
        
        # Getting the type of 'ii' (line 96)
        ii_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 47), 'ii')
        # Applying the binary operator '+' (line 96)
        result_add_358 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 45), '+', result_add_356, ii_357)
        
        # Getting the type of 'self' (line 96)
        self_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'self')
        # Obtaining the member 'points' of a type (line 96)
        points_360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), self_359, 'points')
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), points_360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), getitem___361, result_add_358)
        
        list_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 13), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 13), list_371, subscript_call_result_362)
        # Assigning a type to the variable 'd' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'd', list_371)
        
        # Assigning a Attribute to a Name (line 98):
        
        # Assigning a Attribute to a Name (line 98):
        # Getting the type of 'self' (line 98)
        self_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self')
        # Obtaining the member 'knots' of a type (line 98)
        knots_373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_372, 'knots')
        # Assigning a type to the variable 'U' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'U', knots_373)
        
        
        # Call to range(...): (line 99)
        # Processing the call arguments (line 99)
        int_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 24), 'int')
        # Getting the type of 'self' (line 99)
        self_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'self', False)
        # Obtaining the member 'degree' of a type (line 99)
        degree_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 27), self_376, 'degree')
        int_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 41), 'int')
        # Applying the binary operator '+' (line 99)
        result_add_379 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 27), '+', degree_377, int_378)
        
        # Processing the call keyword arguments (line 99)
        kwargs_380 = {}
        # Getting the type of 'range' (line 99)
        range_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'range', False)
        # Calling range(args, kwargs) (line 99)
        range_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 99, 18), range_374, *[int_375, result_add_379], **kwargs_380)
        
        # Testing the type of a for loop iterable (line 99)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 8), range_call_result_381)
        # Getting the type of the for loop variable (line 99)
        for_loop_var_382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 8), range_call_result_381)
        # Assigning a type to the variable 'ik' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'ik', for_loop_var_382)
        # SSA begins for a for statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'I' (line 100)
        I_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'I', False)
        # Getting the type of 'self' (line 100)
        self_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'self', False)
        # Obtaining the member 'degree' of a type (line 100)
        degree_386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 32), self_385, 'degree')
        # Applying the binary operator '-' (line 100)
        result_sub_387 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 28), '-', I_384, degree_386)
        
        # Getting the type of 'ik' (line 100)
        ik_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 46), 'ik', False)
        # Applying the binary operator '+' (line 100)
        result_add_389 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 44), '+', result_sub_387, ik_388)
        
        int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 51), 'int')
        # Applying the binary operator '+' (line 100)
        result_add_391 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 49), '+', result_add_389, int_390)
        
        # Getting the type of 'I' (line 100)
        I_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 54), 'I', False)
        int_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 58), 'int')
        # Applying the binary operator '+' (line 100)
        result_add_394 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 54), '+', I_392, int_393)
        
        # Processing the call keyword arguments (line 100)
        kwargs_395 = {}
        # Getting the type of 'range' (line 100)
        range_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'range', False)
        # Calling range(args, kwargs) (line 100)
        range_call_result_396 = invoke(stypy.reporting.localization.Localization(__file__, 100, 22), range_383, *[result_add_391, result_add_394], **kwargs_395)
        
        # Testing the type of a for loop iterable (line 100)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 100, 12), range_call_result_396)
        # Getting the type of the for loop variable (line 100)
        for_loop_var_397 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 100, 12), range_call_result_396)
        # Assigning a type to the variable 'ii' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ii', for_loop_var_397)
        # SSA begins for a for statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 101):
        
        # Assigning a Subscript to a Name (line 101):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ii' (line 101)
        ii_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'ii')
        # Getting the type of 'self' (line 101)
        self_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'self')
        # Obtaining the member 'degree' of a type (line 101)
        degree_400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), self_399, 'degree')
        # Applying the binary operator '+' (line 101)
        result_add_401 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 23), '+', ii_398, degree_400)
        
        # Getting the type of 'ik' (line 101)
        ik_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'ik')
        # Applying the binary operator '-' (line 101)
        result_sub_403 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 40), '-', result_add_401, ik_402)
        
        # Getting the type of 'U' (line 101)
        U_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'U')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 21), U_404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_406 = invoke(stypy.reporting.localization.Localization(__file__, 101, 21), getitem___405, result_sub_403)
        
        # Assigning a type to the variable 'ua' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'ua', subscript_call_result_406)
        
        # Assigning a Subscript to a Name (line 102):
        
        # Assigning a Subscript to a Name (line 102):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ii' (line 102)
        ii_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'ii')
        int_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 28), 'int')
        # Applying the binary operator '-' (line 102)
        result_sub_409 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 23), '-', ii_407, int_408)
        
        # Getting the type of 'U' (line 102)
        U_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 21), 'U')
        # Obtaining the member '__getitem__' of a type (line 102)
        getitem___411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 21), U_410, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 102)
        subscript_call_result_412 = invoke(stypy.reporting.localization.Localization(__file__, 102, 21), getitem___411, result_sub_409)
        
        # Assigning a type to the variable 'ub' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'ub', subscript_call_result_412)
        
        # Assigning a BinOp to a Name (line 103):
        
        # Assigning a BinOp to a Name (line 103):
        # Getting the type of 'ua' (line 103)
        ua_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'ua')
        # Getting the type of 'u' (line 103)
        u_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'u')
        # Applying the binary operator '-' (line 103)
        result_sub_415 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 23), '-', ua_413, u_414)
        
        
        # Call to float(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'ua' (line 103)
        ua_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'ua', False)
        # Getting the type of 'ub' (line 103)
        ub_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 44), 'ub', False)
        # Applying the binary operator '-' (line 103)
        result_sub_419 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 39), '-', ua_417, ub_418)
        
        # Processing the call keyword arguments (line 103)
        kwargs_420 = {}
        # Getting the type of 'float' (line 103)
        float_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 33), 'float', False)
        # Calling float(args, kwargs) (line 103)
        float_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 103, 33), float_416, *[result_sub_419], **kwargs_420)
        
        # Applying the binary operator 'div' (line 103)
        result_div_422 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 22), 'div', result_sub_415, float_call_result_421)
        
        # Assigning a type to the variable 'co1' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'co1', result_div_422)
        
        # Assigning a BinOp to a Name (line 104):
        
        # Assigning a BinOp to a Name (line 104):
        # Getting the type of 'u' (line 104)
        u_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'u')
        # Getting the type of 'ub' (line 104)
        ub_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'ub')
        # Applying the binary operator '-' (line 104)
        result_sub_425 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 23), '-', u_423, ub_424)
        
        
        # Call to float(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'ua' (line 104)
        ua_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'ua', False)
        # Getting the type of 'ub' (line 104)
        ub_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 44), 'ub', False)
        # Applying the binary operator '-' (line 104)
        result_sub_429 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 39), '-', ua_427, ub_428)
        
        # Processing the call keyword arguments (line 104)
        kwargs_430 = {}
        # Getting the type of 'float' (line 104)
        float_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'float', False)
        # Calling float(args, kwargs) (line 104)
        float_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 104, 33), float_426, *[result_sub_429], **kwargs_430)
        
        # Applying the binary operator 'div' (line 104)
        result_div_432 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 22), 'div', result_sub_425, float_call_result_431)
        
        # Assigning a type to the variable 'co2' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'co2', result_div_432)
        
        # Assigning a BinOp to a Name (line 105):
        
        # Assigning a BinOp to a Name (line 105):
        # Getting the type of 'ii' (line 105)
        ii_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'ii')
        # Getting the type of 'I' (line 105)
        I_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'I')
        # Applying the binary operator '-' (line 105)
        result_sub_435 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 24), '-', ii_433, I_434)
        
        # Getting the type of 'self' (line 105)
        self_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'self')
        # Obtaining the member 'degree' of a type (line 105)
        degree_437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 33), self_436, 'degree')
        # Applying the binary operator '+' (line 105)
        result_add_438 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 31), '+', result_sub_435, degree_437)
        
        # Getting the type of 'ik' (line 105)
        ik_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 47), 'ik')
        # Applying the binary operator '-' (line 105)
        result_sub_440 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 45), '-', result_add_438, ik_439)
        
        int_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 52), 'int')
        # Applying the binary operator '-' (line 105)
        result_sub_442 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 50), '-', result_sub_440, int_441)
        
        # Assigning a type to the variable 'index' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'index', result_sub_442)
        
        # Assigning a Call to a Subscript (line 106):
        
        # Assigning a Call to a Subscript (line 106):
        
        # Call to linear_combination(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 106)
        index_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 57), 'index', False)
        int_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 65), 'int')
        # Applying the binary operator '+' (line 106)
        result_add_450 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 57), '+', index_448, int_449)
        
        # Getting the type of 'd' (line 106)
        d_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 55), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 55), d_451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_453 = invoke(stypy.reporting.localization.Localization(__file__, 106, 55), getitem___452, result_add_450)
        
        # Getting the type of 'co1' (line 106)
        co1_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 69), 'co1', False)
        # Getting the type of 'co2' (line 106)
        co2_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 74), 'co2', False)
        # Processing the call keyword arguments (line 106)
        kwargs_456 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 106)
        index_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'index', False)
        # Getting the type of 'd' (line 106)
        d_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 27), d_444, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 106)
        subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 106, 27), getitem___445, index_443)
        
        # Obtaining the member 'linear_combination' of a type (line 106)
        linear_combination_447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 27), subscript_call_result_446, 'linear_combination')
        # Calling linear_combination(args, kwargs) (line 106)
        linear_combination_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 106, 27), linear_combination_447, *[subscript_call_result_453, co1_454, co2_455], **kwargs_456)
        
        # Getting the type of 'd' (line 106)
        d_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'd')
        # Getting the type of 'index' (line 106)
        index_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'index')
        # Storing an element on a container (line 106)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 16), d_458, (index_459, linear_combination_call_result_457))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 17), 'int')
        # Getting the type of 'd' (line 107)
        d_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'd')
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 15), d_461, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 107, 15), getitem___462, int_460)
        
        # Assigning a type to the variable 'stypy_return_type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', subscript_call_result_463)
        
        # ################# End of 'call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'call' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_464)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'call'
        return stypy_return_type_464


    @norecursion
    def GetIndex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'GetIndex'
        module_type_store = module_type_store.open_function_context('GetIndex', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.GetIndex.__dict__.__setitem__('stypy_localization', localization)
        Spline.GetIndex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.GetIndex.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.GetIndex.__dict__.__setitem__('stypy_function_name', 'Spline.GetIndex')
        Spline.GetIndex.__dict__.__setitem__('stypy_param_names_list', ['u'])
        Spline.GetIndex.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.GetIndex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.GetIndex.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.GetIndex.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.GetIndex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.GetIndex.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.GetIndex', ['u'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'GetIndex', localization, ['u'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'GetIndex(...)' code ##################

        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to GetDomain(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_467 = {}
        # Getting the type of 'self' (line 110)
        self_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'self', False)
        # Obtaining the member 'GetDomain' of a type (line 110)
        GetDomain_466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 14), self_465, 'GetDomain')
        # Calling GetDomain(args, kwargs) (line 110)
        GetDomain_call_result_468 = invoke(stypy.reporting.localization.Localization(__file__, 110, 14), GetDomain_466, *[], **kwargs_467)
        
        # Assigning a type to the variable 'dom' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'dom', GetDomain_call_result_468)
        
        
        # Call to range(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'self', False)
        # Obtaining the member 'degree' of a type (line 111)
        degree_471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), self_470, 'degree')
        int_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'int')
        # Applying the binary operator '-' (line 111)
        result_sub_473 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 24), '-', degree_471, int_472)
        
        
        # Call to len(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 45), 'self', False)
        # Obtaining the member 'knots' of a type (line 111)
        knots_476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 45), self_475, 'knots')
        # Processing the call keyword arguments (line 111)
        kwargs_477 = {}
        # Getting the type of 'len' (line 111)
        len_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'len', False)
        # Calling len(args, kwargs) (line 111)
        len_call_result_478 = invoke(stypy.reporting.localization.Localization(__file__, 111, 41), len_474, *[knots_476], **kwargs_477)
        
        # Getting the type of 'self' (line 111)
        self_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 59), 'self', False)
        # Obtaining the member 'degree' of a type (line 111)
        degree_480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 59), self_479, 'degree')
        # Applying the binary operator '-' (line 111)
        result_sub_481 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 41), '-', len_call_result_478, degree_480)
        
        # Processing the call keyword arguments (line 111)
        kwargs_482 = {}
        # Getting the type of 'range' (line 111)
        range_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'range', False)
        # Calling range(args, kwargs) (line 111)
        range_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 111, 18), range_469, *[result_sub_473, result_sub_481], **kwargs_482)
        
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 8), range_call_result_483)
        # Getting the type of the for loop variable (line 111)
        for_loop_var_484 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 8), range_call_result_483)
        # Assigning a type to the variable 'ii' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'ii', for_loop_var_484)
        # SSA begins for a for statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'u' (line 112)
        u_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'u')
        
        # Obtaining the type of the subscript
        # Getting the type of 'ii' (line 112)
        ii_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'ii')
        # Getting the type of 'self' (line 112)
        self_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'self')
        # Obtaining the member 'knots' of a type (line 112)
        knots_488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), self_487, 'knots')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), knots_488, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_490 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), getitem___489, ii_486)
        
        # Applying the binary operator '>=' (line 112)
        result_ge_491 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), '>=', u_485, subscript_call_result_490)
        
        
        # Getting the type of 'u' (line 112)
        u_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 39), 'u')
        
        # Obtaining the type of the subscript
        # Getting the type of 'ii' (line 112)
        ii_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 54), 'ii')
        int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 59), 'int')
        # Applying the binary operator '+' (line 112)
        result_add_495 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 54), '+', ii_493, int_494)
        
        # Getting the type of 'self' (line 112)
        self_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 'self')
        # Obtaining the member 'knots' of a type (line 112)
        knots_497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 43), self_496, 'knots')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 43), knots_497, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_499 = invoke(stypy.reporting.localization.Localization(__file__, 112, 43), getitem___498, result_add_495)
        
        # Applying the binary operator '<' (line 112)
        result_lt_500 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 39), '<', u_492, subscript_call_result_499)
        
        # Applying the binary operator 'and' (line 112)
        result_and_keyword_501 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), 'and', result_ge_491, result_lt_500)
        
        # Testing the type of an if condition (line 112)
        if_condition_502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 12), result_and_keyword_501)
        # Assigning a type to the variable 'if_condition_502' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'if_condition_502', if_condition_502)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 113):
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'ii' (line 113)
        ii_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'ii')
        # Assigning a type to the variable 'I' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'I', ii_503)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 111)
        module_type_store.open_ssa_branch('for loop else')
        
        # Assigning a BinOp to a Name (line 116):
        
        # Assigning a BinOp to a Name (line 116):
        
        # Obtaining the type of the subscript
        int_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
        # Getting the type of 'dom' (line 116)
        dom_505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'dom')
        # Obtaining the member '__getitem__' of a type (line 116)
        getitem___506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), dom_505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 116)
        subscript_call_result_507 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), getitem___506, int_504)
        
        int_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'int')
        # Applying the binary operator '-' (line 116)
        result_sub_509 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '-', subscript_call_result_507, int_508)
        
        # Assigning a type to the variable 'I' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'I', result_sub_509)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'I' (line 117)
        I_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'I')
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', I_510)
        
        # ################# End of 'GetIndex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'GetIndex' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_511)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'GetIndex'
        return stypy_return_type_511


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.__len__.__dict__.__setitem__('stypy_localization', localization)
        Spline.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.__len__.__dict__.__setitem__('stypy_function_name', 'Spline.__len__')
        Spline.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        Spline.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        
        # Call to len(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'self', False)
        # Obtaining the member 'points' of a type (line 120)
        points_514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 19), self_513, 'points')
        # Processing the call keyword arguments (line 120)
        kwargs_515 = {}
        # Getting the type of 'len' (line 120)
        len_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'len', False)
        # Calling len(args, kwargs) (line 120)
        len_call_result_516 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), len_512, *[points_514], **kwargs_515)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', len_call_result_516)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_517)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_517


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Spline.stypy__repr__')
        Spline.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Spline.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 15), 'str', 'Spline(%r, %r, %r)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 'self' (line 123)
        self_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'self')
        # Obtaining the member 'points' of a type (line 123)
        points_521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 39), self_520, 'points')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 39), tuple_519, points_521)
        # Adding element type (line 123)
        # Getting the type of 'self' (line 123)
        self_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 52), 'self')
        # Obtaining the member 'degree' of a type (line 123)
        degree_523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 52), self_522, 'degree')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 39), tuple_519, degree_523)
        # Adding element type (line 123)
        # Getting the type of 'self' (line 123)
        self_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 65), 'self')
        # Obtaining the member 'knots' of a type (line 123)
        knots_525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 65), self_524, 'knots')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 39), tuple_519, knots_525)
        
        # Applying the binary operator '%' (line 123)
        result_mod_526 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), '%', str_518, tuple_519)
        
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', result_mod_526)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_527)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_527


# Assigning a type to the variable 'Spline' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'Spline', Spline)

@norecursion
def save_im(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'save_im'
    module_type_store = module_type_store.open_function_context('save_im', 126, 0, False)
    
    # Passed parameters checking function
    save_im.stypy_localization = localization
    save_im.stypy_type_of_self = None
    save_im.stypy_type_store = module_type_store
    save_im.stypy_function_name = 'save_im'
    save_im.stypy_param_names_list = ['im', 'fn']
    save_im.stypy_varargs_param_name = None
    save_im.stypy_kwargs_param_name = None
    save_im.stypy_call_defaults = defaults
    save_im.stypy_call_varargs = varargs
    save_im.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'save_im', ['im', 'fn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'save_im', localization, ['im', 'fn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'save_im(...)' code ##################

    
    # Assigning a Call to a Name (line 127):
    
    # Assigning a Call to a Name (line 127):
    
    # Call to open(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'fn' (line 127)
    fn_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'fn', False)
    str_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'str', 'wb')
    # Processing the call keyword arguments (line 127)
    kwargs_531 = {}
    # Getting the type of 'open' (line 127)
    open_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'open', False)
    # Calling open(args, kwargs) (line 127)
    open_call_result_532 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), open_528, *[fn_529, str_530], **kwargs_531)
    
    # Assigning a type to the variable 'f' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'f', open_call_result_532)
    
    # Assigning a Str to a Name (line 128):
    
    # Assigning a Str to a Name (line 128):
    str_533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 12), 'str', 'P6\n')
    # Assigning a type to the variable 'magic' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'magic', str_533)
    
    # Assigning a Num to a Name (line 129):
    
    # Assigning a Num to a Name (line 129):
    int_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 13), 'int')
    # Assigning a type to the variable 'maxval' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'maxval', int_534)
    
    # Assigning a Call to a Name (line 130):
    
    # Assigning a Call to a Name (line 130):
    
    # Call to len(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'im' (line 130)
    im_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'im', False)
    # Processing the call keyword arguments (line 130)
    kwargs_537 = {}
    # Getting the type of 'len' (line 130)
    len_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'len', False)
    # Calling len(args, kwargs) (line 130)
    len_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), len_535, *[im_536], **kwargs_537)
    
    # Assigning a type to the variable 'w' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'w', len_call_result_538)
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to len(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Obtaining the type of the subscript
    int_540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'int')
    # Getting the type of 'im' (line 131)
    im_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'im', False)
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), im_541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_543 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), getitem___542, int_540)
    
    # Processing the call keyword arguments (line 131)
    kwargs_544 = {}
    # Getting the type of 'len' (line 131)
    len_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'len', False)
    # Calling len(args, kwargs) (line 131)
    len_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), len_539, *[subscript_call_result_543], **kwargs_544)
    
    # Assigning a type to the variable 'h' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'h', len_call_result_545)
    
    # Call to write(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'magic' (line 132)
    magic_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'magic', False)
    # Processing the call keyword arguments (line 132)
    kwargs_549 = {}
    # Getting the type of 'f' (line 132)
    f_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 132)
    write_547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), f_546, 'write')
    # Calling write(args, kwargs) (line 132)
    write_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), write_547, *[magic_548], **kwargs_549)
    
    
    # Call to write(...): (line 133)
    # Processing the call arguments (line 133)
    str_553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 12), 'str', '%i %i\n%i\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    # Adding element type (line 133)
    # Getting the type of 'w' (line 133)
    w_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'w', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 29), tuple_554, w_555)
    # Adding element type (line 133)
    # Getting the type of 'h' (line 133)
    h_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 32), 'h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 29), tuple_554, h_556)
    # Adding element type (line 133)
    # Getting the type of 'maxval' (line 133)
    maxval_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 35), 'maxval', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 29), tuple_554, maxval_557)
    
    # Applying the binary operator '%' (line 133)
    result_mod_558 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 12), '%', str_553, tuple_554)
    
    # Processing the call keyword arguments (line 133)
    kwargs_559 = {}
    # Getting the type of 'f' (line 133)
    f_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 133)
    write_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), f_551, 'write')
    # Calling write(args, kwargs) (line 133)
    write_call_result_560 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), write_552, *[result_mod_558], **kwargs_559)
    
    
    
    # Call to range(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'h' (line 134)
    h_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 19), 'h', False)
    # Processing the call keyword arguments (line 134)
    kwargs_563 = {}
    # Getting the type of 'range' (line 134)
    range_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'range', False)
    # Calling range(args, kwargs) (line 134)
    range_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), range_561, *[h_562], **kwargs_563)
    
    # Testing the type of a for loop iterable (line 134)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 4), range_call_result_564)
    # Getting the type of the for loop variable (line 134)
    for_loop_var_565 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 4), range_call_result_564)
    # Assigning a type to the variable 'j' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'j', for_loop_var_565)
    # SSA begins for a for statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'w' (line 135)
    w_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'w', False)
    # Processing the call keyword arguments (line 135)
    kwargs_568 = {}
    # Getting the type of 'range' (line 135)
    range_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'range', False)
    # Calling range(args, kwargs) (line 135)
    range_call_result_569 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), range_566, *[w_567], **kwargs_568)
    
    # Testing the type of a for loop iterable (line 135)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_569)
    # Getting the type of the for loop variable (line 135)
    for_loop_var_570 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_569)
    # Assigning a type to the variable 'i' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'i', for_loop_var_570)
    # SSA begins for a for statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 136):
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 136)
    j_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'j')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 136)
    i_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'i')
    # Getting the type of 'im' (line 136)
    im_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'im')
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 18), im_573, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_575 = invoke(stypy.reporting.localization.Localization(__file__, 136, 18), getitem___574, i_572)
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 18), subscript_call_result_575, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_577 = invoke(stypy.reporting.localization.Localization(__file__, 136, 18), getitem___576, j_571)
    
    # Assigning a type to the variable 'val' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'val', subscript_call_result_577)
    
    # Assigning a BinOp to a Name (line 137):
    
    # Assigning a BinOp to a Name (line 137):
    # Getting the type of 'val' (line 137)
    val_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'val')
    int_579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'int')
    # Applying the binary operator '*' (line 137)
    result_mul_580 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 16), '*', val_578, int_579)
    
    # Assigning a type to the variable 'c' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'c', result_mul_580)
    
    # Call to write(...): (line 138)
    # Processing the call arguments (line 138)
    str_583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'str', '%c%c%c')
    
    # Obtaining an instance of the builtin type 'tuple' (line 138)
    tuple_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 138)
    # Adding element type (line 138)
    # Getting the type of 'c' (line 138)
    c_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 32), tuple_584, c_585)
    # Adding element type (line 138)
    # Getting the type of 'c' (line 138)
    c_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 35), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 32), tuple_584, c_586)
    # Adding element type (line 138)
    # Getting the type of 'c' (line 138)
    c_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 38), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 32), tuple_584, c_587)
    
    # Applying the binary operator '%' (line 138)
    result_mod_588 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 20), '%', str_583, tuple_584)
    
    # Processing the call keyword arguments (line 138)
    kwargs_589 = {}
    # Getting the type of 'f' (line 138)
    f_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'f', False)
    # Obtaining the member 'write' of a type (line 138)
    write_582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), f_581, 'write')
    # Calling write(args, kwargs) (line 138)
    write_call_result_590 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), write_582, *[result_mod_588], **kwargs_589)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 139)
    # Processing the call keyword arguments (line 139)
    kwargs_593 = {}
    # Getting the type of 'f' (line 139)
    f_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 139)
    close_592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 4), f_591, 'close')
    # Calling close(args, kwargs) (line 139)
    close_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 139, 4), close_592, *[], **kwargs_593)
    
    
    # ################# End of 'save_im(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'save_im' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_595)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'save_im'
    return stypy_return_type_595

# Assigning a type to the variable 'save_im' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'save_im', save_im)
# Declaration of the 'Chaosgame' class

class Chaosgame(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 42), 'float')
        defaults = [float_596]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.__init__', ['splines', 'thickness'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['splines', 'thickness'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 143):
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'splines' (line 143)
        splines_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'splines')
        # Getting the type of 'self' (line 143)
        self_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member 'splines' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_598, 'splines', splines_597)
        
        # Assigning a Name to a Attribute (line 144):
        
        # Assigning a Name to a Attribute (line 144):
        # Getting the type of 'thickness' (line 144)
        thickness_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'thickness')
        # Getting the type of 'self' (line 144)
        self_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self')
        # Setting the type of the member 'thickness' of a type (line 144)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_600, 'thickness', thickness_599)
        
        # Assigning a Call to a Attribute (line 145):
        
        # Assigning a Call to a Attribute (line 145):
        
        # Call to min(...): (line 145)
        # Processing the call arguments (line 145)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'splines' (line 145)
        splines_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 40), 'splines', False)
        comprehension_605 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), splines_604)
        # Assigning a type to the variable 'spl' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'spl', comprehension_605)
        # Calculating comprehension expression
        # Getting the type of 'spl' (line 145)
        spl_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 57), 'spl', False)
        # Obtaining the member 'points' of a type (line 145)
        points_607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 57), spl_606, 'points')
        comprehension_608 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), points_607)
        # Assigning a type to the variable 'p' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'p', comprehension_608)
        # Getting the type of 'p' (line 145)
        p_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'p', False)
        # Obtaining the member 'x' of a type (line 145)
        x_603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), p_602, 'x')
        list_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), list_609, x_603)
        # Processing the call keyword arguments (line 145)
        kwargs_610 = {}
        # Getting the type of 'min' (line 145)
        min_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'min', False)
        # Calling min(args, kwargs) (line 145)
        min_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 145, 20), min_601, *[list_609], **kwargs_610)
        
        # Getting the type of 'self' (line 145)
        self_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'minx' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_612, 'minx', min_call_result_611)
        
        # Assigning a Call to a Attribute (line 146):
        
        # Assigning a Call to a Attribute (line 146):
        
        # Call to min(...): (line 146)
        # Processing the call arguments (line 146)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'splines' (line 146)
        splines_616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'splines', False)
        comprehension_617 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 25), splines_616)
        # Assigning a type to the variable 'spl' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'spl', comprehension_617)
        # Calculating comprehension expression
        # Getting the type of 'spl' (line 146)
        spl_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 57), 'spl', False)
        # Obtaining the member 'points' of a type (line 146)
        points_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 57), spl_618, 'points')
        comprehension_620 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 25), points_619)
        # Assigning a type to the variable 'p' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'p', comprehension_620)
        # Getting the type of 'p' (line 146)
        p_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'p', False)
        # Obtaining the member 'y' of a type (line 146)
        y_615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 25), p_614, 'y')
        list_621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 25), list_621, y_615)
        # Processing the call keyword arguments (line 146)
        kwargs_622 = {}
        # Getting the type of 'min' (line 146)
        min_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'min', False)
        # Calling min(args, kwargs) (line 146)
        min_call_result_623 = invoke(stypy.reporting.localization.Localization(__file__, 146, 20), min_613, *[list_621], **kwargs_622)
        
        # Getting the type of 'self' (line 146)
        self_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'miny' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_624, 'miny', min_call_result_623)
        
        # Assigning a Call to a Attribute (line 147):
        
        # Assigning a Call to a Attribute (line 147):
        
        # Call to max(...): (line 147)
        # Processing the call arguments (line 147)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'splines' (line 147)
        splines_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 40), 'splines', False)
        comprehension_629 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), splines_628)
        # Assigning a type to the variable 'spl' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'spl', comprehension_629)
        # Calculating comprehension expression
        # Getting the type of 'spl' (line 147)
        spl_630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 57), 'spl', False)
        # Obtaining the member 'points' of a type (line 147)
        points_631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 57), spl_630, 'points')
        comprehension_632 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), points_631)
        # Assigning a type to the variable 'p' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'p', comprehension_632)
        # Getting the type of 'p' (line 147)
        p_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'p', False)
        # Obtaining the member 'x' of a type (line 147)
        x_627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), p_626, 'x')
        list_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), list_633, x_627)
        # Processing the call keyword arguments (line 147)
        kwargs_634 = {}
        # Getting the type of 'max' (line 147)
        max_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'max', False)
        # Calling max(args, kwargs) (line 147)
        max_call_result_635 = invoke(stypy.reporting.localization.Localization(__file__, 147, 20), max_625, *[list_633], **kwargs_634)
        
        # Getting the type of 'self' (line 147)
        self_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member 'maxx' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_636, 'maxx', max_call_result_635)
        
        # Assigning a Call to a Attribute (line 148):
        
        # Assigning a Call to a Attribute (line 148):
        
        # Call to max(...): (line 148)
        # Processing the call arguments (line 148)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'splines' (line 148)
        splines_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'splines', False)
        comprehension_641 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 25), splines_640)
        # Assigning a type to the variable 'spl' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'spl', comprehension_641)
        # Calculating comprehension expression
        # Getting the type of 'spl' (line 148)
        spl_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 57), 'spl', False)
        # Obtaining the member 'points' of a type (line 148)
        points_643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 57), spl_642, 'points')
        comprehension_644 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 25), points_643)
        # Assigning a type to the variable 'p' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'p', comprehension_644)
        # Getting the type of 'p' (line 148)
        p_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'p', False)
        # Obtaining the member 'y' of a type (line 148)
        y_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 25), p_638, 'y')
        list_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 25), list_645, y_639)
        # Processing the call keyword arguments (line 148)
        kwargs_646 = {}
        # Getting the type of 'max' (line 148)
        max_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'max', False)
        # Calling max(args, kwargs) (line 148)
        max_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), max_637, *[list_645], **kwargs_646)
        
        # Getting the type of 'self' (line 148)
        self_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self')
        # Setting the type of the member 'maxy' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_648, 'maxy', max_call_result_647)
        
        # Assigning a BinOp to a Attribute (line 149):
        
        # Assigning a BinOp to a Attribute (line 149):
        # Getting the type of 'self' (line 149)
        self_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'self')
        # Obtaining the member 'maxy' of a type (line 149)
        maxy_650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 22), self_649, 'maxy')
        # Getting the type of 'self' (line 149)
        self_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 34), 'self')
        # Obtaining the member 'miny' of a type (line 149)
        miny_652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 34), self_651, 'miny')
        # Applying the binary operator '-' (line 149)
        result_sub_653 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 22), '-', maxy_650, miny_652)
        
        # Getting the type of 'self' (line 149)
        self_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Setting the type of the member 'height' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_654, 'height', result_sub_653)
        
        # Assigning a BinOp to a Attribute (line 150):
        
        # Assigning a BinOp to a Attribute (line 150):
        # Getting the type of 'self' (line 150)
        self_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'self')
        # Obtaining the member 'maxx' of a type (line 150)
        maxx_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 21), self_655, 'maxx')
        # Getting the type of 'self' (line 150)
        self_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 33), 'self')
        # Obtaining the member 'minx' of a type (line 150)
        minx_658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 33), self_657, 'minx')
        # Applying the binary operator '-' (line 150)
        result_sub_659 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 21), '-', maxx_656, minx_658)
        
        # Getting the type of 'self' (line 150)
        self_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'width' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_660, 'width', result_sub_659)
        
        # Assigning a List to a Attribute (line 151):
        
        # Assigning a List to a Attribute (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        
        # Getting the type of 'self' (line 151)
        self_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'num_trafos' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_662, 'num_trafos', list_661)
        
        # Assigning a BinOp to a Name (line 152):
        
        # Assigning a BinOp to a Name (line 152):
        # Getting the type of 'thickness' (line 152)
        thickness_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'thickness')
        # Getting the type of 'self' (line 152)
        self_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'self')
        # Obtaining the member 'width' of a type (line 152)
        width_665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 32), self_664, 'width')
        # Applying the binary operator '*' (line 152)
        result_mul_666 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 20), '*', thickness_663, width_665)
        
        
        # Call to float(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'self' (line 152)
        self_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 51), 'self', False)
        # Obtaining the member 'height' of a type (line 152)
        height_669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 51), self_668, 'height')
        # Processing the call keyword arguments (line 152)
        kwargs_670 = {}
        # Getting the type of 'float' (line 152)
        float_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 45), 'float', False)
        # Calling float(args, kwargs) (line 152)
        float_call_result_671 = invoke(stypy.reporting.localization.Localization(__file__, 152, 45), float_667, *[height_669], **kwargs_670)
        
        # Applying the binary operator 'div' (line 152)
        result_div_672 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 43), 'div', result_mul_666, float_call_result_671)
        
        # Assigning a type to the variable 'maxlength' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'maxlength', result_div_672)
        
        # Getting the type of 'splines' (line 153)
        splines_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'splines')
        # Testing the type of a for loop iterable (line 153)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 8), splines_673)
        # Getting the type of the for loop variable (line 153)
        for_loop_var_674 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 8), splines_673)
        # Assigning a type to the variable 'spl' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'spl', for_loop_var_674)
        # SSA begins for a for statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Name (line 154):
        
        # Assigning a Num to a Name (line 154):
        int_675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'int')
        # Assigning a type to the variable 'length' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'length', int_675)
        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to call(...): (line 155)
        # Processing the call arguments (line 155)
        int_678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 28), 'int')
        # Processing the call keyword arguments (line 155)
        kwargs_679 = {}
        # Getting the type of 'spl' (line 155)
        spl_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'spl', False)
        # Obtaining the member 'call' of a type (line 155)
        call_677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), spl_676, 'call')
        # Calling call(args, kwargs) (line 155)
        call_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 155, 19), call_677, *[int_678], **kwargs_679)
        
        # Assigning a type to the variable 'curr' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'curr', call_call_result_680)
        
        
        # Call to range(...): (line 156)
        # Processing the call arguments (line 156)
        int_682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 27), 'int')
        int_683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 30), 'int')
        # Processing the call keyword arguments (line 156)
        kwargs_684 = {}
        # Getting the type of 'range' (line 156)
        range_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'range', False)
        # Calling range(args, kwargs) (line 156)
        range_call_result_685 = invoke(stypy.reporting.localization.Localization(__file__, 156, 21), range_681, *[int_682, int_683], **kwargs_684)
        
        # Testing the type of a for loop iterable (line 156)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 12), range_call_result_685)
        # Getting the type of the for loop variable (line 156)
        for_loop_var_686 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 12), range_call_result_685)
        # Assigning a type to the variable 'i' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'i', for_loop_var_686)
        # SSA begins for a for statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Name (line 157):
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'curr' (line 157)
        curr_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 23), 'curr')
        # Assigning a type to the variable 'last' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'last', curr_687)
        
        # Assigning a BinOp to a Name (line 158):
        
        # Assigning a BinOp to a Name (line 158):
        float_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 20), 'float')
        int_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'int')
        # Applying the binary operator 'div' (line 158)
        result_div_690 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 20), 'div', float_688, int_689)
        
        # Getting the type of 'i' (line 158)
        i_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'i')
        # Applying the binary operator '*' (line 158)
        result_mul_692 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 30), '*', result_div_690, i_691)
        
        # Assigning a type to the variable 't' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 't', result_mul_692)
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to call(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 't' (line 159)
        t_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 't', False)
        # Processing the call keyword arguments (line 159)
        kwargs_696 = {}
        # Getting the type of 'spl' (line 159)
        spl_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'spl', False)
        # Obtaining the member 'call' of a type (line 159)
        call_694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 23), spl_693, 'call')
        # Calling call(args, kwargs) (line 159)
        call_call_result_697 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), call_694, *[t_695], **kwargs_696)
        
        # Assigning a type to the variable 'curr' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'curr', call_call_result_697)
        
        # Getting the type of 'length' (line 160)
        length_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'length')
        
        # Call to dist(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'last' (line 160)
        last_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'last', False)
        # Processing the call keyword arguments (line 160)
        kwargs_702 = {}
        # Getting the type of 'curr' (line 160)
        curr_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'curr', False)
        # Obtaining the member 'dist' of a type (line 160)
        dist_700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 26), curr_699, 'dist')
        # Calling dist(args, kwargs) (line 160)
        dist_call_result_703 = invoke(stypy.reporting.localization.Localization(__file__, 160, 26), dist_700, *[last_701], **kwargs_702)
        
        # Applying the binary operator '+=' (line 160)
        result_iadd_704 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 16), '+=', length_698, dist_call_result_703)
        # Assigning a type to the variable 'length' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'length', result_iadd_704)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Call to max(...): (line 161)
        # Processing the call arguments (line 161)
        int_709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 39), 'int')
        
        # Call to int(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'length' (line 161)
        length_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 46), 'length', False)
        # Getting the type of 'maxlength' (line 161)
        maxlength_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 55), 'maxlength', False)
        # Applying the binary operator 'div' (line 161)
        result_div_713 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 46), 'div', length_711, maxlength_712)
        
        float_714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 67), 'float')
        # Applying the binary operator '*' (line 161)
        result_mul_715 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 65), '*', result_div_713, float_714)
        
        # Processing the call keyword arguments (line 161)
        kwargs_716 = {}
        # Getting the type of 'int' (line 161)
        int_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 42), 'int', False)
        # Calling int(args, kwargs) (line 161)
        int_call_result_717 = invoke(stypy.reporting.localization.Localization(__file__, 161, 42), int_710, *[result_mul_715], **kwargs_716)
        
        # Processing the call keyword arguments (line 161)
        kwargs_718 = {}
        # Getting the type of 'max' (line 161)
        max_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'max', False)
        # Calling max(args, kwargs) (line 161)
        max_call_result_719 = invoke(stypy.reporting.localization.Localization(__file__, 161, 35), max_708, *[int_709, int_call_result_717], **kwargs_718)
        
        # Processing the call keyword arguments (line 161)
        kwargs_720 = {}
        # Getting the type of 'self' (line 161)
        self_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 161)
        num_trafos_706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), self_705, 'num_trafos')
        # Obtaining the member 'append' of a type (line 161)
        append_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), num_trafos_706, 'append')
        # Calling append(args, kwargs) (line 161)
        append_call_result_721 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), append_707, *[max_call_result_719], **kwargs_720)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 162):
        
        # Assigning a Call to a Attribute (line 162):
        
        # Call to reduce(...): (line 162)
        # Processing the call arguments (line 162)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 162, 32, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['a', 'b']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['a', 'b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['a', 'b'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'a' (line 162)
            a_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 44), 'a', False)
            # Getting the type of 'b' (line 162)
            b_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 46), 'b', False)
            # Applying the binary operator '+' (line 162)
            result_add_725 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 44), '+', a_723, b_724)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'stypy_return_type', result_add_725)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 162)
            stypy_return_type_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_726)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_726

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 162)
        _stypy_temp_lambda_1_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), '_stypy_temp_lambda_1')
        # Getting the type of 'self' (line 162)
        self_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 49), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 162)
        num_trafos_729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 49), self_728, 'num_trafos')
        int_730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 66), 'int')
        # Processing the call keyword arguments (line 162)
        kwargs_731 = {}
        # Getting the type of 'reduce' (line 162)
        reduce_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'reduce', False)
        # Calling reduce(args, kwargs) (line 162)
        reduce_call_result_732 = invoke(stypy.reporting.localization.Localization(__file__, 162, 25), reduce_722, *[_stypy_temp_lambda_1_727, num_trafos_729, int_730], **kwargs_731)
        
        # Getting the type of 'self' (line 162)
        self_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Setting the type of the member 'num_total' of a type (line 162)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_733, 'num_total', reduce_call_result_732)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_random_trafo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_random_trafo'
        module_type_store = module_type_store.open_function_context('get_random_trafo', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_localization', localization)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_function_name', 'Chaosgame.get_random_trafo')
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_param_names_list', [])
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.get_random_trafo', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_random_trafo', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_random_trafo(...)' code ##################

        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to randrange(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to int(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'self' (line 166)
        self_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'self', False)
        # Obtaining the member 'num_total' of a type (line 166)
        num_total_738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 33), self_737, 'num_total')
        # Processing the call keyword arguments (line 166)
        kwargs_739 = {}
        # Getting the type of 'int' (line 166)
        int_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'int', False)
        # Calling int(args, kwargs) (line 166)
        int_call_result_740 = invoke(stypy.reporting.localization.Localization(__file__, 166, 29), int_736, *[num_total_738], **kwargs_739)
        
        int_741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 51), 'int')
        # Applying the binary operator '+' (line 166)
        result_add_742 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 29), '+', int_call_result_740, int_741)
        
        # Processing the call keyword arguments (line 166)
        kwargs_743 = {}
        # Getting the type of 'random' (line 166)
        random_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'random', False)
        # Obtaining the member 'randrange' of a type (line 166)
        randrange_735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), random_734, 'randrange')
        # Calling randrange(args, kwargs) (line 166)
        randrange_call_result_744 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), randrange_735, *[result_add_742], **kwargs_743)
        
        # Assigning a type to the variable 'r' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'r', randrange_call_result_744)
        
        # Assigning a Num to a Name (line 167):
        
        # Assigning a Num to a Name (line 167):
        int_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'int')
        # Assigning a type to the variable 'l' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'l', int_745)
        
        
        # Call to range(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to len(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 168)
        num_trafos_749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 27), self_748, 'num_trafos')
        # Processing the call keyword arguments (line 168)
        kwargs_750 = {}
        # Getting the type of 'len' (line 168)
        len_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), 'len', False)
        # Calling len(args, kwargs) (line 168)
        len_call_result_751 = invoke(stypy.reporting.localization.Localization(__file__, 168, 23), len_747, *[num_trafos_749], **kwargs_750)
        
        # Processing the call keyword arguments (line 168)
        kwargs_752 = {}
        # Getting the type of 'range' (line 168)
        range_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'range', False)
        # Calling range(args, kwargs) (line 168)
        range_call_result_753 = invoke(stypy.reporting.localization.Localization(__file__, 168, 17), range_746, *[len_call_result_751], **kwargs_752)
        
        # Testing the type of a for loop iterable (line 168)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 8), range_call_result_753)
        # Getting the type of the for loop variable (line 168)
        for_loop_var_754 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 8), range_call_result_753)
        # Assigning a type to the variable 'i' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'i', for_loop_var_754)
        # SSA begins for a for statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'r' (line 169)
        r_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'r')
        # Getting the type of 'l' (line 169)
        l_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'l')
        # Applying the binary operator '>=' (line 169)
        result_ge_757 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), '>=', r_755, l_756)
        
        
        # Getting the type of 'r' (line 169)
        r_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'r')
        # Getting the type of 'l' (line 169)
        l_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 30), 'l')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 169)
        i_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 50), 'i')
        # Getting the type of 'self' (line 169)
        self_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'self')
        # Obtaining the member 'num_trafos' of a type (line 169)
        num_trafos_762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 34), self_761, 'num_trafos')
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 34), num_trafos_762, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_764 = invoke(stypy.reporting.localization.Localization(__file__, 169, 34), getitem___763, i_760)
        
        # Applying the binary operator '+' (line 169)
        result_add_765 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 30), '+', l_759, subscript_call_result_764)
        
        # Applying the binary operator '<' (line 169)
        result_lt_766 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 26), '<', r_758, result_add_765)
        
        # Applying the binary operator 'and' (line 169)
        result_and_keyword_767 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), 'and', result_ge_757, result_lt_766)
        
        # Testing the type of an if condition (line 169)
        if_condition_768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 12), result_and_keyword_767)
        # Assigning a type to the variable 'if_condition_768' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'if_condition_768', if_condition_768)
        # SSA begins for if statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 170)
        tuple_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 170)
        # Adding element type (line 170)
        # Getting the type of 'i' (line 170)
        i_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 23), tuple_769, i_770)
        # Adding element type (line 170)
        
        # Call to randrange(...): (line 170)
        # Processing the call arguments (line 170)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 170)
        i_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 59), 'i', False)
        # Getting the type of 'self' (line 170)
        self_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 43), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 170)
        num_trafos_775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 43), self_774, 'num_trafos')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 43), num_trafos_775, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_777 = invoke(stypy.reporting.localization.Localization(__file__, 170, 43), getitem___776, i_773)
        
        # Processing the call keyword arguments (line 170)
        kwargs_778 = {}
        # Getting the type of 'random' (line 170)
        random_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'random', False)
        # Obtaining the member 'randrange' of a type (line 170)
        randrange_772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 26), random_771, 'randrange')
        # Calling randrange(args, kwargs) (line 170)
        randrange_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 170, 26), randrange_772, *[subscript_call_result_777], **kwargs_778)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 23), tuple_769, randrange_call_result_779)
        
        # Assigning a type to the variable 'stypy_return_type' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'stypy_return_type', tuple_769)
        # SSA join for if statement (line 169)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'l' (line 171)
        l_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'l')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 171)
        i_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'i')
        # Getting the type of 'self' (line 171)
        self_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'self')
        # Obtaining the member 'num_trafos' of a type (line 171)
        num_trafos_783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), self_782, 'num_trafos')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), num_trafos_783, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_785 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), getitem___784, i_781)
        
        # Applying the binary operator '+=' (line 171)
        result_iadd_786 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 12), '+=', l_780, subscript_call_result_785)
        # Assigning a type to the variable 'l' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'l', result_iadd_786)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 172)
        tuple_787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 172)
        # Adding element type (line 172)
        
        # Call to len(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'self' (line 172)
        self_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 172)
        num_trafos_790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 19), self_789, 'num_trafos')
        # Processing the call keyword arguments (line 172)
        kwargs_791 = {}
        # Getting the type of 'len' (line 172)
        len_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'len', False)
        # Calling len(args, kwargs) (line 172)
        len_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 172, 15), len_788, *[num_trafos_790], **kwargs_791)
        
        int_793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 38), 'int')
        # Applying the binary operator '-' (line 172)
        result_sub_794 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), '-', len_call_result_792, int_793)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 15), tuple_787, result_sub_794)
        # Adding element type (line 172)
        
        # Call to randrange(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Obtaining the type of the subscript
        int_797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 74), 'int')
        # Getting the type of 'self' (line 172)
        self_798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 58), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 172)
        num_trafos_799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 58), self_798, 'num_trafos')
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 58), num_trafos_799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_801 = invoke(stypy.reporting.localization.Localization(__file__, 172, 58), getitem___800, int_797)
        
        # Processing the call keyword arguments (line 172)
        kwargs_802 = {}
        # Getting the type of 'random' (line 172)
        random_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 41), 'random', False)
        # Obtaining the member 'randrange' of a type (line 172)
        randrange_796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 41), random_795, 'randrange')
        # Calling randrange(args, kwargs) (line 172)
        randrange_call_result_803 = invoke(stypy.reporting.localization.Localization(__file__, 172, 41), randrange_796, *[subscript_call_result_801], **kwargs_802)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 15), tuple_787, randrange_call_result_803)
        
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', tuple_787)
        
        # ################# End of 'get_random_trafo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_random_trafo' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_804)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_random_trafo'
        return stypy_return_type_804


    @norecursion
    def transform_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 174)
        None_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'None')
        defaults = [None_805]
        # Create a new context for function 'transform_point'
        module_type_store = module_type_store.open_function_context('transform_point', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chaosgame.transform_point.__dict__.__setitem__('stypy_localization', localization)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_function_name', 'Chaosgame.transform_point')
        Chaosgame.transform_point.__dict__.__setitem__('stypy_param_names_list', ['point', 'trafo'])
        Chaosgame.transform_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.transform_point', ['point', 'trafo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform_point', localization, ['point', 'trafo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform_point(...)' code ##################

        
        # Assigning a BinOp to a Name (line 175):
        
        # Assigning a BinOp to a Name (line 175):
        # Getting the type of 'point' (line 175)
        point_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), 'point')
        # Obtaining the member 'x' of a type (line 175)
        x_807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 13), point_806, 'x')
        # Getting the type of 'self' (line 175)
        self_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'self')
        # Obtaining the member 'minx' of a type (line 175)
        minx_809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), self_808, 'minx')
        # Applying the binary operator '-' (line 175)
        result_sub_810 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 13), '-', x_807, minx_809)
        
        
        # Call to float(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'self' (line 175)
        self_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), 'self', False)
        # Obtaining the member 'width' of a type (line 175)
        width_813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 42), self_812, 'width')
        # Processing the call keyword arguments (line 175)
        kwargs_814 = {}
        # Getting the type of 'float' (line 175)
        float_811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 36), 'float', False)
        # Calling float(args, kwargs) (line 175)
        float_call_result_815 = invoke(stypy.reporting.localization.Localization(__file__, 175, 36), float_811, *[width_813], **kwargs_814)
        
        # Applying the binary operator 'div' (line 175)
        result_div_816 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 12), 'div', result_sub_810, float_call_result_815)
        
        # Assigning a type to the variable 'x' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'x', result_div_816)
        
        # Assigning a BinOp to a Name (line 176):
        
        # Assigning a BinOp to a Name (line 176):
        # Getting the type of 'point' (line 176)
        point_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'point')
        # Obtaining the member 'y' of a type (line 176)
        y_818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 13), point_817, 'y')
        # Getting the type of 'self' (line 176)
        self_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'self')
        # Obtaining the member 'miny' of a type (line 176)
        miny_820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 23), self_819, 'miny')
        # Applying the binary operator '-' (line 176)
        result_sub_821 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 13), '-', y_818, miny_820)
        
        
        # Call to float(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), 'self', False)
        # Obtaining the member 'height' of a type (line 176)
        height_824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 42), self_823, 'height')
        # Processing the call keyword arguments (line 176)
        kwargs_825 = {}
        # Getting the type of 'float' (line 176)
        float_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'float', False)
        # Calling float(args, kwargs) (line 176)
        float_call_result_826 = invoke(stypy.reporting.localization.Localization(__file__, 176, 36), float_822, *[height_824], **kwargs_825)
        
        # Applying the binary operator 'div' (line 176)
        result_div_827 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 12), 'div', result_sub_821, float_call_result_826)
        
        # Assigning a type to the variable 'y' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'y', result_div_827)
        
        # Type idiom detected: calculating its left and rigth part (line 177)
        # Getting the type of 'trafo' (line 177)
        trafo_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'trafo')
        # Getting the type of 'None' (line 177)
        None_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'None')
        
        (may_be_830, more_types_in_union_831) = may_be_none(trafo_828, None_829)

        if may_be_830:

            if more_types_in_union_831:
                # Runtime conditional SSA (line 177)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 178):
            
            # Assigning a Call to a Name (line 178):
            
            # Call to get_random_trafo(...): (line 178)
            # Processing the call keyword arguments (line 178)
            kwargs_834 = {}
            # Getting the type of 'self' (line 178)
            self_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'self', False)
            # Obtaining the member 'get_random_trafo' of a type (line 178)
            get_random_trafo_833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 20), self_832, 'get_random_trafo')
            # Calling get_random_trafo(args, kwargs) (line 178)
            get_random_trafo_call_result_835 = invoke(stypy.reporting.localization.Localization(__file__, 178, 20), get_random_trafo_833, *[], **kwargs_834)
            
            # Assigning a type to the variable 'trafo' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'trafo', get_random_trafo_call_result_835)

            if more_types_in_union_831:
                # SSA join for if statement (line 177)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 179):
        
        # Assigning a Subscript to a Name (line 179):
        
        # Obtaining the type of the subscript
        int_836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'int')
        
        # Call to GetDomain(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_846 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 40), 'int')
        # Getting the type of 'trafo' (line 179)
        trafo_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 34), trafo_838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_840 = invoke(stypy.reporting.localization.Localization(__file__, 179, 34), getitem___839, int_837)
        
        # Getting the type of 'self' (line 179)
        self_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'self', False)
        # Obtaining the member 'splines' of a type (line 179)
        splines_842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), self_841, 'splines')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), splines_842, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_844 = invoke(stypy.reporting.localization.Localization(__file__, 179, 21), getitem___843, subscript_call_result_840)
        
        # Obtaining the member 'GetDomain' of a type (line 179)
        GetDomain_845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), subscript_call_result_844, 'GetDomain')
        # Calling GetDomain(args, kwargs) (line 179)
        GetDomain_call_result_847 = invoke(stypy.reporting.localization.Localization(__file__, 179, 21), GetDomain_845, *[], **kwargs_846)
        
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), GetDomain_call_result_847, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_849 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), getitem___848, int_836)
        
        # Assigning a type to the variable 'tuple_var_assignment_1' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'tuple_var_assignment_1', subscript_call_result_849)
        
        # Assigning a Subscript to a Name (line 179):
        
        # Obtaining the type of the subscript
        int_850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 8), 'int')
        
        # Call to GetDomain(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_860 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 40), 'int')
        # Getting the type of 'trafo' (line 179)
        trafo_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 34), trafo_852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_854 = invoke(stypy.reporting.localization.Localization(__file__, 179, 34), getitem___853, int_851)
        
        # Getting the type of 'self' (line 179)
        self_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'self', False)
        # Obtaining the member 'splines' of a type (line 179)
        splines_856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), self_855, 'splines')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), splines_856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_858 = invoke(stypy.reporting.localization.Localization(__file__, 179, 21), getitem___857, subscript_call_result_854)
        
        # Obtaining the member 'GetDomain' of a type (line 179)
        GetDomain_859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), subscript_call_result_858, 'GetDomain')
        # Calling GetDomain(args, kwargs) (line 179)
        GetDomain_call_result_861 = invoke(stypy.reporting.localization.Localization(__file__, 179, 21), GetDomain_859, *[], **kwargs_860)
        
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), GetDomain_call_result_861, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_863 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), getitem___862, int_850)
        
        # Assigning a type to the variable 'tuple_var_assignment_2' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'tuple_var_assignment_2', subscript_call_result_863)
        
        # Assigning a Name to a Name (line 179):
        # Getting the type of 'tuple_var_assignment_1' (line 179)
        tuple_var_assignment_1_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'tuple_var_assignment_1')
        # Assigning a type to the variable 'start' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'start', tuple_var_assignment_1_864)
        
        # Assigning a Name to a Name (line 179):
        # Getting the type of 'tuple_var_assignment_2' (line 179)
        tuple_var_assignment_2_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'tuple_var_assignment_2')
        # Assigning a type to the variable 'end' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'end', tuple_var_assignment_2_865)
        
        # Assigning a BinOp to a Name (line 180):
        
        # Assigning a BinOp to a Name (line 180):
        # Getting the type of 'end' (line 180)
        end_866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'end')
        # Getting the type of 'start' (line 180)
        start_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'start')
        # Applying the binary operator '-' (line 180)
        result_sub_868 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 17), '-', end_866, start_867)
        
        # Assigning a type to the variable 'length' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'length', result_sub_868)
        
        # Assigning a BinOp to a Name (line 181):
        
        # Assigning a BinOp to a Name (line 181):
        # Getting the type of 'length' (line 181)
        length_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'length')
        
        # Call to float(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 58), 'int')
        # Getting the type of 'trafo' (line 181)
        trafo_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 52), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 52), trafo_872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_874 = invoke(stypy.reporting.localization.Localization(__file__, 181, 52), getitem___873, int_871)
        
        # Getting the type of 'self' (line 181)
        self_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 181)
        num_trafos_876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 36), self_875, 'num_trafos')
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 36), num_trafos_876, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_878 = invoke(stypy.reporting.localization.Localization(__file__, 181, 36), getitem___877, subscript_call_result_874)
        
        # Processing the call keyword arguments (line 181)
        kwargs_879 = {}
        # Getting the type of 'float' (line 181)
        float_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'float', False)
        # Calling float(args, kwargs) (line 181)
        float_call_result_880 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), float_870, *[subscript_call_result_878], **kwargs_879)
        
        # Applying the binary operator 'div' (line 181)
        result_div_881 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 21), 'div', length_869, float_call_result_880)
        
        # Assigning a type to the variable 'seg_length' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'seg_length', result_div_881)
        
        # Assigning a BinOp to a Name (line 182):
        
        # Assigning a BinOp to a Name (line 182):
        # Getting the type of 'start' (line 182)
        start_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'start')
        # Getting the type of 'seg_length' (line 182)
        seg_length_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'seg_length')
        
        # Obtaining the type of the subscript
        int_884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 39), 'int')
        # Getting the type of 'trafo' (line 182)
        trafo_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 'trafo')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 33), trafo_885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_887 = invoke(stypy.reporting.localization.Localization(__file__, 182, 33), getitem___886, int_884)
        
        # Applying the binary operator '*' (line 182)
        result_mul_888 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 20), '*', seg_length_883, subscript_call_result_887)
        
        # Applying the binary operator '+' (line 182)
        result_add_889 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 12), '+', start_882, result_mul_888)
        
        # Getting the type of 'seg_length' (line 182)
        seg_length_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'seg_length')
        # Getting the type of 'x' (line 182)
        x_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 57), 'x')
        # Applying the binary operator '*' (line 182)
        result_mul_892 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 44), '*', seg_length_890, x_891)
        
        # Applying the binary operator '+' (line 182)
        result_add_893 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 42), '+', result_add_889, result_mul_892)
        
        # Assigning a type to the variable 't' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 't', result_add_893)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to call(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 't' (line 183)
        t_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 48), 't', False)
        # Processing the call keyword arguments (line 183)
        kwargs_904 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 39), 'int')
        # Getting the type of 'trafo' (line 183)
        trafo_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 33), trafo_895, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_897 = invoke(stypy.reporting.localization.Localization(__file__, 183, 33), getitem___896, int_894)
        
        # Getting the type of 'self' (line 183)
        self_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'self', False)
        # Obtaining the member 'splines' of a type (line 183)
        splines_899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), self_898, 'splines')
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), splines_899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_901 = invoke(stypy.reporting.localization.Localization(__file__, 183, 20), getitem___900, subscript_call_result_897)
        
        # Obtaining the member 'call' of a type (line 183)
        call_902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), subscript_call_result_901, 'call')
        # Calling call(args, kwargs) (line 183)
        call_call_result_905 = invoke(stypy.reporting.localization.Localization(__file__, 183, 20), call_902, *[t_903], **kwargs_904)
        
        # Assigning a type to the variable 'basepoint' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'basepoint', call_call_result_905)
        
        
        # Getting the type of 't' (line 184)
        t_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 't')
        float_907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 15), 'float')
        int_908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 19), 'int')
        # Applying the binary operator 'div' (line 184)
        result_div_909 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), 'div', float_907, int_908)
        
        # Applying the binary operator '+' (line 184)
        result_add_910 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), '+', t_906, result_div_909)
        
        # Getting the type of 'end' (line 184)
        end_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'end')
        # Applying the binary operator '>' (line 184)
        result_gt_912 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), '>', result_add_910, end_911)
        
        # Testing the type of an if condition (line 184)
        if_condition_913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), result_gt_912)
        # Assigning a type to the variable 'if_condition_913' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'if_condition_913', if_condition_913)
        # SSA begins for if statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to call(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 't' (line 185)
        t_923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 52), 't', False)
        float_924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 56), 'float')
        int_925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 60), 'int')
        # Applying the binary operator 'div' (line 185)
        result_div_926 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 56), 'div', float_924, int_925)
        
        # Applying the binary operator '-' (line 185)
        result_sub_927 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 52), '-', t_923, result_div_926)
        
        # Processing the call keyword arguments (line 185)
        kwargs_928 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 43), 'int')
        # Getting the type of 'trafo' (line 185)
        trafo_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 37), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 37), trafo_915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_917 = invoke(stypy.reporting.localization.Localization(__file__, 185, 37), getitem___916, int_914)
        
        # Getting the type of 'self' (line 185)
        self_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'self', False)
        # Obtaining the member 'splines' of a type (line 185)
        splines_919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 24), self_918, 'splines')
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 24), splines_919, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_921 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), getitem___920, subscript_call_result_917)
        
        # Obtaining the member 'call' of a type (line 185)
        call_922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 24), subscript_call_result_921, 'call')
        # Calling call(args, kwargs) (line 185)
        call_call_result_929 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), call_922, *[result_sub_927], **kwargs_928)
        
        # Assigning a type to the variable 'neighbour' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'neighbour', call_call_result_929)
        
        # Assigning a BinOp to a Name (line 186):
        
        # Assigning a BinOp to a Name (line 186):
        # Getting the type of 'neighbour' (line 186)
        neighbour_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'neighbour')
        # Getting the type of 'basepoint' (line 186)
        basepoint_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 37), 'basepoint')
        # Applying the binary operator '-' (line 186)
        result_sub_932 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 25), '-', neighbour_930, basepoint_931)
        
        # Assigning a type to the variable 'derivative' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'derivative', result_sub_932)
        # SSA branch for the else part of an if statement (line 184)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to call(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 't' (line 188)
        t_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 52), 't', False)
        float_943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 56), 'float')
        int_944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 60), 'int')
        # Applying the binary operator 'div' (line 188)
        result_div_945 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 56), 'div', float_943, int_944)
        
        # Applying the binary operator '+' (line 188)
        result_add_946 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 52), '+', t_942, result_div_945)
        
        # Processing the call keyword arguments (line 188)
        kwargs_947 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 43), 'int')
        # Getting the type of 'trafo' (line 188)
        trafo_934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 37), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 37), trafo_934, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 188, 37), getitem___935, int_933)
        
        # Getting the type of 'self' (line 188)
        self_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'self', False)
        # Obtaining the member 'splines' of a type (line 188)
        splines_938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), self_937, 'splines')
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), splines_938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), getitem___939, subscript_call_result_936)
        
        # Obtaining the member 'call' of a type (line 188)
        call_941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), subscript_call_result_940, 'call')
        # Calling call(args, kwargs) (line 188)
        call_call_result_948 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), call_941, *[result_add_946], **kwargs_947)
        
        # Assigning a type to the variable 'neighbour' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'neighbour', call_call_result_948)
        
        # Assigning a BinOp to a Name (line 189):
        
        # Assigning a BinOp to a Name (line 189):
        # Getting the type of 'basepoint' (line 189)
        basepoint_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'basepoint')
        # Getting the type of 'neighbour' (line 189)
        neighbour_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'neighbour')
        # Applying the binary operator '-' (line 189)
        result_sub_951 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 25), '-', basepoint_949, neighbour_950)
        
        # Assigning a type to the variable 'derivative' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'derivative', result_sub_951)
        # SSA join for if statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to Mag(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_954 = {}
        # Getting the type of 'derivative' (line 190)
        derivative_952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'derivative', False)
        # Obtaining the member 'Mag' of a type (line 190)
        Mag_953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), derivative_952, 'Mag')
        # Calling Mag(args, kwargs) (line 190)
        Mag_call_result_955 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), Mag_953, *[], **kwargs_954)
        
        int_956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 31), 'int')
        # Applying the binary operator '!=' (line 190)
        result_ne_957 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), '!=', Mag_call_result_955, int_956)
        
        # Testing the type of an if condition (line 190)
        if_condition_958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), result_ne_957)
        # Assigning a type to the variable 'if_condition_958' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_958', if_condition_958)
        # SSA begins for if statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'basepoint' (line 191)
        basepoint_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'basepoint')
        # Obtaining the member 'x' of a type (line 191)
        x_960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), basepoint_959, 'x')
        # Getting the type of 'derivative' (line 191)
        derivative_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'derivative')
        # Obtaining the member 'y' of a type (line 191)
        y_962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), derivative_961, 'y')
        
        # Call to Mag(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_965 = {}
        # Getting the type of 'derivative' (line 191)
        derivative_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 42), 'derivative', False)
        # Obtaining the member 'Mag' of a type (line 191)
        Mag_964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 42), derivative_963, 'Mag')
        # Calling Mag(args, kwargs) (line 191)
        Mag_call_result_966 = invoke(stypy.reporting.localization.Localization(__file__, 191, 42), Mag_964, *[], **kwargs_965)
        
        # Applying the binary operator 'div' (line 191)
        result_div_967 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 27), 'div', y_962, Mag_call_result_966)
        
        # Getting the type of 'y' (line 191)
        y_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 62), 'y')
        float_969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 66), 'float')
        # Applying the binary operator '-' (line 191)
        result_sub_970 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 62), '-', y_968, float_969)
        
        # Applying the binary operator '*' (line 191)
        result_mul_971 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 59), '*', result_div_967, result_sub_970)
        
        # Getting the type of 'self' (line 192)
        self_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'self')
        # Obtaining the member 'thickness' of a type (line 192)
        thickness_973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 27), self_972, 'thickness')
        # Applying the binary operator '*' (line 191)
        result_mul_974 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 71), '*', result_mul_971, thickness_973)
        
        # Applying the binary operator '+=' (line 191)
        result_iadd_975 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 12), '+=', x_960, result_mul_974)
        # Getting the type of 'basepoint' (line 191)
        basepoint_976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'basepoint')
        # Setting the type of the member 'x' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), basepoint_976, 'x', result_iadd_975)
        
        
        # Getting the type of 'basepoint' (line 193)
        basepoint_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'basepoint')
        # Obtaining the member 'y' of a type (line 193)
        y_978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), basepoint_977, 'y')
        
        # Getting the type of 'derivative' (line 193)
        derivative_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'derivative')
        # Obtaining the member 'x' of a type (line 193)
        x_980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 28), derivative_979, 'x')
        # Applying the 'usub' unary operator (line 193)
        result___neg___981 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 27), 'usub', x_980)
        
        
        # Call to Mag(...): (line 193)
        # Processing the call keyword arguments (line 193)
        kwargs_984 = {}
        # Getting the type of 'derivative' (line 193)
        derivative_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 43), 'derivative', False)
        # Obtaining the member 'Mag' of a type (line 193)
        Mag_983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 43), derivative_982, 'Mag')
        # Calling Mag(args, kwargs) (line 193)
        Mag_call_result_985 = invoke(stypy.reporting.localization.Localization(__file__, 193, 43), Mag_983, *[], **kwargs_984)
        
        # Applying the binary operator 'div' (line 193)
        result_div_986 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 27), 'div', result___neg___981, Mag_call_result_985)
        
        # Getting the type of 'y' (line 193)
        y_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 63), 'y')
        float_988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 67), 'float')
        # Applying the binary operator '-' (line 193)
        result_sub_989 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 63), '-', y_987, float_988)
        
        # Applying the binary operator '*' (line 193)
        result_mul_990 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 60), '*', result_div_986, result_sub_989)
        
        # Getting the type of 'self' (line 194)
        self_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'self')
        # Obtaining the member 'thickness' of a type (line 194)
        thickness_992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 27), self_991, 'thickness')
        # Applying the binary operator '*' (line 193)
        result_mul_993 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 72), '*', result_mul_990, thickness_992)
        
        # Applying the binary operator '+=' (line 193)
        result_iadd_994 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 12), '+=', y_978, result_mul_993)
        # Getting the type of 'basepoint' (line 193)
        basepoint_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'basepoint')
        # Setting the type of the member 'y' of a type (line 193)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), basepoint_995, 'y', result_iadd_994)
        
        # SSA branch for the else part of an if statement (line 190)
        module_type_store.open_ssa_branch('else')
        pass
        # SSA join for if statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to truncate(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'basepoint' (line 197)
        basepoint_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 22), 'basepoint', False)
        # Processing the call keyword arguments (line 197)
        kwargs_999 = {}
        # Getting the type of 'self' (line 197)
        self_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self', False)
        # Obtaining the member 'truncate' of a type (line 197)
        truncate_997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_996, 'truncate')
        # Calling truncate(args, kwargs) (line 197)
        truncate_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), truncate_997, *[basepoint_998], **kwargs_999)
        
        # Getting the type of 'basepoint' (line 198)
        basepoint_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'basepoint')
        # Assigning a type to the variable 'stypy_return_type' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'stypy_return_type', basepoint_1001)
        
        # ################# End of 'transform_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform_point' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1002)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform_point'
        return stypy_return_type_1002


    @norecursion
    def truncate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'truncate'
        module_type_store = module_type_store.open_function_context('truncate', 200, 4, False)
        # Assigning a type to the variable 'self' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chaosgame.truncate.__dict__.__setitem__('stypy_localization', localization)
        Chaosgame.truncate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chaosgame.truncate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chaosgame.truncate.__dict__.__setitem__('stypy_function_name', 'Chaosgame.truncate')
        Chaosgame.truncate.__dict__.__setitem__('stypy_param_names_list', ['point'])
        Chaosgame.truncate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chaosgame.truncate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chaosgame.truncate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chaosgame.truncate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chaosgame.truncate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chaosgame.truncate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.truncate', ['point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'truncate', localization, ['point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'truncate(...)' code ##################

        
        
        # Getting the type of 'point' (line 201)
        point_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'point')
        # Obtaining the member 'x' of a type (line 201)
        x_1004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), point_1003, 'x')
        # Getting the type of 'self' (line 201)
        self_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 22), 'self')
        # Obtaining the member 'maxx' of a type (line 201)
        maxx_1006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 22), self_1005, 'maxx')
        # Applying the binary operator '>=' (line 201)
        result_ge_1007 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), '>=', x_1004, maxx_1006)
        
        # Testing the type of an if condition (line 201)
        if_condition_1008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), result_ge_1007)
        # Assigning a type to the variable 'if_condition_1008' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_1008', if_condition_1008)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 202):
        
        # Assigning a Attribute to a Attribute (line 202):
        # Getting the type of 'self' (line 202)
        self_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'self')
        # Obtaining the member 'maxx' of a type (line 202)
        maxx_1010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 22), self_1009, 'maxx')
        # Getting the type of 'point' (line 202)
        point_1011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'point')
        # Setting the type of the member 'x' of a type (line 202)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), point_1011, 'x', maxx_1010)
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'point' (line 203)
        point_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'point')
        # Obtaining the member 'y' of a type (line 203)
        y_1013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), point_1012, 'y')
        # Getting the type of 'self' (line 203)
        self_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'self')
        # Obtaining the member 'maxy' of a type (line 203)
        maxy_1015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 22), self_1014, 'maxy')
        # Applying the binary operator '>=' (line 203)
        result_ge_1016 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), '>=', y_1013, maxy_1015)
        
        # Testing the type of an if condition (line 203)
        if_condition_1017 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_ge_1016)
        # Assigning a type to the variable 'if_condition_1017' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_1017', if_condition_1017)
        # SSA begins for if statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 204):
        
        # Assigning a Attribute to a Attribute (line 204):
        # Getting the type of 'self' (line 204)
        self_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'self')
        # Obtaining the member 'maxy' of a type (line 204)
        maxy_1019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 22), self_1018, 'maxy')
        # Getting the type of 'point' (line 204)
        point_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'point')
        # Setting the type of the member 'y' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), point_1020, 'y', maxy_1019)
        # SSA join for if statement (line 203)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'point' (line 205)
        point_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'point')
        # Obtaining the member 'x' of a type (line 205)
        x_1022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 11), point_1021, 'x')
        # Getting the type of 'self' (line 205)
        self_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'self')
        # Obtaining the member 'minx' of a type (line 205)
        minx_1024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 21), self_1023, 'minx')
        # Applying the binary operator '<' (line 205)
        result_lt_1025 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), '<', x_1022, minx_1024)
        
        # Testing the type of an if condition (line 205)
        if_condition_1026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_lt_1025)
        # Assigning a type to the variable 'if_condition_1026' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_1026', if_condition_1026)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 206):
        
        # Assigning a Attribute to a Attribute (line 206):
        # Getting the type of 'self' (line 206)
        self_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 22), 'self')
        # Obtaining the member 'minx' of a type (line 206)
        minx_1028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 22), self_1027, 'minx')
        # Getting the type of 'point' (line 206)
        point_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'point')
        # Setting the type of the member 'x' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), point_1029, 'x', minx_1028)
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'point' (line 207)
        point_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'point')
        # Obtaining the member 'y' of a type (line 207)
        y_1031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 11), point_1030, 'y')
        # Getting the type of 'self' (line 207)
        self_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), 'self')
        # Obtaining the member 'miny' of a type (line 207)
        miny_1033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 21), self_1032, 'miny')
        # Applying the binary operator '<' (line 207)
        result_lt_1034 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), '<', y_1031, miny_1033)
        
        # Testing the type of an if condition (line 207)
        if_condition_1035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_lt_1034)
        # Assigning a type to the variable 'if_condition_1035' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_1035', if_condition_1035)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 208):
        
        # Assigning a Attribute to a Attribute (line 208):
        # Getting the type of 'self' (line 208)
        self_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), 'self')
        # Obtaining the member 'miny' of a type (line 208)
        miny_1037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 22), self_1036, 'miny')
        # Getting the type of 'point' (line 208)
        point_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'point')
        # Setting the type of the member 'y' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), point_1038, 'y', miny_1037)
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'truncate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'truncate' in the type store
        # Getting the type of 'stypy_return_type' (line 200)
        stypy_return_type_1039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'truncate'
        return stypy_return_type_1039


    @norecursion
    def create_image_chaos(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_image_chaos'
        module_type_store = module_type_store.open_function_context('create_image_chaos', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_localization', localization)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_function_name', 'Chaosgame.create_image_chaos')
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_param_names_list', ['w', 'h', 'name', 'n'])
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.create_image_chaos', ['w', 'h', 'name', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_image_chaos', localization, ['w', 'h', 'name', 'n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_image_chaos(...)' code ##################

        
        # Assigning a ListComp to a Name (line 211):
        
        # Assigning a ListComp to a Name (line 211):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'w' (line 211)
        w_1045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 37), 'w', False)
        # Processing the call keyword arguments (line 211)
        kwargs_1046 = {}
        # Getting the type of 'range' (line 211)
        range_1044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 31), 'range', False)
        # Calling range(args, kwargs) (line 211)
        range_call_result_1047 = invoke(stypy.reporting.localization.Localization(__file__, 211, 31), range_1044, *[w_1045], **kwargs_1046)
        
        comprehension_1048 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), range_call_result_1047)
        # Assigning a type to the variable 'i' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 14), 'i', comprehension_1048)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_1040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        int_1041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), list_1040, int_1041)
        
        # Getting the type of 'h' (line 211)
        h_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'h')
        # Applying the binary operator '*' (line 211)
        result_mul_1043 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 14), '*', list_1040, h_1042)
        
        list_1049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 14), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), list_1049, result_mul_1043)
        # Assigning a type to the variable 'im' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'im', list_1049)
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to GVector(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'self' (line 212)
        self_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 25), 'self', False)
        # Obtaining the member 'maxx' of a type (line 212)
        maxx_1052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 25), self_1051, 'maxx')
        # Getting the type of 'self' (line 212)
        self_1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'self', False)
        # Obtaining the member 'minx' of a type (line 212)
        minx_1054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 37), self_1053, 'minx')
        # Applying the binary operator '+' (line 212)
        result_add_1055 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 25), '+', maxx_1052, minx_1054)
        
        float_1056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 50), 'float')
        # Applying the binary operator 'div' (line 212)
        result_div_1057 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 24), 'div', result_add_1055, float_1056)
        
        # Getting the type of 'self' (line 213)
        self_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'self', False)
        # Obtaining the member 'maxy' of a type (line 213)
        maxy_1059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 25), self_1058, 'maxy')
        # Getting the type of 'self' (line 213)
        self_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 37), 'self', False)
        # Obtaining the member 'miny' of a type (line 213)
        miny_1061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 37), self_1060, 'miny')
        # Applying the binary operator '+' (line 213)
        result_add_1062 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 25), '+', maxy_1059, miny_1061)
        
        float_1063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 50), 'float')
        # Applying the binary operator 'div' (line 213)
        result_div_1064 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 24), 'div', result_add_1062, float_1063)
        
        int_1065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 55), 'int')
        # Processing the call keyword arguments (line 212)
        kwargs_1066 = {}
        # Getting the type of 'GVector' (line 212)
        GVector_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'GVector', False)
        # Calling GVector(args, kwargs) (line 212)
        GVector_call_result_1067 = invoke(stypy.reporting.localization.Localization(__file__, 212, 16), GVector_1050, *[result_div_1057, result_div_1064, int_1065], **kwargs_1066)
        
        # Assigning a type to the variable 'point' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'point', GVector_call_result_1067)
        
        # Assigning a Num to a Name (line 214):
        
        # Assigning a Num to a Name (line 214):
        int_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 18), 'int')
        # Assigning a type to the variable 'colored' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'colored', int_1068)
        
        # Assigning a List to a Name (line 215):
        
        # Assigning a List to a Name (line 215):
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_1069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        
        # Assigning a type to the variable 'times' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'times', list_1069)
        
        
        # Call to range(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'n' (line 216)
        n_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 23), 'n', False)
        # Processing the call keyword arguments (line 216)
        kwargs_1072 = {}
        # Getting the type of 'range' (line 216)
        range_1070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'range', False)
        # Calling range(args, kwargs) (line 216)
        range_call_result_1073 = invoke(stypy.reporting.localization.Localization(__file__, 216, 17), range_1070, *[n_1071], **kwargs_1072)
        
        # Testing the type of a for loop iterable (line 216)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 8), range_call_result_1073)
        # Getting the type of the for loop variable (line 216)
        for_loop_var_1074 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 8), range_call_result_1073)
        # Assigning a type to the variable '_' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), '_', for_loop_var_1074)
        # SSA begins for a for statement (line 216)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to time(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_1077 = {}
        # Getting the type of 'time' (line 217)
        time_1075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), 'time', False)
        # Obtaining the member 'time' of a type (line 217)
        time_1076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 17), time_1075, 'time')
        # Calling time(args, kwargs) (line 217)
        time_call_result_1078 = invoke(stypy.reporting.localization.Localization(__file__, 217, 17), time_1076, *[], **kwargs_1077)
        
        # Assigning a type to the variable 't1' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 't1', time_call_result_1078)
        
        
        # Call to xrange(...): (line 218)
        # Processing the call arguments (line 218)
        int_1080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 28), 'int')
        # Processing the call keyword arguments (line 218)
        kwargs_1081 = {}
        # Getting the type of 'xrange' (line 218)
        xrange_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'xrange', False)
        # Calling xrange(args, kwargs) (line 218)
        xrange_call_result_1082 = invoke(stypy.reporting.localization.Localization(__file__, 218, 21), xrange_1079, *[int_1080], **kwargs_1081)
        
        # Testing the type of a for loop iterable (line 218)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 218, 12), xrange_call_result_1082)
        # Getting the type of the for loop variable (line 218)
        for_loop_var_1083 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 218, 12), xrange_call_result_1082)
        # Assigning a type to the variable 'i' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'i', for_loop_var_1083)
        # SSA begins for a for statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to transform_point(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'point' (line 219)
        point_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 45), 'point', False)
        # Processing the call keyword arguments (line 219)
        kwargs_1087 = {}
        # Getting the type of 'self' (line 219)
        self_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'self', False)
        # Obtaining the member 'transform_point' of a type (line 219)
        transform_point_1085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 24), self_1084, 'transform_point')
        # Calling transform_point(args, kwargs) (line 219)
        transform_point_call_result_1088 = invoke(stypy.reporting.localization.Localization(__file__, 219, 24), transform_point_1085, *[point_1086], **kwargs_1087)
        
        # Assigning a type to the variable 'point' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'point', transform_point_call_result_1088)
        
        # Assigning a BinOp to a Name (line 220):
        
        # Assigning a BinOp to a Name (line 220):
        # Getting the type of 'point' (line 220)
        point_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'point')
        # Obtaining the member 'x' of a type (line 220)
        x_1090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), point_1089, 'x')
        # Getting the type of 'self' (line 220)
        self_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 31), 'self')
        # Obtaining the member 'minx' of a type (line 220)
        minx_1092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 31), self_1091, 'minx')
        # Applying the binary operator '-' (line 220)
        result_sub_1093 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 21), '-', x_1090, minx_1092)
        
        # Getting the type of 'self' (line 220)
        self_1094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 44), 'self')
        # Obtaining the member 'width' of a type (line 220)
        width_1095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 44), self_1094, 'width')
        # Applying the binary operator 'div' (line 220)
        result_div_1096 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 20), 'div', result_sub_1093, width_1095)
        
        # Getting the type of 'w' (line 220)
        w_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 57), 'w')
        # Applying the binary operator '*' (line 220)
        result_mul_1098 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 55), '*', result_div_1096, w_1097)
        
        # Assigning a type to the variable 'x' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'x', result_mul_1098)
        
        # Assigning a BinOp to a Name (line 221):
        
        # Assigning a BinOp to a Name (line 221):
        # Getting the type of 'point' (line 221)
        point_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'point')
        # Obtaining the member 'y' of a type (line 221)
        y_1100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 21), point_1099, 'y')
        # Getting the type of 'self' (line 221)
        self_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 31), 'self')
        # Obtaining the member 'miny' of a type (line 221)
        miny_1102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 31), self_1101, 'miny')
        # Applying the binary operator '-' (line 221)
        result_sub_1103 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 21), '-', y_1100, miny_1102)
        
        # Getting the type of 'self' (line 221)
        self_1104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'self')
        # Obtaining the member 'height' of a type (line 221)
        height_1105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 44), self_1104, 'height')
        # Applying the binary operator 'div' (line 221)
        result_div_1106 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 20), 'div', result_sub_1103, height_1105)
        
        # Getting the type of 'h' (line 221)
        h_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 58), 'h')
        # Applying the binary operator '*' (line 221)
        result_mul_1108 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 56), '*', result_div_1106, h_1107)
        
        # Assigning a type to the variable 'y' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'y', result_mul_1108)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to int(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'x' (line 222)
        x_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'x', False)
        # Processing the call keyword arguments (line 222)
        kwargs_1111 = {}
        # Getting the type of 'int' (line 222)
        int_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'int', False)
        # Calling int(args, kwargs) (line 222)
        int_call_result_1112 = invoke(stypy.reporting.localization.Localization(__file__, 222, 20), int_1109, *[x_1110], **kwargs_1111)
        
        # Assigning a type to the variable 'x' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'x', int_call_result_1112)
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to int(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'y' (line 223)
        y_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 24), 'y', False)
        # Processing the call keyword arguments (line 223)
        kwargs_1115 = {}
        # Getting the type of 'int' (line 223)
        int_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'int', False)
        # Calling int(args, kwargs) (line 223)
        int_call_result_1116 = invoke(stypy.reporting.localization.Localization(__file__, 223, 20), int_1113, *[y_1114], **kwargs_1115)
        
        # Assigning a type to the variable 'y' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'y', int_call_result_1116)
        
        
        # Getting the type of 'x' (line 224)
        x_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'x')
        # Getting the type of 'w' (line 224)
        w_1118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'w')
        # Applying the binary operator '==' (line 224)
        result_eq_1119 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 19), '==', x_1117, w_1118)
        
        # Testing the type of an if condition (line 224)
        if_condition_1120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 16), result_eq_1119)
        # Assigning a type to the variable 'if_condition_1120' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'if_condition_1120', if_condition_1120)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'x' (line 225)
        x_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'x')
        int_1122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 25), 'int')
        # Applying the binary operator '-=' (line 225)
        result_isub_1123 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 20), '-=', x_1121, int_1122)
        # Assigning a type to the variable 'x' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'x', result_isub_1123)
        
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'y' (line 226)
        y_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'y')
        # Getting the type of 'h' (line 226)
        h_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'h')
        # Applying the binary operator '==' (line 226)
        result_eq_1126 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 19), '==', y_1124, h_1125)
        
        # Testing the type of an if condition (line 226)
        if_condition_1127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 16), result_eq_1126)
        # Assigning a type to the variable 'if_condition_1127' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'if_condition_1127', if_condition_1127)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'y' (line 227)
        y_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'y')
        int_1129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 25), 'int')
        # Applying the binary operator '-=' (line 227)
        result_isub_1130 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 20), '-=', y_1128, int_1129)
        # Assigning a type to the variable 'y' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'y', result_isub_1130)
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Num to a Subscript (line 228):
        
        # Assigning a Num to a Subscript (line 228):
        int_1131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 35), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'x' (line 228)
        x_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'x')
        # Getting the type of 'im' (line 228)
        im_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'im')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___1134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), im_1133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_1135 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), getitem___1134, x_1132)
        
        # Getting the type of 'h' (line 228)
        h_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'h')
        # Getting the type of 'y' (line 228)
        y_1137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'y')
        # Applying the binary operator '-' (line 228)
        result_sub_1138 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 22), '-', h_1136, y_1137)
        
        int_1139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 30), 'int')
        # Applying the binary operator '-' (line 228)
        result_sub_1140 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 28), '-', result_sub_1138, int_1139)
        
        # Storing an element on a container (line 228)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 16), subscript_call_result_1135, (result_sub_1140, int_1131))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to time(...): (line 229)
        # Processing the call keyword arguments (line 229)
        kwargs_1143 = {}
        # Getting the type of 'time' (line 229)
        time_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), 'time', False)
        # Obtaining the member 'time' of a type (line 229)
        time_1142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 17), time_1141, 'time')
        # Calling time(args, kwargs) (line 229)
        time_call_result_1144 = invoke(stypy.reporting.localization.Localization(__file__, 229, 17), time_1142, *[], **kwargs_1143)
        
        # Assigning a type to the variable 't2' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 't2', time_call_result_1144)
        
        # Call to append(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 't2' (line 230)
        t2_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 't2', False)
        # Getting the type of 't1' (line 230)
        t1_1148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 30), 't1', False)
        # Applying the binary operator '-' (line 230)
        result_sub_1149 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 25), '-', t2_1147, t1_1148)
        
        # Processing the call keyword arguments (line 230)
        kwargs_1150 = {}
        # Getting the type of 'times' (line 230)
        times_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'times', False)
        # Obtaining the member 'append' of a type (line 230)
        append_1146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), times_1145, 'append')
        # Calling append(args, kwargs) (line 230)
        append_call_result_1151 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), append_1146, *[result_sub_1149], **kwargs_1150)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to save_im(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'im' (line 231)
        im_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'im', False)
        # Getting the type of 'name' (line 231)
        name_1154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'name', False)
        # Processing the call keyword arguments (line 231)
        kwargs_1155 = {}
        # Getting the type of 'save_im' (line 231)
        save_im_1152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'save_im', False)
        # Calling save_im(args, kwargs) (line 231)
        save_im_call_result_1156 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), save_im_1152, *[im_1153, name_1154], **kwargs_1155)
        
        # Getting the type of 'times' (line 232)
        times_1157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'times')
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', times_1157)
        
        # ################# End of 'create_image_chaos(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_image_chaos' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_image_chaos'
        return stypy_return_type_1158


# Assigning a type to the variable 'Chaosgame' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'Chaosgame', Chaosgame)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 235, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = ['n']
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Assigning a List to a Name (line 236):
    
    # Assigning a List to a Name (line 236):
    
    # Obtaining an instance of the builtin type 'list' (line 236)
    list_1159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 236)
    # Adding element type (line 236)
    
    # Call to Spline(...): (line 237)
    # Processing the call arguments (line 237)
    
    # Obtaining an instance of the builtin type 'list' (line 237)
    list_1161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 237)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 238)
    # Processing the call arguments (line 238)
    float_1163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 20), 'float')
    float_1164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 30), 'float')
    float_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 40), 'float')
    # Processing the call keyword arguments (line 238)
    kwargs_1166 = {}
    # Getting the type of 'GVector' (line 238)
    GVector_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 238)
    GVector_call_result_1167 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), GVector_1162, *[float_1163, float_1164, float_1165], **kwargs_1166)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1161, GVector_call_result_1167)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 239)
    # Processing the call arguments (line 239)
    float_1169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 20), 'float')
    float_1170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 30), 'float')
    float_1171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 40), 'float')
    # Processing the call keyword arguments (line 239)
    kwargs_1172 = {}
    # Getting the type of 'GVector' (line 239)
    GVector_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 239)
    GVector_call_result_1173 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), GVector_1168, *[float_1169, float_1170, float_1171], **kwargs_1172)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1161, GVector_call_result_1173)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 240)
    # Processing the call arguments (line 240)
    float_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 20), 'float')
    float_1176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 30), 'float')
    float_1177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 40), 'float')
    # Processing the call keyword arguments (line 240)
    kwargs_1178 = {}
    # Getting the type of 'GVector' (line 240)
    GVector_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 240)
    GVector_call_result_1179 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), GVector_1174, *[float_1175, float_1176, float_1177], **kwargs_1178)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1161, GVector_call_result_1179)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 241)
    # Processing the call arguments (line 241)
    float_1181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'float')
    float_1182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'float')
    float_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 40), 'float')
    # Processing the call keyword arguments (line 241)
    kwargs_1184 = {}
    # Getting the type of 'GVector' (line 241)
    GVector_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 241)
    GVector_call_result_1185 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), GVector_1180, *[float_1181, float_1182, float_1183], **kwargs_1184)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1161, GVector_call_result_1185)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 242)
    # Processing the call arguments (line 242)
    float_1187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 20), 'float')
    float_1188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 30), 'float')
    float_1189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 40), 'float')
    # Processing the call keyword arguments (line 242)
    kwargs_1190 = {}
    # Getting the type of 'GVector' (line 242)
    GVector_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 242)
    GVector_call_result_1191 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), GVector_1186, *[float_1187, float_1188, float_1189], **kwargs_1190)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1161, GVector_call_result_1191)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 243)
    # Processing the call arguments (line 243)
    float_1193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 20), 'float')
    float_1194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 30), 'float')
    float_1195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 40), 'float')
    # Processing the call keyword arguments (line 243)
    kwargs_1196 = {}
    # Getting the type of 'GVector' (line 243)
    GVector_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 243)
    GVector_call_result_1197 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), GVector_1192, *[float_1193, float_1194, float_1195], **kwargs_1196)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1161, GVector_call_result_1197)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 244)
    # Processing the call arguments (line 244)
    float_1199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'float')
    float_1200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'float')
    float_1201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 40), 'float')
    # Processing the call keyword arguments (line 244)
    kwargs_1202 = {}
    # Getting the type of 'GVector' (line 244)
    GVector_1198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 244)
    GVector_call_result_1203 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), GVector_1198, *[float_1199, float_1200, float_1201], **kwargs_1202)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1161, GVector_call_result_1203)
    
    int_1204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 12), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 245)
    list_1205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 245)
    # Adding element type (line 245)
    int_1206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1206)
    # Adding element type (line 245)
    int_1207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1207)
    # Adding element type (line 245)
    int_1208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1208)
    # Adding element type (line 245)
    int_1209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1209)
    # Adding element type (line 245)
    int_1210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1210)
    # Adding element type (line 245)
    int_1211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1211)
    # Adding element type (line 245)
    int_1212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1212)
    # Adding element type (line 245)
    int_1213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1213)
    # Adding element type (line 245)
    int_1214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1205, int_1214)
    
    # Processing the call keyword arguments (line 237)
    kwargs_1215 = {}
    # Getting the type of 'Spline' (line 237)
    Spline_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'Spline', False)
    # Calling Spline(args, kwargs) (line 237)
    Spline_call_result_1216 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), Spline_1160, *[list_1161, int_1204, list_1205], **kwargs_1215)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 14), list_1159, Spline_call_result_1216)
    # Adding element type (line 236)
    
    # Call to Spline(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Obtaining an instance of the builtin type 'list' (line 246)
    list_1218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 246)
    # Adding element type (line 246)
    
    # Call to GVector(...): (line 247)
    # Processing the call arguments (line 247)
    float_1220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'float')
    float_1221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 30), 'float')
    float_1222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 40), 'float')
    # Processing the call keyword arguments (line 247)
    kwargs_1223 = {}
    # Getting the type of 'GVector' (line 247)
    GVector_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 247)
    GVector_call_result_1224 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), GVector_1219, *[float_1220, float_1221, float_1222], **kwargs_1223)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), list_1218, GVector_call_result_1224)
    # Adding element type (line 246)
    
    # Call to GVector(...): (line 248)
    # Processing the call arguments (line 248)
    float_1226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 20), 'float')
    float_1227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 30), 'float')
    float_1228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 40), 'float')
    # Processing the call keyword arguments (line 248)
    kwargs_1229 = {}
    # Getting the type of 'GVector' (line 248)
    GVector_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 248)
    GVector_call_result_1230 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), GVector_1225, *[float_1226, float_1227, float_1228], **kwargs_1229)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), list_1218, GVector_call_result_1230)
    # Adding element type (line 246)
    
    # Call to GVector(...): (line 249)
    # Processing the call arguments (line 249)
    float_1232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'float')
    float_1233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 30), 'float')
    float_1234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 40), 'float')
    # Processing the call keyword arguments (line 249)
    kwargs_1235 = {}
    # Getting the type of 'GVector' (line 249)
    GVector_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 249)
    GVector_call_result_1236 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), GVector_1231, *[float_1232, float_1233, float_1234], **kwargs_1235)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), list_1218, GVector_call_result_1236)
    # Adding element type (line 246)
    
    # Call to GVector(...): (line 250)
    # Processing the call arguments (line 250)
    float_1238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'float')
    float_1239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 30), 'float')
    float_1240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 40), 'float')
    # Processing the call keyword arguments (line 250)
    kwargs_1241 = {}
    # Getting the type of 'GVector' (line 250)
    GVector_1237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 250)
    GVector_call_result_1242 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), GVector_1237, *[float_1238, float_1239, float_1240], **kwargs_1241)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), list_1218, GVector_call_result_1242)
    
    int_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 12), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 251)
    list_1244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 251)
    # Adding element type (line 251)
    int_1245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1244, int_1245)
    # Adding element type (line 251)
    int_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1244, int_1246)
    # Adding element type (line 251)
    int_1247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1244, int_1247)
    # Adding element type (line 251)
    int_1248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1244, int_1248)
    # Adding element type (line 251)
    int_1249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1244, int_1249)
    # Adding element type (line 251)
    int_1250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1244, int_1250)
    
    # Processing the call keyword arguments (line 246)
    kwargs_1251 = {}
    # Getting the type of 'Spline' (line 246)
    Spline_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'Spline', False)
    # Calling Spline(args, kwargs) (line 246)
    Spline_call_result_1252 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), Spline_1217, *[list_1218, int_1243, list_1244], **kwargs_1251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 14), list_1159, Spline_call_result_1252)
    # Adding element type (line 236)
    
    # Call to Spline(...): (line 252)
    # Processing the call arguments (line 252)
    
    # Obtaining an instance of the builtin type 'list' (line 252)
    list_1254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 252)
    # Adding element type (line 252)
    
    # Call to GVector(...): (line 253)
    # Processing the call arguments (line 253)
    float_1256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'float')
    float_1257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 30), 'float')
    float_1258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 40), 'float')
    # Processing the call keyword arguments (line 253)
    kwargs_1259 = {}
    # Getting the type of 'GVector' (line 253)
    GVector_1255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 253)
    GVector_call_result_1260 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), GVector_1255, *[float_1256, float_1257, float_1258], **kwargs_1259)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 15), list_1254, GVector_call_result_1260)
    # Adding element type (line 252)
    
    # Call to GVector(...): (line 254)
    # Processing the call arguments (line 254)
    float_1262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'float')
    float_1263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 30), 'float')
    float_1264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 40), 'float')
    # Processing the call keyword arguments (line 254)
    kwargs_1265 = {}
    # Getting the type of 'GVector' (line 254)
    GVector_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 254)
    GVector_call_result_1266 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), GVector_1261, *[float_1262, float_1263, float_1264], **kwargs_1265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 15), list_1254, GVector_call_result_1266)
    # Adding element type (line 252)
    
    # Call to GVector(...): (line 255)
    # Processing the call arguments (line 255)
    float_1268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 20), 'float')
    float_1269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 30), 'float')
    float_1270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 40), 'float')
    # Processing the call keyword arguments (line 255)
    kwargs_1271 = {}
    # Getting the type of 'GVector' (line 255)
    GVector_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 255)
    GVector_call_result_1272 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), GVector_1267, *[float_1268, float_1269, float_1270], **kwargs_1271)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 15), list_1254, GVector_call_result_1272)
    # Adding element type (line 252)
    
    # Call to GVector(...): (line 256)
    # Processing the call arguments (line 256)
    float_1274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'float')
    float_1275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 30), 'float')
    float_1276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'float')
    # Processing the call keyword arguments (line 256)
    kwargs_1277 = {}
    # Getting the type of 'GVector' (line 256)
    GVector_1273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 256)
    GVector_call_result_1278 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), GVector_1273, *[float_1274, float_1275, float_1276], **kwargs_1277)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 15), list_1254, GVector_call_result_1278)
    
    int_1279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 257)
    list_1280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 257)
    # Adding element type (line 257)
    int_1281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1280, int_1281)
    # Adding element type (line 257)
    int_1282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1280, int_1282)
    # Adding element type (line 257)
    int_1283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1280, int_1283)
    # Adding element type (line 257)
    int_1284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1280, int_1284)
    # Adding element type (line 257)
    int_1285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1280, int_1285)
    # Adding element type (line 257)
    int_1286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1280, int_1286)
    
    # Processing the call keyword arguments (line 252)
    kwargs_1287 = {}
    # Getting the type of 'Spline' (line 252)
    Spline_1253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'Spline', False)
    # Calling Spline(args, kwargs) (line 252)
    Spline_call_result_1288 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), Spline_1253, *[list_1254, int_1279, list_1280], **kwargs_1287)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 14), list_1159, Spline_call_result_1288)
    
    # Assigning a type to the variable 'splines' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'splines', list_1159)
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 259):
    
    # Call to Chaosgame(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'splines' (line 259)
    splines_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'splines', False)
    float_1291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'float')
    # Processing the call keyword arguments (line 259)
    kwargs_1292 = {}
    # Getting the type of 'Chaosgame' (line 259)
    Chaosgame_1289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'Chaosgame', False)
    # Calling Chaosgame(args, kwargs) (line 259)
    Chaosgame_call_result_1293 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), Chaosgame_1289, *[splines_1290, float_1291], **kwargs_1292)
    
    # Assigning a type to the variable 'c' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'c', Chaosgame_call_result_1293)
    
    # Call to create_image_chaos(...): (line 260)
    # Processing the call arguments (line 260)
    int_1296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'int')
    int_1297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 38), 'int')
    
    # Call to Relative(...): (line 260)
    # Processing the call arguments (line 260)
    str_1299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 53), 'str', 'py.ppm')
    # Processing the call keyword arguments (line 260)
    kwargs_1300 = {}
    # Getting the type of 'Relative' (line 260)
    Relative_1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 44), 'Relative', False)
    # Calling Relative(args, kwargs) (line 260)
    Relative_call_result_1301 = invoke(stypy.reporting.localization.Localization(__file__, 260, 44), Relative_1298, *[str_1299], **kwargs_1300)
    
    # Getting the type of 'n' (line 260)
    n_1302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 64), 'n', False)
    # Processing the call keyword arguments (line 260)
    kwargs_1303 = {}
    # Getting the type of 'c' (line 260)
    c_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'c', False)
    # Obtaining the member 'create_image_chaos' of a type (line 260)
    create_image_chaos_1295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 11), c_1294, 'create_image_chaos')
    # Calling create_image_chaos(args, kwargs) (line 260)
    create_image_chaos_call_result_1304 = invoke(stypy.reporting.localization.Localization(__file__, 260, 11), create_image_chaos_1295, *[int_1296, int_1297, Relative_call_result_1301, n_1302], **kwargs_1303)
    
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type', create_image_chaos_call_result_1304)
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 235)
    stypy_return_type_1305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_1305

# Assigning a type to the variable 'main' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 263, 0, False)
    
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

    
    # Call to main(...): (line 264)
    # Processing the call arguments (line 264)
    int_1307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 9), 'int')
    # Processing the call keyword arguments (line 264)
    kwargs_1308 = {}
    # Getting the type of 'main' (line 264)
    main_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'main', False)
    # Calling main(args, kwargs) (line 264)
    main_call_result_1309 = invoke(stypy.reporting.localization.Localization(__file__, 264, 4), main_1306, *[int_1307], **kwargs_1308)
    
    # Getting the type of 'True' (line 265)
    True_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type', True_1310)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_1311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1311)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1311

# Assigning a type to the variable 'run' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'run', run)

# Call to run(...): (line 267)
# Processing the call keyword arguments (line 267)
kwargs_1313 = {}
# Getting the type of 'run' (line 267)
run_1312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'run', False)
# Calling run(args, kwargs) (line 267)
run_call_result_1314 = invoke(stypy.reporting.localization.Localization(__file__, 267, 0), run_1312, *[], **kwargs_1313)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
